"""Analyze the internal learning process of one Task 3 training run."""

import argparse
import json
import math
from pathlib import Path
import re
from statistics import mean, pstdev

import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.exploration_metrics import (
    exploration_strategy, exploration_summary, print_exploration_summary,
    validate_exploration_round,
)

from evaluation.task3.block_metrics import summarize_block_diagnostics
from evaluation.task3.combat_metrics import combat_round_row


DEFAULT_AGENT_NAME = "q_learning_agent"
SUPPORTED_TRAINING_METRICS_SCHEMA_VERSIONS = {2}
TASK_3_LEARNING_ACTIONS = 6
TASK_3_ACTION_SPACE_VERSION = "task2_all_actions_v1"
LEGACY_TD_TARGET_VERSION = "q_learning_1_step_v1"
ACTION_ORDER = (
    "UP",
    "RIGHT",
    "DOWN",
    "LEFT",
    "WAIT",
    "BOMB",
)
ANALYSIS_WINDOW_FRACTION = 0.1
MAX_ANALYSIS_WINDOW_ROUNDS = 100

ROUND_FIELDS = {
    "schema_version",
    "round",
    "undiscounted_return",
    "q_updates",
    "td_error_abs_sum",
    "td_error_squared_sum",
    "td_error_abs_max",
    "q_table_states",
    "new_q_table_states",
    "updated_state_action_pairs",
    "new_updated_state_action_pairs",
    "reward_components",
}

SCHEMA_2_ROUND_FIELDS = {
    "action_counts",
    "action_update_counts",
}


def load_training_results(file_path: Path) -> dict:
    """Load one framework statistics file containing training data."""
    try:
        with file_path.open("r", encoding="utf-8") as file:
            results = json.load(file)
    except FileNotFoundError as error:
        raise ValueError(f"Result file does not exist: {file_path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Result file is not valid JSON: {error}") from error

    if not isinstance(results, dict):
        raise ValueError("The top-level JSON value must be an object.")

    return results


def require_integer(value, location: str, minimum: int = 0) -> int:
    """Return one validated integer."""
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
    ):
        raise ValueError(
            f"'{location}' must be an integer of at least {minimum}."
        )
    return value


def require_number(value, location: str) -> float:
    """Return one validated finite number."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"'{location}' must be a finite number.")
    return float(value)


def get_gameplay_rounds(results: dict) -> dict[int, dict]:
    """Index the existing framework statistics by round number."""
    round_statistics = results.get("by_round")
    if not isinstance(round_statistics, dict):
        raise ValueError("'by_round' must be an object.")

    indexed_rounds = {}
    for identifier, round_data in round_statistics.items():
        match = re.search(r"\bRound\s+(\d+)\b", identifier)
        if match is None:
            raise ValueError(f"Cannot read round number from {identifier!r}.")

        round_number = int(match.group(1))
        if round_number in indexed_rounds:
            raise ValueError(f"Duplicate gameplay round {round_number}.")
        if not isinstance(round_data, dict):
            raise ValueError(f"Gameplay round {round_number} must be an object.")

        indexed_rounds[round_number] = round_data

    return indexed_rounds


def validate_metadata(metadata: dict) -> None:
    """Validate metadata supplied by the experiment framework."""
    if metadata.get("mode") != "training":
        raise ValueError("'training_metadata.mode' must be 'training'.")

    for name in ("world_seed", "agent_seed"):
        require_integer(metadata.get(name), f"training_metadata.{name}")

    for name in (
        "scenario",
        "feature_version",
        "reward_version",
        "learning_rate_version",
        "exploration_version",
    ):
        value = metadata.get(name)
        if not isinstance(value, str) or not value:
            raise ValueError(
                f"'training_metadata.{name}' must be a non-empty string."
            )

    discount_factor_version = metadata.get(
        "discount_factor_version"
    )
    if (
        discount_factor_version is not None
        and (
            not isinstance(discount_factor_version, str)
            or not discount_factor_version
        )
    ):
        raise ValueError(
            "'training_metadata.discount_factor_version' must be "
            "a non-empty string when present."
        )

    td_target_version = metadata.get(
        "td_target_version",
        LEGACY_TD_TARGET_VERSION,
    )
    if not isinstance(td_target_version, str) or not td_target_version:
        raise ValueError(
            "'training_metadata.td_target_version' must be a "
            "non-empty string when present."
        )

    # Training result files created before TD-target versioning used the
    # historical one-step update exclusively. Normalize that implicit
    # contract so old and new analyses expose the same metadata field.
    metadata.setdefault(
        "td_target_version",
        LEGACY_TD_TARGET_VERSION,
    )

    if metadata.get("action_space_version") != TASK_3_ACTION_SPACE_VERSION:
        raise ValueError(
            "'training_metadata.action_space_version' must be "
            f"{TASK_3_ACTION_SPACE_VERSION!r}."
        )

    wall_time = require_number(
        metadata.get("training_wall_time_seconds"),
        "training_metadata.training_wall_time_seconds",
    )
    if wall_time < 0:
        raise ValueError("Training wall time cannot be negative.")


def validate_reward_components(round_data: dict, location: str) -> None:
    """Validate dynamic event contributions and their return sum."""
    components = round_data["reward_components"]
    if not isinstance(components, dict):
        raise ValueError(f"'{location}.reward_components' must be an object.")

    component_sum = 0.0
    for event, component in components.items():
        component_location = f"{location}.reward_components.{event}"
        if not isinstance(event, str) or not event:
            raise ValueError("Reward event names must be non-empty strings.")
        if not isinstance(component, dict):
            raise ValueError(f"'{component_location}' must be an object.")

        require_integer(component.get("count"), f"{component_location}.count")
        component_sum += require_number(
            component.get("reward_sum"),
            f"{component_location}.reward_sum",
        )

    if not math.isclose(
        component_sum,
        round_data["undiscounted_return"],
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise ValueError(
            f"Reward components do not sum to the return at '{location}'."
        )


def validate_action_counts(
    round_data: dict,
    field_name: str,
    location: str,
) -> int:
    """Validate one complete per-action counter and return its sum."""
    action_counts = round_data[field_name]
    field_location = f"{location}.{field_name}"

    if not isinstance(action_counts, dict):
        raise ValueError(f"'{field_location}' must be an object.")

    if set(action_counts) != set(ACTION_ORDER):
        raise ValueError(
            f"'{field_location}' must contain exactly "
            f"{list(ACTION_ORDER)}."
        )

    return sum(
        require_integer(
            action_counts[action],
            f"{field_location}.{action}",
        )
        for action in ACTION_ORDER
    )

def validate_training_results(
    results: dict,
    agent_name: str = DEFAULT_AGENT_NAME,
) -> tuple[dict, list[dict]]:
    """Validate and return metadata and per-round telemetry."""
    metadata = results.get("training_metadata")
    if not isinstance(metadata, dict):
        raise ValueError("The file does not contain 'training_metadata'.")
    validate_metadata(metadata)

    training_by_agent = results.get("training_by_agent")
    if not isinstance(training_by_agent, dict):
        raise ValueError("The file does not contain 'training_by_agent'.")

    rounds = training_by_agent.get(agent_name)
    if not isinstance(rounds, list) or not rounds:
        raise ValueError(f"No telemetry found for agent {agent_name!r}.")

    gameplay_rounds = get_gameplay_rounds(results)
    if len(gameplay_rounds) != len(rounds):
        raise ValueError(
            "Gameplay and training telemetry contain different round counts."
        )

    previous_states = 0
    previous_pairs = 0
    strategy = exploration_strategy(metadata)
    previous_exploration_end = None

    for expected_round, round_data in enumerate(rounds, start=1):
        location = (
            f"training_by_agent.{agent_name}[{expected_round - 1}]"
        )
        if not isinstance(round_data, dict):
            raise ValueError(f"'{location}' must be an object.")

        missing = ROUND_FIELDS - round_data.keys()
        if missing:
            raise ValueError(
                f"'{location}' is missing fields: {sorted(missing)}."
            )

        schema = require_integer(
            round_data["schema_version"],
            f"{location}.schema_version",
            minimum=1,
        )
        if schema not in SUPPORTED_TRAINING_METRICS_SCHEMA_VERSIONS:
            raise ValueError(
                f"Unsupported training schema {schema}; "
                "supported versions are "
                f"{sorted(SUPPORTED_TRAINING_METRICS_SCHEMA_VERSIONS)}."
            )

        if schema >= 2:
            missing_schema_2 = (
                SCHEMA_2_ROUND_FIELDS - round_data.keys()
            )
            if missing_schema_2:
                raise ValueError(
                    f"'{location}' is missing schema-2 fields: "
                    f"{sorted(missing_schema_2)}."
                )

        round_number = require_integer(
            round_data["round"], f"{location}.round", minimum=1
        )
        if round_number != expected_round:
            raise ValueError("Training rounds must be consecutive from 1.")

        for name in (
            "undiscounted_return",
            "td_error_abs_sum",
            "td_error_squared_sum",
            "td_error_abs_max",
        ):
            require_number(round_data[name], f"{location}.{name}")

        previous_exploration_end = validate_exploration_round(
            round_data, strategy, previous_exploration_end, location,
        )

        q_updates = require_integer(
            round_data["q_updates"], f"{location}.q_updates", minimum=1
        )
        states = require_integer(
            round_data["q_table_states"], f"{location}.q_table_states"
        )
        new_states = require_integer(
            round_data["new_q_table_states"],
            f"{location}.new_q_table_states",
        )
        pairs = require_integer(
            round_data["updated_state_action_pairs"],
            f"{location}.updated_state_action_pairs",
        )
        new_pairs = require_integer(
            round_data["new_updated_state_action_pairs"],
            f"{location}.new_updated_state_action_pairs",
        )

        if states - previous_states != new_states:
            raise ValueError(f"Inconsistent Q-table growth in round {round_number}.")
        if pairs - previous_pairs != new_pairs:
            raise ValueError(
                f"Inconsistent state-action growth in round {round_number}."
            )
        if pairs > states * TASK_3_LEARNING_ACTIONS:
            raise ValueError(f"Impossible state-action count in round {round_number}.")
        if new_pairs > q_updates:
            raise ValueError(f"More new pairs than updates in round {round_number}.")
        if min(
            round_data["td_error_abs_sum"],
            round_data["td_error_squared_sum"],
            round_data["td_error_abs_max"],
        ) < 0:
            raise ValueError(f"Negative TD-error aggregate in round {round_number}.")
        if round_data["td_error_abs_max"] > (
            round_data["td_error_abs_sum"] + 1e-12
        ):
            raise ValueError(f"Impossible TD-error maximum in round {round_number}.")

        validate_reward_components(round_data, location)

        gameplay_round = gameplay_rounds.get(round_number, {})
        gameplay_agent = gameplay_round.get("by_agent", {}).get(agent_name)
        if not isinstance(gameplay_agent, dict):
            raise ValueError(
                f"Missing gameplay data for agent in round {round_number}."
            )
        steps = require_integer(
            gameplay_agent.get("steps"),
            f"by_round[{round_number}].by_agent.{agent_name}.steps",
            minimum=1,
        )
        if q_updates != steps:
            raise ValueError(
                f"Round {round_number} has {steps} steps but "
                f"{q_updates} Q-updates."
            )

        if schema >= 2:
            selected_actions = validate_action_counts(
                round_data,
                "action_counts",
                location,
            )
            updated_actions = validate_action_counts(
                round_data,
                "action_update_counts",
                location,
            )

            if selected_actions != steps:
                raise ValueError(
                    f"Round {round_number} has {steps} steps but "
                    f"{selected_actions} selected actions."
                )
            if updated_actions != q_updates:
                raise ValueError(
                    f"Round {round_number} has {q_updates} Q-updates "
                    f"but {updated_actions} action-labelled updates."
                )
        previous_states = states
        previous_pairs = pairs

    return metadata, rounds


def calculate_window_size(number_of_rounds: int) -> int:
    """Return the shared early, late, and smoothing window size."""
    return min(
        number_of_rounds,
        MAX_ANALYSIS_WINDOW_ROUNDS,
        max(1, math.ceil(ANALYSIS_WINDOW_FRACTION * number_of_rounds)),
    )


def calculate_moving_average(
    values: list[float], window_size: int
) -> list[float]:
    """Return a trailing moving average with a growing initial window."""
    result = []
    running_sum = 0.0

    for index, value in enumerate(values):
        running_sum += value
        if index >= window_size:
            running_sum -= values[index - window_size]
        result.append(running_sum / min(index + 1, window_size))

    return result


def calculate_gameplay_behavior(
    results: dict,
    rounds: list[dict],
    agent_name: str,
    window_size: int,
) -> tuple[list[dict], list[str]]:
    """Return inherited and combat outcomes for every training round."""
    gameplay_rounds = get_gameplay_rounds(results)
    behavior_rows = []
    opponent_names = None

    for round_data in rounds:
        round_number = round_data["round"]
        gameplay_round = gameplay_rounds[round_number]
        combat = combat_round_row(
            f"Round {round_number}",
            gameplay_round,
            agent_name,
        )

        if opponent_names is None:
            opponent_names = combat["opponents"]
        elif combat["opponents"] != opponent_names:
            raise ValueError(
                "Every Task 3 training round must use the same "
                "ordered opponent roster."
            )

        if combat["kill_auc"] is None:
            raise ValueError(
                "Task 3 training results require per-round "
                "'kill_events' telemetry."
            )

        gameplay_agent = gameplay_round.get(
            "by_agent",
            {},
        ).get(agent_name)

        if not isinstance(gameplay_agent, dict):
            raise ValueError(
                "Missing gameplay data for agent in round "
                f"{round_number}."
            )

        initial_crates = require_integer(
            gameplay_round.get("initial_crates"),
            f"by_round[{round_number}].initial_crates",
        )
        remaining_crates = require_integer(
            gameplay_round.get("remaining_crates"),
            f"by_round[{round_number}].remaining_crates",
        )
        crates_destroyed = require_integer(
            gameplay_agent.get("crates"),
            (
                f"by_round[{round_number}].by_agent."
                f"{agent_name}.crates"
            ),
        )
        coins_collected = require_integer(
            gameplay_agent.get("coins"),
            (
                f"by_round[{round_number}].by_agent."
                f"{agent_name}.coins"
            ),
        )
        survived = gameplay_agent.get("survived")

        if not isinstance(survived, bool):
            raise ValueError(
                f"Training round {round_number} has no boolean "
                "survival outcome."
            )
        if remaining_crates > initial_crates:
            raise ValueError(
                f"Training round {round_number} contains more "
                "remaining than initial crates."
            )
        globally_destroyed_crates = initial_crates - remaining_crates

        if crates_destroyed > globally_destroyed_crates:
            raise ValueError(
                f"Training round {round_number} attributes more "
                "destroyed crates to the learning agent than were "
                "destroyed globally."
            )

        behavior_rows.append(
            {
                "coins_collected": coins_collected,
                "crate_destruction_rate": (
                    crates_destroyed / initial_crates
                    if initial_crates
                    else 0.0
                ),
                "survived": float(survived),
                "kills": combat["kills"],
                "all_opponents_eliminated": float(
                    combat["all_opponents_eliminated"]
                ),
                "death_by_opponent": float(
                    combat["death_by_opponent"]
                ),
                "kill_auc": combat["kill_auc"],
            }
        )

    moving_coins = calculate_moving_average(
        [row["coins_collected"] for row in behavior_rows],
        window_size,
    )
    moving_crate_rates = calculate_moving_average(
        [row["crate_destruction_rate"] for row in behavior_rows],
        window_size,
    )
    moving_survival_rates = calculate_moving_average(
        [row["survived"] for row in behavior_rows],
        window_size,
    )
    moving_kills = calculate_moving_average(
        [row["kills"] for row in behavior_rows],
        window_size,
    )
    moving_all_eliminated = calculate_moving_average(
        [row["all_opponents_eliminated"] for row in behavior_rows],
        window_size,
    )
    moving_death_by_opponent = calculate_moving_average(
        [row["death_by_opponent"] for row in behavior_rows],
        window_size,
    )
    moving_kill_auc = calculate_moving_average(
        [row["kill_auc"] for row in behavior_rows],
        window_size,
    )

    for (
        row,
        coins,
        crates,
        survival,
        kills,
        all_eliminated,
        death_by_opponent,
        kill_auc,
    ) in zip(
        behavior_rows,
        moving_coins,
        moving_crate_rates,
        moving_survival_rates,
        moving_kills,
        moving_all_eliminated,
        moving_death_by_opponent,
        moving_kill_auc,
    ):
        row.update(
            {
                "coins_collected_moving_average": coins,
                "crate_destruction_rate_moving_average": crates,
                "survival_rate_moving_average": survival,
                "kills_moving_average": kills,
                "all_opponents_eliminated_moving_average": (
                    all_eliminated
                ),
                "death_by_opponent_moving_average": (
                    death_by_opponent
                ),
                "kill_auc_moving_average": kill_auc,
            }
        )

    return behavior_rows, list(opponent_names or ())


def calculate_td_metrics(rounds: list[dict]) -> dict:
    """Aggregate TD errors with individual Q-updates as observations."""
    updates = sum(round_data["q_updates"] for round_data in rounds)
    absolute_sum = sum(
        round_data["td_error_abs_sum"] for round_data in rounds
    )
    squared_sum = sum(
        round_data["td_error_squared_sum"] for round_data in rounds
    )

    return {
        "q_updates": updates,
        "mae": absolute_sum / updates,
        "rmse": math.sqrt(squared_sum / updates),
        "absolute_max": max(
            round_data["td_error_abs_max"] for round_data in rounds
        ),
    }


def aggregate_reward_components(rounds: list[dict]) -> dict:
    """Aggregate arbitrary standard and custom reward events."""
    aggregated = {}
    for round_data in rounds:
        for event, component in round_data["reward_components"].items():
            target = aggregated.setdefault(
                event, {"count": 0, "reward_sum": 0.0}
            )
            target["count"] += component["count"]
            target["reward_sum"] += component["reward_sum"]

    number_of_rounds = len(rounds)
    return {
        event: {
            **component,
            "mean_count_per_round": component["count"] / number_of_rounds,
            "mean_reward_per_round": (
                component["reward_sum"] / number_of_rounds
            ),
        }
        for event, component in sorted(aggregated.items())
    }


def calculate_training_metrics(
    results: dict,
    agent_name: str = DEFAULT_AGENT_NAME,
) -> dict:
    """Calculate the central learning-process metrics."""
    metadata, rounds = validate_training_results(results, agent_name)
    number_of_rounds = len(rounds)
    window_size = calculate_window_size(number_of_rounds)
    early_rounds = rounds[:window_size]
    late_rounds = rounds[-window_size:]

    returns = [row["undiscounted_return"] for row in rounds]
    moving_returns = calculate_moving_average(returns, window_size)
    gameplay_behavior, opponents = calculate_gameplay_behavior(
        results,
        rounds,
        agent_name,
        window_size,
    )
    overall_td = calculate_td_metrics(rounds)
    early_td = calculate_td_metrics(early_rounds)
    late_td = calculate_td_metrics(late_rounds)

    final_states = rounds[-1]["q_table_states"]
    final_pairs = rounds[-1]["updated_state_action_pairs"]
    possible_pairs = final_states * TASK_3_LEARNING_ACTIONS
    final_coverage = final_pairs / possible_pairs if possible_pairs else 0.0

    wall_time = metadata["training_wall_time_seconds"]
    rounds_per_second = number_of_rounds / wall_time if wall_time else None
    updates_per_second = overall_td["q_updates"] / wall_time if wall_time else None

    per_round = []
    for round_data, moving_return, behavior in zip(
        rounds,
        moving_returns,
        gameplay_behavior,
    ):
        round_td = calculate_td_metrics([round_data])
        states = round_data["q_table_states"]
        pairs = round_data["updated_state_action_pairs"]

        row = {
            "round": round_data["round"],
            "undiscounted_return": round_data[
                "undiscounted_return"
            ],
            "return_moving_average": moving_return,
            "epsilon_end": round_data.get("epsilon_end"),
            "temperature_end": round_data.get("temperature_end"),
            "q_updates": round_data["q_updates"],
            "td_error_mae": round_td["mae"],
            "td_error_rmse": round_td["rmse"],
            "td_error_abs_max": round_td["absolute_max"],
            "q_table_states": states,
            "new_q_table_states": round_data[
                "new_q_table_states"
            ],
            "updated_state_action_pairs": pairs,
            "new_updated_state_action_pairs": round_data[
                "new_updated_state_action_pairs"
            ],
            "state_action_coverage": (
                pairs / (states * TASK_3_LEARNING_ACTIONS)
                if states else 0.0
            ),
        }

        if "action_counts" in round_data:
            row["action_counts"] = dict(
                round_data["action_counts"]
            )
            row["action_update_counts"] = dict(
                round_data["action_update_counts"]
            )

        row.update(behavior)

        per_round.append(row)

    early_return = mean(row["undiscounted_return"] for row in early_rounds)
    late_return = mean(row["undiscounted_return"] for row in late_rounds)
    early_behavior = gameplay_behavior[:window_size]
    late_behavior = gameplay_behavior[-window_size:]
    mean_kills = mean(row["kills"] for row in gameplay_behavior)
    early_mean_kills = mean(row["kills"] for row in early_behavior)
    late_mean_kills = mean(row["kills"] for row in late_behavior)

    summary = {
        "rounds": number_of_rounds,
        "analysis_window_rounds": window_size,
        "mean_return": mean(returns),
        "return_standard_deviation": pstdev(returns),
        "early_mean_return": early_return,
        "late_mean_return": late_return,
        "return_improvement": late_return - early_return,
        "td_error_mae": overall_td["mae"],
        "td_error_rmse": overall_td["rmse"],
        "td_error_abs_max": overall_td["absolute_max"],
        "early_td_error_mae": early_td["mae"],
        "late_td_error_mae": late_td["mae"],
        "td_error_mae_change": late_td["mae"] - early_td["mae"],
        **exploration_summary(metadata, rounds),
        "total_q_updates": overall_td["q_updates"],
        "mean_q_updates_per_round": overall_td["q_updates"] / number_of_rounds,
        "final_q_table_states": final_states,
        "mean_new_q_table_states_per_round": mean(
            row["new_q_table_states"] for row in rounds
        ),
        "late_mean_new_q_table_states": mean(
            row["new_q_table_states"] for row in late_rounds
        ),
        "final_updated_state_action_pairs": final_pairs,
        "final_state_action_coverage": final_coverage,
        "mean_new_updated_pairs_per_round": mean(
            row["new_updated_state_action_pairs"] for row in rounds
        ),
        "late_mean_new_updated_pairs": mean(
            row["new_updated_state_action_pairs"] for row in late_rounds
        ),
        "training_wall_time_seconds": wall_time,
        "rounds_per_second": rounds_per_second,
        "updates_per_second": updates_per_second,
        "mean_kills_per_round": mean_kills,
        "early_mean_kills_per_round": early_mean_kills,
        "late_mean_kills_per_round": late_mean_kills,
        "kills_per_round_improvement": (
            late_mean_kills - early_mean_kills
        ),
        "all_opponents_eliminated_rate": mean(
            row["all_opponents_eliminated"]
            for row in gameplay_behavior
        ),
        "late_all_opponents_eliminated_rate": mean(
            row["all_opponents_eliminated"]
            for row in late_behavior
        ),
        "death_by_opponent_rate": mean(
            row["death_by_opponent"]
            for row in gameplay_behavior
        ),
        "late_death_by_opponent_rate": mean(
            row["death_by_opponent"]
            for row in late_behavior
        ),
        "mean_kill_auc": mean(
            row["kill_auc"] for row in gameplay_behavior
        ),
        "late_mean_kill_auc": mean(
            row["kill_auc"] for row in late_behavior
        ),
    }

    action_counts = {
        action: sum(
            row.get("action_counts", {}).get(action, 0)
            for row in rounds
        )
        for action in ACTION_ORDER
    }
    action_update_counts = {
        action: sum(
            row.get("action_update_counts", {}).get(action, 0)
            for row in rounds
        )
        for action in ACTION_ORDER
    }
    total_selected_actions = sum(action_counts.values())
    total_action_updates = sum(action_update_counts.values())
    summary.update(
        {
            "bomb_action_count": action_counts["BOMB"],
            "bomb_action_rate": (
                action_counts["BOMB"] / total_selected_actions
                if total_selected_actions
                else 0.0
            ),
            "bomb_update_count": action_update_counts["BOMB"],
            "bomb_update_rate": (
                action_update_counts["BOMB"] / total_action_updates
                if total_action_updates
                else 0.0
            ),
        }
    )

    summary.update(summarize_block_diagnostics(
        [row.get("block_diagnostics", {}) for row in rounds]))

    return {
        "metadata": dict(metadata),
        "opponents": opponents,
        "summary": summary,
        "reward_components": aggregate_reward_components(rounds),
        "action_counts": action_counts,
        "action_update_counts": action_update_counts,
        "per_round": per_round,
    }

def analyze_training_file(
    file_path: Path,
    agent_name: str = DEFAULT_AGENT_NAME,
) -> dict:
    """Load, validate, and analyze one training statistics file."""
    results = load_training_results(file_path)

    return calculate_training_metrics(
        results,
        agent_name,
    )

def print_training_metrics(metrics: dict) -> None:
    """Print one compact human-readable report."""
    metadata = metrics["metadata"]
    summary = metrics["summary"]

    print("Training run")
    print(
        f"  Seeds: world={metadata['world_seed']}, "
        f"agent={metadata['agent_seed']}"
    )
    print(f"  Scenario: {metadata['scenario']}")
    print("  Opponents: " + ", ".join(metrics["opponents"]))
    print(f"  Feature version: {metadata['feature_version']}")
    print(f"  Reward version: {metadata['reward_version']}")
    print(
        "  Learning-rate / exploration / discount: "
        f"{metadata['learning_rate_version']} / "
        f"{metadata['exploration_version']} / "
        f"{metadata.get('discount_factor_version', 'legacy')}"
    )
    print(
        "  TD-target version: "
        f"{metadata['td_target_version']}"
    )

    print("\nReturn development")
    print(f"  Rounds: {summary['rounds']}")
    print(f"  Analysis window: {summary['analysis_window_rounds']} round(s)")
    print(f"  Mean return: {summary['mean_return']:.6f}")
    print(
        "  Return standard deviation: "
        f"{summary['return_standard_deviation']:.6f}"
    )
    print(f"  Early mean return: {summary['early_mean_return']:.6f}")
    print(f"  Late mean return: {summary['late_mean_return']:.6f}")
    print(
        "  Return improvement (late - early): "
        f"{summary['return_improvement']:.6f}"
    )

    print("\nCombat development")
    print(
        "  Mean / early / late kills per round: "
        f"{summary['mean_kills_per_round']:.3f} / "
        f"{summary['early_mean_kills_per_round']:.3f} / "
        f"{summary['late_mean_kills_per_round']:.3f}"
    )
    print(
        "  All-opponents-eliminated overall / late: "
        f"{100 * summary['all_opponents_eliminated_rate']:.2f}% / "
        f"{100 * summary['late_all_opponents_eliminated_rate']:.2f}%"
    )
    print(
        "  Death by opponent overall / late: "
        f"{100 * summary['death_by_opponent_rate']:.2f}% / "
        f"{100 * summary['late_death_by_opponent_rate']:.2f}%"
    )
    print(
        "  Kill AUC overall / late: "
        f"{100 * summary['mean_kill_auc']:.2f}% / "
        f"{100 * summary['late_mean_kill_auc']:.2f}%"
    )

    print("\nTD-error development")
    print(f"  Overall MAE: {summary['td_error_mae']:.6f}")
    print(f"  Overall RMSE: {summary['td_error_rmse']:.6f}")
    print(f"  Maximum absolute: {summary['td_error_abs_max']:.6f}")
    print(f"  Early MAE: {summary['early_td_error_mae']:.6f}")
    print(f"  Late MAE: {summary['late_td_error_mae']:.6f}")
    print(
        "  MAE change (late - early): "
        f"{summary['td_error_mae_change']:.6f}"
    )

    print("\nExploration and state space")
    print_exploration_summary(summary)
    print(f"  Final Q-table states: {summary['final_q_table_states']}")
    print(
        "  Final updated state-action pairs: "
        f"{summary['final_updated_state_action_pairs']}"
    )
    print(
        "  Final state-action coverage: "
        f"{100 * summary['final_state_action_coverage']:.2f}%"
    )
    print(
        "  Mean / late new states per round: "
        f"{summary['mean_new_q_table_states_per_round']:.2f} / "
        f"{summary['late_mean_new_q_table_states']:.2f}"
    )
    print(
        "  Mean / late new pairs per round: "
        f"{summary['mean_new_updated_pairs_per_round']:.2f} / "
        f"{summary['late_mean_new_updated_pairs']:.2f}"
    )

    print("\nTraining efficiency")
    print(f"  Q-updates: {summary['total_q_updates']}")
    print(
        "  Mean Q-updates per round: "
        f"{summary['mean_q_updates_per_round']:.2f}"
    )
    print(f"  Wall time: {summary['training_wall_time_seconds']:.6f} s")
    if summary["rounds_per_second"] is None:
        print("  Rounds / Q-updates per second: N/A")
    else:
        print(f"  Rounds per second: {summary['rounds_per_second']:.2f}")
        print(f"  Q-updates per second: {summary['updates_per_second']:.2f}")

    if any(metrics.get("action_counts", {}).values()):
        print("\nSelected actions / Q-updates")
        for action in ACTION_ORDER:
            print(
                f"  {action}: "
                f"{metrics['action_counts'][action]} / "
                f"{metrics['action_update_counts'][action]}"
            )
        print(
            "  BOMB selection rate: "
            f"{100 * summary['bomb_action_rate']:.2f}%"
        )

    print("\nReward components")
    for event, component in metrics["reward_components"].items():
        print(
            f"  {event}: count={component['count']}, "
            f"mean_count/round={component['mean_count_per_round']:.2f}, "
            f"reward_sum={component['reward_sum']:.6f}, "
            f"mean_reward/round={component['mean_reward_per_round']:.6f}"
        )


def parse_arguments():
    """Parse the training result and optional agent name."""
    parser = argparse.ArgumentParser(
        description="Analyze one Task 3 training result."
    )
    parser.add_argument(
        "result_file",
        type=Path,
        help="Training JSON file created with --save-stats.",
    )
    parser.add_argument(
        "--agent",
        default=DEFAULT_AGENT_NAME,
        help="Agent key inside training_by_agent.",
    )
    return parser.parse_args()


def main():
    """Load, validate, analyze, and print one training run."""
    args = parse_arguments()
    try:
        metrics = analyze_training_file(
            args.result_file,
            args.agent,
        )
    except ValueError as error:
        raise SystemExit(f"Training analysis failed: {error}")

    print_training_metrics(metrics)


if __name__ == "__main__":
    main()
