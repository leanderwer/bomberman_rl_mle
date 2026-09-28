"""Analyze the internal learning process of one Task 1 training run."""

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


DEFAULT_AGENT_NAME = "q_learning_agent"
SUPPORTED_TRAINING_METRICS_SCHEMA_VERSIONS = {1, 2}
TASK_1_LEARNING_ACTIONS = 5
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

    for name in ("scenario", "feature_version", "reward_version"):
        value = metadata.get(name)
        if not isinstance(value, str) or not value:
            raise ValueError(
                f"'training_metadata.{name}' must be a non-empty string."
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
        if pairs > states * TASK_1_LEARNING_ACTIONS:
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
            if round_data["action_counts"]["BOMB"] != 0:
                raise ValueError(
                    "Task 1 telemetry contains BOMB actions."
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
    overall_td = calculate_td_metrics(rounds)
    early_td = calculate_td_metrics(early_rounds)
    late_td = calculate_td_metrics(late_rounds)

    final_states = rounds[-1]["q_table_states"]
    final_pairs = rounds[-1]["updated_state_action_pairs"]
    possible_pairs = final_states * TASK_1_LEARNING_ACTIONS
    final_coverage = final_pairs / possible_pairs if possible_pairs else 0.0

    wall_time = metadata["training_wall_time_seconds"]
    rounds_per_second = number_of_rounds / wall_time if wall_time else None
    updates_per_second = overall_td["q_updates"] / wall_time if wall_time else None

    per_round = []
    for round_data, moving_return in zip(rounds, moving_returns):
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
                pairs / (states * TASK_1_LEARNING_ACTIONS)
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

        per_round.append(row)

    early_return = mean(row["undiscounted_return"] for row in early_rounds)
    late_return = mean(row["undiscounted_return"] for row in late_rounds)

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

    return {
        "metadata": dict(metadata),
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
    print(f"  Feature version: {metadata['feature_version']}")
    print(f"  Reward version: {metadata['reward_version']}")

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
        description="Analyze one Task 1 training result."
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
