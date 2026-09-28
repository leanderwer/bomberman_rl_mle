"""Analyze one Task 2 evaluation result."""

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev


AGENT_NAME = "q_learning_agent"
DEFAULT_COINS_PER_ROUND = 9
MAX_STEPS_PER_ROUND = 400
DECISION_TIME_LIMIT_SECONDS = 0.5

ACTION_DIAGNOSTIC_PAIRS = (
    (
        "coin_objective",
        "coin_objective_opportunities",
        "coin_objective_followed",
    ),
    (
        "crate_objective",
        "crate_objective_opportunities",
        "crate_objective_followed",
    ),
    (
        "escape_objective",
        "escape_objective_opportunities",
        "escape_objective_followed",
    ),
    (
        "safe_productive_bomb",
        "safe_productive_bomb_opportunities",
        "safe_productive_bomb_followed",
    ),
    (
        "threat_urgency_0",
        "threat_urgency_0_opportunities",
        "threat_urgency_0_followed",
    ),
    (
        "threat_urgency_1",
        "threat_urgency_1_opportunities",
        "threat_urgency_1_followed",
    ),
    (
        "threat_urgency_2",
        "threat_urgency_2_opportunities",
        "threat_urgency_2_followed",
    ),
    (
        "threat_urgency_3",
        "threat_urgency_3_opportunities",
        "threat_urgency_3_followed",
    ),
    (
        "target_conflict_coin_strong",
        "target_conflict_coin_strong_opportunities",
        "target_conflict_coin_strong_followed",
    ),
    (
        "target_conflict_coin_weak",
        "target_conflict_coin_weak_opportunities",
        "target_conflict_coin_weak_followed",
    ),
    (
        "target_conflict_bomb_weak",
        "target_conflict_bomb_weak_opportunities",
        "target_conflict_bomb_weak_followed",
    ),
    (
        "target_conflict_bomb_strong",
        "target_conflict_bomb_strong_opportunities",
        "target_conflict_bomb_strong_followed",
    ),
    (
        "safe_joint_target",
        "safe_joint_target_opportunities",
        "safe_joint_target_followed",
    ),
    (
        "safe_coin_only_target",
        "safe_coin_only_target_opportunities",
        "safe_coin_only_target_followed",
    ),
    (
        "safe_bomb_only_target",
        "safe_bomb_only_target_opportunities",
        "safe_bomb_only_target_followed",
    ),
    (
        "bomb_target_precision",
        "valid_bomb_drops",
        "bomb_drops_at_target",
    ),
    (
        "bomb_target_response",
        "bomb_target_opportunities",
        "bomb_drops_at_primary_target",
    ),
    (
        "coin_route_bomb_joint_conversion",
        "coin_route_bomb_joint_conversion_opportunities",
        "coin_route_bomb_joint_conversion_followed",
    ),
    (
        "coin_route_bomb_alternative_target",
        "coin_route_bomb_alternative_target_opportunities",
        "coin_route_bomb_alternative_target_followed",
    ),
)

TARGET_CONFLICT_CLASS_PAIRS = tuple(
    (coin_class, bomb_class)
    for coin_class in range(4)
    for bomb_class in range(3)
)
TARGET_CONFLICT_DIAGNOSTIC_PAIRS = tuple(
    diagnostic_pair
    for coin_class, bomb_class in TARGET_CONFLICT_CLASS_PAIRS
    for diagnostic_pair in (
        (
            (
                "target_conflict_"
                f"coin_class_{coin_class}_bomb_class_{bomb_class}_coin"
            ),
            (
                "target_conflict_"
                f"coin_class_{coin_class}_bomb_class_{bomb_class}_"
                "opportunities"
            ),
            (
                "target_conflict_"
                f"coin_class_{coin_class}_bomb_class_{bomb_class}_"
                "coin_followed"
            ),
        ),
        (
            (
                "target_conflict_"
                f"coin_class_{coin_class}_bomb_class_{bomb_class}_bomb"
            ),
            (
                "target_conflict_"
                f"coin_class_{coin_class}_bomb_class_{bomb_class}_"
                "opportunities"
            ),
            (
                "target_conflict_"
                f"coin_class_{coin_class}_bomb_class_{bomb_class}_"
                "bomb_followed"
            ),
        ),
    )
)
ACTION_DIAGNOSTIC_PAIRS += TARGET_CONFLICT_DIAGNOSTIC_PAIRS

ACTION_DIAGNOSTIC_COUNTERS = (
    (
        "target_conflict_equal",
        "target_conflict_equal_opportunities",
    ),
    (
        "no_safe_target",
        "no_safe_target_opportunities",
    ),
    (
        "coin_route_bomb_alternative",
        "coin_route_bomb_alternative_opportunities",
    ),
)

COIN_ROUTE_BOMB_SCORE_RATIO_SUM = (
    "coin_route_bomb_alternative_score_ratio_milli_sum"
)
COIN_ROUTE_BOMB_DIAGNOSTIC_MARKER = (
    "coin_route_bomb_diagnostic_observations"
)

SELF_KILL_DIAGNOSTIC_COUNTERS = (
    "self_kill_unsafe_action_with_safe_alternative",
    "self_kill_no_survivable_action_available",
    "self_kill_not_explained_by_final_action",
)
SELF_KILL_DIAGNOSTIC_MARKER = "self_kill_diagnostic_observations"
INDEPENDENT_TARGET_DIAGNOSTIC_MARKER = (
    "independent_target_diagnostic_observations"
)


def load_results(file_path: Path) -> dict:
    """Load one JSON result file created by the framework."""
    try:
        with file_path.open("r", encoding="utf-8") as file:
            results = json.load(file)
    except FileNotFoundError as error:
        raise ValueError(
            f"Result file does not exist: {file_path}"
        ) from error
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Result file is not valid JSON: {error}"
        ) from error

    if not isinstance(results, dict):
        raise ValueError("The top-level JSON value must be an object.")

    return results


def calculate_coin_auc(
    coin_collection_steps: list[int],
    coins_per_round: int = DEFAULT_COINS_PER_ROUND,
) -> float:
    """Return normalized area under one cumulative coin curve."""
    if not isinstance(coins_per_round, int) or coins_per_round <= 0:
        raise ValueError("'coins_per_round' must be a positive integer.")

    if len(coin_collection_steps) > coins_per_round:
        raise ValueError(
            "More coin collection steps than available coins."
        )

    if coin_collection_steps != sorted(coin_collection_steps):
        raise ValueError("Coin collection steps are not chronological.")

    for step in coin_collection_steps:
        if not isinstance(step, int) or not 1 <= step <= MAX_STEPS_PER_ROUND:
            raise ValueError(f"Invalid coin collection step: {step!r}")

    accumulated_coin_time = sum(
        MAX_STEPS_PER_ROUND - step + 1
        for step in coin_collection_steps
    )

    return (
        accumulated_coin_time
        / (coins_per_round * MAX_STEPS_PER_ROUND)
    )


def calculate_percentile(
    values: list[float],
    quantile: float,
) -> float:
    """Return a linearly interpolated quantile."""
    if not values:
        raise ValueError(
            "Cannot calculate a percentile of empty data."
        )
    if not 0 <= quantile <= 1:
        raise ValueError("Quantile must be between 0 and 1.")

    sorted_values = sorted(values)
    position = (len(sorted_values) - 1) * quantile
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    weight = position - lower_index

    return (
        sorted_values[lower_index]
        + weight
        * (sorted_values[upper_index] - sorted_values[lower_index])
    )


def _round_rows(
    round_statistics: dict,
    agent_name: str,
    coins_per_round: int,
) -> tuple[list[dict], list[float]]:
    """Validate and normalize the Task 2 fields of every round."""
    rows = []
    decision_times = []

    for round_identifier, round_data in round_statistics.items():
        if not isinstance(round_data, dict):
            raise ValueError(
                f"Round {round_identifier!r} must be an object."
            )

        agent_data = round_data.get("by_agent", {}).get(agent_name)
        if not isinstance(agent_data, dict):
            raise ValueError(
                f"Missing statistics for agent {agent_name!r} "
                f"in {round_identifier!r}."
            )

        for field_name in (
            "initial_crates",
            "remaining_crates",
            "initial_coins",
        ):
            if field_name not in round_data:
                raise ValueError(
                    f"Missing Task 2 field {field_name!r} "
                    f"in {round_identifier!r}."
                )

        if round_data["initial_coins"] != coins_per_round:
            raise ValueError(
                f"{round_identifier!r} contains "
                f"{round_data['initial_coins']} coins, but the "
                f"analysis expects {coins_per_round}."
            )

        coin_steps = agent_data.get("coin_collection_steps")
        round_decision_times = agent_data.get("decision_times")
        if not isinstance(coin_steps, list):
            raise ValueError(
                f"Missing coin collection steps in {round_identifier!r}."
            )
        if not isinstance(round_decision_times, list):
            raise ValueError(
                f"Missing decision times in {round_identifier!r}."
            )

        action_diagnostics = agent_data.get("action_diagnostics")
        diagnostics_available = action_diagnostics is not None

        if diagnostics_available:
            if not isinstance(action_diagnostics, dict):
                raise ValueError(
                    "Action diagnostics in "
                    f"{round_identifier!r} must be an object."
                )

            for name, value in action_diagnostics.items():
                if (
                    not isinstance(name, str)
                    or isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < 0
                ):
                    raise ValueError(
                        "Action diagnostics in "
                        f"{round_identifier!r} must contain "
                        "non-negative integer counters."
                    )
        else:
            action_diagnostics = {}

        action_diagnostics = dict(action_diagnostics)
        coin_route_bomb_diagnostic_observations = (
            action_diagnostics.get(
                COIN_ROUTE_BOMB_DIAGNOSTIC_MARKER,
                0,
            )
        )

        # Results produced before V6 have only one bomb target, so their
        # historical numerator is also the primary-target numerator. V6's
        # marker prevents an alternative-target bomb from entering the
        # primary response rate merely because the new counter is absent.
        if not coin_route_bomb_diagnostic_observations:
            action_diagnostics[
                "bomb_drops_at_primary_target"
            ] = action_diagnostics.get("bomb_drops_at_target", 0)

        diagnostic_counts = {}

        for _, opportunity_name, followed_name in (
            ACTION_DIAGNOSTIC_PAIRS
        ):
            opportunities = action_diagnostics.get(
                opportunity_name,
                0,
            )
            followed = action_diagnostics.get(
                followed_name,
                0,
            )

            if followed > opportunities:
                raise ValueError(
                    f"{followed_name!r} exceeds "
                    f"{opportunity_name!r} in "
                    f"{round_identifier!r}."
                )

            diagnostic_counts[opportunity_name] = opportunities
            diagnostic_counts[followed_name] = followed

        for coin_class, bomb_class in TARGET_CONFLICT_CLASS_PAIRS:
            conflict_prefix = (
                "target_conflict_"
                f"coin_class_{coin_class}_bomb_class_{bomb_class}"
            )
            opportunities = diagnostic_counts[
                f"{conflict_prefix}_opportunities"
            ]
            coin_followed = diagnostic_counts[
                f"{conflict_prefix}_coin_followed"
            ]
            bomb_followed = diagnostic_counts[
                f"{conflict_prefix}_bomb_followed"
            ]

            if coin_followed + bomb_followed > opportunities:
                raise ValueError(
                    "Coin and bomb choices exceed true-conflict "
                    f"opportunities for {conflict_prefix!r} in "
                    f"{round_identifier!r}."
                )

        for _, counter_name in ACTION_DIAGNOSTIC_COUNTERS:
            diagnostic_counts[counter_name] = (
                action_diagnostics.get(counter_name, 0)
            )

        score_ratio_sum = action_diagnostics.get(
            COIN_ROUTE_BOMB_SCORE_RATIO_SUM,
            0,
        )
        alternative_observations = diagnostic_counts[
            "coin_route_bomb_alternative_opportunities"
        ]

        if score_ratio_sum > 1000 * alternative_observations:
            raise ValueError(
                "Coin-route bomb score-ratio sum exceeds its "
                f"observation bound in {round_identifier!r}."
            )

        diagnostic_counts[
            COIN_ROUTE_BOMB_SCORE_RATIO_SUM
        ] = score_ratio_sum
        diagnostic_counts[COIN_ROUTE_BOMB_DIAGNOSTIC_MARKER] = (
            coin_route_bomb_diagnostic_observations
        )

        self_kill_diagnostic_observations = action_diagnostics.get(
            SELF_KILL_DIAGNOSTIC_MARKER,
            0,
        )
        diagnostic_counts[SELF_KILL_DIAGNOSTIC_MARKER] = (
            self_kill_diagnostic_observations
        )
        diagnostic_counts[INDEPENDENT_TARGET_DIAGNOSTIC_MARKER] = (
            action_diagnostics.get(
                INDEPENDENT_TARGET_DIAGNOSTIC_MARKER,
                0,
            )
        )

        for counter_name in SELF_KILL_DIAGNOSTIC_COUNTERS:
            diagnostic_counts[counter_name] = action_diagnostics.get(
                counter_name,
                0,
            )

        suicides = agent_data.get("suicides", 0)
        classified_self_kills = sum(
            diagnostic_counts[counter_name]
            for counter_name in SELF_KILL_DIAGNOSTIC_COUNTERS
        )

        if self_kill_diagnostic_observations:
            if classified_self_kills != suicides:
                raise ValueError(
                    "Self-kill diagnostic categories do not sum to "
                    f"the suicide count in {round_identifier!r}."
                )
        elif classified_self_kills:
            raise ValueError(
                "Self-kill categories were recorded without the V2 "
                f"diagnostic marker in {round_identifier!r}."
            )

        survived = bool(agent_data.get("survived", False))
        rows.append(
            {
                "score": agent_data.get("score", 0),
                "coins": agent_data.get("coins", 0),
                "steps": agent_data.get("steps", 0),
                "crates": agent_data.get("crates", 0),
                "bombs": agent_data.get("bombs", 0),
                "bombs_exploded": agent_data.get(
                    "bombs_exploded",
                    0,
                ),
                "coins_found": agent_data.get("coins_found", 0),
                "waits": agent_data.get("waits", 0),
                "invalid": agent_data.get("invalid", 0),
                "suicides": suicides,
                "deaths": agent_data.get(
                    "deaths",
                    int(not survived),
                ),
                "survived": survived,
                "initial_crates": round_data["initial_crates"],
                "remaining_crates": round_data["remaining_crates"],
                "coin_auc": calculate_coin_auc(
                    coin_steps,
                    coins_per_round,
                ),
                "action_diagnostics_available": (
                    diagnostics_available
                ),
                **diagnostic_counts,
            }
        )
        decision_times.extend(round_decision_times)

    return rows, decision_times


def calculate_metrics(
    agent_statistics: dict,
    round_statistics: dict,
    agent_name: str = AGENT_NAME,
    coins_per_round: int = DEFAULT_COINS_PER_ROUND,
) -> dict:
    """Calculate navigation, bomb-use, crate, and safety metrics."""
    if not isinstance(coins_per_round, int) or coins_per_round <= 0:
        raise ValueError("'coins_per_round' must be a positive integer.")
    if not isinstance(agent_statistics, dict):
        raise ValueError("Agent statistics must be an object.")
    if not isinstance(round_statistics, dict) or not round_statistics:
        raise ValueError("'by_round' must be a non-empty object.")

    rows, decision_times = _round_rows(
        round_statistics,
        agent_name,
        coins_per_round,
    )
    number_of_rounds = len(rows)

    total_coins = sum(row["coins"] for row in rows)
    total_steps = sum(row["steps"] for row in rows)
    total_crates = sum(row["crates"] for row in rows)
    total_initial_crates = sum(
        row["initial_crates"] for row in rows
    )
    total_bombs = sum(row["bombs"] for row in rows)
    total_bombs_exploded = sum(
        row["bombs_exploded"] for row in rows
    )
    total_coins_found = sum(row["coins_found"] for row in rows)
    total_waits = sum(row["waits"] for row in rows)
    total_invalid = sum(row["invalid"] for row in rows)

    diagnostic_rows = [
        row
        for row in rows
        if row["action_diagnostics_available"]
    ]
    diagnostic_metrics = {
        "action_diagnostic_rounds": len(diagnostic_rows),
        INDEPENDENT_TARGET_DIAGNOSTIC_MARKER: sum(
            row[INDEPENDENT_TARGET_DIAGNOSTIC_MARKER]
            for row in diagnostic_rows
        ),
    }

    for metric_prefix, opportunity_name, followed_name in (
        ACTION_DIAGNOSTIC_PAIRS
    ):
        if not diagnostic_rows:
            total_opportunities = None
            total_followed = None
            mean_opportunities = None
            follow_rate = None
        else:
            total_opportunities = sum(
                row[opportunity_name]
                for row in diagnostic_rows
            )
            total_followed = sum(
                row[followed_name]
                for row in diagnostic_rows
            )
            mean_opportunities = (
                total_opportunities / len(diagnostic_rows)
            )
            follow_rate = (
                total_followed / total_opportunities
                if total_opportunities
                else None
            )

        diagnostic_metrics[
            f"total_{metric_prefix}_opportunities"
        ] = total_opportunities
        diagnostic_metrics[
            f"total_{metric_prefix}_followed"
        ] = total_followed
        diagnostic_metrics[
            f"mean_{metric_prefix}_opportunities_per_round"
        ] = mean_opportunities
        diagnostic_metrics[
            f"{metric_prefix}_follow_rate"
        ] = follow_rate

    for coin_class, bomb_class in TARGET_CONFLICT_CLASS_PAIRS:
        conflict_prefix = (
            "target_conflict_"
            f"coin_class_{coin_class}_bomb_class_{bomb_class}"
        )
        opportunity_name = f"{conflict_prefix}_opportunities"
        coin_followed_name = f"{conflict_prefix}_coin_followed"
        bomb_followed_name = f"{conflict_prefix}_bomb_followed"

        if not diagnostic_rows:
            total_opportunities = None
            total_coin_followed = None
            total_bomb_followed = None
            mean_opportunities = None
            target_follow_rate = None
            coin_choice_rate = None
        else:
            total_opportunities = sum(
                row[opportunity_name]
                for row in diagnostic_rows
            )
            total_coin_followed = sum(
                row[coin_followed_name]
                for row in diagnostic_rows
            )
            total_bomb_followed = sum(
                row[bomb_followed_name]
                for row in diagnostic_rows
            )
            total_target_followed = (
                total_coin_followed + total_bomb_followed
            )
            mean_opportunities = (
                total_opportunities / len(diagnostic_rows)
            )
            target_follow_rate = (
                total_target_followed / total_opportunities
                if total_opportunities
                else None
            )
            coin_choice_rate = (
                total_coin_followed / total_target_followed
                if total_target_followed
                else None
            )

        diagnostic_metrics[
            f"total_{conflict_prefix}_opportunities"
        ] = total_opportunities
        diagnostic_metrics[
            f"total_{conflict_prefix}_coin_followed"
        ] = total_coin_followed
        diagnostic_metrics[
            f"total_{conflict_prefix}_bomb_followed"
        ] = total_bomb_followed
        diagnostic_metrics[
            f"mean_{conflict_prefix}_opportunities_per_round"
        ] = mean_opportunities
        diagnostic_metrics[
            f"{conflict_prefix}_target_follow_rate"
        ] = target_follow_rate
        diagnostic_metrics[
            f"{conflict_prefix}_coin_choice_rate"
        ] = coin_choice_rate

    for metric_prefix, counter_name in ACTION_DIAGNOSTIC_COUNTERS:
        if not diagnostic_rows:
            total_count = None
            mean_count = None
        else:
            total_count = sum(
                row[counter_name]
                for row in diagnostic_rows
            )
            mean_count = total_count / len(diagnostic_rows)

        diagnostic_metrics[
            f"total_{metric_prefix}_opportunities"
        ] = total_count
        diagnostic_metrics[
            f"mean_{metric_prefix}_opportunities_per_round"
        ] = mean_count

    if not diagnostic_rows:
        mean_coin_route_bomb_score_ratio = None
    else:
        alternative_observations = sum(
            row["coin_route_bomb_alternative_opportunities"]
            for row in diagnostic_rows
        )
        score_ratio_sum = sum(
            row[COIN_ROUTE_BOMB_SCORE_RATIO_SUM]
            for row in diagnostic_rows
        )
        mean_coin_route_bomb_score_ratio = (
            score_ratio_sum / (1000 * alternative_observations)
            if alternative_observations
            else None
        )

    diagnostic_metrics[
        "mean_coin_route_bomb_alternative_score_ratio"
    ] = mean_coin_route_bomb_score_ratio

    threat_urgency_0_follow_rate = diagnostic_metrics[
        "threat_urgency_0_follow_rate"
    ]
    diagnostic_metrics["no_threat_unsafe_action_rate"] = (
        None
        if threat_urgency_0_follow_rate is None
        else 1.0 - threat_urgency_0_follow_rate
    )

    self_kill_diagnostic_rows = [
        row
        for row in rows
        if row[SELF_KILL_DIAGNOSTIC_MARKER] > 0
    ]
    complete_self_kill_diagnostics = (
        len(self_kill_diagnostic_rows) == number_of_rounds
    )
    diagnostic_metrics["self_kill_diagnostic_rounds"] = len(
        self_kill_diagnostic_rows
    )

    for counter_name in SELF_KILL_DIAGNOSTIC_COUNTERS:
        total_count = sum(
            row[counter_name]
            for row in self_kill_diagnostic_rows
        )
        diagnostic_metrics[f"total_{counter_name}"] = (
            total_count
            if self_kill_diagnostic_rows
            else None
        )
        diagnostic_metrics[f"{counter_name}_rate"] = (
            total_count / number_of_rounds
            if complete_self_kill_diagnostics
            else None
        )

    coins_per_round_values = [row["coins"] for row in rows]
    completed_rows = [
        row for row in rows
        if row["coins"] == coins_per_round
    ]
    coin_auc_values = [row["coin_auc"] for row in rows]

    if decision_times:
        mean_decision_time_ms = 1000 * mean(decision_times)
        decision_time_p95_ms = 1000 * calculate_percentile(
            decision_times,
            0.95,
        )
        maximum_decision_time_ms = 1000 * max(decision_times)
        timeout_rate = (
            sum(
                decision_time > DECISION_TIME_LIMIT_SECONDS
                for decision_time in decision_times
            )
            / len(decision_times)
        )
    else:
        mean_decision_time_ms = None
        decision_time_p95_ms = None
        maximum_decision_time_ms = None
        timeout_rate = None

    return {
        "rounds": number_of_rounds,
        "coins_per_round_target": coins_per_round,
        "mean_score_per_round": mean(row["score"] for row in rows),
        "mean_coins_per_round": mean(coins_per_round_values),
        "coin_standard_deviation": pstdev(coins_per_round_values),
        "coin_collection_rate": (
            total_coins / (coins_per_round * number_of_rounds)
        ),
        "completion_rate": len(completed_rows) / number_of_rounds,
        "mean_steps_per_round": mean(row["steps"] for row in rows),
        "mean_steps_to_completion": (
            mean(row["steps"] for row in completed_rows)
            if completed_rows
            else None
        ),
        "mean_coin_auc": mean(coin_auc_values),
        "coin_auc_standard_deviation": pstdev(coin_auc_values),
        "mean_crates_destroyed_per_round": (
            total_crates / number_of_rounds
        ),
        "crate_destruction_rate": (
            total_crates / total_initial_crates
            if total_initial_crates
            else 0.0
        ),
        "all_crates_cleared_rate": (
            sum(row["remaining_crates"] == 0 for row in rows)
            / number_of_rounds
        ),
        "mean_bombs_dropped_per_round": (
            total_bombs / number_of_rounds
        ),
        "rounds_with_bomb_rate": (
            sum(row["bombs"] > 0 for row in rows)
            / number_of_rounds
        ),
        "bomb_explosion_rate": (
            total_bombs_exploded / total_bombs
            if total_bombs
            else 0.0
        ),
        "crates_per_bomb": (
            total_crates / total_bombs
            if total_bombs
            else 0.0
        ),
        "mean_coins_found_per_round": (
            total_coins_found / number_of_rounds
        ),
        "coins_found_per_bomb": (
            total_coins_found / total_bombs
            if total_bombs
            else 0.0
        ),
        "survival_rate": (
            sum(row["survived"] for row in rows)
            / number_of_rounds
        ),
        "self_kill_rate": (
            sum(row["suicides"] > 0 for row in rows)
            / number_of_rounds
        ),
        "wait_action_rate": (
            total_waits / total_steps
            if total_steps
            else 0.0
        ),
        "invalid_action_rate": (
            total_invalid / total_steps
            if total_steps
            else 0.0
        ),
        "coins_per_100_steps": (
            100 * total_coins / total_steps
            if total_steps
            else 0.0
        ),
        "crates_per_100_steps": (
            100 * total_crates / total_steps
            if total_steps
            else 0.0
        ),
        "mean_decision_time_ms": mean_decision_time_ms,
        "decision_time_p95_ms": decision_time_p95_ms,
        "maximum_decision_time_ms": maximum_decision_time_ms,
        "timeout_rate": timeout_rate,
        **diagnostic_metrics,
    }


def analyze_evaluation_file(
    file_path: Path,
    agent_name: str = AGENT_NAME,
    coins_per_round: int = DEFAULT_COINS_PER_ROUND,
) -> dict:
    """Load and analyze one evaluation statistics file."""
    results = load_results(file_path)
    by_agent = results.get("by_agent")
    round_statistics = results.get("by_round")

    if not isinstance(by_agent, dict):
        raise ValueError("The file does not contain 'by_agent'.")
    if not isinstance(round_statistics, dict):
        raise ValueError("The file does not contain 'by_round'.")

    agent_statistics = by_agent.get(agent_name)
    if not isinstance(agent_statistics, dict):
        raise ValueError(
            f"No statistics found for agent {agent_name!r}."
        )

    return calculate_metrics(
        agent_statistics,
        round_statistics,
        agent_name,
        coins_per_round,
    )


def _format_diagnostic_rate(metrics: dict, metric_prefix: str) -> str:
    """Format one adherence rate together with its denominator."""
    rate = metrics[f"{metric_prefix}_follow_rate"]
    opportunities = metrics.get(
        f"total_{metric_prefix}_opportunities"
    )
    rate_text = "N/A" if rate is None else f"{100 * rate:.2f}%"
    opportunity_text = "N/A" if opportunities is None else opportunities

    return f"{rate_text} (n={opportunity_text})"


def print_metrics(metrics: dict) -> None:
    """Print a compact Task 2 report."""
    print(f"Rounds: {metrics['rounds']}")
    print(
        "Coins / completion / coin AUC: "
        f"{metrics['mean_coins_per_round']:.2f} / "
        f"{100 * metrics['completion_rate']:.2f}% / "
        f"{100 * metrics['mean_coin_auc']:.2f}%"
    )
    print(
        "Crates / destruction rate / crates per bomb: "
        f"{metrics['mean_crates_destroyed_per_round']:.2f} / "
        f"{100 * metrics['crate_destruction_rate']:.2f}% / "
        f"{metrics['crates_per_bomb']:.3f}"
    )
    print(
        "Bombs per round / rounds with bomb: "
        f"{metrics['mean_bombs_dropped_per_round']:.2f} / "
        f"{100 * metrics['rounds_with_bomb_rate']:.2f}%"
    )
    print(
        "Survival / self-kill rate: "
        f"{100 * metrics['survival_rate']:.2f}% / "
        f"{100 * metrics['self_kill_rate']:.2f}%"
    )
    print(
        "Coins / crates per 100 steps: "
        f"{metrics['coins_per_100_steps']:.2f} / "
        f"{metrics['crates_per_100_steps']:.2f}"
    )
    print(
        "WAIT / invalid action rate: "
        f"{100 * metrics['wait_action_rate']:.2f}% / "
        f"{100 * metrics['invalid_action_rate']:.2f}%"
    )

    if metrics["action_diagnostic_rounds"]:
        v2_prefixes = (
            "threat_urgency_0",
            "threat_urgency_1",
            "threat_urgency_2",
            "threat_urgency_3",
            "target_conflict_coin_strong",
            "target_conflict_coin_weak",
            "target_conflict_bomb_weak",
            "target_conflict_bomb_strong",
            "safe_joint_target",
            "safe_coin_only_target",
            "safe_bomb_only_target",
            "bomb_target_precision",
            "bomb_target_response",
        )
        independent_target_diagnostics = bool(
            metrics[INDEPENDENT_TARGET_DIAGNOSTIC_MARKER]
        )
        v2_diagnostics_recorded = independent_target_diagnostics or any(
            (
                metrics[
                    f"mean_{metric_prefix}_opportunities_per_round"
                ]
                or 0.0
            )
            > 0.0
            for metric_prefix in v2_prefixes
        )

        if v2_diagnostics_recorded:
            threat_rates = [
                _format_diagnostic_rate(
                    metrics,
                    f"threat_urgency_{urgency}",
                )
                for urgency in (0, 1, 2, 3)
            ]

            print(
                "Safety adherence urgency 0 / 1 / 2 / 3: "
                + " / ".join(threat_rates)
            )
            if independent_target_diagnostics:
                print(
                    "True-conflict coin choice by coin class "
                    "(far / mid-far / near / very near) and bomb "
                    "class (weak / medium / strong):"
                )

                for coin_class in range(4):
                    class_values = []

                    for bomb_class in range(3):
                        conflict_prefix = (
                            "target_conflict_"
                            f"coin_class_{coin_class}_"
                            f"bomb_class_{bomb_class}"
                        )
                        choice_rate = metrics[
                            f"{conflict_prefix}_coin_choice_rate"
                        ]
                        target_rate = metrics[
                            f"{conflict_prefix}_target_follow_rate"
                        ]
                        opportunities = metrics[
                            f"total_{conflict_prefix}_opportunities"
                        ]
                        class_values.append(
                            "coin=N/A, use=N/A, n=N/A"
                            if opportunities is None
                            else (
                                "coin="
                                + (
                                    "N/A"
                                    if choice_rate is None
                                    else f"{100 * choice_rate:.1f}%"
                                )
                                + ", use="
                                + (
                                    "N/A"
                                    if target_rate is None
                                    else f"{100 * target_rate:.1f}%"
                                )
                                + f", n={opportunities}"
                            )
                        )

                    print(
                        f"  coin class {coin_class}: "
                        + " / ".join(class_values)
                    )
            else:
                print(
                    "True-conflict priority coin strong / coin weak / "
                    "bomb weak / bomb strong: "
                    + " / ".join(
                        _format_diagnostic_rate(metrics, prefix)
                        for prefix in (
                            "target_conflict_coin_strong",
                            "target_conflict_coin_weak",
                            "target_conflict_bomb_weak",
                            "target_conflict_bomb_strong",
                        )
                    )
                )
            print(
                "Safe-target utilization joint / newly created joint / "
                "coin only / bomb only: "
                + " / ".join(
                    _format_diagnostic_rate(metrics, prefix)
                    for prefix in (
                        "safe_joint_target",
                        "coin_route_bomb_joint_conversion",
                        "safe_coin_only_target",
                        "safe_bomb_only_target",
                    )
                )
            )

            if independent_target_diagnostics:
                no_safe_target_mean = metrics[
                    "mean_no_safe_target_opportunities_per_round"
                ]
                print(
                    "No safe target per round: "
                    + (
                        "N/A"
                        if no_safe_target_mean is None
                        else f"{no_safe_target_mean:.3f}"
                    )
                )
            else:
                excluded_state_means = (
                    metrics[
                        "mean_target_conflict_equal_opportunities_per_round"
                    ],
                    metrics[
                        "mean_no_safe_target_opportunities_per_round"
                    ],
                )
                print(
                    "Excluded navigation states equal-distance conflict / "
                    "no safe target per round: "
                    + " / ".join(
                        "N/A" if value is None else f"{value:.3f}"
                        for value in excluded_state_means
                    )
                )

            print(
                "BFS bomb-target precision (either target) / primary "
                "response / alternative response: "
                + " / ".join(
                    _format_diagnostic_rate(metrics, prefix)
                    for prefix in (
                        "bomb_target_precision",
                        "bomb_target_response",
                        "coin_route_bomb_alternative_target",
                    )
                )
            )

            alternative_opportunities = metrics[
                "total_coin_route_bomb_alternative_opportunities"
            ]

            if alternative_opportunities:
                score_ratio = metrics[
                    "mean_coin_route_bomb_alternative_score_ratio"
                ]
                print(
                    "Coin-route bomb alternatives / score ratio: "
                    f"{alternative_opportunities} / "
                    + (
                        "N/A"
                        if score_ratio is None
                        else f"{100 * score_ratio:.2f}%"
                    )
                )

            self_kill_rates = [
                metrics[f"{counter_name}_rate"]
                for counter_name in SELF_KILL_DIAGNOSTIC_COUNTERS
            ]
            if any(rate is not None for rate in self_kill_rates):
                print(
                    "Self-kill diagnosis unsafe final action / "
                    "already trapped / unexplained: "
                    + " / ".join(
                        "N/A"
                        if rate is None
                        else f"{100 * rate:.2f}%"
                        for rate in self_kill_rates
                    )
                )
        else:
            objective_rates = []

            for metric_prefix in (
                "coin_objective",
                "crate_objective",
                "escape_objective",
            ):
                rate = metrics[f"{metric_prefix}_follow_rate"]
                objective_rates.append(
                    "N/A"
                    if rate is None
                    else f"{100 * rate:.2f}%"
                )

            print(
                "Objective-mask adherence coin / crate / escape: "
                + " / ".join(objective_rates)
            )

            bomb_rate = metrics[
                "safe_productive_bomb_follow_rate"
            ]
            bomb_rate_text = (
                "N/A"
                if bomb_rate is None
                else f"{100 * bomb_rate:.2f}%"
            )
            print(
                "Safe productive BOMB response / opportunities "
                "per round: "
                f"{bomb_rate_text} / "
                f"{metrics['mean_safe_productive_bomb_opportunities_per_round']:.3f}"
            )
    else:
        print("Objective-action diagnostics: not recorded")

    if metrics["mean_decision_time_ms"] is None:
        print("Decision times: N/A")
    else:
        print(
            "Decision time mean / p95 / max: "
            f"{metrics['mean_decision_time_ms']:.3f} / "
            f"{metrics['decision_time_p95_ms']:.3f} / "
            f"{metrics['maximum_decision_time_ms']:.3f} ms"
        )
        print(
            "Timeout rate: "
            f"{100 * metrics['timeout_rate']:.2f}%"
        )


def parse_arguments() -> argparse.Namespace:
    """Parse one result file and its coin count."""
    parser = argparse.ArgumentParser(
        description="Analyze one Task 2 evaluation result."
    )
    parser.add_argument(
        "result_file",
        type=Path,
        help="JSON result file created with --save-stats.",
    )
    parser.add_argument(
        "--coins-per-round",
        type=int,
        default=DEFAULT_COINS_PER_ROUND,
        help=(
            "Number of coins available in each round "
            f"(default: {DEFAULT_COINS_PER_ROUND})."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Analyze and print one Task 2 result file."""
    args = parse_arguments()
    try:
        metrics = analyze_evaluation_file(
            args.result_file,
            coins_per_round=args.coins_per_round,
        )
    except ValueError as error:
        raise SystemExit(f"Evaluation analysis failed: {error}")

    print_metrics(metrics)


if __name__ == "__main__":
    main()
