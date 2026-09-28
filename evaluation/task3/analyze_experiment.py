"""Analyze one complete controlled Task 3 experiment."""

import argparse
import json
from pathlib import Path
import re
from statistics import mean, pstdev, stdev

import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.task3.analyze_results import analyze_evaluation_file
from evaluation.task3.analyze_training import analyze_training_file
from evaluation.exploration_metrics import training_strategy
from evaluation.task3.block_metrics import BLOCK_METRICS
from evaluation.task3.combat_metrics import (
    EVALUATION_COMBAT_METRICS,
    TRAINING_COMBAT_CURVE_METRICS,
    TRAINING_COMBAT_METRICS,
    combat_round_row,
    opponent_instance_names,
)

SCENARIO_COIN_COUNTS = {
    "coin-heaven": 50,
    "coin-heaven-9": 9,
    "loot-crate": 50,
    "classic": 9,
}

LEGACY_TD_TARGET_VERSION = "q_learning_1_step_v1"
TRAIN_AND_EVALUATE_MODE = "train_and_evaluate"
EVALUATE_CHECKPOINT_MODE = "evaluate_checkpoint"


def experiment_mode(configuration: dict) -> str:
    """Return the indexed experiment mode with legacy fallback."""
    return configuration.get(
        "experiment_mode",
        TRAIN_AND_EVALUATE_MODE,
    )

EVALUATION_METRICS = (
    "mean_score_per_round",
    "mean_coins_per_round",
    "coin_collection_rate",
    "completion_rate",
    "mean_steps_per_round",
    "mean_steps_to_completion",
    "mean_crates_destroyed_per_round",
    "crate_destruction_rate",
    "all_crates_cleared_rate",
    "mean_bombs_dropped_per_round",
    "rounds_with_bomb_rate",
    "crates_per_bomb",
    "mean_coins_found_per_round",
    "coins_found_per_bomb",
    "survival_rate",
    "self_kill_rate",
    "wait_action_rate",
    "invalid_action_rate",
    "coins_per_100_steps",
    "crates_per_100_steps",
    "mean_decision_time_ms",
    "decision_time_p95_ms",
    "maximum_decision_time_ms",
    "timeout_rate",
    "mean_coin_auc",
    "mean_coin_objective_opportunities_per_round",
    "coin_objective_follow_rate",
    "mean_crate_objective_opportunities_per_round",
    "crate_objective_follow_rate",
    "mean_escape_objective_opportunities_per_round",
    "escape_objective_follow_rate",
    "mean_safe_productive_bomb_opportunities_per_round",
    "safe_productive_bomb_follow_rate",
    "mean_threat_urgency_0_opportunities_per_round",
    "threat_urgency_0_follow_rate",
    "no_threat_unsafe_action_rate",
    "mean_threat_urgency_1_opportunities_per_round",
    "threat_urgency_1_follow_rate",
    "mean_threat_urgency_2_opportunities_per_round",
    "threat_urgency_2_follow_rate",
    "mean_threat_urgency_3_opportunities_per_round",
    "threat_urgency_3_follow_rate",
    "mean_target_conflict_coin_strong_opportunities_per_round",
    "target_conflict_coin_strong_follow_rate",
    "mean_target_conflict_coin_weak_opportunities_per_round",
    "target_conflict_coin_weak_follow_rate",
    "mean_target_conflict_bomb_weak_opportunities_per_round",
    "target_conflict_bomb_weak_follow_rate",
    "mean_target_conflict_bomb_strong_opportunities_per_round",
    "target_conflict_bomb_strong_follow_rate",
    "mean_safe_joint_target_opportunities_per_round",
    "safe_joint_target_follow_rate",
    "mean_safe_coin_only_target_opportunities_per_round",
    "safe_coin_only_target_follow_rate",
    "mean_safe_bomb_only_target_opportunities_per_round",
    "safe_bomb_only_target_follow_rate",
    "mean_target_conflict_equal_opportunities_per_round",
    "mean_no_safe_target_opportunities_per_round",
    "mean_bomb_target_precision_opportunities_per_round",
    "bomb_target_precision_follow_rate",
    "mean_bomb_target_response_opportunities_per_round",
    "bomb_target_response_follow_rate",
    "mean_coin_route_bomb_alternative_opportunities_per_round",
    "mean_coin_route_bomb_alternative_score_ratio",
    "mean_coin_route_bomb_joint_conversion_opportunities_per_round",
    "coin_route_bomb_joint_conversion_follow_rate",
    "mean_coin_route_bomb_alternative_target_opportunities_per_round",
    "coin_route_bomb_alternative_target_follow_rate",
    "self_kill_diagnostic_rounds",
    "self_kill_unsafe_action_with_safe_alternative_rate",
    "self_kill_no_survivable_action_available_rate",
    "self_kill_not_explained_by_final_action_rate",
)

TARGET_CONFLICT_CLASS_METRICS = tuple(
    metric_name
    for coin_class in range(4)
    for bomb_class in range(3)
    for metric_name in (
        (
            "mean_target_conflict_"
            f"coin_class_{coin_class}_bomb_class_{bomb_class}_"
            "opportunities_per_round"
        ),
        (
            "target_conflict_"
            f"coin_class_{coin_class}_bomb_class_{bomb_class}_"
            "coin_choice_rate"
        ),
        (
            "target_conflict_"
            f"coin_class_{coin_class}_bomb_class_{bomb_class}_"
            "target_follow_rate"
        ),
    )
)
EVALUATION_METRICS += tuple(
    metric
    for prefix in ("pressure_target", "pressure_local_bomb", "bomb_blockade_check")
    for metric in (f"mean_{prefix}_opportunities_per_round", f"{prefix}_follow_rate")
)
EVALUATION_METRICS += (
    TARGET_CONFLICT_CLASS_METRICS
    + EVALUATION_COMBAT_METRICS
)
EVALUATION_METRICS += BLOCK_METRICS

TRAINING_METRICS = BLOCK_METRICS + (
    "mean_return",
    "early_mean_return",
    "late_mean_return",
    "return_improvement",
    "td_error_mae",
    "td_error_rmse",
    "td_error_abs_max",
    "early_td_error_mae",
    "late_td_error_mae",
    "td_error_mae_change",
    "initial_epsilon",
    "final_epsilon",
    "initial_temperature",
    "final_temperature",
    "total_q_updates",
    "mean_q_updates_per_round",
    "final_q_table_states",
    "mean_new_q_table_states_per_round",
    "late_mean_new_q_table_states",
    "final_updated_state_action_pairs",
    "final_state_action_coverage",
    "mean_new_updated_pairs_per_round",
    "late_mean_new_updated_pairs",
    "training_wall_time_seconds",
    "rounds_per_second",
    "updates_per_second",
    "bomb_action_count",
    "bomb_action_rate",
    "bomb_update_count",
    "bomb_update_rate",
) + TRAINING_COMBAT_METRICS

TRAINING_CURVE_METRICS = (
    "undiscounted_return",
    "return_moving_average",
    "epsilon_end",
    "temperature_end",
    "q_updates",
    "td_error_mae",
    "td_error_rmse",
    "td_error_abs_max",
    "q_table_states",
    "new_q_table_states",
    "updated_state_action_pairs",
    "new_updated_state_action_pairs",
    "state_action_coverage",
    "coins_collected",
    "coins_collected_moving_average",
    "crate_destruction_rate",
    "crate_destruction_rate_moving_average",
    "survived",
    "survival_rate_moving_average",
) + TRAINING_COMBAT_CURVE_METRICS

# A PDF cannot resolve one point for every training round in a long run.
# Keeping a bounded, evenly sampled curve payload makes summary.json
# portable while the original training result remains the exact source.
MAX_TRAINING_CURVE_POINTS = 500


def load_experiment_index(file_path: Path) -> dict:
    """Load one experiment index."""
    try:
        with file_path.open("r", encoding="utf-8") as file:
            experiment_index = json.load(file)
    except FileNotFoundError as error:
        raise ValueError(
            f"Experiment index does not exist: {file_path}"
        ) from error
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Experiment index is not valid JSON: {error}"
        ) from error

    if not isinstance(experiment_index, dict):
        raise ValueError(
            "The top-level JSON value must be an object."
        )

    return experiment_index


def validate_experiment_index(
    experiment_index: dict,
) -> tuple[dict, list[dict]]:
    """Validate the basic experiment-index structure."""
    status = experiment_index.get("status")

    if status != "completed":
        raise ValueError(
            "Only completed experiments can be analyzed. "
            f"Current status: {status!r}"
        )

    configuration = experiment_index.get("configuration")

    if not isinstance(configuration, dict):
        raise ValueError(
            "The experiment index does not contain a configuration."
        )

    agent_name = configuration.get("agent")

    if not isinstance(agent_name, str) or not agent_name:
        raise ValueError(
            "The experiment configuration contains no valid agent."
        )

    opponents = configuration.get("opponents")

    if (
        not isinstance(opponents, list)
        or not opponents
        or any(
            not isinstance(opponent, str) or not opponent
            for opponent in opponents
        )
    ):
        raise ValueError(
            "The Task 3 configuration contains no valid fixed "
            "opponent roster."
        )

    if agent_name in opponents or agent_name in opponent_instance_names(opponents):
        raise ValueError(
            "The learning agent must not appear in 'opponents'."
        )

    mode = experiment_mode(configuration)
    if mode not in {
        TRAIN_AND_EVALUATE_MODE,
        EVALUATE_CHECKPOINT_MODE,
    }:
        raise ValueError(
            f"Unsupported Task 3 experiment mode: {mode!r}."
        )

    if experiment_index.get("experiment_mode", mode) != mode:
        raise ValueError(
            "Experiment mode differs between index and configuration."
        )

    runs = experiment_index.get("runs")

    if not isinstance(runs, list) or not runs:
        raise ValueError(
            "The experiment index contains no model runs."
        )

    if mode == EVALUATE_CHECKPOINT_MODE and len(runs) != 1:
        raise ValueError(
            "A checkpoint-evaluation experiment must contain "
            "exactly one model run."
        )

    for run_number, run in enumerate(runs, start=1):
        if not isinstance(run, dict):
            raise ValueError(
                f"Model run {run_number} must be an object."
            )

        evaluations = run.get("evaluations")

        if not isinstance(evaluations, list) or not evaluations:
            raise ValueError(
                f"Model run {run_number} contains "
                "no evaluation runs."
            )

        training_results = run.get("training_results")
        if mode == TRAIN_AND_EVALUATE_MODE and (
            not isinstance(training_results, str)
            or not training_results
        ):
            raise ValueError(
                f"Training run {run_number} contains no "
                "training-results path."
            )

        if mode == EVALUATE_CHECKPOINT_MODE and (
            training_results is not None
        ):
            raise ValueError(
                "A checkpoint-evaluation run must not contain "
                "training results."
            )

    return configuration, runs


def resolve_coins_per_round(configuration: dict) -> int:
    """Return the configured or scenario-defined coin count."""
    coins_per_round = configuration.get("coins_per_round")

    if coins_per_round is None:
        coins_per_round = SCENARIO_COIN_COUNTS.get(
            configuration.get("scenario")
        )

    if not isinstance(coins_per_round, int) or coins_per_round <= 0:
        raise ValueError(
            "The experiment must define a positive 'coins_per_round' "
            "or use a scenario with a known coin count."
        )

    return coins_per_round

def resolve_result_file(
    experiment_directory: Path,
    relative_path,
    location: str,
) -> Path:
    """Resolve and require one indexed result file."""
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError(
            f"'{location}' must be a non-empty path string."
        )

    result_file = experiment_directory / relative_path

    if not result_file.is_file():
        raise ValueError(
            f"Indexed result file does not exist: {result_file}"
        )

    return result_file

def validate_run_consistency(
    configuration: dict,
    analyzed_run: dict,
    run_number: int,
) -> None:
    """Verify that one analyzed run matches the experiment index."""
    mode = experiment_mode(configuration)
    training_metrics = analyzed_run["training_metrics"]

    if mode == TRAIN_AND_EVALUATE_MODE:
        if training_metrics is None:
            raise ValueError(
                f"Training run {run_number} has no training metrics."
            )

        metadata = training_metrics["metadata"]
        summary = training_metrics["summary"]

        expected_metadata = {
            "world_seed": analyzed_run["training_world_seed"],
            "agent_seed": analyzed_run["training_agent_seed"],
            "scenario": configuration["scenario"],
            "feature_version": configuration["feature_version"],
            "reward_version": configuration["reward_version"],
            "action_space_version": configuration[
                "action_space_version"
            ],
            "learning_rate_version": configuration[
                "learning_rate_version"
            ],
            "exploration_version": configuration[
                "exploration_version"
            ],
            "td_target_version": configuration.get(
                "td_target_version",
                LEGACY_TD_TARGET_VERSION,
            ),
        }

        if "discount_factor_version" in configuration:
            expected_metadata["discount_factor_version"] = (
                configuration["discount_factor_version"]
            )

        for name, expected_value in expected_metadata.items():
            actual_value = metadata.get(name)

            if actual_value != expected_value:
                raise ValueError(
                    f"Training run {run_number} has inconsistent "
                    f"{name}: expected {expected_value!r}, "
                    f"found {actual_value!r}."
                )

        expected_training_rounds = configuration[
            "training"
        ]["rounds"]

        if summary["rounds"] != expected_training_rounds:
            raise ValueError(
                f"Training run {run_number} contains "
                f"{summary['rounds']} rounds; expected "
                f"{expected_training_rounds}."
            )

        expected_opponents = opponent_instance_names(configuration["opponents"])

        if training_metrics["opponents"] != expected_opponents:
            raise ValueError(
                f"Training run {run_number} used opponents "
                f"{training_metrics['opponents']!r}; expected "
                f"{expected_opponents!r}."
            )
    else:
        if training_metrics is not None:
            raise ValueError(
                "A checkpoint-evaluation run must not contain "
                "training metrics."
            )

        checkpoint = configuration["checkpoint"]
        expected_model_seeds = (
            checkpoint["training_world_seed"],
            checkpoint["training_agent_seed"],
        )
        actual_model_seeds = (
            analyzed_run["training_world_seed"],
            analyzed_run["training_agent_seed"],
        )

        if actual_model_seeds != expected_model_seeds:
            raise ValueError(
                "Checkpoint model seeds differ from the configuration."
            )

        if analyzed_run.get("model_source") != "external_checkpoint":
            raise ValueError(
                "Checkpoint run lacks external-checkpoint provenance."
            )

        model_sha256 = analyzed_run.get("model_sha256")
        if (
            not isinstance(model_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", model_sha256) is None
        ):
            raise ValueError(
                "Checkpoint run contains no valid model SHA-256."
            )

    expected_opponents = opponent_instance_names(configuration["opponents"])

    expected_evaluation_runs = configuration[
        "evaluation"
    ]["runs"]

    if len(analyzed_run["evaluations"]) != len(
        expected_evaluation_runs
    ):
        raise ValueError(
            f"Model run {run_number} contains the wrong "
            "number of evaluation runs."
        )

    expected_seed_pairs = {
        (
            evaluation["world_seed"],
            evaluation["agent_seed"],
        )
        for evaluation in expected_evaluation_runs
    }

    actual_seed_pairs = {
        (
            evaluation["evaluation_world_seed"],
            evaluation["evaluation_agent_seed"],
        )
        for evaluation in analyzed_run["evaluations"]
    }

    if actual_seed_pairs != expected_seed_pairs:
        raise ValueError(
            f"Model run {run_number} contains unexpected "
            "evaluation seed pairs."
        )

    expected_evaluation_rounds = configuration[
        "evaluation"
    ]["rounds_per_seed"]

    for evaluation in analyzed_run["evaluations"]:
        actual_rounds = evaluation["metrics"]["rounds"]

        if actual_rounds != expected_evaluation_rounds:
            raise ValueError(
                f"Evaluation {evaluation['results_file']} contains "
                f"{actual_rounds} rounds; expected "
                f"{expected_evaluation_rounds}."
            )

        actual_opponents = evaluation["metrics"]["opponents"]

        if actual_opponents != expected_opponents:
            raise ValueError(
                f"Evaluation {evaluation['results_file']} used "
                f"opponents {actual_opponents!r}; expected "
                f"{expected_opponents!r}."
            )

def analyze_experiment_runs(
    experiment_file: Path,
    configuration: dict,
    runs: list[dict],
) -> list[dict]:
    """Analyze every indexed model and evaluation result file."""
    experiment_directory = experiment_file.resolve().parent
    agent_name = configuration["agent"]
    coins_per_round = resolve_coins_per_round(configuration)
    analyzed_runs = []

    for run_number, run in enumerate(runs, start=1):
        training_results = run.get("training_results")

        if training_results is None:
            training_metrics = None
        else:
            training_file = resolve_result_file(
                experiment_directory,
                training_results,
                f"runs[{run_number - 1}].training_results",
            )

            training_metrics = analyze_training_file(
                training_file,
                agent_name,
            )

        analyzed_evaluations = []

        for evaluation_number, evaluation in enumerate(
            run["evaluations"],
            start=1,
        ):
            if not isinstance(evaluation, dict):
                raise ValueError(
                    "Evaluation entry "
                    f"{evaluation_number} of model run "
                    f"{run_number} must be an object."
                )

            evaluation_file = resolve_result_file(
                experiment_directory,
                evaluation.get("results"),
                (
                    f"runs[{run_number - 1}].evaluations"
                    f"[{evaluation_number - 1}].results"
                ),
            )

            evaluation_metrics = analyze_evaluation_file(
                evaluation_file,
                agent_name,
                coins_per_round,
            )

            analyzed_evaluations.append(
                {
                    "evaluation_world_seed": evaluation.get(
                        "evaluation_world_seed"
                    ),
                    "evaluation_agent_seed": evaluation.get(
                        "evaluation_agent_seed"
                    ),
                    "results_file": evaluation["results"],
                    "metrics": evaluation_metrics,
                }
            )

        analyzed_run = {
            "training_world_seed": run.get(
                "training_world_seed"
            ),
            "training_agent_seed": run.get(
                "training_agent_seed"
            ),
            "training_results_file": run[
                "training_results"
            ],
            "training_metrics": training_metrics,
            "model": run.get("model"),
            "model_source": run.get("model_source"),
            "model_sha256": run.get("model_sha256"),
            "evaluations": analyzed_evaluations,
        }

        validate_run_consistency(
            configuration,
            analyzed_run,
            run_number,
        )

        analyzed_runs.append(analyzed_run)

    return analyzed_runs

def summarize_values(
    values: list,
    use_sample_standard_deviation: bool,
) -> dict:
    """Summarize available numerical observations."""
    available_values = [
        float(value)
        for value in values
        if value is not None
    ]

    if not available_values:
        return {
            "observations": 0,
            "mean": None,
            "standard_deviation": None,
        }

    if use_sample_standard_deviation:
        standard_deviation = (
            stdev(available_values)
            if len(available_values) >= 2
            else None
        )
    else:
        standard_deviation = pstdev(available_values)

    return {
        "observations": len(available_values),
        "mean": mean(available_values),
        "standard_deviation": standard_deviation,
    }

def summarize_metric_dictionaries(
    metric_dictionaries: list[dict],
    use_sample_standard_deviation: bool,
) -> dict:
    """Summarize the selected evaluation metrics."""
    return {
        metric_name: summarize_values(
            [
                metrics.get(metric_name)
                for metrics in metric_dictionaries
            ],
            use_sample_standard_deviation,
        )
        for metric_name in EVALUATION_METRICS
    }

def aggregate_evaluation_metrics(
    analyzed_runs: list[dict],
) -> dict:
    """Aggregate evaluation seeds first and trained models second."""
    model_mean_metrics = []

    for run in analyzed_runs:
        evaluation_metrics = [
            evaluation["metrics"]
            for evaluation in run["evaluations"]
        ]

        run["evaluation_summary"] = (
            summarize_metric_dictionaries(
                evaluation_metrics,
                use_sample_standard_deviation=False,
            )
        )

        model_mean_metrics.append(
            {
                metric_name: run["evaluation_summary"][
                    metric_name
                ]["mean"]
                for metric_name in EVALUATION_METRICS
            }
        )

    return summarize_metric_dictionaries(
        model_mean_metrics,
        use_sample_standard_deviation=True,
    )

def aggregate_evaluations_by_seed(
    analyzed_runs: list[dict],
) -> list[dict]:
    """Aggregate each evaluation seed across trained models."""
    grouped_metrics = {}

    for run in analyzed_runs:
        for evaluation in run["evaluations"]:
            seed_pair = (
                evaluation["evaluation_world_seed"],
                evaluation["evaluation_agent_seed"],
            )

            grouped_metrics.setdefault(
                seed_pair,
                [],
            ).append(evaluation["metrics"])

    return [
        {
            "evaluation_world_seed": world_seed,
            "evaluation_agent_seed": agent_seed,
            "trained_models": len(metric_dictionaries),
            "evaluated_models": len(metric_dictionaries),
            "metrics": summarize_metric_dictionaries(
                metric_dictionaries,
                use_sample_standard_deviation=True,
            ),
        }
        for (
            world_seed,
            agent_seed,
        ), metric_dictionaries in grouped_metrics.items()
    ]

def aggregate_training_metrics(
    analyzed_runs: list[dict],
) -> dict:
    """Aggregate training summaries across independent models."""
    training_strategy(analyzed_runs)
    training_summaries = [
        run["training_metrics"]["summary"]
        for run in analyzed_runs
        if run["training_metrics"] is not None
    ]

    if not training_summaries:
        return {}

    return {
        metric_name: summarize_values(
            [
                summary.get(metric_name)
                for summary in training_summaries
            ],
            use_sample_standard_deviation=True,
        )
        for metric_name in TRAINING_METRICS
    }

def aggregate_training_curves(
    analyzed_runs: list[dict],
) -> dict:
    """Aggregate a bounded, column-oriented set of training curves."""
    training_strategy(analyzed_runs)
    per_model_rounds = [
        run["training_metrics"]["per_round"]
        for run in analyzed_runs
        if run["training_metrics"] is not None
    ]

    if not per_model_rounds:
        return {}

    number_of_rounds = len(per_model_rounds[0])

    if any(
        len(rounds) != number_of_rounds
        for rounds in per_model_rounds
    ):
        raise ValueError(
            "Training runs contain different numbers of rounds."
        )

    if number_of_rounds <= MAX_TRAINING_CURVE_POINTS:
        round_indices = list(range(number_of_rounds))
    else:
        round_indices = [
            point_index * (number_of_rounds - 1)
            // (MAX_TRAINING_CURVE_POINTS - 1)
            for point_index in range(MAX_TRAINING_CURVE_POINTS)
        ]

    sampled_rounds = []
    metric_means = {
        metric_name: []
        for metric_name in TRAINING_CURVE_METRICS
    }
    metric_standard_deviations = {
        metric_name: []
        for metric_name in TRAINING_CURVE_METRICS
    }

    for round_index in round_indices:
        round_entries = [
            rounds[round_index]
            for rounds in per_model_rounds
        ]

        round_numbers = {
            entry["round"]
            for entry in round_entries
        }

        if len(round_numbers) != 1:
            raise ValueError(
                "Training runs contain inconsistent round numbers."
            )

        sampled_rounds.append(round_entries[0]["round"])

        for metric_name in TRAINING_CURVE_METRICS:
            metric_summary = summarize_values(
                [
                    entry.get(metric_name)
                    for entry in round_entries
                ],
                use_sample_standard_deviation=True,
            )
            metric_means[metric_name].append(
                metric_summary["mean"]
            )
            metric_standard_deviations[metric_name].append(
                metric_summary["standard_deviation"]
            )

    return {
        "source_rounds": number_of_rounds,
        "points": len(sampled_rounds),
        "sampling": "evenly_spaced_including_endpoints",
        "observations_per_point": len(per_model_rounds),
        "rounds": sampled_rounds,
        "metrics": {
            metric_name: {
                "mean": metric_means[metric_name],
                "standard_deviation": (
                    metric_standard_deviations[metric_name]
                ),
            }
            for metric_name in TRAINING_CURVE_METRICS
        },
    }

def aggregate_reward_components(
    analyzed_runs: list[dict],
) -> dict:
    """Aggregate arbitrary reward events across trained models."""
    training_runs = [
        run
        for run in analyzed_runs
        if run["training_metrics"] is not None
    ]

    if not training_runs:
        return {}

    event_names = sorted(
        {
            event_name
            for run in training_runs
            for event_name in run["training_metrics"][
                "reward_components"
            ]
        }
    )

    aggregated = {}

    for event_name in event_names:
        components = [
            run["training_metrics"]["reward_components"].get(
                event_name,
                {
                    "count": 0,
                    "reward_sum": 0.0,
                    "mean_count_per_round": 0.0,
                    "mean_reward_per_round": 0.0,
                },
            )
            for run in training_runs
        ]

        aggregated[event_name] = {
            "total_count": sum(
                component["count"]
                for component in components
            ),
            "total_reward_sum": sum(
                component["reward_sum"]
                for component in components
            ),
            "mean_count_per_round": summarize_values(
                [
                    component["mean_count_per_round"]
                    for component in components
                ],
                use_sample_standard_deviation=True,
            ),
            "mean_reward_per_round": summarize_values(
                [
                    component["mean_reward_per_round"]
                    for component in components
                ],
                use_sample_standard_deviation=True,
            ),
        }

    return aggregated


def aggregate_action_counts(
    analyzed_runs: list[dict],
) -> dict:
    """Aggregate selected actions and labelled Q-updates."""
    action_order = (
        "UP",
        "RIGHT",
        "DOWN",
        "LEFT",
        "WAIT",
        "BOMB",
    )

    training_runs = [
        run
        for run in analyzed_runs
        if run["training_metrics"] is not None
    ]

    if not training_runs:
        return {}

    return {
        action: {
            "selected_total": sum(
                run["training_metrics"]["action_counts"][action]
                for run in training_runs
            ),
            "updates_total": sum(
                run["training_metrics"]["action_update_counts"][
                    action
                ]
                for run in training_runs
            ),
            "selected_per_model": summarize_values(
                [
                    run["training_metrics"]["action_counts"][action]
                    for run in training_runs
                ],
                use_sample_standard_deviation=True,
            ),
        }
        for action in action_order
    }

def build_analysis_summary(
    experiment_index: dict,
    configuration: dict,
    analyzed_runs: list[dict],
    evaluation_summary: dict,
    evaluation_by_seed: list[dict],
    training_summary: dict,
    training_curves: dict,
    reward_summary: dict,
    action_summary: dict,
) -> dict:
    """Build a compact, machine-readable experiment analysis."""
    number_of_evaluations = sum(
        len(run["evaluations"])
        for run in analyzed_runs
    )

    run_summaries = []

    for run in analyzed_runs:
        training_metrics = run["training_metrics"]
        run_summaries.append(
            {
                "training_world_seed": run[
                    "training_world_seed"
                ],
                "training_agent_seed": run[
                    "training_agent_seed"
                ],
                "training_results_file": run[
                    "training_results_file"
                ],
                "training_metadata": (
                    None
                    if training_metrics is None
                    else training_metrics["metadata"]
                ),
                "opponents": configuration["opponents"],
                "training_summary": (
                    None
                    if training_metrics is None
                    else training_metrics["summary"]
                ),
                "training_action_counts": (
                    None
                    if training_metrics is None
                    else training_metrics["action_counts"]
                ),
                "training_action_update_counts": (
                    None
                    if training_metrics is None
                    else training_metrics["action_update_counts"]
                ),
                "model": run["model"],
                "model_source": run["model_source"],
                "model_sha256": run["model_sha256"],
                "evaluation_summary": run[
                    "evaluation_summary"
                ],
                "evaluations": run["evaluations"],
            }
        )

    return {
        "schema_version": 8,
        "source": {
            "experiment_index": "experiment.json",
            "git_commit": experiment_index.get("git_commit"),
        },
        "experiment": {
            "status": experiment_index["status"],
            "experiment_mode": experiment_mode(configuration),
            "configuration": configuration,
            "training_runs": sum(
                run["training_metrics"] is not None
                for run in analyzed_runs
            ),
            "evaluated_models": len(analyzed_runs),
            "evaluation_files": number_of_evaluations,
        },
        "aggregate": {
            "evaluation": evaluation_summary,
            "evaluation_by_seed": evaluation_by_seed,
            "training": training_summary,
            "training_curves": training_curves,
            "reward_components": reward_summary,
            "training_actions": action_summary,
        },
        "runs": run_summaries,
    }


def retain_completed_diagnostics(summary_file: Path, summary: dict) -> None:
    """Keep a finished historical diagnosis when the identical run predates collection."""
    if summary.get("evaluation_diagnostics", {}).get("status") != "unavailable":
        return
    try:
        previous = json.loads(summary_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    diagnostic = previous.get("evaluation_diagnostics", previous.get("replay_diagnostics", {}))
    if (diagnostic.get("status") == "complete"
            and all(previous.get(key) == summary.get(key) for key in ("source", "experiment", "runs"))):
        summary["evaluation_diagnostics"] = {**diagnostic, "source": "legacy_completed_summary"}


def write_analysis_summary(
    experiment_file: Path,
    analysis_summary: dict,
) -> Path:
    """Write analysis/summary.json atomically."""
    analysis_directory = (
        experiment_file.resolve().parent / "analysis"
    )
    summary_file = analysis_directory / "summary.json"
    temporary_file = analysis_directory / "summary.json.tmp"

    try:
        analysis_directory.mkdir(exist_ok=True)

        with temporary_file.open("w", encoding="utf-8") as file:
            json.dump(
                analysis_summary,
                file,
                indent=2,
                sort_keys=True,
            )
            file.write("\n")

        temporary_file.replace(summary_file)
    except OSError as error:
        raise ValueError(
            f"Could not write analysis summary: {error}"
        ) from error

    return summary_file


def summarize_pressure_counts(evaluations: list[dict]) -> dict:
    """Pool exact round counts; never average percentages from different seeds."""
    totals = {
        name: sum(evaluation[name] for evaluation in evaluations)
        for name in ("rounds", "own_kills", "score_wins", "score_draws", "score_losses")
    }
    if not totals["rounds"]:
        raise ValueError("A pressure-sweep variant contains no evaluation rounds.")
    return {
        **totals,
        "mean_kills_per_round": totals["own_kills"] / totals["rounds"],
        "score_win_rate": totals["score_wins"] / totals["rounds"],
    }


def analyze_pressure_sweep(experiment_file: Path, experiment_index: dict) -> dict:
    """Compare completed frozen-policy variants using the existing combat rules."""
    if experiment_index.get("status") != "completed":
        raise ValueError("Only completed pressure sweeps can be analyzed.")
    configuration = experiment_index.get("configuration", {})
    sweep = configuration.get("pressure_sweep", {})
    expected = {
        (threshold, discount)
        for threshold in sweep.get("thresholds", [])
        for discount in sweep.get("distance_discounts", [])
    }
    variants = experiment_index.get("variants")
    if not expected or not isinstance(variants, list) or not variants:
        raise ValueError("The pressure sweep contains no configured variants.")

    rows, seen, hashes = [], set(), set()
    for variant in variants:
        if not isinstance(variant, dict):
            raise ValueError("A pressure-sweep variant must be an object.")
        pair = (variant.get("pressure_threshold"), variant.get("pressure_distance_discount"))
        if pair not in expected or pair in seen:
            raise ValueError("The pressure sweep contains duplicate or unexpected variants.")
        seen.add(pair)
        child_file = resolve_result_file(
            experiment_file.resolve().parent, variant.get("experiment"), "variant.experiment")
        child = load_experiment_index(child_file)
        child_configuration, runs = validate_experiment_index(child)
        if experiment_mode(child_configuration) != EVALUATE_CHECKPOINT_MODE:
            raise ValueError("Pressure sweeps require frozen-checkpoint evaluation runs.")
        for key in ("agent", "scenario", "opponents", "feature_version", "reward_version",
                    "action_space_version", "learning_rate_version", "exploration_version",
                    "discount_factor_version", "td_target_version", "checkpoint"):
            if child_configuration.get(key) != configuration.get(key):
                raise ValueError(f"Pressure-sweep variant has inconsistent {key}.")
        if child.get("git_commit") != experiment_index.get("git_commit"):
            raise ValueError("Pressure-sweep variants were evaluated with different code commits.")
        evaluation_configuration = child_configuration["evaluation"]
        for key in ("runs", "rounds_per_seed"):
            if evaluation_configuration.get(key) != configuration["evaluation"].get(key):
                raise ValueError(f"Pressure-sweep variant has inconsistent evaluation {key}.")
        actual_pair = (evaluation_configuration.get("pressure_threshold"),
                       evaluation_configuration.get("pressure_distance_discount"))
        if actual_pair != pair:
            raise ValueError("Pressure parameters differ between variant index and configuration.")

        run = runs[0]
        evaluations, validation_evaluations = [], []
        for evaluation in run["evaluations"]:
            if not isinstance(evaluation, dict):
                raise ValueError("A pressure-sweep evaluation must be an object.")
            result_file = resolve_result_file(child_file.parent, evaluation.get("results"),
                                              "evaluation.results")
            result = load_experiment_index(result_file)
            rounds = result.get("by_round")
            if not isinstance(rounds, dict) or not rounds:
                raise ValueError(f"Evaluation contains no rounds: {result_file}")
            outcomes = [combat_round_row(name, data, configuration["agent"])
                        for name, data in rounds.items()]
            opponents = opponent_instance_names(configuration["opponents"])
            if any(list(outcome["opponents"]) != opponents for outcome in outcomes):
                raise ValueError(f"Evaluation used an unexpected opponent roster: {result_file}")
            seed_result = {
                "evaluation_world_seed": evaluation.get("evaluation_world_seed"),
                "evaluation_agent_seed": evaluation.get("evaluation_agent_seed"),
                "results_file": evaluation["results"],
                "rounds": len(outcomes),
                "own_kills": sum(outcome["kills"] for outcome in outcomes),
                "score_wins": sum(outcome["score_win"] for outcome in outcomes),
                "score_draws": sum(outcome["score_draw"] for outcome in outcomes),
                "score_losses": sum(outcome["score_loss"] for outcome in outcomes),
            }
            evaluations.append(seed_result)
            validation_evaluations.append({
                **seed_result, "metrics": {"rounds": len(outcomes), "opponents": opponents}})
        validate_run_consistency(child_configuration, {
            **run, "training_metrics": None, "evaluations": validation_evaluations}, 1)
        hashes.add(run["model_sha256"])
        rows.append({**variant, **summarize_pressure_counts(evaluations),
                     "model_sha256": run["model_sha256"], "evaluations": evaluations})
    if seen != expected:
        raise ValueError("The pressure sweep is missing configured variants.")
    if len(hashes) != 1:
        raise ValueError("Pressure-sweep variants did not use the same checkpoint.")
    if experiment_index.get("checkpoint_sha256") not in (None, next(iter(hashes))):
        raise ValueError("Pressure-sweep checkpoint differs from its parent index.")

    rows.sort(key=lambda row: (-row["pressure_distance_discount"], row["pressure_threshold"]))
    return {
        "schema_version": 8,
        "experiment_kind": "pressure_sweep",
        "source": {"experiment_index": "experiment.json",
                   "git_commit": experiment_index.get("git_commit"),
                   "model_sha256": next(iter(hashes))},
        "experiment": {"status": "completed", "configuration": configuration,
                       "experiment_mode": EVALUATE_CHECKPOINT_MODE,
                       "evaluated_models": 1, "training_runs": 0,
                       "variants": len(rows), "evaluation_rounds": sum(row["rounds"] for row in rows)},
        "metric_definitions": {
            "mean_kills_per_round": "Own engine-credited opponent kills divided by evaluated rounds.",
            "score_win_rate": "Rounds with a strictly higher final engine score than every opponent, "
                              "divided by evaluated rounds. Top-score ties are draws; survival is irrelevant.",
        },
        "variants": rows,
    }

def parse_arguments():
    """Parse the experiment-index path."""
    parser = argparse.ArgumentParser(
        description="Analyze one complete Task 3 experiment."
    )

    parser.add_argument(
        "experiment_file",
        type=Path,
        help="Path to the experiment.json file.",
    )

    return parser.parse_args()


def main():
    """Load and inspect one completed experiment."""
    args = parse_arguments()

    try:
        experiment_index = load_experiment_index(
            args.experiment_file
        )

        if experiment_index.get("experiment_kind") == "pressure_sweep":
            analysis_summary = analyze_pressure_sweep(args.experiment_file, experiment_index)
            summary_file = write_analysis_summary(args.experiment_file, analysis_summary)
            print("Pressure threshold  Distance discount  Rounds  Own kills  Kills/round  Score wins")
            for row in analysis_summary["variants"]:
                print(f"{row['pressure_threshold']:18.2f}  {row['pressure_distance_discount']:17.2f}"
                      f"  {row['rounds']:6d}  {row['own_kills']:9d}"
                      f"  {row['mean_kills_per_round']:11.3f}"
                      f"  {row['score_wins']}/{row['rounds']} ({100 * row['score_win_rate']:.1f}%)")
            print(f"Analysis summary: {summary_file}")
            return

        configuration, runs = validate_experiment_index(
            experiment_index
        )

        analyzed_runs = analyze_experiment_runs(
            args.experiment_file,
            configuration,
            runs,
        )

        evaluation_summary = aggregate_evaluation_metrics(
            analyzed_runs
        )

        evaluation_by_seed = aggregate_evaluations_by_seed(
            analyzed_runs
        )

        training_summary = aggregate_training_metrics(
            analyzed_runs
        )

        training_curves = aggregate_training_curves(
            analyzed_runs
        )

        reward_summary = aggregate_reward_components(
            analyzed_runs
        )

        action_summary = aggregate_action_counts(
            analyzed_runs
        )

        analysis_summary = build_analysis_summary(
            experiment_index=experiment_index,
            configuration=configuration,
            analyzed_runs=analyzed_runs,
            evaluation_summary=evaluation_summary,
            evaluation_by_seed=evaluation_by_seed,
            training_summary=training_summary,
            training_curves=training_curves,
            reward_summary=reward_summary,
            action_summary=action_summary,
        )

        # Only aggregate records already collected during evaluation.
        from evaluation.task3.diagnostics import analyze_evaluation_diagnostics
        analysis_summary["evaluation_diagnostics"] = analyze_evaluation_diagnostics(
            args.experiment_file, experiment_index)

        retain_completed_diagnostics(
            args.experiment_file.resolve().parent / "analysis/summary.json", analysis_summary)

        summary_file = write_analysis_summary(
            args.experiment_file,
            analysis_summary,
        )

    except (ValueError, OSError) as error:
        raise SystemExit(
            f"Experiment analysis failed: {error}"
        )

    number_of_evaluations = sum(
        len(run["evaluations"])
        for run in analyzed_runs
    )

    print("All indexed result files analyzed successfully.")
    print(
        f"Experiment: {configuration['experiment_name']}"
    )
    print(f"Agent: {configuration['agent']}")
    print("Opponents: " + ", ".join(configuration["opponents"]))
    mode = experiment_mode(configuration)
    training_runs = sum(
        run["training_metrics"] is not None
        for run in analyzed_runs
    )

    print(f"Experiment mode: {mode}")
    print(f"Training runs: {training_runs}")
    print(f"Evaluated models: {len(analyzed_runs)}")
    print(f"Evaluation files: {number_of_evaluations}")
    print(f"Analysis summary: {summary_file}")
    for run in analyzed_runs:
        if run["training_metrics"] is None:
            print(
                "  Frozen checkpoint trained with seeds "
                f"w={run['training_world_seed']}, "
                f"a={run['training_agent_seed']}: "
                f"{len(run['evaluations'])} evaluations, "
                f"sha256={run['model_sha256'][:12]}..."
            )
        else:
            summary = run["training_metrics"]["summary"]

            print(
                "  Training seeds "
                f"w={run['training_world_seed']}, "
                f"a={run['training_agent_seed']}: "
                f"{summary['rounds']} rounds, "
                f"{len(run['evaluations'])} evaluations"
            )
    mean_coins = evaluation_summary[
        "mean_coins_per_round"
    ]
    mean_auc = evaluation_summary["mean_coin_auc"]
    completion = evaluation_summary["completion_rate"]
    invalid = evaluation_summary["invalid_action_rate"]
    crates = evaluation_summary[
        "mean_crates_destroyed_per_round"
    ]
    survival = evaluation_summary["survival_rate"]
    self_kill = evaluation_summary["self_kill_rate"]

    print("\nEvaluation summary across evaluated models")
    print(
        "  Mean coins per round: "
        f"{mean_coins['mean']:.2f}"
    )

    if mean_auc["mean"] is None:
        print("  Mean normalized coin AUC: N/A")
    else:
        print(
            "  Mean normalized coin AUC: "
            f"{100 * mean_auc['mean']:.2f}%"
        )

    print(
        "  Completion rate: "
        f"{100 * completion['mean']:.2f}%"
    )

    print(
        "  Invalid action rate: "
        f"{100 * invalid['mean']:.2f}%"
    )
    print(
        "  Mean crates destroyed per round: "
        f"{crates['mean']:.2f}"
    )
    print(
        "  Survival / self-kill rate: "
        f"{100 * survival['mean']:.2f}% / "
        f"{100 * self_kill['mean']:.2f}%"
    )
    print(
        "  Kills per round / rounds with kill: "
        f"{evaluation_summary['mean_kills_per_round']['mean']:.3f} / "
        f"{100 * evaluation_summary['rounds_with_kill_rate']['mean']:.2f}%"
    )
    print(
        "  All opponents eliminated / surviving win / death by opponent: "
        f"{100 * evaluation_summary['all_opponents_eliminated_rate']['mean']:.2f}% / "
        f"{100 * evaluation_summary['win_rate']['mean']:.2f}% / "
        f"{100 * evaluation_summary['death_by_opponent_rate']['mean']:.2f}%"
    )
    print(
        "  Final game-score wins / draws / losses: "
        f"{100 * evaluation_summary['score_win_rate']['mean']:.2f}% / "
        f"{100 * evaluation_summary['score_draw_rate']['mean']:.2f}% / "
        f"{100 * evaluation_summary['score_loss_rate']['mean']:.2f}%"
    )
    print(
        "  Opponent-suicide rounds / normalized kill AUC: "
        f"{100 * evaluation_summary['rounds_with_opponent_suicide_rate']['mean']:.2f}% / "
        f"{100 * evaluation_summary['mean_kill_auc']['mean']:.2f}%"
    )

    print(
        "  Evaluation seed summaries: "
        f"{len(evaluation_by_seed)}"
    )

    if training_summary:
        return_improvement = training_summary[
            "return_improvement"
        ]
        late_td_error = training_summary[
            "late_td_error_mae"
        ]
        final_states = training_summary[
            "final_q_table_states"
        ]
        final_coverage = training_summary[
            "final_state_action_coverage"
        ]
        bomb_action_rate = training_summary[
            "bomb_action_rate"
        ]

        print("\nTraining summary across trained models")
        print(
            "  Mean return improvement: "
            f"{return_improvement['mean']:.6f}"
        )
        print(
            "  Mean late TD-error MAE: "
            f"{late_td_error['mean']:.6f}"
        )
        print(
            "  Mean final Q-table states: "
            f"{final_states['mean']:.2f}"
        )
        print(
            "  Mean final state-action coverage: "
            f"{100 * final_coverage['mean']:.2f}%"
        )
        print(
            "  Aggregated training curve points: "
            f"{training_curves['points']} / "
            f"{training_curves['source_rounds']}"
        )
        print(
            "  Mean BOMB selection rate: "
            f"{100 * bomb_action_rate['mean']:.2f}%"
        )
        print(
            "  Mean kill improvement (late - early): "
            f"{training_summary['kills_per_round_improvement']['mean']:.3f}"
        )
        print(
            "  Late all-opponents-eliminated / death by opponent / "
            "kill AUC: "
            f"{100 * training_summary['late_all_opponents_eliminated_rate']['mean']:.2f}% / "
            f"{100 * training_summary['late_death_by_opponent_rate']['mean']:.2f}% / "
            f"{100 * training_summary['late_mean_kill_auc']['mean']:.2f}%"
        )

    if reward_summary:
        largest_reward_event = max(
            reward_summary,
            key=lambda event: reward_summary[event][
                "mean_reward_per_round"
            ]["mean"],
        )

        smallest_reward_event = min(
            reward_summary,
            key=lambda event: reward_summary[event][
                "mean_reward_per_round"
            ]["mean"],
        )

        largest_reward_value = reward_summary[
            largest_reward_event
        ]["mean_reward_per_round"]["mean"]

        smallest_reward_value = reward_summary[
            smallest_reward_event
        ]["mean_reward_per_round"]["mean"]

        print("\nReward summary across trained models")
        print(
            f"  Aggregated events: {len(reward_summary)}"
        )
        print(
            "  Largest mean contribution per round: "
            f"{largest_reward_event} "
            f"({largest_reward_value:.6f})"
        )
        print(
            "  Smallest mean contribution per round: "
            f"{smallest_reward_event} "
            f"({smallest_reward_value:.6f})"
        )


if __name__ == "__main__":
    main()
