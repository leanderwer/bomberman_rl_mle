"""Analyze one complete controlled Task 1 experiment."""

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev, stdev

import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.task1.analyze_results import analyze_evaluation_file
from evaluation.task1.analyze_training import analyze_training_file
from evaluation.exploration_metrics import training_strategy

SCENARIO_COIN_COUNTS = {
    "coin-heaven": 50,
    "coin-heaven-9": 9,
    "loot-crate": 50,
    "classic": 9,
}

EVALUATION_METRICS = (
    "mean_coins_per_round",
    "coin_collection_rate",
    "completion_rate",
    "mean_steps_per_round",
    "mean_steps_to_completion",
    "invalid_action_rate",
    "coins_per_100_steps",
    "mean_decision_time_ms",
    "decision_time_p95_ms",
    "maximum_decision_time_ms",
    "timeout_rate",
    "mean_coin_auc",
)

TRAINING_METRICS = (
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
)

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
)

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

    runs = experiment_index.get("runs")

    if not isinstance(runs, list) or not runs:
        raise ValueError(
            "The experiment index contains no training runs."
        )

    for run_number, run in enumerate(runs, start=1):
        if not isinstance(run, dict):
            raise ValueError(
                f"Training run {run_number} must be an object."
            )

        evaluations = run.get("evaluations")

        if not isinstance(evaluations, list) or not evaluations:
            raise ValueError(
                f"Training run {run_number} contains "
                "no evaluation runs."
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
    training_metrics = analyzed_run["training_metrics"]
    metadata = training_metrics["metadata"]
    summary = training_metrics["summary"]

    expected_metadata = {
        "world_seed": analyzed_run["training_world_seed"],
        "agent_seed": analyzed_run["training_agent_seed"],
        "scenario": configuration["scenario"],
        "feature_version": configuration["feature_version"],
        "reward_version": configuration["reward_version"],
    }

    if "exploration_version" in configuration and "exploration_version" in metadata:
        expected_metadata["exploration_version"] = configuration["exploration_version"]

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

    expected_evaluation_runs = configuration[
        "evaluation"
    ]["runs"]

    if len(analyzed_run["evaluations"]) != len(
        expected_evaluation_runs
    ):
        raise ValueError(
            f"Training run {run_number} contains the wrong "
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
            f"Training run {run_number} contains unexpected "
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

def analyze_experiment_runs(
    experiment_file: Path,
    configuration: dict,
    runs: list[dict],
) -> list[dict]:
    """Analyze every indexed training and evaluation result file."""
    experiment_directory = experiment_file.resolve().parent
    agent_name = configuration["agent"]
    coins_per_round = resolve_coins_per_round(configuration)
    analyzed_runs = []

    for run_number, run in enumerate(runs, start=1):
        training_file = resolve_result_file(
            experiment_directory,
            run.get("training_results"),
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
                    f"{evaluation_number} of training run "
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
    ]

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
) -> list[dict]:
    """Aggregate per-round training metrics across models."""
    training_strategy(analyzed_runs)
    per_model_rounds = [
        run["training_metrics"]["per_round"]
        for run in analyzed_runs
    ]

    number_of_rounds = len(per_model_rounds[0])

    if any(
        len(rounds) != number_of_rounds
        for rounds in per_model_rounds
    ):
        raise ValueError(
            "Training runs contain different numbers of rounds."
        )

    aggregated_rounds = []

    for round_index in range(number_of_rounds):
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

        aggregated_rounds.append(
            {
                "round": round_entries[0]["round"],
                "metrics": {
                    metric_name: summarize_values(
                        [
                            entry.get(metric_name)
                            for entry in round_entries
                        ],
                        use_sample_standard_deviation=True,
                    )
                    for metric_name in TRAINING_CURVE_METRICS
                },
            }
        )

    return aggregated_rounds

def aggregate_reward_components(
    analyzed_runs: list[dict],
) -> dict:
    """Aggregate arbitrary reward events across trained models."""
    event_names = sorted(
        {
            event_name
            for run in analyzed_runs
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
            for run in analyzed_runs
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

def build_analysis_summary(
    experiment_index: dict,
    configuration: dict,
    analyzed_runs: list[dict],
    evaluation_summary: dict,
    evaluation_by_seed: list[dict],
    training_summary: dict,
    training_curves: list[dict],
    reward_summary: dict,
) -> dict:
    """Build a compact, machine-readable experiment analysis."""
    number_of_evaluations = sum(
        len(run["evaluations"])
        for run in analyzed_runs
    )

    run_summaries = []

    for run in analyzed_runs:
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
                "training_metadata": run[
                    "training_metrics"
                ]["metadata"],
                "training_summary": run[
                    "training_metrics"
                ]["summary"],
                "evaluation_summary": run[
                    "evaluation_summary"
                ],
                "evaluations": run["evaluations"],
            }
        )

    return {
        "schema_version": 1,
        "source": {
            "experiment_index": "experiment.json",
            "git_commit": experiment_index.get("git_commit"),
        },
        "experiment": {
            "status": experiment_index["status"],
            "configuration": configuration,
            "training_runs": len(analyzed_runs),
            "evaluation_files": number_of_evaluations,
        },
        "aggregate": {
            "evaluation": evaluation_summary,
            "evaluation_by_seed": evaluation_by_seed,
            "training": training_summary,
            "training_curves": training_curves,
            "reward_components": reward_summary,
        },
        "runs": run_summaries,
    }


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
                indent=4,
                sort_keys=True,
            )
            file.write("\n")

        temporary_file.replace(summary_file)
    except OSError as error:
        raise ValueError(
            f"Could not write analysis summary: {error}"
        ) from error

    return summary_file

def parse_arguments():
    """Parse the experiment-index path."""
    parser = argparse.ArgumentParser(
        description="Analyze one complete Task 1 experiment."
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

        analysis_summary = build_analysis_summary(
            experiment_index=experiment_index,
            configuration=configuration,
            analyzed_runs=analyzed_runs,
            evaluation_summary=evaluation_summary,
            evaluation_by_seed=evaluation_by_seed,
            training_summary=training_summary,
            training_curves=training_curves,
            reward_summary=reward_summary,
        )

        summary_file = write_analysis_summary(
            args.experiment_file,
            analysis_summary,
        )

    except ValueError as error:
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
    print(f"Training runs: {len(analyzed_runs)}")
    print(f"Evaluation files: {number_of_evaluations}")
    print(f"Analysis summary: {summary_file}")
    for run in analyzed_runs:
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

    print("\nEvaluation summary across trained models")
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
        "  Evaluation seed summaries: "
        f"{len(evaluation_by_seed)}"
    )

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
        f"{len(training_curves)}"
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
