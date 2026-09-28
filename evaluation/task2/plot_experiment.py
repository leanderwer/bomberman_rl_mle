"""Create plots from an analyzed Task 2 experiment."""

import argparse
import json
from pathlib import Path

import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.exploration_metrics import exploration_plot_spec

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator
import numpy as np


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Create plots from an experiment analysis summary."
        )
    )
    parser.add_argument(
        "experiment_file",
        type=Path,
        help="Path to the experiment.json file.",
    )
    return parser.parse_args()


def load_analysis_summary(
    experiment_file: Path,
) -> tuple[dict, Path]:
    """Load analysis/summary.json for an experiment."""
    summary_file = (
        experiment_file.resolve().parent
        / "analysis"
        / "summary.json"
    )

    try:
        with summary_file.open("r", encoding="utf-8") as file:
            summary = json.load(file)
    except OSError as error:
        raise ValueError(
            f"Could not read analysis summary: {error}"
        ) from error
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Invalid analysis summary JSON: {error}"
        ) from error

    if summary.get("schema_version") not in {3, 4, 5, 6}:
        raise ValueError(
            "Unsupported analysis summary schema version."
        )

    return summary, summary_file


def metric_series(
    training_curves: list[dict] | dict,
    metric_name: str,
    scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract means and standard deviations for one metric."""
    if isinstance(training_curves, dict):
        metric = training_curves["metrics"][metric_name]

        return (
            np.asarray(
                [
                    np.nan if value is None else scale * float(value)
                    for value in metric["mean"]
                ],
                dtype=float,
            ),
            np.asarray(
                [
                    0.0 if value is None else scale * float(value)
                    for value in metric["standard_deviation"]
                ],
                dtype=float,
            ),
        )

    means = []
    standard_deviations = []

    for curve_point in training_curves:
        metric = curve_point["metrics"][metric_name]

        mean_value = metric["mean"]
        standard_deviation = metric["standard_deviation"]

        means.append(
            np.nan
            if mean_value is None
            else scale * float(mean_value)
        )
        standard_deviations.append(
            0.0
            if standard_deviation is None
            else scale * float(standard_deviation)
        )

    return (
        np.asarray(means, dtype=float),
        np.asarray(standard_deviations, dtype=float),
    )


def training_round_series(
    training_curves: list[dict] | dict,
) -> np.ndarray:
    """Return round numbers from legacy or compact curve storage."""
    if isinstance(training_curves, dict):
        rounds = training_curves["rounds"]
    else:
        rounds = [
            curve_point["round"]
            for curve_point in training_curves
        ]

    return np.asarray(rounds, dtype=int)


def plot_mean_and_sd(
    axis: plt.Axes,
    rounds: np.ndarray,
    means: np.ndarray,
    standard_deviations: np.ndarray,
    color: str,
    label: str,
    lower_bound: float | None = None,
    upper_bound: float | None = None,
) -> None:
    """Plot a mean curve with a one-standard-deviation band."""
    lower = means - standard_deviations
    upper = means + standard_deviations

    if lower_bound is not None:
        lower = np.maximum(lower, lower_bound)

    if upper_bound is not None:
        upper = np.minimum(upper, upper_bound)

    axis.plot(
        rounds,
        means,
        color=color,
        linewidth=2.0,
        label=label,
    )
    axis.fill_between(
        rounds,
        lower,
        upper,
        color=color,
        alpha=0.18,
        linewidth=0,
    )


def plot_training_overview(
    summary: dict,
    summary_file: Path,
) -> plt.Figure:
    """Create the aggregated training-overview figure."""
    training_curves = summary["aggregate"]["training_curves"]

    if not training_curves:
        raise ValueError(
            "Analysis summary contains no training curves."
        )

    rounds = training_round_series(training_curves)

    raw_returns, _ = metric_series(
        training_curves,
        "undiscounted_return",
    )
    moving_returns, moving_return_sd = metric_series(
        training_curves,
        "return_moving_average",
    )
    raw_coins, _ = metric_series(
        training_curves,
        "coins_collected",
    )
    moving_coins, moving_coins_sd = metric_series(
        training_curves,
        "coins_collected_moving_average",
    )
    raw_crate_rates, _ = metric_series(
        training_curves,
        "crate_destruction_rate",
        scale=100.0,
    )
    moving_crate_rates, moving_crate_rates_sd = metric_series(
        training_curves,
        "crate_destruction_rate_moving_average",
        scale=100.0,
    )
    raw_survival, _ = metric_series(
        training_curves,
        "survived",
        scale=100.0,
    )
    moving_survival, moving_survival_sd = metric_series(
        training_curves,
        "survival_rate_moving_average",
        scale=100.0,
    )
    exploration_metric, exploration_label, exploration_upper = exploration_plot_spec(summary)
    epsilon, epsilon_sd = metric_series(
        training_curves,
        exploration_metric,
    )
    q_updates, q_updates_sd = metric_series(
        training_curves,
        "q_updates",
    )

    figure, axes = plt.subplots(
        2,
        3,
        figsize=(15, 9),
        sharex=True,
    )

    return_axis = axes[0, 0]
    return_axis.plot(
        rounds,
        raw_returns,
        color="0.65",
        linewidth=1.0,
        alpha=0.7,
        label="Mean return per round",
    )
    plot_mean_and_sd(
        axis=return_axis,
        rounds=rounds,
        means=moving_returns,
        standard_deviations=moving_return_sd,
        color="#1f77b4",
        label="Moving-average return",
    )
    return_axis.axhline(
        0.0,
        color="black",
        linewidth=0.8,
        alpha=0.5,
    )
    return_axis.set_title("Training return")
    return_axis.set_ylabel("Undiscounted return")
    return_axis.legend(frameon=False)

    epsilon_axis = axes[0, 1]
    plot_mean_and_sd(
        axis=epsilon_axis,
        rounds=rounds,
        means=epsilon,
        standard_deviations=epsilon_sd,
        color="#9467bd",
        label=exploration_label,
        lower_bound=0.0,
        upper_bound=exploration_upper,
    )
    epsilon_axis.set_title("Exploration schedule")
    epsilon_axis.set_ylabel(exploration_label)
    epsilon_axis.set_ylim(bottom=0.0, top=exploration_upper)

    updates_axis = axes[0, 2]
    plot_mean_and_sd(
        axis=updates_axis,
        rounds=rounds,
        means=q_updates,
        standard_deviations=q_updates_sd,
        color="#1f77b4",
        label="Q-updates",
        lower_bound=0.0,
    )
    updates_axis.set_title("Learning updates")
    updates_axis.set_ylabel("Q-updates per round")
    updates_axis.set_ylim(bottom=0.0)

    coins_axis = axes[1, 0]
    coins_axis.plot(
        rounds,
        raw_coins,
        color="0.65",
        linewidth=1.0,
        alpha=0.7,
        label="Mean per round",
    )
    plot_mean_and_sd(
        axis=coins_axis,
        rounds=rounds,
        means=moving_coins,
        standard_deviations=moving_coins_sd,
        color="#1f77b4",
        label="Moving average",
        lower_bound=0.0,
    )
    coins_axis.set_title("Coins collected during training")
    coins_axis.set_xlabel("Training round")
    coins_axis.set_ylabel("Coins per round")
    coins_axis.set_ylim(bottom=0.0)
    coins_axis.legend(frameon=False)

    crates_axis = axes[1, 1]
    crates_axis.plot(
        rounds,
        raw_crate_rates,
        color="0.65",
        linewidth=1.0,
        alpha=0.7,
        label="Mean per round",
    )
    plot_mean_and_sd(
        axis=crates_axis,
        rounds=rounds,
        means=moving_crate_rates,
        standard_deviations=moving_crate_rates_sd,
        color="#ff7f0e",
        label="Moving average",
        lower_bound=0.0,
    )
    crates_axis.set_title("Crates destroyed during training")
    crates_axis.set_xlabel("Training round")
    crates_axis.set_ylabel("Initial crates destroyed (%)")
    crates_axis.set_ylim(bottom=0.0)
    crates_axis.legend(frameon=False)

    survival_axis = axes[1, 2]
    survival_axis.plot(
        rounds,
        raw_survival,
        color="0.65",
        linewidth=1.0,
        alpha=0.7,
        label="Mean per round",
    )
    plot_mean_and_sd(
        axis=survival_axis,
        rounds=rounds,
        means=moving_survival,
        standard_deviations=moving_survival_sd,
        color="#2ca02c",
        label="Moving average",
        lower_bound=0.0,
        upper_bound=100.0,
    )
    survival_axis.set_title("Survival during training")
    survival_axis.set_xlabel("Training round")
    survival_axis.set_ylabel("Survival rate (%)")
    survival_axis.set_ylim(0.0, 100.0)
    survival_axis.legend(frameon=False)

    for axis in axes.flat:
        axis.xaxis.set_major_locator(
            MaxNLocator(integer=True)
        )
        axis.grid(
            True,
            linewidth=0.6,
            alpha=0.3,
        )

    configuration = summary["experiment"]["configuration"]
    experiment_name = configuration.get(
        "experiment_name",
        summary_file.parent.parent.name,
    )
    training_runs = summary["experiment"].get("evaluated_models", summary["experiment"]["training_runs"])

    figure.suptitle(
        f"Training overview: {experiment_name}",
        fontsize=14,
    )
    figure.text(
        0.5,
        0.01,
        (
            f"Mean across {training_runs} independently trained "
            "models; shaded areas show ±1 sample SD where "
            "available."
        ),
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(
        rect=(0.0, 0.04, 1.0, 0.95),
    )

    return figure


def plot_training_diagnostics(
    summary: dict,
    summary_file: Path,
) -> plt.Figure:
    """Plot additional per-round training diagnostics."""
    training_curves = summary["aggregate"]["training_curves"]

    if not training_curves:
        raise ValueError(
            "Analysis summary contains no training curves."
        )

    rounds = training_round_series(training_curves)

    td_mae, td_mae_sd = metric_series(
        training_curves,
        "td_error_mae",
    )
    td_rmse, td_rmse_sd = metric_series(
        training_curves,
        "td_error_rmse",
    )
    td_max, td_max_sd = metric_series(
        training_curves,
        "td_error_abs_max",
    )
    q_table_states, q_table_states_sd = metric_series(
        training_curves,
        "q_table_states",
    )
    updated_pairs, updated_pairs_sd = metric_series(
        training_curves,
        "updated_state_action_pairs",
    )
    new_states, new_states_sd = metric_series(
        training_curves,
        "new_q_table_states",
    )
    new_pairs, new_pairs_sd = metric_series(
        training_curves,
        "new_updated_state_action_pairs",
    )
    coverage, coverage_sd = metric_series(
        training_curves,
        "state_action_coverage",
        scale=100.0,
    )

    figure, axes = plt.subplots(
        2,
        3,
        figsize=(15, 9),
        sharex=True,
    )

    mae_axis = axes[0, 0]
    plot_mean_and_sd(
        axis=mae_axis,
        rounds=rounds,
        means=td_mae,
        standard_deviations=td_mae_sd,
        color="#d62728",
        label="TD-error MAE",
        lower_bound=0.0,
    )
    mae_axis.set_title("TD-error MAE")
    mae_axis.set_ylabel("TD-error MAE")
    mae_axis.set_ylim(bottom=0.0)

    rmse_axis = axes[0, 1]
    plot_mean_and_sd(
        axis=rmse_axis,
        rounds=rounds,
        means=td_rmse,
        standard_deviations=td_rmse_sd,
        color="#ff7f0e",
        label="TD-error RMSE",
        lower_bound=0.0,
    )
    rmse_axis.set_title("TD-error RMSE")
    rmse_axis.set_ylabel("TD-error RMSE")
    rmse_axis.set_ylim(bottom=0.0)

    maximum_axis = axes[0, 2]
    plot_mean_and_sd(
        axis=maximum_axis,
        rounds=rounds,
        means=td_max,
        standard_deviations=td_max_sd,
        color="#d62728",
        label="Maximum absolute TD error",
        lower_bound=0.0,
    )
    maximum_axis.set_title("Largest TD-error")
    maximum_axis.set_ylabel("Maximum absolute TD error")
    maximum_axis.set_ylim(bottom=0.0)

    cumulative_axis = axes[1, 0]
    plot_mean_and_sd(
        axis=cumulative_axis,
        rounds=rounds,
        means=q_table_states,
        standard_deviations=q_table_states_sd,
        color="#2ca02c",
        label="Known feature states",
        lower_bound=0.0,
    )
    plot_mean_and_sd(
        axis=cumulative_axis,
        rounds=rounds,
        means=updated_pairs,
        standard_deviations=updated_pairs_sd,
        color="#17becf",
        label="Updated state-action pairs",
        lower_bound=0.0,
    )
    cumulative_axis.set_title("Q-table growth")
    cumulative_axis.set_xlabel("Training round")
    cumulative_axis.set_ylabel("Cumulative count")
    cumulative_axis.set_ylim(bottom=0.0)
    cumulative_axis.legend(
        frameon=False,
        loc="upper left",
        fontsize=8,
    )

    coverage_axis = axes[1, 1]
    plot_mean_and_sd(
        axis=coverage_axis,
        rounds=rounds,
        means=coverage,
        standard_deviations=coverage_sd,
        color="#9467bd",
        label="State-action coverage",
        lower_bound=0.0,
        upper_bound=100.0,
    )
    coverage_axis.set_title("State-action coverage")
    coverage_axis.set_xlabel("Training round")
    coverage_axis.set_ylabel("Coverage (%)")
    coverage_axis.set_ylim(0.0, 100.0)

    new_entries_axis = axes[1, 2]
    plot_mean_and_sd(
        axis=new_entries_axis,
        rounds=rounds,
        means=new_states,
        standard_deviations=new_states_sd,
        color="#2ca02c",
        label="New states",
        lower_bound=0.0,
    )
    plot_mean_and_sd(
        axis=new_entries_axis,
        rounds=rounds,
        means=new_pairs,
        standard_deviations=new_pairs_sd,
        color="#17becf",
        label="New state-action pairs",
        lower_bound=0.0,
    )
    new_entries_axis.set_title("Newly explored entries")
    new_entries_axis.set_xlabel("Training round")
    new_entries_axis.set_ylabel("New entries per round")
    new_entries_axis.set_ylim(bottom=0.0)
    new_entries_axis.legend(frameon=False)

    for axis in axes.flat:
        axis.xaxis.set_major_locator(
            MaxNLocator(integer=True)
        )
        axis.grid(
            True,
            linewidth=0.6,
            alpha=0.3,
        )

    configuration = summary["experiment"]["configuration"]
    experiment_name = configuration.get(
        "experiment_name",
        summary_file.parent.parent.name,
    )
    training_runs = summary["experiment"].get("evaluated_models", summary["experiment"]["training_runs"])

    figure.suptitle(
        f"Training diagnostics: {experiment_name}",
        fontsize=14,
    )
    figure.text(
        0.5,
        0.01,
        (
            f"Mean across {training_runs} independently "
            "trained models; shaded areas show ±1 sample SD "
            "where available."
        ),
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(
        rect=(0.0, 0.04, 1.0, 0.95),
    )

    return figure


def evaluation_metric_series(
    evaluation_by_seed: list[dict],
    metric_name: str,
    scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract evaluation means and SDs for each seed."""
    means = []
    standard_deviations = []

    for seed_summary in evaluation_by_seed:
        metric = seed_summary.get("metrics", {}).get(
            metric_name,
            {},
        )

        means.append(
            np.nan
            if metric.get("mean") is None
            else scale * float(metric["mean"])
        )
        standard_deviations.append(
            0.0
            if metric.get("standard_deviation") is None
            else scale * float(
                metric["standard_deviation"]
            )
        )

    return (
        np.asarray(means, dtype=float),
        np.asarray(standard_deviations, dtype=float),
    )


def plot_evaluation_metric_panel(
    axis: plt.Axes,
    evaluation_by_seed: list[dict],
    aggregate_evaluation: dict,
    positions: np.ndarray,
    seed_labels: list[str],
    metric_name: str,
    title: str,
    y_label: str,
    scale: float,
    color: str,
    empty_message: str | None = None,
    zero_message: str | None = None,
) -> None:
    """Plot one non-negative evaluation metric by seed."""
    means, standard_deviations = (
        evaluation_metric_series(
            evaluation_by_seed,
            metric_name,
            scale,
        )
    )

    available = np.isfinite(means)

    if np.any(available):
        available_means = means[available]
        available_standard_deviations = (
            standard_deviations[available]
        )

        # Do not draw error bars below the valid value zero.
        lower_errors = np.minimum(
            available_standard_deviations,
            available_means,
        )
        errors = np.vstack(
            (
                lower_errors,
                available_standard_deviations,
            )
        )

        axis.bar(
            positions[available],
            available_means,
            yerr=errors,
            color=color,
            alpha=0.82,
            capsize=4,
            error_kw={
                "elinewidth": 1.2,
                "capthick": 1.2,
            },
        )

        overall_mean = aggregate_evaluation.get(
            metric_name,
            {},
        ).get("mean")

        if overall_mean is not None:
            axis.axhline(
                scale * float(overall_mean),
                color="black",
                linestyle="--",
                linewidth=1.2,
            )

        axis.set_ylim(bottom=0.0)

        if (
            zero_message is not None
            and np.allclose(available_means, 0.0)
        ):
            axis.set_ylim(0.0, 1.0)
            axis.text(
                0.5,
                0.5,
                zero_message,
                transform=axis.transAxes,
                ha="center",
                va="center",
                fontsize=10,
                color="0.35",
            )
    else:
        axis.set_ylim(0.0, 1.0)
        axis.text(
            0.5,
            0.5,
            empty_message or "No available observations",
            transform=axis.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            color="0.35",
        )

    axis.set_title(title)
    axis.set_ylabel(y_label)
    axis.set_xticks(positions)
    axis.set_xticklabels(seed_labels)


def plot_evaluation_diagnostics(
    summary: dict,
    summary_file: Path,
) -> plt.Figure:
    """Plot additional evaluation diagnostics by seed."""
    evaluation_by_seed = sorted(
        summary["aggregate"]["evaluation_by_seed"],
        key=lambda entry: (
            entry["evaluation_world_seed"],
            entry["evaluation_agent_seed"],
        ),
    )

    if not evaluation_by_seed:
        raise ValueError(
            "Analysis summary contains no evaluations."
        )

    aggregate_evaluation = summary["aggregate"][
        "evaluation"
    ]

    seed_labels = [
        (
            f"w={entry['evaluation_world_seed']}\n"
            f"a={entry['evaluation_agent_seed']}"
        )
        for entry in evaluation_by_seed
    ]
    positions = np.arange(len(seed_labels))

    figure, axes = plt.subplots(
        2,
        3,
        figsize=(15, 9),
    )

    plot_evaluation_metric_panel(
        axis=axes[0, 0],
        evaluation_by_seed=evaluation_by_seed,
        aggregate_evaluation=aggregate_evaluation,
        positions=positions,
        seed_labels=seed_labels,
        metric_name="mean_coin_auc",
        title="Coin collection timing",
        y_label="Normalized coin AUC (%)",
        scale=100.0,
        color="#1f77b4",
    )

    plot_evaluation_metric_panel(
        axis=axes[0, 1],
        evaluation_by_seed=evaluation_by_seed,
        aggregate_evaluation=aggregate_evaluation,
        positions=positions,
        seed_labels=seed_labels,
        metric_name="mean_coins_found_per_round",
        title="Coins found",
        y_label="Mean coins found per round",
        scale=1.0,
        color="#ff7f0e",
        zero_message="No coins found",
    )

    plot_evaluation_metric_panel(
        axis=axes[0, 2],
        evaluation_by_seed=evaluation_by_seed,
        aggregate_evaluation=aggregate_evaluation,
        positions=positions,
        seed_labels=seed_labels,
        metric_name="crates_per_bomb",
        title="Bomb productivity",
        y_label="Crates per bomb",
        scale=1.0,
        color="#2ca02c",
    )

    plot_evaluation_metric_panel(
        axis=axes[1, 0],
        evaluation_by_seed=evaluation_by_seed,
        aggregate_evaluation=aggregate_evaluation,
        positions=positions,
        seed_labels=seed_labels,
        metric_name="mean_bombs_dropped_per_round",
        title="Bomb use",
        y_label="Mean bombs per round",
        scale=1.0,
        color="#9467bd",
        zero_message="No bombs dropped",
    )

    decision_axis = axes[1, 1]
    decision_metrics = (
        (
            "mean_decision_time_ms",
            "Mean",
            "#1f77b4",
        ),
        (
            "decision_time_p95_ms",
            "P95",
            "#ff7f0e",
        ),
        (
            "maximum_decision_time_ms",
            "Maximum",
            "#d62728",
        ),
    )

    bar_width = 0.24
    offsets = (-bar_width, 0.0, bar_width)

    for offset, (
        metric_name,
        label,
        color,
    ) in zip(offsets, decision_metrics):
        means, standard_deviations = (
            evaluation_metric_series(
                evaluation_by_seed,
                metric_name,
            )
        )
        available = np.isfinite(means)

        if not np.any(available):
            continue

        available_means = means[available]
        available_standard_deviations = (
            standard_deviations[available]
        )
        lower_errors = np.minimum(
            available_standard_deviations,
            available_means,
        )
        errors = np.vstack(
            (
                lower_errors,
                available_standard_deviations,
            )
        )

        decision_axis.bar(
            positions[available] + offset,
            available_means,
            width=bar_width,
            yerr=errors,
            color=color,
            alpha=0.82,
            capsize=3,
            label=label,
            error_kw={
                "elinewidth": 1.0,
                "capthick": 1.0,
            },
        )

    decision_axis.set_title("Decision time")
    decision_axis.set_ylabel("Decision time (ms)")
    decision_axis.set_xticks(positions)
    decision_axis.set_xticklabels(seed_labels)
    decision_axis.set_ylim(bottom=0.0)
    decision_axis.legend(frameon=False)

    plot_evaluation_metric_panel(
        axis=axes[1, 2],
        evaluation_by_seed=evaluation_by_seed,
        aggregate_evaluation=aggregate_evaluation,
        positions=positions,
        seed_labels=seed_labels,
        metric_name="invalid_action_rate",
        title="Invalid actions",
        y_label="Invalid action rate (%)",
        scale=100.0,
        color="#d62728",
        zero_message="No invalid actions (0%)",
    )

    for axis in axes.flat:
        axis.grid(
            True,
            axis="y",
            linewidth=0.6,
            alpha=0.3,
        )

    configuration = summary["experiment"]["configuration"]
    experiment_name = configuration.get(
        "experiment_name",
        summary_file.parent.parent.name,
    )
    training_runs = summary["experiment"].get("evaluated_models", summary["experiment"]["training_runs"])

    figure.suptitle(
        f"Evaluation diagnostics: {experiment_name}",
        fontsize=14,
    )
    figure.text(
        0.5,
        0.01,
        (
            f"Bars show means across {training_runs} trained "
            "models for each evaluation-seed pair; error bars "
            "show ±1 sample SD. Dashed lines in single-metric "
            "panels show overall model means."
        ),
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(
        rect=(0.0, 0.05, 1.0, 0.95),
    )

    return figure


def plot_evaluation_overview(
    summary: dict,
    summary_file: Path,
) -> plt.Figure:
    """Create evaluation plots grouped by evaluation seed."""
    evaluation_by_seed = sorted(
        summary["aggregate"]["evaluation_by_seed"],
        key=lambda entry: (
            entry["evaluation_world_seed"],
            entry["evaluation_agent_seed"],
        ),
    )

    if not evaluation_by_seed:
        raise ValueError(
            "Analysis summary contains no evaluations."
        )

    aggregate_evaluation = summary["aggregate"]["evaluation"]

    seed_labels = [
        (
            f"w={entry['evaluation_world_seed']}\n"
            f"a={entry['evaluation_agent_seed']}"
        )
        for entry in evaluation_by_seed
    ]
    positions = np.arange(len(seed_labels))

    panels = (
        (
            "mean_coins_per_round",
            "Coins collected",
            "Mean coins per round",
            1.0,
            "#1f77b4",
            None,
        ),
        (
            "completion_rate",
            "Completed rounds",
            "Completion rate (%)",
            100.0,
            "#9467bd",
            "No completed rounds (0%)",
        ),
        (
            "crate_destruction_rate",
            "Crates destroyed",
            "Crate destruction rate (%)",
            100.0,
            "#ff7f0e",
            "No crates destroyed (0%)",
        ),
        (
            "survival_rate",
            "Survived rounds",
            "Survival rate (%)",
            100.0,
            "#2ca02c",
            "No survived rounds (0%)",
        ),
    )

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(12, 8),
    )

    for axis, panel in zip(axes.flat, panels):
        (
            metric_name,
            title,
            y_label,
            scale,
            color,
            zero_message,
        ) = panel

        plot_evaluation_metric_panel(
            axis=axis,
            evaluation_by_seed=evaluation_by_seed,
            aggregate_evaluation=aggregate_evaluation,
            positions=positions,
            seed_labels=seed_labels,
            metric_name=metric_name,
            title=title,
            y_label=y_label,
            scale=scale,
            color=color,
            zero_message=zero_message,
        )

        axis.grid(
            True,
            axis="y",
            linewidth=0.6,
            alpha=0.3,
        )

    configuration = summary["experiment"]["configuration"]
    experiment_name = configuration.get(
        "experiment_name",
        summary_file.parent.parent.name,
    )
    training_runs = summary["experiment"].get("evaluated_models", summary["experiment"]["training_runs"])

    figure.suptitle(
        f"Evaluation overview: {experiment_name}",
        fontsize=14,
    )
    figure.text(
        0.5,
        0.01,
        (
            f"Bars show means across {training_runs} trained "
            "models for each evaluation-seed pair; error bars "
            "show ±1 sample SD. Dashed lines show the overall "
            "model mean."
        ),
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(
        rect=(0.0, 0.05, 1.0, 0.95),
    )

    return figure


def plot_grouped_evaluation_rates(
    axis: plt.Axes,
    evaluations: list[dict],
    positions: np.ndarray,
    labels: list[str],
    metric_specs: tuple[tuple[str, str, str], ...],
    title: str,
    empty_message: str,
    y_label: str = "Follow rate (%)",
) -> None:
    """Plot several percentage metrics side by side."""
    bar_width = min(0.24, 0.72 / len(metric_specs))
    center = (len(metric_specs) - 1) / 2
    plotted_metric = False

    for metric_index, (
        metric_name,
        metric_label,
        color,
    ) in enumerate(metric_specs):
        means, standard_deviations = evaluation_metric_series(
            evaluations,
            metric_name,
            scale=100.0,
        )
        available = np.isfinite(means)

        if not np.any(available):
            continue

        plotted_metric = True
        available_means = means[available]
        available_standard_deviations = (
            standard_deviations[available]
        )
        lower_errors = np.minimum(
            available_standard_deviations,
            available_means,
        )
        errors = np.vstack(
            (
                lower_errors,
                available_standard_deviations,
            )
        )
        offset = (metric_index - center) * bar_width

        axis.bar(
            positions[available] + offset,
            available_means,
            width=bar_width,
            yerr=errors,
            color=color,
            alpha=0.82,
            capsize=3,
            label=metric_label,
            error_kw={
                "elinewidth": 1.0,
                "capthick": 1.0,
            },
        )

    axis.set_title(title)
    axis.set_ylabel(y_label)
    axis.set_xticks(positions)
    axis.set_xticklabels(labels)
    axis.set_ylim(0.0, 100.0)

    if plotted_metric:
        legend_columns = 2 if len(metric_specs) >= 4 else 1
        axis.legend(
            frameon=False,
            fontsize=8,
            ncol=legend_columns,
        )
    else:
        axis.text(
            0.5,
            0.5,
            empty_message,
            transform=axis.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            color="0.35",
        )


def plot_evaluation_by_model(
    summary: dict,
    summary_file: Path,
) -> plt.Figure:
    """Create evaluation plots grouped by trained model."""
    analyzed_runs = sorted(
        summary["runs"],
        key=lambda run: (
            run["training_world_seed"],
            run["training_agent_seed"],
        ),
    )

    if not analyzed_runs:
        raise ValueError(
            "Analysis summary contains no trained models."
        )

    aggregate_evaluation = summary["aggregate"]["evaluation"]
    model_evaluations = [
        {
            "metrics": run["evaluation_summary"],
        }
        for run in analyzed_runs
    ]

    model_labels = [
        (
            f"w={run['training_world_seed']}\n"
            f"a={run['training_agent_seed']}"
        )
        for run in analyzed_runs
    ]
    positions = np.arange(len(model_labels))

    panels = (
        (
            "mean_coins_per_round",
            "Coins collected",
            "Mean coins per round",
            1.0,
            "#1f77b4",
            None,
        ),
        (
            "completion_rate",
            "Completed rounds",
            "Completion rate (%)",
            100.0,
            "#9467bd",
            "No completed rounds (0%)",
        ),
        (
            "crate_destruction_rate",
            "Crates destroyed",
            "Crate destruction rate (%)",
            100.0,
            "#ff7f0e",
            "No crates destroyed (0%)",
        ),
        (
            "survival_rate",
            "Survived rounds",
            "Survival rate (%)",
            100.0,
            "#2ca02c",
            "No survived rounds (0%)",
        ),
    )

    figure_width = max(
        12.0,
        min(20.0, 7.5 + 0.9 * len(model_labels)),
    )
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(figure_width, 8),
    )

    for axis, panel in zip(axes.flat, panels):
        (
            metric_name,
            title,
            y_label,
            scale,
            color,
            zero_message,
        ) = panel

        plot_evaluation_metric_panel(
            axis=axis,
            evaluation_by_seed=model_evaluations,
            aggregate_evaluation=aggregate_evaluation,
            positions=positions,
            seed_labels=model_labels,
            metric_name=metric_name,
            title=title,
            y_label=y_label,
            scale=scale,
            color=color,
            zero_message=zero_message,
        )

        axis.grid(
            True,
            axis="y",
            linewidth=0.6,
            alpha=0.3,
        )

    configuration = summary["experiment"]["configuration"]

    if len(model_labels) > 10:
        for axis in axes.flat:
            plt.setp(
                axis.get_xticklabels(),
                rotation=45,
                ha="right",
                rotation_mode="anchor",
            )
            axis.tick_params(axis="x", labelsize=8)

    experiment_name = configuration.get(
        "experiment_name",
        summary_file.parent.parent.name,
    )

    evaluation_counts = {
        len(run["evaluations"])
        for run in analyzed_runs
    }

    if len(evaluation_counts) == 1:
        evaluation_count = next(iter(evaluation_counts))
        evaluation_description = (
            f"{evaluation_count} evaluation-seed "
            f"pair{'s' if evaluation_count != 1 else ''}"
        )
    else:
        evaluation_description = (
            "the available evaluation-seed pairs"
        )

    figure.suptitle(
        f"Evaluation by trained model: {experiment_name}",
        fontsize=14,
    )
    figure.text(
        0.5,
        0.01,
        (
            f"Bars show means across {evaluation_description} "
            "for each independently trained model; error bars "
            "show ±1 population SD across evaluation seeds. "
            "Dashed lines in single-metric panels show the "
            "overall mean across trained models."
        ),
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(
        rect=(0.0, 0.05, 1.0, 0.95),
    )

    return figure


def plot_self_kill_diagnosis(
    axis: plt.Axes,
    evaluations: list[dict],
    positions: np.ndarray,
    labels: list[str],
) -> None:
    """Plot mutually exclusive self-kill causes as stacked rates."""
    category_specs = (
        (
            "self_kill_unsafe_action_with_safe_alternative_rate",
            "Unsafe final action",
            "#d62728",
        ),
        (
            "self_kill_no_survivable_action_available_rate",
            "Already trapped",
            "#ff7f0e",
        ),
        (
            "self_kill_not_explained_by_final_action_rate",
            "Not explained by final action",
            "#7f7f7f",
        ),
    )
    bottoms = np.zeros(len(positions), dtype=float)
    diagnostics_available = np.zeros(
        len(positions),
        dtype=bool,
    )

    for metric_name, metric_label, color in category_specs:
        means, _ = evaluation_metric_series(
            evaluations,
            metric_name,
            scale=100.0,
        )
        available = np.isfinite(means)

        if not np.any(available):
            continue

        diagnostics_available |= available
        values = np.where(available, means, 0.0)
        axis.bar(
            positions[available],
            values[available],
            bottom=bottoms[available],
            color=color,
            alpha=0.86,
            label=metric_label,
        )
        bottoms += values

    if np.any(diagnostics_available):
        total_means, total_standard_deviations = (
            evaluation_metric_series(
                evaluations,
                "self_kill_rate",
                scale=100.0,
            )
        )
        total_available = (
            diagnostics_available & np.isfinite(total_means)
        )

        if np.any(total_available):
            lower_errors = np.minimum(
                total_standard_deviations[total_available],
                total_means[total_available],
            )
            errors = np.vstack(
                (
                    lower_errors,
                    total_standard_deviations[total_available],
                )
            )
            axis.errorbar(
                positions[total_available],
                bottoms[total_available],
                yerr=errors,
                fmt="none",
                ecolor="black",
                capsize=3,
                linewidth=1.0,
            )

        axis.legend(frameon=False, fontsize=7)
        axis.set_ylim(bottom=0.0)
    else:
        axis.set_ylim(0.0, 1.0)
        axis.text(
            0.5,
            0.5,
            "Self-kill diagnostics not recorded",
            transform=axis.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            color="0.35",
        )

    axis.set_title("Self-kill diagnosis")
    axis.set_ylabel("Rounds (%)")
    axis.set_xticks(positions)
    axis.set_xticklabels(labels)


def plot_independent_target_conflict_heatmap(
    axis: plt.Axes,
    analyzed_runs: list[dict],
) -> None:
    """Plot pooled coin preference for the 4x3 true-conflict matrix.

    Color conditions on the policy selecting either safe target. Each cell
    also reports unconditional safe-target utilization and the complete true-
    conflict opportunity count, so ignoring both targets cannot masquerade as
    a learned coin/bomb preference.
    """
    opportunities = np.zeros((4, 3), dtype=int)
    coin_followed = np.zeros((4, 3), dtype=int)
    bomb_followed = np.zeros((4, 3), dtype=int)

    for run in analyzed_runs:
        for evaluation in run["evaluations"]:
            metrics = evaluation["metrics"]

            for coin_class in range(4):
                for bomb_class in range(3):
                    conflict_prefix = (
                        "target_conflict_"
                        f"coin_class_{coin_class}_"
                        f"bomb_class_{bomb_class}"
                    )
                    metric_values = (
                        metrics.get(
                            f"total_{conflict_prefix}_opportunities"
                        ),
                        metrics.get(
                            f"total_{conflict_prefix}_coin_followed"
                        ),
                        metrics.get(
                            f"total_{conflict_prefix}_bomb_followed"
                        ),
                    )

                    if any(value is None for value in metric_values):
                        continue

                    opportunities[coin_class, bomb_class] += int(
                        metric_values[0]
                    )
                    coin_followed[coin_class, bomb_class] += int(
                        metric_values[1]
                    )
                    bomb_followed[coin_class, bomb_class] += int(
                        metric_values[2]
                    )

    target_followed = coin_followed + bomb_followed
    coin_preference = np.full((4, 3), np.nan, dtype=float)
    target_utilization = np.full((4, 3), np.nan, dtype=float)
    selected = target_followed > 0
    observed = opportunities > 0
    coin_preference[selected] = (
        100.0 * coin_followed[selected] / target_followed[selected]
    )
    target_utilization[observed] = (
        100.0 * target_followed[observed] / opportunities[observed]
    )
    color_map = plt.get_cmap("RdBu").copy()
    color_map.set_bad("#eeeeee")
    image = axis.imshow(
        np.ma.masked_invalid(coin_preference),
        cmap=color_map,
        vmin=0.0,
        vmax=100.0,
        aspect="auto",
    )

    for coin_class in range(4):
        for bomb_class in range(3):
            count = opportunities[coin_class, bomb_class]
            preference = coin_preference[coin_class, bomb_class]
            utilization = target_utilization[coin_class, bomb_class]

            if count == 0:
                annotation = "No data\nn=0"
                text_color = "0.3"
            elif not np.isfinite(preference):
                annotation = f"Coin N/A\nUse 0.0%\nn={count}"
                text_color = "0.2"
            else:
                annotation = (
                    f"Coin {preference:.1f}%\n"
                    f"Use {utilization:.1f}%\n"
                    f"n={count}"
                )
                text_color = (
                    "white"
                    if preference <= 25.0 or preference >= 75.0
                    else "black"
                )

            axis.text(
                bomb_class,
                coin_class,
                annotation,
                ha="center",
                va="center",
                fontsize=7,
                color=text_color,
            )

    axis.set_title("True-conflict target preference (pooled)")
    axis.set_xlabel("Best-bomb quality class")
    axis.set_ylabel("Coin-distance class")
    axis.set_xticks(range(3))
    axis.set_xticklabels(
        ("0 Weak\n<3.5", "1 Medium\n3.5–<5", "2 Strong\n≥5")
    )
    axis.set_yticks(range(4))
    axis.set_yticklabels(
        (
            "0 Far\n≥7",
            "1 Mid-far\n5–6",
            "2 Near\n3–4",
            "3 Very near\n1–2",
        )
    )
    color_bar = axis.figure.colorbar(
        image,
        ax=axis,
        fraction=0.05,
        pad=0.03,
    )
    color_bar.set_label(
        "Coin choice among followed targets (%)",
        fontsize=8,
    )


def plot_policy_diagnostics_by_model(
    summary: dict,
    summary_file: Path,
) -> plt.Figure:
    """Create the policy-diagnostics page grouped by trained model."""
    analyzed_runs = sorted(
        summary["runs"],
        key=lambda run: (
            run["training_world_seed"],
            run["training_agent_seed"],
        ),
    )

    if not analyzed_runs:
        raise ValueError(
            "Analysis summary contains no trained models."
        )

    aggregate_evaluation = summary["aggregate"]["evaluation"]
    model_evaluations = [
        {"metrics": run["evaluation_summary"]}
        for run in analyzed_runs
    ]
    model_labels = [
        (
            f"w={run['training_world_seed']}\n"
            f"a={run['training_agent_seed']}"
        )
        for run in analyzed_runs
    ]
    positions = np.arange(len(model_labels))
    configuration = summary["experiment"]["configuration"]
    experiment_name = configuration.get(
        "experiment_name",
        summary_file.parent.parent.name,
    )
    figure_width = max(
        15.0,
        min(22.0, 9.5 + 1.0 * len(model_labels)),
    )
    feature_version = configuration.get("feature_version")
    parallel_masks_v2 = feature_version == "task2_parallel_masks_v2"
    independent_targets_v3 = (
        feature_version
        == "task2_parallel_masks_independent_targets_v3"
    )
    compact_symmetric_v4 = (
        feature_version
        == "task2_parallel_masks_compact_symmetric_v4"
    )
    compact_symmetric_v5 = (
        feature_version
        == "task2_parallel_masks_compact_symmetric_v5"
    )
    coin_route_bomb_v6 = (
        feature_version
        in (
            "task2_parallel_masks_coin_route_bomb_v6",
            "task2_report_v6_no_d4_v1",
            "task2_report_v6_conflict_classes_v1",
            "task2_report_v6_coin_priority_v1",
        )
    )
    independent_targets = (
        independent_targets_v3
        or compact_symmetric_v4
        or compact_symmetric_v5
        or coin_route_bomb_v6
    )
    parallel_masks = parallel_masks_v2 or independent_targets

    if not parallel_masks:
        figure, axes = plt.subplots(
            1,
            2,
            figsize=(figure_width, 5.5),
        )
        plot_grouped_evaluation_rates(
            axis=axes[0],
            evaluations=model_evaluations,
            positions=positions,
            labels=model_labels,
            metric_specs=(
                (
                    "coin_objective_follow_rate",
                    "Coin",
                    "#1f77b4",
                ),
                (
                    "crate_objective_follow_rate",
                    "Crate",
                    "#ff7f0e",
                ),
                (
                    "escape_objective_follow_rate",
                    "Escape",
                    "#2ca02c",
                ),
            ),
            title="Objective-mask adherence",
            empty_message="Objective diagnostics not recorded",
        )
        plot_evaluation_metric_panel(
            axis=axes[1],
            evaluation_by_seed=model_evaluations,
            aggregate_evaluation=aggregate_evaluation,
            positions=positions,
            seed_labels=model_labels,
            metric_name="safe_productive_bomb_follow_rate",
            title="Safe productive BOMB response",
            y_label="BOMB selection rate at opportunity (%)",
            scale=100.0,
            color="#d62728",
            empty_message="BOMB diagnostics not recorded",
        )
        axes[1].set_ylim(0.0, 100.0)
        policy_axes = list(axes)
    else:
        figure = plt.figure(
            figsize=(figure_width, 9.5),
        )
        grid = figure.add_gridspec(
            2,
            6,
            height_ratios=(1.05, 1.0),
        )
        safety_axis = figure.add_subplot(grid[0, 0:3])
        priority_axis = figure.add_subplot(grid[0, 3:6])
        target_axis = figure.add_subplot(grid[1, 0:2])
        bomb_axis = figure.add_subplot(grid[1, 2:4])
        self_kill_axis = figure.add_subplot(grid[1, 4:6])
        policy_axes = [
            safety_axis,
            priority_axis,
            target_axis,
            bomb_axis,
            self_kill_axis,
        ]

        plot_grouped_evaluation_rates(
            axis=safety_axis,
            evaluations=model_evaluations,
            positions=positions,
            labels=model_labels,
            metric_specs=(
                (
                    "threat_urgency_0_follow_rate",
                    "u=0 (no threat)",
                    "#bdbdbd",
                ),
                (
                    "threat_urgency_1_follow_rate",
                    "u=1 (≥3 steps)",
                    "#8dd3c7",
                ),
                (
                    "threat_urgency_2_follow_rate",
                    "u=2 (2 steps)",
                    "#2ca02c",
                ),
                (
                    "threat_urgency_3_follow_rate",
                    "u=3 (next step)",
                    "#006d2c",
                ),
            ),
            title="Movement/WAIT safety adherence by urgency",
            empty_message="Safety diagnostics not recorded",
            y_label="Safe-action follow rate (%)",
        )
        if independent_targets:
            plot_independent_target_conflict_heatmap(
                axis=priority_axis,
                analyzed_runs=analyzed_runs,
            )
        else:
            plot_grouped_evaluation_rates(
                axis=priority_axis,
                evaluations=model_evaluations,
                positions=positions,
                labels=model_labels,
                metric_specs=(
                    (
                        "target_conflict_coin_strong_follow_rate",
                        "Coin ≥2 closer",
                        "#08519c",
                    ),
                    (
                        "target_conflict_coin_weak_follow_rate",
                        "Coin 1 closer",
                        "#6baed6",
                    ),
                    (
                        "target_conflict_bomb_weak_follow_rate",
                        "Bomb 1 closer",
                        "#fdae6b",
                    ),
                    (
                        "target_conflict_bomb_strong_follow_rate",
                        "Bomb ≥2 closer",
                        "#e6550d",
                    ),
                ),
                title="Target priority in true conflicts",
                empty_message=(
                    "Target-conflict diagnostics not recorded"
                ),
                y_label="Prioritized safe-target follow rate (%)",
            )
        plot_grouped_evaluation_rates(
            axis=target_axis,
            evaluations=model_evaluations,
            positions=positions,
            labels=model_labels,
            metric_specs=(
                (
                    "safe_joint_target_follow_rate",
                    "Joint coin + bomb",
                    "#756bb1",
                ),
                (
                    "coin_route_bomb_joint_conversion_follow_rate",
                    "New joint via alternative",
                    "#31a354",
                ),
                (
                    "safe_coin_only_target_follow_rate",
                    "Coin only",
                    "#3182bd",
                ),
                (
                    "safe_bomb_only_target_follow_rate",
                    "Bomb only",
                    "#fd8d3c",
                ),
            ),
            title="Safe-target utilization",
            empty_message="Safe-target diagnostics not recorded",
            y_label="Available safe-target follow rate (%)",
        )
        plot_grouped_evaluation_rates(
            axis=bomb_axis,
            evaluations=model_evaluations,
            positions=positions,
            labels=model_labels,
            metric_specs=(
                (
                    "bomb_target_precision_follow_rate",
                    "Bombs at either BFS target",
                    "#d62728",
                ),
                (
                    "bomb_target_response_follow_rate",
                    "Primary BFS target used",
                    "#9467bd",
                ),
                (
                    "coin_route_bomb_alternative_target_follow_rate",
                    "Alternative BFS target used",
                    "#17becf",
                ),
            ),
            title="BFS bomb-target use",
            empty_message="BFS bomb-target diagnostics not recorded",
            y_label="Rate (%)",
        )
        plot_self_kill_diagnosis(
            axis=self_kill_axis,
            evaluations=model_evaluations,
            positions=positions,
            labels=model_labels,
        )

    for axis in policy_axes:
        if independent_targets and axis is priority_axis:
            continue

        axis.grid(
            True,
            axis="y",
            linewidth=0.6,
            alpha=0.3,
        )

    if len(model_labels) > 10:
        for axis in policy_axes:
            if independent_targets and axis is priority_axis:
                continue

            plt.setp(
                axis.get_xticklabels(),
                rotation=45,
                ha="right",
                rotation_mode="anchor",
            )
            axis.tick_params(axis="x", labelsize=8)

    evaluation_counts = {
        len(run["evaluations"])
        for run in analyzed_runs
    }
    if len(evaluation_counts) == 1:
        evaluation_count = next(iter(evaluation_counts))
        evaluation_description = (
            f"{evaluation_count} evaluation-seed "
            f"pair{'s' if evaluation_count != 1 else ''}"
        )
    else:
        evaluation_description = "the available evaluation-seed pairs"

    figure.suptitle(
        f"Policy diagnostics by trained model: {experiment_name}",
        fontsize=14,
    )
    heatmap_description = (
        " The heatmap pools all true-conflict decisions; color is "
        "the coin share conditional on following either target, "
        "while cell text reports unconditional target use and n."
        if independent_targets
        else ""
    )
    figure.text(
        0.5,
        0.01,
        (
            f"Bars show means across {evaluation_description}; "
            "error bars show ±1 population SD across evaluation "
            "seeds. Target-navigation categories are safety-filtered "
            "and mutually exclusive; local BOMB opportunities are "
            "measured separately. Self-kill segments are percentages "
            "of all rounds."
            + heatmap_description
        ),
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(
        rect=(0.0, 0.05, 1.0, 0.95),
    )

    return figure


def plot_reward_components(
    summary: dict,
    summary_file: Path,
) -> plt.Figure:
    """Plot reward contributions and event frequencies."""
    reward_components = summary["aggregate"][
        "reward_components"
    ]

    if not reward_components:
        raise ValueError(
            "Analysis summary contains no reward components."
        )

    event_items = sorted(
        (
            item
            for item in reward_components.items()
            if item[0] != "BOMB_EXPLODED"
        ),
        key=lambda item: (
            item[1]["mean_reward_per_round"]["mean"]
            if item[1]["mean_reward_per_round"]["mean"]
            is not None
            else 0.0
        ),
    )

    event_names = [
        event_name
        for event_name, _ in event_items
    ]
    positions = np.arange(len(event_names))

    reward_means = np.asarray(
        [
            0.0
            if component[
                "mean_reward_per_round"
            ]["mean"] is None
            else float(
                component[
                    "mean_reward_per_round"
                ]["mean"]
            )
            for _, component in event_items
        ],
        dtype=float,
    )
    reward_standard_deviations = np.asarray(
        [
            0.0
            if component[
                "mean_reward_per_round"
            ]["standard_deviation"] is None
            else float(
                component[
                    "mean_reward_per_round"
                ]["standard_deviation"]
            )
            for _, component in event_items
        ],
        dtype=float,
    )

    count_means = np.asarray(
        [
            0.0
            if component[
                "mean_count_per_round"
            ]["mean"] is None
            else float(
                component[
                    "mean_count_per_round"
                ]["mean"]
            )
            for _, component in event_items
        ],
        dtype=float,
    )
    count_standard_deviations = np.asarray(
        [
            0.0
            if component[
                "mean_count_per_round"
            ]["standard_deviation"] is None
            else float(
                component[
                    "mean_count_per_round"
                ]["standard_deviation"]
            )
            for _, component in event_items
        ],
        dtype=float,
    )

    reward_colors = [
        (
            "#2ca02c"
            if value > 0.0
            else "#d62728"
            if value < 0.0
            else "#7f7f7f"
        )
        for value in reward_means
    ]

    # Prevent frequency error bars from extending below zero.
    count_lower_errors = np.minimum(
        count_standard_deviations,
        count_means,
    )
    count_errors = np.vstack(
        (
            count_lower_errors,
            count_standard_deviations,
        )
    )

    figure_height = max(
        6.0,
        0.55 * len(event_names) + 2.5,
    )
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(13, figure_height),
        sharey=True,
    )

    reward_axis = axes[0]
    reward_axis.barh(
        positions,
        reward_means,
        xerr=reward_standard_deviations,
        color=reward_colors,
        alpha=0.82,
        capsize=4,
        error_kw={
            "elinewidth": 1.2,
            "capthick": 1.2,
        },
    )
    reward_axis.axvline(
        0.0,
        color="black",
        linewidth=1.0,
    )
    reward_axis.set_title("Reward contribution")
    reward_axis.set_xlabel(
        "Mean reward contribution per training round"
    )
    reward_axis.set_yticks(positions)
    reward_axis.set_yticklabels(event_names)
    reward_axis.grid(
        True,
        axis="x",
        linewidth=0.6,
        alpha=0.3,
    )

    count_axis = axes[1]
    count_axis.barh(
        positions,
        count_means,
        xerr=count_errors,
        color="#1f77b4",
        alpha=0.82,
        capsize=4,
        error_kw={
            "elinewidth": 1.2,
            "capthick": 1.2,
        },
    )
    count_axis.set_title("Event frequency")
    count_axis.set_xlabel(
        "Mean occurrences per training round"
    )
    count_axis.set_xlim(left=0.0)
    count_axis.grid(
        True,
        axis="x",
        linewidth=0.6,
        alpha=0.3,
    )

    configuration = summary["experiment"]["configuration"]
    experiment_name = configuration.get(
        "experiment_name",
        summary_file.parent.parent.name,
    )
    training_runs = summary["experiment"].get("evaluated_models", summary["experiment"]["training_runs"])

    figure.suptitle(
        f"Reward decomposition: {experiment_name}",
        fontsize=14,
    )
    figure.text(
        0.5,
        0.01,
        (
            f"Means across {training_runs} independently "
            "trained models; error bars show ±1 sample SD. "
            "Green and red denote positive and negative "
            "reward contributions."
        ),
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(
        rect=(0.0, 0.05, 1.0, 0.95),
    )

    return figure


def write_plot_document(
    summary: dict,
    summary_file: Path,
) -> Path:
    """Write all experiment figures to one multi-page PDF."""
    plot_directory = summary_file.parent / "plots"
    pdf_file = plot_directory / "experiment_plots.pdf"
    temporary_file = plot_directory / "experiment_plots.pdf.tmp"

    plot_functions = (
        plot_training_overview,
        plot_training_diagnostics,
        plot_evaluation_overview,
        plot_evaluation_by_model,
        plot_policy_diagnostics_by_model,
        plot_evaluation_diagnostics,
        plot_reward_components,
    )

    if summary["experiment"].get("experiment_mode") == "evaluate_checkpoint":
        plot_functions = (plot_evaluation_overview, plot_evaluation_by_model,
                          plot_policy_diagnostics_by_model, plot_evaluation_diagnostics)

    legacy_plot_files = (
        "training_overview.png",
        "training_overview.pdf",
        "training_diagnostics.png",
        "training_diagnostics.pdf",
        "evaluation_overview.png",
        "evaluation_overview.pdf",
        "evaluation_diagnostics.png",
        "evaluation_diagnostics.pdf",
        "reward_components.png",
        "reward_components.pdf",
    )

    try:
        plot_directory.mkdir(exist_ok=True)

        with PdfPages(temporary_file) as pdf:
            for plot_function in plot_functions:
                figure = plot_function(
                    summary,
                    summary_file,
                )

                try:
                    pdf.savefig(
                        figure,
                        bbox_inches="tight",
                    )
                finally:
                    plt.close(figure)

        temporary_file.replace(pdf_file)

        for filename in legacy_plot_files:
            (plot_directory / filename).unlink(
                missing_ok=True
            )
    except OSError as error:
        raise ValueError(
            f"Could not write plot document: {error}"
        ) from error
    finally:
        try:
            temporary_file.unlink(missing_ok=True)
        except OSError:
            pass

    return pdf_file


def main() -> int:
    """Create all experiment plots."""
    args = parse_arguments()

    try:
        summary, summary_file = load_analysis_summary(
            args.experiment_file
        )
        pdf_file = write_plot_document(
            summary,
            summary_file,
        )
    except (KeyError, TypeError, ValueError) as error:
        print(f"Error: {error}")
        return 1

    print("Experiment plot document created successfully.")
    print(f"PDF: {pdf_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
