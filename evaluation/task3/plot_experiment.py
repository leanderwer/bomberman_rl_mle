"""Create plots from an analyzed Task 3 experiment."""

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

PRESSURE_FEATURE_VERSIONS = (
    "task3_parallel_masks_pressure_bomb_v7", "task3_parallel_masks_robust_safety_v8",
    "task3_parallel_masks_tactical_bombs_v9",
    "task3_parallel_masks_tactical_bombs_v10",
    "task3_parallel_masks_tactical_bombs_v11_1",
    "task3_parallel_masks_tactical_bombs_v11_2",
    "task3_parallel_masks_tactical_bombs_v11_3",
    "task3_parallel_masks_tactical_bombs_v12",
    "task3_report_v12_no_hold_pooling_v1",
    "task3_report_v12_nominal_opponent_safety_v1",
    "task3_report_v12_no_coin_route_targets_v1",
    "task3_report_v12_no_block_targets_v1",
)
from matplotlib.ticker import MaxNLocator
import numpy as np

TRAIN_AND_EVALUATE_MODE = "train_and_evaluate"
EVALUATE_CHECKPOINT_MODE = "evaluate_checkpoint"


def _diagnostics(summary):
    # Finished summaries from the previous patch remain directly plottable.
    return summary.get("evaluation_diagnostics", summary.get("replay_diagnostics", {}))


PRESSURE_COLORS = plt.get_cmap("viridis")(np.linspace(.12, .88, 10))


def plot_self_kill_history(axis, histories, positions, labels):
    specs = (("unsafe_despite_safe", "0 despite an available 1", "#d95f02"),
             ("lost_safety_despite_following", "Safety lost while following 1", "#7570b3"),
             ("other", "Other self-kills", "#888888"))
    if not histories:
        axis.text(.5, .5, "Escape history not recorded during evaluation.",
                  ha="center", va="center", transform=axis.transAxes, fontsize=9)
        axis.set_ylim(0, 100)
    else:
        bottoms = np.zeros(len(histories))
        for key, label, color in specs:
            totals = [h["self_kills"]["total"] for h in histories]
            counts = [h["self_kills"][key] for h in histories]
            heights = np.array([100 * n / total if total else 0 for n, total in zip(counts, totals)])
            axis.bar(positions, heights, bottom=bottoms, color=color, label=label, width=.65)
            for x, bottom, height, count in zip(positions, bottoms, heights, counts):
                if count:
                    axis.text(x, bottom + height / 2, str(count), ha="center", va="center",
                              color="white", fontsize=9, fontweight="bold")
            bottoms += heights
        for x, total in zip(positions, totals):
            axis.text(x, 104, f"N={total}", ha="center", fontsize=9)
        axis.set_ylim(0, 112)
        axis.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(.5, -.17))
    axis.set_title("Self-kills: complete escape history")
    axis.set_ylabel("Share of self-kills (%)")
    axis.set_xticks(positions, labels)


def _pressure(summary):
    return _diagnostics(summary).get("pressure", {})


def plot_pressure_chain_histogram(summary, summary_file):
    figure, axis = plt.subplots(figsize=(12, 6))
    bins = _pressure(summary).get("bins", [])
    for i, group in enumerate(bins):
        x = (group["lower"] + group["upper"]) / 2
        count = group["count"]
        height = 100 * group["success_rate"] if count else 0
        axis.bar(x, height, width=.09, color=PRESSURE_COLORS[i], edgecolor="white", linewidth=.8)
        axis.text(x, height / 2 if height > 0 else 1.5, f"N={count}",
                  ha="center", va="center" if height > 0 else "bottom",
                  color="white" if height > 0 and i < 6 else "#333333", fontsize=9,
                  fontweight="bold" if count else "normal", zorder=4,
                  bbox={"facecolor": "white", "edgecolor": "none", "pad": 1}
                  if height == 0 else None)
    threshold = _diagnostics(summary).get("pressure_threshold")
    if threshold is not None:
        axis.axvline(threshold, color="#b22222", linestyle="--", linewidth=1.2, zorder=0,
                     label=f"Target-score threshold {threshold:g}")
    axis.set(xlim=(0, 1), ylim=(0, 105), xlabel="Pressure target score at chain start (bin width 0.1)",
             ylabel="Successful chains (%)", title="Pressure chains: success by starting score")
    axis.set_xticks(np.linspace(0, 1, 11))
    axis.grid(axis="y", alpha=.25)
    axis.set_axisbelow(True)
    if threshold is not None:
        axis.legend(frameon=False, loc="upper left")
    n = _pressure(summary).get("eligible_chains", 0)
    diagnostic = _diagnostics(summary)
    missing = (diagnostic.get("missing_diagnostic_rounds", diagnostic.get("missing_replay_rounds", 0))
               + diagnostic.get("incomplete_pressure_rounds", 0))
    coverage = f" Excluded rounds without complete pressure records: {missing}." if missing else ""
    figure.text(.5, .025,
                 f"N={n} chains with a valid crate-bomb alternative at START. Counts are inside bars "
                 "(above zero bars).\nSuccess: at least one kill by a bomb placed in that chain, "
                 "including after its end. Re-entry always starts a new chain." + coverage,
                 ha="center", fontsize=9)
    figure.tight_layout(rect=(0, .09, 1, 1))
    return figure


def pressure_chain_lengths(curve):
    """Recover exact chain lengths from N(t), including historical summaries."""
    counts = np.asarray(curve.get("count", []), dtype=int)
    if not len(counts):
        return np.array([], dtype=int)
    ended = counts - np.r_[counts[1:], 0]
    if np.any(ended < 0) or counts[0] != curve["chain_count"]:
        raise ValueError("Inconsistent pressure-chain survivor counts")
    return np.repeat(np.arange(1, len(counts) + 1), ended)


def plot_pressure_chain_profiles(summary, summary_file):
    figure = plt.figure(figsize=(17, 9.5))
    grid = figure.add_gridspec(2, 5)
    curves = _pressure(summary).get("curves", [])
    all_lengths = []
    for i in range(10):
        panel = grid[i // 5, i % 5].subgridspec(2, 1, height_ratios=(3.2, 1), hspace=.10)
        axis = figure.add_subplot(panel[0])
        sample_axis = figure.add_subplot(panel[1], sharex=axis)
        curve = curves[i] if i < len(curves) else None
        lower, upper = i / 10, (i + 1) / 10
        count = curve["chain_count"] if curve else 0
        title = f"[{lower:.1f}, {upper:.1f}{']' if i == 9 else ')'}  |  N(start)={count}"
        if count:
            lengths = pressure_chain_lengths(curve)
            all_lengths.extend(lengths)
            title += (f"\nLength: median {np.median(lengths):g}, "
                      f"P95 {np.percentile(lengths, 95):g}, max {max(lengths)}")
            x = np.arange(len(curve["mean"]))
            mean = np.asarray(curve["mean"])
            sd = np.array([v if v is not None else np.nan for v in curve["sd"]])
            axis.plot(x, mean, color=PRESSURE_COLORS[i], linewidth=1.8,
                      marker="o" if len(x) == 1 else None)
            axis.fill_between(x, np.maximum(0, mean - sd), np.minimum(1, mean + sd),
                              color=PRESSURE_COLORS[i], alpha=.23, linewidth=0)
            valid = np.isfinite(sd)
            if valid.sum() == 1:
                axis.errorbar(x[valid], mean[valid],
                              yerr=[np.minimum(sd[valid], mean[valid]),
                                    np.minimum(sd[valid], 1 - mean[valid])],
                              color=PRESSURE_COLORS[i], capsize=3, fmt="none")
            samples = np.asarray(curve["count"])
            sample_axis.step(x, samples, where="post", color="#444444", linewidth=1.3)
            sample_axis.set_yscale("log")
            sample_axis.set_ylim(.8, max(1.5, count * 1.25))
            sample_axis.set_yticks(sorted({1, count}), [str(n) for n in sorted({1, count})])
            sample_axis.minorticks_off()
            single = np.flatnonzero(samples == 1)
            if count > 1 and len(single):
                axis.axvspan(single[0], max(1, len(x) - 1), color=".5", alpha=.12)
                axis.text(.98, .96, "Shaded tail: N=1", transform=axis.transAxes,
                          ha="right", va="top", fontsize=8, color=".35")
            axis.set_xlim(0, max(1, len(x) - 1))
        else:
            axis.text(.5, .5, "No chains", transform=axis.transAxes, ha="center", color=".5")
            axis.set_xlim(0, 1)
            sample_axis.set_yticks([])
        axis.set_title(title, fontsize=9)
        axis.set_ylim(0, 1)
        axis.grid(alpha=.2)
        axis.tick_params(labelsize=8, labelbottom=False)
        sample_axis.tick_params(labelsize=8)
        sample_axis.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
        sample_axis.set_ylabel("N(t)", fontsize=8)
        sample_axis.grid(axis="x", alpha=.2)
        if i >= 5:
            sample_axis.set_xlabel("Steps since chain start", fontsize=9)
        if i % 5 == 0:
            axis.set_ylabel("Current pressure target score", fontsize=9)
    figure.suptitle("Pressure score development by starting-score interval", fontsize=15)
    length_note = (f"All {len(all_lengths)} chains: median {np.median(all_lengths):g}, "
                   f"P95 {np.percentile(all_lengths, 95):g}, max {max(all_lengths)} steps; "
                   f"{sum(n > 100 for n in all_lengths)} chains >100 steps.\n") if all_lengths else ""
    figure.text(.5, .02,
                 length_note + "Colored line: mean score; band: +/-1 sample SD across surviving chains; no SD for N=1. "
                 "Gray tail: only one chain remains.\n"
                 "N(t): contributing chains (log scale). Each interval has its own time axis. "
                 "Ended chains are not padded; valid crate target required only at START.",
                 ha="center", fontsize=9)
    figure.tight_layout(rect=(0, .10, 1, .94), h_pad=2.5)
    return figure


def experiment_mode(summary: dict) -> str:
    """Return the experiment mode with legacy fallback."""
    experiment = summary["experiment"]
    return experiment.get(
        "experiment_mode",
        experiment["configuration"].get(
            "experiment_mode",
            TRAIN_AND_EVALUATE_MODE,
        ),
    )


def evaluated_model_count(summary: dict) -> int:
    """Return the number of source models used for evaluation."""
    experiment = summary["experiment"]
    return experiment.get(
        "evaluated_models",
        experiment["training_runs"],
    )


def evaluated_model_description(summary: dict) -> str:
    """Return a grammatically correct source-model description."""
    count = evaluated_model_count(summary)
    noun = "model" if count == 1 else "models"
    return f"{count} evaluated source {noun}"


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

    if summary.get("schema_version") not in {3, 4, 5, 6, 7, 8}:
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
    training_runs = summary["experiment"]["training_runs"]

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


def plot_combat_training(
    summary: dict,
    summary_file: Path,
) -> plt.Figure:
    """Plot the compact Task 3 combat-learning diagnostics."""
    training_curves = summary["aggregate"]["training_curves"]
    rounds = training_round_series(training_curves)
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(13, 9),
        sharex=True,
    )
    panels = (
        (
            "kills_moving_average",
            "Kills during training",
            "Kills per round",
            1.0,
            "#d62728",
            None,
        ),
        (
            "all_opponents_eliminated_moving_average",
            "All opponents eliminated",
            "Round rate (%)",
            100.0,
            "#2ca02c",
            100.0,
        ),
        (
            "death_by_opponent_moving_average",
            "Deaths caused by opponents",
            "Round rate (%)",
            100.0,
            "#9467bd",
            100.0,
        ),
        (
            "kill_auc_moving_average",
            "Kill timing quality",
            "Normalized kill AUC (%)",
            100.0,
            "#ff7f0e",
            100.0,
        ),
    )

    for axis, (
        metric_name,
        title,
        y_label,
        scale,
        color,
        upper_bound,
    ) in zip(axes.flat, panels):
        means, standard_deviations = metric_series(
            training_curves,
            metric_name,
            scale=scale,
        )
        plot_mean_and_sd(
            axis=axis,
            rounds=rounds,
            means=means,
            standard_deviations=standard_deviations,
            color=color,
            label="Moving average",
            lower_bound=0.0,
            upper_bound=upper_bound,
        )
        axis.set_title(title)
        axis.set_xlabel("Training round")
        axis.set_ylabel(y_label)
        axis.set_ylim(bottom=0.0)
        if upper_bound is not None:
            axis.set_ylim(0.0, upper_bound)
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.grid(True, linewidth=0.6, alpha=0.3)
        axis.legend(frameon=False)

    configuration = summary["experiment"]["configuration"]
    experiment_name = configuration.get(
        "experiment_name",
        summary_file.parent.parent.name,
    )
    opponents = ", ".join(configuration.get("training_opponents", configuration["opponents"]))
    opponent_label = (
        "Training opponents (including the other learner)"
        if "training_roster" in configuration else "Fixed opponents"
    )
    training_runs = summary["experiment"]["training_runs"]

    figure.suptitle(
        f"Combat learning: {experiment_name}",
        fontsize=14,
    )
    figure.text(
        0.5,
        0.01,
        (
            f"{opponent_label}: {opponents}. Mean across "
            f"{training_runs} independently trained models; shaded "
            "areas show ±1 sample SD where available."
        ),
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0.0, 0.04, 1.0, 0.95))

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
    training_runs = summary["experiment"]["training_runs"]

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


def plot_combat_evaluation(
    summary: dict,
    summary_file: Path,
) -> plt.Figure:
    """Plot Task 3 combat quality for each evaluation seed."""
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
    positions = np.arange(len(evaluation_by_seed))
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(16, 9),
        sharex=True,
    )
    panels = (
        (
            "mean_kills_per_round",
            "Kills",
            "Kills per round",
            1.0,
            "#d62728",
        ),
        (
            "rounds_with_kill_rate",
            "Rounds with a kill",
            "Rate (%)",
            100.0,
            "#e377c2",
        ),
        (
            "all_opponents_eliminated_rate",
            "All opponents eliminated",
            "Rate (%)",
            100.0,
            "#2ca02c",
        ),
        (
            "score_win_rate",
            "Wins (game score)",
            "Rounds won (%)",
            100.0,
            "#17becf",
        ),
        (
            "death_by_opponent_rate",
            "Deaths caused by opponents",
            "Rate (%)",
            100.0,
            "#9467bd",
        ),
        (
            "mean_kill_auc",
            "Kill timing quality",
            "Normalized kill AUC (%)",
            100.0,
            "#ff7f0e",
        ),
    )

    for axis, (
        metric_name,
        title,
        y_label,
        scale,
        color,
    ) in zip(axes.flat, panels):
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
            empty_message=(
                "Score wins unavailable:\nrerun analyze_experiment.py"
                if metric_name == "score_win_rate" else "No combat observations"
            ),
            zero_message="No observed events",
        )
        axis.grid(True, axis="y", linewidth=0.6, alpha=0.3)
        if scale == 100.0:
            axis.set_ylim(0.0, 100.0)

    configuration = summary["experiment"]["configuration"]
    experiment_name = configuration.get(
        "experiment_name",
        summary_file.parent.parent.name,
    )
    opponents = ", ".join(configuration["opponents"])
    evaluated_models = evaluated_model_description(summary)

    figure.suptitle(
        f"Combat evaluation: {experiment_name}",
        fontsize=14,
    )
    figure.text(
        0.5,
        0.01,
        (
            f"Fixed opponents: {opponents}. Bars separate held-out "
            f"world/agent seeds; values aggregate {evaluated_models}."
            "\nWin: final score higher than every opponent, regardless of survival. "
            "Tied highest scores are draws."
        ),
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0.0, 0.06, 1.0, 0.95))

    return figure


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
    evaluated_models = evaluated_model_description(summary)

    figure.suptitle(
        f"Evaluation diagnostics: {experiment_name}",
        fontsize=14,
    )
    figure.text(
        0.5,
        0.01,
        (
            f"Bars show means across {evaluated_models} for each "
            "evaluation-seed pair; error bars "
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
    evaluated_models = evaluated_model_description(summary)

    figure.suptitle(
        f"Evaluation overview: {experiment_name}",
        fontsize=14,
    )
    figure.text(
        0.5,
        0.01,
        (
            f"Bars show means across {evaluated_models} for each "
            "evaluation-seed pair; error bars "
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
    """Create evaluation plots grouped by evaluated source model."""
    analyzed_runs = sorted(
        summary["runs"],
        key=lambda run: (
            run["training_world_seed"],
            run["training_agent_seed"],
        ),
    )

    if not analyzed_runs:
        raise ValueError(
            "Analysis summary contains no evaluated models."
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
        f"Evaluation by source model: {experiment_name}",
        fontsize=14,
    )
    figure.text(
        0.5,
        0.01,
        (
            f"Bars show means across {evaluation_description} "
            "for each evaluated source model; error bars "
            "show ±1 population SD across evaluation seeds. "
            "Dashed lines in single-metric panels show the "
            "overall mean across evaluated models."
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
    histories=None,
) -> None:
    """Replace the old final-action classification with complete escape histories."""
    plot_self_kill_history(axis, histories, positions, labels)


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
    """Create policy diagnostics grouped by evaluated source model."""
    analyzed_runs = sorted(
        summary["runs"],
        key=lambda run: (
            run["training_world_seed"],
            run["training_agent_seed"],
        ),
    )

    if not analyzed_runs:
        raise ValueError(
            "Analysis summary contains no evaluated models."
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
        in ("task2_parallel_masks_coin_route_bomb_v6", *PRESSURE_FEATURE_VERSIONS)
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
        diagnostic_models = _diagnostics(summary).get("models", [])
        history_by_model = {(r["training_world_seed"], r["training_agent_seed"]): r
                            for r in diagnostic_models}
        histories = [history_by_model.get((r["training_world_seed"], r["training_agent_seed"]))
                     for r in analyzed_runs]
        plot_self_kill_diagnosis(
            axis=self_kill_axis,
            evaluations=model_evaluations,
            positions=positions,
            labels=model_labels,
            histories=histories if all(h is not None for h in histories) else None,
        )

    if feature_version in PRESSURE_FEATURE_VERSIONS:
        priority_axis.set_title("Coin / crate conflicts (pressure targets excluded)")

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
        f"Policy diagnostics by source model: {experiment_name}",
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
            f"Other bars show means across {evaluation_description}; "
            "error bars show ±1 population SD across evaluation "
            "seeds.\nTarget-navigation categories are safety-filtered "
            "and mutually exclusive; local BOMB opportunities are "
            "measured separately. Self-kill segments are percentages "
            "of self-kills; text gives counts.\n"
            + heatmap_description.strip()
        ),
        ha="center",
        fontsize=8,
    )
    figure.tight_layout(
        rect=(0.0, 0.09, 1.0, 0.95),
    )

    return figure


def plot_reward_components(
    summary: dict,
    summary_file: Path,
) -> plt.Figure:
    """Plot reward contributions and event frequencies."""
    reward_components = dict(summary["aggregate"][
        "reward_components"
    ])
    reward_version = summary["experiment"]["configuration"].get("reward_version")
    if reward_version in ("task3_hierarchical_tactical_hold_v3_10",
                          "task3_hierarchical_tactical_v4", "task3_hierarchical_tactical_v5"):
        # An absent event in the generic reward aggregate has zero occurrences.
        # Display that explicitly, including when replotting an older V9 run.
        registered = ("WAIT_AT_BLOCK_TARGET", "SKIPPED_BLOCK_TARGET")
        if reward_version in ("task3_hierarchical_tactical_v4", "task3_hierarchical_tactical_v5"):
            registered += ("KILLED_BY_OPPONENT",)
        for name in registered:
            reward_components.setdefault(name, {
                "mean_reward_per_round": {"mean": 0.0, "standard_deviation": None},
                "mean_count_per_round": {"mean": 0.0, "standard_deviation": None},
            })

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
    training_runs = summary["experiment"]["training_runs"]

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


def plot_tactical_use(axis, runs, diagnostics, evaluations):
    """Global adherence plus the own/opponent WAIT split, with pooled counts."""
    models = {(m["training_world_seed"], m["training_agent_seed"]): m.get("tactical", {})
              for m in diagnostics.get("models", [])}
    if not any(m.get("schema_version", 0) >= 2 and m.get("complete_rounds") for m in models.values()):
        plot_grouped_evaluation_rates(
            axis, evaluations, np.arange(len(runs)),
            [f"w={r['training_world_seed']}\na={r['training_agent_seed']}" for r in runs],
            (("block_target_follow_rate", "All block targets", "#3182bd"),
             ("block_local_wait_follow_rate", "WAIT all", "#31a354")),
            "Tactical blockade use", "No measured blockade opportunities")
        axis.set_xlim(-.6, len(runs) - .4)
        axis.text(0, -.24, "Own/opponent WAIT split: rerun analysis with episode records.",
                  transform=axis.transAxes, va="top", fontsize=8)
        return
    positions, labels, notes = [], [], []
    for i, run in enumerate(runs):
        key = run["training_world_seed"], run["training_agent_seed"]
        model = models.get(key, {})
        scenarios = model.get("scenarios", {}) if model.get("schema_version", 0) >= 2 else {}
        for j, (label, categories, denominator, numerator, color) in enumerate((
                ("All targets\n(incl. HOLD)", tuple(scenarios), "target_steps", "target_followed", "#3182bd"),
                ("WAIT\nall", tuple(scenarios), "wait_steps", "wait_followed", "#31a354"),
                ("WAIT\nown bombs", ("own_preplanned", "own_later"), "wait_steps", "wait_followed", "#e6550d"),
                ("WAIT\nopponent", ("opponent",), "wait_steps", "wait_followed", "#756bb1"))):
            x = 5 * i + j
            positions.append(x)
            labels.append(("All\ntargets", "WAIT\nall", "WAIT\nown", "WAIT\nopponent")[j]
                          if len(runs) > 1 else label)
            n = sum(scenarios.get(c, {}).get(denominator, 0) for c in categories)
            used = sum(scenarios.get(c, {}).get(numerator, 0) for c in categories)
            if n:
                height = 100 * used / n
                axis.bar(x, height, width=.65, color=color)
                axis.text(x, height + 3, f"{height:.1f}%\n{used}/{n}", ha="center",
                          fontsize=7 if len(runs) > 1 else 8)
            else:
                axis.text(x, 3, "N=0" if scenarios else "n/a", ha="center", fontsize=8, color=".45")
        if len(runs) > 1:
            axis.text(5 * i + 1.5, -.17, f"w={key[0]}, a={key[1]}",
                      transform=axis.get_xaxis_transform(), ha="center", fontsize=8)
        prefix = f"w={key[0]}, a={key[1]}: " if len(runs) > 1 else ""
        for label, categories in (("own", ("own_preplanned", "own_later")), ("opponent", ("opponent",))):
            records = [scenarios.get(c, {}).get("hold_execution", {}) for c in categories]
            observed = sum(r.get("observed_steps", 0) for r in records)
            total = sum(scenarios.get(c, {}).get("wait_steps", 0) for c in categories)
            if observed:
                stayed = sum(r.get("stayed_steps", 0) for r in records)
                non_wait = sum(r.get("non_wait_stayed_steps", 0) for r in records)
                invalid = sum(r.get("invalid_steps", 0) for r in records)
                detail = f"{stayed}/{observed} kept position; non-WAIT stays {non_wait}; invalid {invalid}"
                if observed < total:
                    detail += f"; coverage {observed}/{total}"
            else:
                detail = "not recorded" if total or not scenarios else "no HOLD opportunities"
            notes.append(f"{prefix}Physical HOLD {label}: {detail}")
    axis.set(title="Tactical blockade use", ylabel="Follow rate (%)", ylim=(0, 125),
             xticks=positions, xticklabels=labels, xlim=(-.6, max(positions, default=0) + .6))
    axis.set_yticks([0, 25, 50, 75, 100])
    axis.tick_params(axis="x", labelsize=8)
    axis.text(0, -.28 if len(runs) > 1 else -.24, "\n".join(notes),
              transform=axis.transAxes, va="top", fontsize=8)


def plot_tactical_opportunities(axis, runs, diagnostics):
    """Three scenario groups; HOLD remains a subset of all target steps."""
    categories = (("own_preplanned", "Own bomb\npreplanned"),
                  ("own_later", "Own bomb\nrecognized later"),
                  ("opponent", "Opponent\nbomb"))
    models = {(m["training_world_seed"], m["training_agent_seed"]): m.get("tactical", {})
              for m in diagnostics.get("models", [])}
    labels, positions, notes = [], [], []
    available = False
    maximum = 0.0
    for i, run in enumerate(runs):
        key = run["training_world_seed"], run["training_agent_seed"]
        model = models.get(key, {})
        usable = model.get("complete_rounds") and model.get("schema_version", 0) >= 2
        if not usable:
            notes.append(f"w={key[0]}, a={key[1]}: scenario attribution not recorded; re-evaluate")
            if model.get("complete_rounds"):
                notes[-1] = f"w={key[0]}, a={key[1]}: old attribution; rerun analyze_experiment.py"
        else:
            available = True
            combo = model["local_combos"]
            mixed = model["scenarios"]["mixed_unknown"]["target_steps"]
            notes.append(f"w={key[0]}, a={key[1]}: local certificate / target selected / bomb placed")
            notes.append(f"{combo['opportunities']} / {combo['selected_opportunities']} / {combo['placements']} steps. "
                         f"Unresolved: {mixed} block steps; complete rounds: "
                         f"{model['complete_rounds']}/{model['complete_rounds'] + model['incomplete_rounds'] + model['missing_rounds']}")
            if model.get("legacy_unresolved_rounds"):
                notes.append(f"Legacy origins unresolved in {model['legacy_unresolved_rounds']} rounds; re-evaluate.")
        for j, (category, label) in enumerate(categories):
            x = 4 * i + j
            positions.append(x)
            labels.append(("Own\npreplanned", "Own\nlater", "Opponent\nbomb")[j]
                          if len(runs) > 1 else label)
            if not usable:
                continue
            for offset, name, bar_label, color in (
                    (-.17, "target_steps_per_round", "All block target steps", "#3182bd"),
                    (.17, "wait_steps_per_round", "Safe local WAIT steps", "#31a354")):
                value = model["scenarios"][category][name]["mean"]
                maximum = max(maximum, value)
                bars = axis.bar(x + offset, value, width=.31, color=color,
                                label=bar_label if i == 0 and j == 0 else None)
                axis.bar_label(bars, labels=[f"{value:.3f}"], padding=3, fontsize=8,
                               rotation=90 if len(runs) > 1 else 0)
        if len(runs) > 1:
            axis.text(4 * i + 1, -.17, f"w={key[0]}, a={key[1]}",
                      transform=axis.get_xaxis_transform(), ha="center", fontsize=8)
    axis.set(title="Tactical blockade opportunities", ylabel="Decision steps per round",
             xticks=positions, xticklabels=labels, xlim=(-.6, max(positions, default=0) + .6))
    axis.tick_params(axis="x", labelsize=8)
    axis.set_ylim(0, max(maximum * 1.55, .1))
    if available:
        # Explicit handles also work when the first model has legacy data.
        from matplotlib.patches import Patch
        axis.legend(handles=[Patch(color="#3182bd", label="All block target steps"),
                             Patch(color="#31a354", label="Safe local WAIT steps")],
                    frameon=False, fontsize=8, loc="upper right")
    axis.text(.0, -.28 if len(runs) > 1 else -.24, "\n".join(notes),
              transform=axis.transAxes, va="top", fontsize=8)


def plot_bomb_target_round_shares(axis, runs, diagnostics):
    """Mean of each round's target_steps / world_steps, never renormalized."""
    models = {(m["training_world_seed"], m["training_agent_seed"]): m.get("tactical", {})
              for m in diagnostics.get("models", [])}
    positions = np.arange(len(runs))
    labels = [f"w={r['training_world_seed']}\na={r['training_agent_seed']}" for r in runs]
    for i, run in enumerate(runs):
        model = models.get((run["training_world_seed"], run["training_agent_seed"]), {})
        if model.get("complete_rounds"):
            remainder = model["mode_shares"]["none"]["mean"] + model["inactive_share"]["mean"]
            labels[i] += f"\nRemainder: {100 * remainder:.2f}%"
    available = False
    for i, (mode, label, color) in enumerate((
            ("pressure", "Pressure", "#e6550d"), ("crates", "Crates", "#756bb1"),
            ("block", "Blockade", "#3182bd"))):
        for j, run in enumerate(runs):
            model = models.get((run["training_world_seed"], run["training_agent_seed"]), {})
            mean = model.get("mode_shares", {}).get(mode, {}).get("mean")
            if mean is None:
                continue
            available = True
            bars = axis.bar(j + (i - 1) * .24, mean * 100, width=.22, color=color)
            axis.bar_label(bars, labels=[f"{100 * mean:.2f}%"], padding=3, fontsize=9)
    axis.set(title="Bomb BFS target share of round", ylabel="Mean share of round steps (%)",
             ylim=(0, 105), xticks=positions, xticklabels=labels,
             xlim=(-.6, len(runs) - .4))
    if available:
        from matplotlib.patches import Patch
        axis.legend(handles=[Patch(color=c, label=l) for c, l in (
            ("#e6550d", "Pressure"), ("#756bb1", "Crates"), ("#3182bd", "Blockade"))],
            frameon=False, loc="upper right")
    else:
        axis.text(.5, .5, "Round target shares were not recorded; re-evaluate the checkpoint",
                  transform=axis.transAxes, ha="center", va="center", fontsize=10)


def plot_pressure_bomb_diagnostics(summary: dict, summary_file: Path) -> plt.Figure:
    """V7 target adherence and acceptance of the local blockade test."""
    runs = sorted(summary["runs"], key=lambda r: (r["training_world_seed"], r["training_agent_seed"]))
    labels = [f"w={r['training_world_seed']}\na={r['training_agent_seed']}" for r in runs]
    evaluations = [{"metrics": r["evaluation_summary"]} for r in runs]
    positions = np.arange(len(runs))
    feature_version = summary["experiment"]["configuration"].get("feature_version")
    tactical = feature_version in PRESSURE_FEATURE_VERSIONS[2:]
    if tactical:
        figure = plt.figure(figsize=(max(14, 7 * len(runs)), max(14, 11.5 + len(runs))))
        grid = figure.add_gridspec(3, 2, height_ratios=(1, 1.15, .85))
        axes = [figure.add_subplot(grid[i, j]) for i in range(2) for j in range(2)]
        axes.append(figure.add_subplot(grid[2, :]))
    else:
        figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    plot_grouped_evaluation_rates(
        axis=axes[0], evaluations=evaluations, positions=positions, labels=labels,
        metric_specs=(("pressure_target_follow_rate", "Pressure target followed", "#e6550d"),
                      ("pressure_local_bomb_follow_rate", "Local pressure bomb used", "#756bb1")),
        title="Pressure target use", empty_message="No pressure opportunities recorded",
        y_label="Rate (%)")
    plot_grouped_evaluation_rates(
        axis=axes[1], evaluations=evaluations, positions=positions, labels=labels,
        metric_specs=(("bomb_blockade_check_follow_rate", "Local checks passed", "#31a354"),),
        title=("Local bomb safety checks" if feature_version in PRESSURE_FEATURE_VERSIONS[3:]
               else "Body-blockade check at selected local sites"),
        empty_message="No local blockade checks recorded", y_label="Accepted placements (%)")
    if tactical:
        plot_tactical_use(axes[2], runs, _diagnostics(summary), evaluations)
        plot_tactical_opportunities(axes[3], runs, _diagnostics(summary))
        plot_bomb_target_round_shares(axes[4], runs, _diagnostics(summary))
    for i, axis in enumerate(axes):
        if i < 2:
            axis.set_xlim(-.6, len(runs) - .4)
        axis.grid(True, axis="y", linewidth=.6, alpha=.3)
    figure.suptitle("Pressure targets and own-bomb escape")
    note = "Checks assess a surviving strategy; the learned policy chooses all actions. Unplaced enemy bombs are outside this check."
    if tactical:
        note += ("\nShares average target steps / total round steps separately for each round, irrespective of the chosen action."
                 "\nNo target and steps after death remain outside the three shares. New diagnostic counts use complete evaluation rounds only.")
    figure.text(.5, .01, note, ha="center", fontsize=9)
    figure.tight_layout(rect=(0, .065 if tactical else .06, 1, .96), h_pad=3)
    return figure


def plot_pressure_sweep(summary: dict, summary_file: Path) -> plt.Figure:
    """One compact comparison page with the two requested outcomes and exact counts."""
    variants = summary["variants"]
    thresholds = sorted({row["pressure_threshold"] for row in variants})
    discounts = sorted({row["pressure_distance_discount"] for row in variants}, reverse=True)
    figure = plt.figure(figsize=(11.7, 8.3))
    grid = figure.add_gridspec(2, 2, height_ratios=(1, 1.25), hspace=.48, wspace=.25)
    axes = [figure.add_subplot(grid[0, column]) for column in range(2)]
    metrics = (("mean_kills_per_round", "Own kills per round", 1),
               ("score_win_rate", "Score wins (%)", 100))
    for axis, (metric, label, scale) in zip(axes, metrics):
        for discount in discounts:
            rows = sorted((row for row in variants if row["pressure_distance_discount"] == discount),
                          key=lambda row: row["pressure_threshold"])
            axis.plot([row["pressure_threshold"] for row in rows],
                      [scale * row[metric] for row in rows], marker="o", linewidth=1.7,
                      label=f"Distance discount {discount:.2f}")
        axis.set(xlabel="Pressure threshold", ylabel=label, title=label)
        axis.set_xticks(thresholds, [f"{threshold:.2f}" for threshold in thresholds])
        axis.set_ylim(bottom=0)
        if metric == "score_win_rate":
            axis.set_ylim(0, 100)
        axis.grid(alpha=.25)
        axis.legend(frameon=False, fontsize=9)

    table_axis = figure.add_subplot(grid[1, :])
    table_axis.axis("off")
    rows = sorted(variants, key=lambda row: (row["pressure_threshold"], -row["pressure_distance_discount"]))
    table = table_axis.table(
        cellText=[[f"{row['pressure_threshold']:.2f}", f"{row['pressure_distance_discount']:.2f}",
                   str(row["rounds"]), str(row["own_kills"]), f"{row['mean_kills_per_round']:.3f}",
                   f"{row['score_wins']}/{row['rounds']}", f"{100 * row['score_win_rate']:.1f}%"]
                  for row in rows],
        colLabels=["Threshold", "Discount", "Rounds", "Own kills", "Kills / round",
                   "Wins / rounds", "Win rate"],
        cellLoc="center", colLoc="center", loc="center", bbox=(0, 0, 1, 1))
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (row_index, column), cell in table.get_celld().items():
        cell.set_edgecolor("#dddddd")
        if row_index == 0:
            cell.set_facecolor("#e7edf3")
            cell.set_text_props(weight="bold")
        elif (rows[row_index - 1]["pressure_threshold"],
              rows[row_index - 1]["pressure_distance_discount"]) == (.25, .95):
            cell.set_facecolor("#eef5e8")
        elif row_index % 2:
            cell.set_facecolor("#f8f8f8")
    configuration = summary["experiment"]["configuration"]
    figure.suptitle("Pressure threshold and distance discount: frozen-policy evaluation", fontsize=14)
    checkpoint = summary["source"]["model_sha256"][:12]
    figure.text(.5, .93, f"Same checkpoint {checkpoint} | {len(configuration['opponents'])} opponents | "
                f"{sum(row['rounds'] for row in variants)} evaluated rounds", ha="center", fontsize=9)
    figure.text(.5, .025,
                "Own kills are engine-credited opponent kills. Score wins require strictly the highest final score; ties are draws.\n"
                "Rates pool all evaluated rounds. Green row: baseline 0.25 / 0.95. Pressure fallback without crate targets is unchanged.",
                ha="center", fontsize=8)
    figure.subplots_adjust(top=.86, bottom=.1)
    return figure


def plot_certified_blockade_opportunities(summary: dict, summary_file: Path) -> plt.Figure:
    """Separate available certificates, chosen actions, and engine execution."""
    models = _diagnostics(summary).get("models", [])
    figure, axes = plt.subplots(max(1, len(models)), 1,
                               figsize=(11.7, max(8.3, 2.8 * len(models) + 2)),
                               squeeze=False)
    labels = (("own_preplanned", "Own bomb\ncertified at placement"),
              ("own_later", "Own bomb\nrecognized later"),
              ("opponent", "Opponent bomb"), ("mixed_unknown", "Mixed / unknown"))

    def fraction(numerator, denominator, opportunities=0):
        if denominator:
            return f"{numerator}/{denominator}\n{100 * numerator / denominator:.1f}%"
        return "0/0\nnot observed" if opportunities else "0/0\nno opportunities"

    for axis, model in zip(axes[:, 0], models):
        axis.axis("off")
        certified = model.get("tactical", {}).get("certified_opportunities", {})
        usable = bool(certified.get("complete_rounds"))
        rows, notes = [], []
        for category, label in labels:
            counts = certified.get("scenarios", {}).get(category, {})
            if not usable:
                rows.append([label] + ["not recorded"] * 7)
                continue
            execution = counts["hold_execution"]
            rows.append([
                label, str(counts["approach_steps"]),
                fraction(counts["approach_followed"], counts["approach_steps"]),
                fraction(counts["approach_executed_steps"], counts["approach_observed_steps"], counts["approach_steps"]),
                str(counts["wait_steps"]), fraction(counts["wait_followed"], counts["wait_steps"]),
                fraction(execution["stayed_steps"], execution["observed_steps"], counts["wait_steps"]),
                fraction(counts["selected_target_steps"], counts["target_steps"]),
            ])
            notes.append(f"{label.replace(chr(10), ' ')}: non-WAIT stays {execution['non_wait_stayed_steps']}, "
                         f"invalid HOLD actions {execution['invalid_steps']}, "
                         f"observed HOLD {execution['observed_steps']}/{counts['wait_steps']}")
        table = axis.table(cellText=rows,
            colLabels=["Certificate source", "Approach\nopportunities", "Move chosen\n/ opportunity",
                       "Move executed\n/ observed", "HOLD\nopportunities", "WAIT chosen\n/ opportunity",
                       "Position kept\n/ observed", "Target selected\n/ opportunity"],
            colWidths=[.17, .10, .12, .12, .095, .12, .13, .145],
            cellLoc="center", loc="center", bbox=(0, .28, 1, .57))
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        for (row, column), cell in table.get_celld().items():
            cell.set_edgecolor("#dddddd")
            if row == 0:
                cell.set_facecolor("#e7edf3")
                cell.set_text_props(weight="bold")
        axis.set_title(f"w={model['training_world_seed']}, a={model['training_agent_seed']} | "
                       f"complete recorded rounds: {certified.get('complete_rounds', 0)} | "
                       f"status: {certified.get('status', 'unavailable')}", fontsize=10)
        axis.text(0, .19, "\n".join(notes), transform=axis.transAxes, fontsize=8, va="top")
    figure.suptitle("Certified blockade opportunities, independent of selected targets", fontsize=13)
    figure.text(.5, .045,
        "Opportunities are decision steps with a certified plan, including when block targets are disabled.\n"
        "Chosen actions and physical execution have separate denominators; invalid actions can keep the agent in place.\n"
        "Counts describe behaviour on each policy's trajectories; they do not establish that a blockade caused an elimination.",
        ha="center", fontsize=8)
    figure.subplots_adjust(top=.88, bottom=.14, left=.03, right=.97, hspace=.5)
    return figure


def has_certified_opportunity_records(summary):
    return any(model.get("tactical", {}).get("certified_opportunities", {}).get("status")
               in ("complete", "partial") for model in _diagnostics(summary).get("models", []))


def experiment_plot_functions(summary: dict) -> list:
    """Select plot pages appropriate for the experiment mode."""
    if summary.get("experiment_kind") == "pressure_sweep":
        return [plot_pressure_sweep]
    if experiment_mode(summary) == EVALUATE_CHECKPOINT_MODE:
        functions = [
            plot_evaluation_overview,
            plot_combat_evaluation,
            plot_evaluation_by_model,
            plot_policy_diagnostics_by_model,
            plot_evaluation_diagnostics,
        ]
        if summary["experiment"]["configuration"].get("feature_version") in PRESSURE_FEATURE_VERSIONS:
            functions.append(plot_pressure_bomb_diagnostics)
            if "pressure" in _diagnostics(summary):
                functions.extend((plot_pressure_chain_histogram, plot_pressure_chain_profiles))
        if has_certified_opportunity_records(summary):
            functions.append(plot_certified_blockade_opportunities)
        return functions

    plot_functions = [
        plot_training_overview,
        plot_training_diagnostics,
        plot_evaluation_overview,
        plot_evaluation_by_model,
        plot_policy_diagnostics_by_model,
        plot_evaluation_diagnostics,
    ]

    if summary.get("schema_version", 0) >= 7:
        plot_functions.insert(1, plot_combat_training)
        plot_functions.insert(4, plot_combat_evaluation)

    if summary["experiment"]["configuration"].get("feature_version") in PRESSURE_FEATURE_VERSIONS:
        plot_functions.append(plot_pressure_bomb_diagnostics)
        if "pressure" in _diagnostics(summary):
            plot_functions.extend((plot_pressure_chain_histogram, plot_pressure_chain_profiles))

    plot_functions.append(plot_reward_components)
    if has_certified_opportunity_records(summary):
        plot_functions.append(plot_certified_blockade_opportunities)
    return plot_functions


def write_plot_document(
    summary: dict,
    summary_file: Path,
) -> Path:
    """Write all applicable experiment figures to one PDF."""
    plot_directory = summary_file.parent / "plots"
    pdf_file = plot_directory / "experiment_plots.pdf"
    temporary_file = plot_directory / "experiment_plots.pdf.tmp"
    plot_functions = experiment_plot_functions(summary)

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
