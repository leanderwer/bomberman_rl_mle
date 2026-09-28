"""Shared exploration telemetry rules for all task analyses and plots."""

import math


def exploration_strategy(metadata: dict) -> str:
    """Resolve the recorded strategy, retaining epsilon-only legacy results."""
    strategies = set()
    declared = metadata.get("exploration_strategy")
    if declared is not None:
        if declared not in ("epsilon_greedy", "softmax"):
            raise ValueError(f"Unknown exploration strategy: {declared!r}.")
        strategies.add(declared)
    version = metadata.get("exploration_version") or ""
    if not isinstance(version, str):
        raise ValueError("Exploration version must be a string.")
    if "softmax" in version:
        strategies.add("softmax")
    elif "epsilon" in version:
        strategies.add("epsilon_greedy")
    parameters = metadata.get("exploration_parameters") or {}
    if not isinstance(parameters, dict):
        raise ValueError("Exploration parameters must be an object.")
    schedule = parameters.get("schedule")
    if schedule == "multiplicative_temperature_decay":
        strategies.add("softmax")
    elif schedule == "multiplicative_epsilon_decay":
        strategies.add("epsilon_greedy")
    if len(strategies) > 1:
        raise ValueError("Conflicting exploration strategy metadata.")
    return next(iter(strategies), "epsilon_greedy")


def validate_exploration_round(
    row: dict, strategy: str, previous_end, location: str,
) -> float:
    """Validate exactly one active, finite and continuous schedule."""
    field = "temperature" if strategy == "softmax" else "epsilon"
    inactive = "epsilon" if strategy == "softmax" else "temperature"
    if any(
        row.get(f"{inactive}_{suffix}") is not None
        for suffix in ("start", "end")
    ):
        raise ValueError(
            f"'{location}' contains incompatible {inactive} telemetry for {strategy}."
        )
    values = []
    for suffix in ("start", "end"):
        name = f"{field}_{suffix}"
        value = row.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"'{location}.{name}' must be a finite number.")
        if (
            value < 0
            or (field == "epsilon" and value > 1)
            or (field == "temperature" and value == 0)
        ):
            raise ValueError(f"Invalid {field} value in '{location}.{name}'.")
        values.append(float(value))
    if previous_end is not None and not math.isclose(
        values[0], previous_end, rel_tol=1e-12, abs_tol=1e-12,
    ):
        raise ValueError(f"{field.title()} is discontinuous before '{location}'.")
    return values[1]


def exploration_summary(metadata: dict, rounds: list[dict]) -> dict:
    """Keep legacy epsilon fields and report temperature without substitution."""
    return {
        "exploration_strategy": exploration_strategy(metadata),
        "initial_epsilon": rounds[0].get("epsilon_start"),
        "final_epsilon": rounds[-1].get("epsilon_end"),
        "initial_temperature": rounds[0].get("temperature_start"),
        "final_temperature": rounds[-1].get("temperature_end"),
    }


def training_strategy(analyzed_runs: list[dict]) -> str:
    """Do not combine epsilon and temperature observations in one curve."""
    strategies = {
        exploration_strategy(run["training_metrics"].get("metadata", {}))
        for run in analyzed_runs if run.get("training_metrics") is not None
    }
    if len(strategies) > 1:
        raise ValueError("Training runs contain mixed exploration strategies.")
    return next(iter(strategies), "epsilon_greedy")


def exploration_plot_spec(summary: dict) -> tuple[str, str, float | None]:
    """Return the active metric, axis label and optional probability ceiling."""
    configuration = summary.get("experiment", {}).get("configuration", {})
    strategy = exploration_strategy(configuration)
    if strategy == "softmax":
        return "temperature_end", "Temperature", None
    return "epsilon_end", "Epsilon", 1.0


def print_exploration_summary(summary: dict) -> None:
    """Print the active schedule, including temperatures greater than one."""
    field = (
        "temperature" if summary.get("exploration_strategy") == "softmax"
        else "epsilon"
    )
    print(
        f"  {field.title()}: {summary[f'initial_{field}']:.6f} -> "
        f"{summary[f'final_{field}']:.6f}"
    )
