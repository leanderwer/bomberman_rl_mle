"""Versioned learning-rate schedules for tabular SARSA."""

from collections.abc import Callable


LearningRateFunction = Callable[[int], float]

CONSTANT_LEARNING_RATE_VERSION = "task1_constant_v1"
VISIT_DECAY_LEARNING_RATE_VERSION = "task1_visit_decay_v1"
TASK2_LONG_HORIZON_LEARNING_RATE_VERSION = (
    "task2_visit_decay_power_060_v1"
)
DEFAULT_LEARNING_RATE_VERSION = CONSTANT_LEARNING_RATE_VERSION

CONSTANT_ALPHA = 0.1
VISIT_DECAY_NUMERATOR = 10.0
VISIT_DECAY_OFFSET = 99.0
TASK2_LONG_HORIZON_INITIAL_ALPHA = 0.1
TASK2_LONG_HORIZON_REFERENCE = 100.0
TASK2_LONG_HORIZON_OFFSET = 99.0
TASK2_LONG_HORIZON_EXPONENT = 0.6


def _validate_visit_count(visit_count: int) -> None:
    """Require a positive state-action update count."""
    if (
        isinstance(visit_count, bool)
        or not isinstance(visit_count, int)
        or visit_count < 1
    ):
        raise ValueError(
            "The state-action visit count must be a positive integer."
        )


def constant_learning_rate_v1(visit_count: int) -> float:
    """Return the original constant learning rate."""
    _validate_visit_count(visit_count)
    return CONSTANT_ALPHA


def visit_decay_learning_rate_v1(visit_count: int) -> float:
    """Return an inverse state-action visit-count learning rate."""
    _validate_visit_count(visit_count)

    return (
        VISIT_DECAY_NUMERATOR
        / (VISIT_DECAY_OFFSET + visit_count)
    )


def task2_long_horizon_learning_rate_v1(
    visit_count: int,
) -> float:
    """Return a slower power-law decay for long Task 2 runs."""
    _validate_visit_count(visit_count)

    return TASK2_LONG_HORIZON_INITIAL_ALPHA * (
        TASK2_LONG_HORIZON_REFERENCE
        / (TASK2_LONG_HORIZON_OFFSET + visit_count)
    ) ** TASK2_LONG_HORIZON_EXPONENT


LEARNING_RATE_FUNCTIONS: dict[str, LearningRateFunction] = {
    CONSTANT_LEARNING_RATE_VERSION: constant_learning_rate_v1,
    VISIT_DECAY_LEARNING_RATE_VERSION: visit_decay_learning_rate_v1,
    TASK2_LONG_HORIZON_LEARNING_RATE_VERSION: (
        task2_long_horizon_learning_rate_v1
    ),
}

LEARNING_RATE_PARAMETERS: dict[str, dict[str, float | str]] = {
    CONSTANT_LEARNING_RATE_VERSION: {
        "schedule": "constant",
        "alpha": CONSTANT_ALPHA,
    },
    VISIT_DECAY_LEARNING_RATE_VERSION: {
        "schedule": "inverse_visit_count",
        "numerator": VISIT_DECAY_NUMERATOR,
        "offset": VISIT_DECAY_OFFSET,
        "initial_alpha": CONSTANT_ALPHA,
    },
    TASK2_LONG_HORIZON_LEARNING_RATE_VERSION: {
        "schedule": "power_visit_count",
        "initial_alpha": TASK2_LONG_HORIZON_INITIAL_ALPHA,
        "reference": TASK2_LONG_HORIZON_REFERENCE,
        "offset": TASK2_LONG_HORIZON_OFFSET,
        "exponent": TASK2_LONG_HORIZON_EXPONENT,
    },
}


def get_learning_rate_function(
    learning_rate_version: str,
) -> LearningRateFunction:
    """Return the registered learning-rate function."""
    try:
        return LEARNING_RATE_FUNCTIONS[learning_rate_version]

    except KeyError as error:
        available_versions = ", ".join(
            sorted(LEARNING_RATE_FUNCTIONS)
        )

        raise ValueError(
            f"Unknown learning-rate version "
            f"{learning_rate_version!r}. "
            f"Available versions: {available_versions}."
        ) from error


def get_learning_rate_parameters(
    learning_rate_version: str,
) -> dict[str, float | str]:
    """Return a copy of the schedule's checkpoint parameters."""
    get_learning_rate_function(learning_rate_version)

    return dict(
        LEARNING_RATE_PARAMETERS[learning_rate_version]
    )
