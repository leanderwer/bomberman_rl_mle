"""Versioned epsilon and temperature schedules for the training behavior policy."""

from math import nextafter

import numpy as np


STANDARD_EXPLORATION_VERSION = "task1_epsilon_standard_v1"
SLOW_3000_EXPLORATION_VERSION = "task1_epsilon_3000_v1"
TASK2_BASELINE_EXPLORATION_VERSION = "task2_epsilon_3000_v1"
SLOW_10000_EXPLORATION_VERSION = "task2_epsilon_10000_v1"
TASK2_LONG_15000_EXPLORATION_VERSION = "task2_epsilon_10500_v1"
TASK2_SLOW_15000_EXPLORATION_VERSION = "task2_epsilon_15000_v1"
TASK2_LONG_25000_EXPLORATION_VERSION = "task2_epsilon_17500_v1"
TASK2_LONG_50000_EXPLORATION_VERSION = "task2_epsilon_35000_v1"
TASK2_LONG_100000_EXPLORATION_VERSION = "task2_epsilon_70000_v1"
TASK3_NIGHT_20000_EXPLORATION_VERSION = "task3_epsilon_20000_v1"
TASK3_SAFETY_6500_EXPLORATION_VERSION = "task3_epsilon_6500_v1"
TASK3_SHORT_7000_EXPLORATION_VERSION = "task3_epsilon_7000_v1"
TASK3_MEDIUM_13500_EXPLORATION_VERSION = "task3_epsilon_13500_v1"
TASK3_HOLD_15000_EXPLORATION_VERSION = "task3_epsilon_15000_v1"
TASK3_RULE_BASED_22000_EXPLORATION_VERSION = "task3_epsilon_22000_v1"
DEFAULT_EXPLORATION_VERSION = STANDARD_EXPLORATION_VERSION

EPSILON_START = 1.0
EPSILON_MIN = 0.05
STANDARD_EPSILON_DECAY = 0.995
SLOW_DECAY_ROUNDS = 3000
SLOW_3000_EPSILON_DECAY = nextafter(
    (EPSILON_MIN / EPSILON_START)
    ** (1.0 / SLOW_DECAY_ROUNDS),
    0.0,
)
TASK2_SLOW_DECAY_ROUNDS = 10000
SLOW_10000_EPSILON_DECAY = nextafter(
    (EPSILON_MIN / EPSILON_START)
    ** (1.0 / TASK2_SLOW_DECAY_ROUNDS),
    0.0,
)
TASK2_LONG_15000_DECAY_ROUNDS = 10500
TASK2_LONG_15000_EPSILON_DECAY = nextafter(
    (EPSILON_MIN / EPSILON_START)
    ** (1.0 / TASK2_LONG_15000_DECAY_ROUNDS),
    0.0,
)
TASK2_SLOW_15000_DECAY_ROUNDS = 15000
TASK2_SLOW_15000_EPSILON_DECAY = nextafter(
    (EPSILON_MIN / EPSILON_START)
    ** (1.0 / TASK2_SLOW_15000_DECAY_ROUNDS),
    0.0,
)
TASK2_LONG_DECAY_ROUNDS = 17500
TASK2_LONG_EPSILON_DECAY = nextafter(
    (EPSILON_MIN / EPSILON_START)
    ** (1.0 / TASK2_LONG_DECAY_ROUNDS),
    0.0,
)
TASK2_LONG_50000_DECAY_ROUNDS = 35000
TASK2_LONG_50000_EPSILON_DECAY = nextafter(
    (EPSILON_MIN / EPSILON_START)
    ** (1.0 / TASK2_LONG_50000_DECAY_ROUNDS),
    0.0,
)
TASK2_LONG_100000_DECAY_ROUNDS = 70000
TASK2_LONG_100000_EPSILON_DECAY = nextafter(
    (EPSILON_MIN / EPSILON_START)
    ** (1.0 / TASK2_LONG_100000_DECAY_ROUNDS),
    0.0,
)
TASK3_NIGHT_DECAY_ROUNDS = 20000
TASK3_NIGHT_EPSILON_DECAY = nextafter(
    (EPSILON_MIN / EPSILON_START)
    ** (1.0 / TASK3_NIGHT_DECAY_ROUNDS),
    0.0,
)
TASK3_SAFETY_DECAY_ROUNDS = 6500
TASK3_SAFETY_EPSILON_DECAY = nextafter(
    (EPSILON_MIN / EPSILON_START)
    ** (1.0 / TASK3_SAFETY_DECAY_ROUNDS),
    0.0,
)
TASK3_SHORT_DECAY_ROUNDS = 7000
TASK3_SHORT_EPSILON_DECAY = nextafter(
    (EPSILON_MIN / EPSILON_START)
    ** (1.0 / TASK3_SHORT_DECAY_ROUNDS),
    0.0,
)
TASK3_MEDIUM_DECAY_ROUNDS = 13500
TASK3_MEDIUM_EPSILON_DECAY = nextafter(
    (EPSILON_MIN / EPSILON_START)
    ** (1.0 / TASK3_MEDIUM_DECAY_ROUNDS),
    0.0,
)
TASK3_HOLD_DECAY_ROUNDS = 15000
TASK3_HOLD_EPSILON_DECAY = nextafter(
    (EPSILON_MIN / EPSILON_START)
    ** (1.0 / TASK3_HOLD_DECAY_ROUNDS),
    0.0,
)
TASK3_RULE_BASED_DECAY_ROUNDS = 22000
TASK3_RULE_BASED_EPSILON_DECAY = nextafter(
    (EPSILON_MIN / EPSILON_START)
    ** (1.0 / TASK3_RULE_BASED_DECAY_ROUNDS),
    0.0,
)


EXPLORATION_PARAMETERS: dict[str, dict[str, float | int | str]] = {
    STANDARD_EXPLORATION_VERSION: {
        "schedule": "multiplicative_epsilon_decay",
        "epsilon_start": EPSILON_START,
        "epsilon_min": EPSILON_MIN,
        "epsilon_decay": STANDARD_EPSILON_DECAY,
    },
    SLOW_3000_EXPLORATION_VERSION: {
        "schedule": "multiplicative_epsilon_decay",
        "epsilon_start": EPSILON_START,
        "epsilon_min": EPSILON_MIN,
        "epsilon_decay": SLOW_3000_EPSILON_DECAY,
        "rounds_to_minimum": SLOW_DECAY_ROUNDS,
    },
    TASK2_BASELINE_EXPLORATION_VERSION: {
        "schedule": "multiplicative_epsilon_decay",
        "epsilon_start": EPSILON_START,
        "epsilon_min": EPSILON_MIN,
        "epsilon_decay": SLOW_3000_EPSILON_DECAY,
        "rounds_to_minimum": SLOW_DECAY_ROUNDS,
    },
    SLOW_10000_EXPLORATION_VERSION: {
        "schedule": "multiplicative_epsilon_decay",
        "epsilon_start": EPSILON_START,
        "epsilon_min": EPSILON_MIN,
        "epsilon_decay": SLOW_10000_EPSILON_DECAY,
        "rounds_to_minimum": TASK2_SLOW_DECAY_ROUNDS,
    },
    TASK2_LONG_15000_EXPLORATION_VERSION: {
        "schedule": "multiplicative_epsilon_decay",
        "epsilon_start": EPSILON_START,
        "epsilon_min": EPSILON_MIN,
        "epsilon_decay": TASK2_LONG_15000_EPSILON_DECAY,
        "rounds_to_minimum": TASK2_LONG_15000_DECAY_ROUNDS,
    },
    TASK2_SLOW_15000_EXPLORATION_VERSION: {
        "schedule": "multiplicative_epsilon_decay",
        "epsilon_start": EPSILON_START,
        "epsilon_min": EPSILON_MIN,
        "epsilon_decay": TASK2_SLOW_15000_EPSILON_DECAY,
        "rounds_to_minimum": TASK2_SLOW_15000_DECAY_ROUNDS,
    },
    TASK2_LONG_25000_EXPLORATION_VERSION: {
        "schedule": "multiplicative_epsilon_decay",
        "epsilon_start": EPSILON_START,
        "epsilon_min": EPSILON_MIN,
        "epsilon_decay": TASK2_LONG_EPSILON_DECAY,
        "rounds_to_minimum": TASK2_LONG_DECAY_ROUNDS,
    },
    TASK2_LONG_50000_EXPLORATION_VERSION: {
        "schedule": "multiplicative_epsilon_decay",
        "epsilon_start": EPSILON_START,
        "epsilon_min": EPSILON_MIN,
        "epsilon_decay": TASK2_LONG_50000_EPSILON_DECAY,
        "rounds_to_minimum": TASK2_LONG_50000_DECAY_ROUNDS,
    },
    TASK2_LONG_100000_EXPLORATION_VERSION: {
        "schedule": "multiplicative_epsilon_decay",
        "epsilon_start": EPSILON_START,
        "epsilon_min": EPSILON_MIN,
        "epsilon_decay": TASK2_LONG_100000_EPSILON_DECAY,
        "rounds_to_minimum": TASK2_LONG_100000_DECAY_ROUNDS,
    },
    TASK3_NIGHT_20000_EXPLORATION_VERSION: {
        "schedule": "multiplicative_epsilon_decay",
        "epsilon_start": EPSILON_START,
        "epsilon_min": EPSILON_MIN,
        "epsilon_decay": TASK3_NIGHT_EPSILON_DECAY,
        "rounds_to_minimum": TASK3_NIGHT_DECAY_ROUNDS,
    },
    TASK3_SAFETY_6500_EXPLORATION_VERSION: {
        "schedule": "multiplicative_epsilon_decay",
        "epsilon_start": EPSILON_START,
        "epsilon_min": EPSILON_MIN,
        "epsilon_decay": TASK3_SAFETY_EPSILON_DECAY,
        "rounds_to_minimum": TASK3_SAFETY_DECAY_ROUNDS,
    },
    TASK3_SHORT_7000_EXPLORATION_VERSION: {
        "schedule": "multiplicative_epsilon_decay",
        "epsilon_start": EPSILON_START,
        "epsilon_min": EPSILON_MIN,
        "epsilon_decay": TASK3_SHORT_EPSILON_DECAY,
        "rounds_to_minimum": TASK3_SHORT_DECAY_ROUNDS,
    },
    TASK3_MEDIUM_13500_EXPLORATION_VERSION: {
        "schedule": "multiplicative_epsilon_decay",
        "epsilon_start": EPSILON_START,
        "epsilon_min": EPSILON_MIN,
        "epsilon_decay": TASK3_MEDIUM_EPSILON_DECAY,
        "rounds_to_minimum": TASK3_MEDIUM_DECAY_ROUNDS,
    },
    TASK3_HOLD_15000_EXPLORATION_VERSION: {
        "schedule": "multiplicative_epsilon_decay",
        "epsilon_start": EPSILON_START,
        "epsilon_min": EPSILON_MIN,
        "epsilon_decay": TASK3_HOLD_EPSILON_DECAY,
        "rounds_to_minimum": TASK3_HOLD_DECAY_ROUNDS,
    },
    TASK3_RULE_BASED_22000_EXPLORATION_VERSION: {
        "schedule": "multiplicative_epsilon_decay",
        "epsilon_start": EPSILON_START,
        "epsilon_min": EPSILON_MIN,
        "epsilon_decay": TASK3_RULE_BASED_EPSILON_DECAY,
        "rounds_to_minimum": TASK3_RULE_BASED_DECAY_ROUNDS,
    },
}


# Both learners support the same policies and decay horizons. Temperature
# schedules have their own version identifiers: choosing SARSA does not imply
# choosing Softmax, nor does choosing Q-learning imply epsilon-greedy.
TEMPERATURE_START = 1.0
TEMPERATURE_MIN = 0.05
TASK2_SOFTMAX_15000_VERSION = "task2_softmax_15000_v1"
TASK2_SOFTMAX_15000_ROUNDS = TASK2_SLOW_15000_DECAY_ROUNDS
TASK2_SOFTMAX_15000_DECAY = TASK2_SLOW_15000_EPSILON_DECAY
TASK3_RULE_BASED_SOFTMAX_22000_VERSION = "task3_softmax_22000_v1"

for _epsilon_version, _epsilon_parameters in tuple(EXPLORATION_PARAMETERS.items()):
    _temperature_parameters = {
        "schedule": "multiplicative_temperature_decay",
        "temperature_start": TEMPERATURE_START,
        "temperature_min": TEMPERATURE_MIN,
        "temperature_decay": _epsilon_parameters["epsilon_decay"],
    }
    if "rounds_to_minimum" in _epsilon_parameters:
        _temperature_parameters["rounds_to_minimum"] = (
            _epsilon_parameters["rounds_to_minimum"]
        )
    EXPLORATION_PARAMETERS[_epsilon_version.replace("_epsilon_", "_softmax_")] = (
        _temperature_parameters
    )
del _epsilon_version, _epsilon_parameters, _temperature_parameters


def softmax_probabilities(q_values, temperature: float) -> np.ndarray:
    """Return finite probabilities, including for extreme finite Q values.

    Subtracting the maximum before exponentiation prevents overflow. A very
    negative shifted value may underflow to zero, which is the correct limit.
    """
    values = np.asarray(q_values, dtype=float)
    if values.ndim != 1 or not values.size or not np.all(np.isfinite(values)):
        raise ValueError("Softmax requires a nonempty vector of finite Q values.")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Softmax temperature must be finite and positive.")
    with np.errstate(over="ignore", under="ignore"):
        weights = np.exp((values - np.max(values)) / temperature)
    return weights / np.sum(weights)


def get_exploration_parameters(
    exploration_version: str,
) -> dict[str, float | int | str]:
    """Return a copy of one registered exploration schedule."""
    try:
        return dict(EXPLORATION_PARAMETERS[exploration_version])

    except KeyError as error:
        available_versions = ", ".join(
            sorted(EXPLORATION_PARAMETERS)
        )

        raise ValueError(
            f"Unknown exploration version {exploration_version!r}. "
            f"Available versions: {available_versions}."
        ) from error
