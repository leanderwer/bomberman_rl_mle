"""Versioned temporal-difference targets for tabular SARSA."""

from collections.abc import Sequence
import math
from numbers import Real


ONE_STEP_SARSA_TARGET_VERSION = "sarsa_1_step_v1"
FIVE_STEP_SARSA_TARGET_VERSION = "sarsa_5_step_v1"
DEFAULT_TD_TARGET_VERSION = ONE_STEP_SARSA_TARGET_VERSION


TD_TARGET_PARAMETERS: dict[
    str,
    dict[str, int | str],
] = {
    ONE_STEP_SARSA_TARGET_VERSION: {
        "algorithm": "sampled_n_step_sarsa",
        "n_steps": 1,
        "reward_source": "sampled_behavior_trajectory",
        "bootstrap_operator": "on_policy_next_action_q",
        "terminal_bootstrap": "none",
        "importance_sampling": "none",
    },
    FIVE_STEP_SARSA_TARGET_VERSION: {
        "algorithm": "sampled_n_step_sarsa",
        "n_steps": 5,
        "reward_source": "sampled_behavior_trajectory",
        "bootstrap_operator": "on_policy_next_action_q",
        "terminal_bootstrap": "none",
        "importance_sampling": "none",
    },
}


def get_td_target_parameters(
    td_target_version: str,
) -> dict[str, int | str]:
    """Return a copy of one registered TD-target definition."""
    try:
        return dict(TD_TARGET_PARAMETERS[td_target_version])

    except KeyError as error:
        available_versions = ", ".join(
            sorted(TD_TARGET_PARAMETERS)
        )

        raise ValueError(
            f"Unknown TD-target version {td_target_version!r}. "
            f"Available versions: {available_versions}."
        ) from error


def get_td_target_n_steps(td_target_version: str) -> int:
    """Return the strictly positive return horizon of one version."""
    parameters = get_td_target_parameters(td_target_version)
    n_steps = parameters["n_steps"]

    if (
        isinstance(n_steps, bool)
        or not isinstance(n_steps, int)
        or n_steps <= 0
    ):
        raise ValueError(
            f"TD-target version {td_target_version!r} has an "
            f"invalid n-step horizon: {n_steps!r}."
        )

    return n_steps


def calculate_sampled_n_step_target(
    rewards: Sequence[float],
    discount_factor: float,
    bootstrap_value: float | None,
) -> float:
    r"""Return a sampled n-step target.

    For ``m = len(rewards)``, this evaluates

    ``sum(gamma**k * rewards[k], k=0..m-1)``

    and, for a non-terminal sequence, adds

    ``gamma**m * bootstrap_value``.

    A ``None`` bootstrap value marks a terminal sequence and therefore
    contributes no value beyond its final observed reward.
    """
    if not rewards:
        raise ValueError(
            "An n-step target requires at least one reward."
        )

    if (
        isinstance(discount_factor, bool)
        or not isinstance(discount_factor, Real)
        or not math.isfinite(float(discount_factor))
        or not 0.0 <= float(discount_factor) <= 1.0
    ):
        raise ValueError(
            "The discount factor must be a finite number in [0, 1]."
        )

    gamma = float(discount_factor)
    discounted_return = 0.0
    discount = 1.0

    for index, reward in enumerate(rewards):
        if (
            isinstance(reward, bool)
            or not isinstance(reward, Real)
            or not math.isfinite(float(reward))
        ):
            raise ValueError(
                f"Reward {index} must be a finite number."
            )

        discounted_return += discount * float(reward)
        discount *= gamma

    if bootstrap_value is not None:
        if (
            isinstance(bootstrap_value, bool)
            or not isinstance(bootstrap_value, Real)
            or not math.isfinite(float(bootstrap_value))
        ):
            raise ValueError(
                "The bootstrap value must be a finite number or None."
            )

        discounted_return += discount * float(bootstrap_value)

    return float(discounted_return)
