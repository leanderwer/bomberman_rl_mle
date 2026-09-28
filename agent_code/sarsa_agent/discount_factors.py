"""Versioned discount factors for tabular SARSA."""


GAMMA_095_VERSION = "gamma_095_v1"
DEFAULT_DISCOUNT_FACTOR_VERSION = GAMMA_095_VERSION

DISCOUNT_FACTORS: dict[str, float] = {
    GAMMA_095_VERSION: 0.95,
}


def get_discount_factor(
    discount_factor_version: str,
) -> float:
    """Return the discount factor registered for one version."""
    try:
        return float(DISCOUNT_FACTORS[discount_factor_version])

    except KeyError as error:
        available_versions = ", ".join(
            sorted(DISCOUNT_FACTORS)
        )

        raise ValueError(
            "Unknown discount-factor version "
            f"{discount_factor_version!r}. "
            f"Available versions: {available_versions}."
        ) from error
