"""Versioned learning-action spaces for the Q-learning agent."""


ACTION_ORDER = (
    "UP",
    "RIGHT",
    "DOWN",
    "LEFT",
    "WAIT",
    "BOMB",
)

TASK_1_ACTION_SPACE_VERSION = "task1_without_bomb_v1"
TASK_2_ACTION_SPACE_VERSION = "task2_all_actions_v1"
DEFAULT_ACTION_SPACE_VERSION = TASK_1_ACTION_SPACE_VERSION

ACTION_SPACES = {
    TASK_1_ACTION_SPACE_VERSION: ACTION_ORDER[:-1],
    TASK_2_ACTION_SPACE_VERSION: ACTION_ORDER,
}


def get_learning_actions(
    action_space_version: str,
) -> tuple[str, ...]:
    """Return the ordered actions learned by one version."""
    try:
        return ACTION_SPACES[action_space_version]
    except KeyError as error:
        available_versions = ", ".join(sorted(ACTION_SPACES))
        raise ValueError(
            f"Unknown action-space version {action_space_version!r}. "
            f"Available versions: {available_versions}."
        ) from error


def get_learning_action_indices(
    action_space_version: str,
) -> tuple[int, ...]:
    """Return canonical Q-vector indices for one action-space version."""
    return tuple(
        ACTION_ORDER.index(action)
        for action in get_learning_actions(action_space_version)
    )
