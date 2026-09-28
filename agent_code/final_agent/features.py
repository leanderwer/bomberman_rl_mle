"""Encode the three V12 masks in one canonical, 15-entry Q-table state.

Direction order is UP, RIGHT, DOWN, LEFT. Entries 0:5 contain movement/WAIT
safety, 5:9 coin directions, and 9:14 bomb or blockade targets. Entry 13 is
0 for no local target, 1 for BOMB here, or 2 for HOLD here. Entry 14 records
bomb availability, pooled to 1 during HOLD to share the learned WAIT action.

The masks provide observations; they do not restrict the six policy actions.
Optional callback contexts retain physical availability and diagnostic data
without adding either to the learned state.
"""

from dataclasses import dataclass

from .bomb_mechanics import PressureBombDiagnosticContext
from .tactics import feature_data


FEATURE_VERSION = "task3_parallel_masks_tactical_bombs_v12"
ACTIONS = ("UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB")
FeatureState = tuple | None

# Callbacks opt in by adding these keys to a shallow observation copy.
FEATURE_ACTION_TRANSFORM_CONTEXT_KEY = "_feature_action_transform_context"
THREAT_URGENCY_DIAGNOSTIC_CONTEXT_KEY = "_threat_urgency_diagnostic_context"
TARGET_CLASS_DIAGNOSTIC_CONTEXT_KEY = "_target_class_diagnostic_context"
COIN_ROUTE_BOMB_DIAGNOSTIC_CONTEXT_KEY = "_coin_route_bomb_diagnostic_context"
EVALUATION_DIAGNOSTIC_CONTEXT_KEY = "_evaluation_diagnostic_context"


@dataclass(frozen=True)
class FeatureActionTransform:
    """Map directional actions between board and canonical orientation."""

    name: str
    board_to_canonical_directions: tuple[int, int, int, int]
    canonical_to_board_directions: tuple[int, int, int, int]

    def to_canonical_action(self, action: str | None) -> str | None:
        """Return the model-space form of one board-space action."""
        if action is None or action in ACTIONS[4:]:
            return action
        try:
            direction_index = ACTIONS[:4].index(action)
        except ValueError as error:
            raise ValueError(f"Unknown action {action!r}.") from error
        return ACTIONS[self.board_to_canonical_directions[direction_index]]

    def to_board_action(self, action: str | None) -> str | None:
        """Return the board-space form of one model-space action."""
        if action is None or action in ACTIONS[4:]:
            return action
        try:
            direction_index = ACTIONS[:4].index(action)
        except ValueError as error:
            raise ValueError(f"Unknown action {action!r}.") from error
        return ACTIONS[self.canonical_to_board_directions[direction_index]]


def _feature_action_transform(
    name: str,
    permutation: tuple[int, int, int, int],
) -> FeatureActionTransform:
    inverse = [0, 0, 0, 0]
    for source_index, target_index in enumerate(permutation):
        inverse[target_index] = source_index
    return FeatureActionTransform(name, permutation, tuple(inverse))


# Fixed ordering preserves the learned tie-break between equivalent D4 images.
# WAIT, BOMB, and the local bomb/HOLD indicator are invariant.
D4_FEATURE_ACTION_TRANSFORMS = (
    _feature_action_transform("identity", (0, 1, 2, 3)),
    _feature_action_transform("rotate_90", (1, 2, 3, 0)),
    _feature_action_transform("rotate_180", (2, 3, 0, 1)),
    _feature_action_transform("rotate_270", (3, 0, 1, 2)),
    _feature_action_transform("reflect", (0, 3, 2, 1)),
    _feature_action_transform("reflect_then_rotate_90", (1, 0, 3, 2)),
    _feature_action_transform("reflect_then_rotate_180", (2, 1, 0, 3)),
    _feature_action_transform("reflect_then_rotate_270", (3, 2, 1, 0)),
)
IDENTITY_FEATURE_ACTION_TRANSFORM = D4_FEATURE_ACTION_TRANSFORMS[0]


def transform_direction_mask(
    mask: tuple[int, int, int, int],
    transform: FeatureActionTransform,
) -> tuple[int, int, int, int]:
    """Transform a directional mask into one D4 orientation."""
    transformed_mask = [0, 0, 0, 0]
    for board_index, value in enumerate(mask):
        canonical_index = transform.board_to_canonical_directions[board_index]
        transformed_mask[canonical_index] = value
    return tuple(transformed_mask)


def transform_state(state: tuple, transform: FeatureActionTransform) -> tuple:
    """Rotate/reflect all three directional masks, keeping scalar bits fixed."""
    if len(state) != 15:
        raise ValueError("A feature state must contain exactly 15 entries.")
    return (
        *transform_direction_mask(state[0:4], transform),
        state[4],
        *transform_direction_mask(state[5:9], transform),
        *transform_direction_mask(state[9:13], transform),
        state[13],
        state[14],
    )


def canonicalize_state(state: tuple) -> tuple[tuple, FeatureActionTransform]:
    """Return the lexicographically smallest D4 image and its transform."""
    candidates = tuple(
        (transform_state(state, transform), transform_index, transform)
        for transform_index, transform in enumerate(D4_FEATURE_ACTION_TRANSFORMS)
    )
    canonical, _, selected_transform = min(
        candidates, key=lambda candidate: (candidate[0], candidate[1])
    )
    return canonical, selected_transform


def state_to_features(game_state: dict | None) -> FeatureState:
    """Compute V12 features and fill only explicitly requested contexts."""
    if game_state is None:
        return None

    evaluation = game_state.get(EVALUATION_DIAGNOSTIC_CONTEXT_KEY)
    data, diagnostic = feature_data(game_state, diagnostic_context=evaluation)
    raw = (
        *data.safety_mask,
        *data.coin_mask,
        *data.bomb_target_mask,
        1 if data.bomb_target_mask[4] == 2 else data.bomb_available,
    )
    canonical, transform = canonicalize_state(raw)
    context = {
        FEATURE_ACTION_TRANSFORM_CONTEXT_KEY: transform,
        THREAT_URGENCY_DIAGNOSTIC_CONTEXT_KEY: data.threat_urgency,
        TARGET_CLASS_DIAGNOSTIC_CONTEXT_KEY: (
            data.coin_distance_class, data.bomb_quality_class
        ),
        COIN_ROUTE_BOMB_DIAGNOSTIC_CONTEXT_KEY: PressureBombDiagnosticContext(
            primary_mask=(
                *transform_direction_mask(data.primary_bomb_target_mask[:4], transform),
                data.primary_bomb_target_mask[4],
            ),
            alternative_mask=(
                *transform_direction_mask(data.coin_route_alternative_mask[:4], transform),
                data.coin_route_alternative_mask[4],
            ),
            alternative_score_ratio_milli=data.coin_route_alternative_score_ratio_milli,
            pressure=diagnostic,
            evaluation=evaluation,
            bomb_available=bool(data.bomb_available),
        ),
    }
    for key, value in context.items():
        if key in game_state:
            game_state[key] = value
    return canonical
