from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .action_spaces import ACTION_ORDER
from .task3_tactical_mechanics import (
    task3_v9_feature_data, task3_v10_feature_data, task3_v11_feature_data,
    task3_v12_feature_data,
)
from .task3_bomb_mechanics import (
    PressureBombDiagnosticContext,
    task3_v7_feature_data,
    task3_v8_feature_data,
)
from .task2_mechanics import (
    CoinRouteBombDiagnosticContext,
    TASK2_TARGET_CASE_TRUE_CONFLICT,
    bomb_placement_status,
    classify_safe_target_navigation,
    movement_neighbor_statuses,
    navigation_data as task2_navigation_data,
    nearest_safe_bomb_position_mask,
    nearest_target_direction_mask,
    occupied_positions,
    position_is_threatened,
    safe_escape_first_step_mask,
    task2_v2_feature_data,
    task2_v3_feature_data,
    task2_v6_feature_data,
)


FeatureState = tuple | None
FeatureExtractor = Callable[[dict | None], FeatureState]
NavigationData = tuple[
    dict[tuple[int, int], int],
    dict[tuple[int, int], int],
]


@dataclass(frozen=True)
class PotentialGuidanceContext:
    """Potential values already computed for one guided feature state."""

    hint_code: int
    current_potential: float | None
    direction_potentials: tuple[
        float | None,
        float | None,
        float | None,
        float | None,
    ]
    shortest_path_mask: tuple[int, int, int, int] = (
        0,
        0,
        0,
        0,
    )

BASELINE_FEATURE_VERSION = "task1_baseline_v1"
BFS_MASK_FEATURE_VERSION = "task1_bfs_mask_v1"
BFS_POTENTIAL_MASK_FEATURE_VERSION = (
    "task1_bfs_potential_mask_v1"
)
BFS_POTENTIAL_PRIORITY_FEATURE_VERSION = (
    "task1_bfs_potential_priority_v1"
)
BFS_POTENTIAL_PRIORITY_V2_FEATURE_VERSION = (
    "task1_bfs_potential_priority_v2"
)
BFS_POTENTIAL_TIE_BREAK_FEATURE_VERSION = (
    "task1_bfs_potential_tie_break_v1"
)
TASK2_BASELINE_FEATURE_VERSION = "task2_baseline_v1"
TASK2_PARALLEL_MASKS_FEATURE_VERSION = "task2_parallel_masks_v2"
TASK2_INDEPENDENT_TARGETS_FEATURE_VERSION = (
    "task2_parallel_masks_independent_targets_v3"
)
TASK2_COMPACT_SYMMETRIC_FEATURE_VERSION = (
    "task2_parallel_masks_compact_symmetric_v4"
)
TASK2_COMPACT_SYMMETRIC_V5_FEATURE_VERSION = (
    "task2_parallel_masks_compact_symmetric_v5"
)
TASK2_COIN_ROUTE_BOMB_V6_FEATURE_VERSION = (
    "task2_parallel_masks_coin_route_bomb_v6"
)
TASK3_PRESSURE_BOMB_V7_FEATURE_VERSION = "task3_parallel_masks_pressure_bomb_v7"
TASK3_ROBUST_SAFETY_V8_FEATURE_VERSION = "task3_parallel_masks_robust_safety_v8"
TASK3_TACTICAL_BOMB_V9_FEATURE_VERSION = "task3_parallel_masks_tactical_bombs_v9"
TASK3_TACTICAL_BOMB_V10_FEATURE_VERSION = "task3_parallel_masks_tactical_bombs_v10"
TASK3_TACTICAL_BOMB_V11_1_FEATURE_VERSION = "task3_parallel_masks_tactical_bombs_v11_1"
TASK3_TACTICAL_BOMB_V11_2_FEATURE_VERSION = "task3_parallel_masks_tactical_bombs_v11_2"
TASK3_TACTICAL_BOMB_V11_3_FEATURE_VERSION = "task3_parallel_masks_tactical_bombs_v11_3"
TASK3_TACTICAL_BOMB_V12_FEATURE_VERSION = "task3_parallel_masks_tactical_bombs_v12"
TASK3_RETRY_TACTICAL_FEATURE_VERSIONS = (
    TASK3_TACTICAL_BOMB_V10_FEATURE_VERSION,
    TASK3_TACTICAL_BOMB_V11_1_FEATURE_VERSION,
    TASK3_TACTICAL_BOMB_V11_2_FEATURE_VERSION,
    TASK3_TACTICAL_BOMB_V11_3_FEATURE_VERSION,
)
TASK3_ARRIVAL_TACTICAL_FEATURE_VERSIONS = (
    *TASK3_RETRY_TACTICAL_FEATURE_VERSIONS,
    TASK3_TACTICAL_BOMB_V12_FEATURE_VERSION,
)
DEFAULT_FEATURE_VERSION = BASELINE_FEATURE_VERSION

COIN_POTENTIAL_DECAY = 0.95
POTENTIAL_OVERRIDE_RELATIVE_MARGIN = 0.05
POTENTIAL_CONFIDENCE_THRESHOLDS = (
    0.02,
    0.05,
    0.10,
)

# Categorical direction features use zero for "no signal" and the
# one-based direction index otherwise:
# 1 = UP, 2 = RIGHT, 3 = DOWN, 4 = LEFT.
NO_DIRECTION = 0

# Task 2 navigation modes. The direction mask in the feature state is
# interpreted according to exactly one of these categorical objectives.
TASK2_NO_OBJECTIVE = 0
TASK2_ESCAPE_OBJECTIVE = 1
TASK2_COIN_OBJECTIVE = 2
TASK2_CRATE_OBJECTIVE = 3

# The callback layer adds this private context field to a shallow copy of
# the visible game state. It is used only to suppress an immediate move
# back to the tile occupied in the preceding step.
PREVIOUS_AGENT_POSITION_KEY = "_previous_agent_position"
POTENTIAL_GUIDANCE_CONTEXT_KEY = "_potential_guidance_context"
FEATURE_ACTION_TRANSFORM_CONTEXT_KEY = (
    "_feature_action_transform_context"
)
THREAT_URGENCY_DIAGNOSTIC_CONTEXT_KEY = (
    "_threat_urgency_diagnostic_context"
)
TARGET_CLASS_DIAGNOSTIC_CONTEXT_KEY = (
    "_target_class_diagnostic_context"
)
COIN_ROUTE_BOMB_DIAGNOSTIC_CONTEXT_KEY = (
    "_coin_route_bomb_diagnostic_context"
)
EVALUATION_DIAGNOSTIC_CONTEXT_KEY = "_evaluation_diagnostic_context"
TACTICAL_PLANNER_CONTEXT_KEY = "_tactical_planner_context"

# V4 activates the target-quality categories only in a true conflict.
# Dedicated out-of-range values avoid conflating "inactive" with V3's
# real far/weak class zero.
TASK2_V4_INACTIVE_COIN_CLASS = 4
TASK2_V4_INACTIVE_BOMB_CLASS = 3

# Direction order: UP, RIGHT, DOWN, LEFT.
NEIGHBOR_DIRECTIONS = (
    (0, -1),
    (1, 0),
    (0, 1),
    (-1, 0),
)


@dataclass(frozen=True)
class FeatureActionTransform:
    """Map directional actions between board and canonical orientation."""

    name: str
    board_to_canonical_directions: tuple[int, int, int, int]
    canonical_to_board_directions: tuple[int, int, int, int]

    def to_canonical_action(self, action: str | None) -> str | None:
        """Return the model-space form of one board-space action."""
        if action is None or action in ACTION_ORDER[4:]:
            return action

        try:
            direction_index = ACTION_ORDER[:4].index(action)
        except ValueError as error:
            raise ValueError(f"Unknown action {action!r}.") from error

        canonical_index = self.board_to_canonical_directions[
            direction_index
        ]
        return ACTION_ORDER[canonical_index]

    def to_board_action(self, action: str | None) -> str | None:
        """Return the board-space form of one model-space action."""
        if action is None or action in ACTION_ORDER[4:]:
            return action

        try:
            direction_index = ACTION_ORDER[:4].index(action)
        except ValueError as error:
            raise ValueError(f"Unknown action {action!r}.") from error

        board_index = self.canonical_to_board_directions[
            direction_index
        ]
        return ACTION_ORDER[board_index]


def _inverse_direction_permutation(
    permutation: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    inverse = [0, 0, 0, 0]

    for source_index, target_index in enumerate(permutation):
        inverse[target_index] = source_index

    return tuple(inverse)


def _feature_action_transform(
    name: str,
    board_to_canonical_directions: tuple[int, int, int, int],
) -> FeatureActionTransform:
    return FeatureActionTransform(
        name=name,
        board_to_canonical_directions=(
            board_to_canonical_directions
        ),
        canonical_to_board_directions=(
            _inverse_direction_permutation(
                board_to_canonical_directions
            )
        ),
    )


# The four rotations followed by the same rotations after a left-right
# reflection form the dihedral group D4. Each permutation maps the indices
# UP, RIGHT, DOWN, LEFT from board orientation into candidate canonical
# orientation. WAIT, BOMB, and the "bomb here" bit are invariant.
D4_FEATURE_ACTION_TRANSFORMS = (
    _feature_action_transform("identity", (0, 1, 2, 3)),
    _feature_action_transform("rotate_90", (1, 2, 3, 0)),
    _feature_action_transform("rotate_180", (2, 3, 0, 1)),
    _feature_action_transform("rotate_270", (3, 0, 1, 2)),
    _feature_action_transform("reflect", (0, 3, 2, 1)),
    _feature_action_transform(
        "reflect_then_rotate_90",
        (1, 0, 3, 2),
    ),
    _feature_action_transform(
        "reflect_then_rotate_180",
        (2, 1, 0, 3),
    ),
    _feature_action_transform(
        "reflect_then_rotate_270",
        (3, 2, 1, 0),
    ),
)
IDENTITY_FEATURE_ACTION_TRANSFORM = D4_FEATURE_ACTION_TRANSFORMS[0]


def transform_direction_mask(
    mask: tuple[int, int, int, int],
    transform: FeatureActionTransform,
) -> tuple[int, int, int, int]:
    """Transform a directional mask into one D4 orientation."""
    transformed_mask = [0, 0, 0, 0]

    for board_index, value in enumerate(mask):
        canonical_index = transform.board_to_canonical_directions[
            board_index
        ]
        transformed_mask[canonical_index] = value

    return tuple(transformed_mask)


def transform_task2_v4_state(
    state: tuple,
    transform: FeatureActionTransform,
) -> tuple:
    """Apply one D4 transform to a raw 17-entry V4 state."""
    if len(state) != 17:
        raise ValueError(
            "A Task 2 V4 state must contain exactly 17 entries."
        )

    safety_mask = transform_direction_mask(state[0:4], transform)
    coin_mask = transform_direction_mask(state[5:9], transform)
    bomb_direction_mask = transform_direction_mask(
        state[9:13],
        transform,
    )

    return (
        *safety_mask,
        state[4],
        *coin_mask,
        *bomb_direction_mask,
        state[13],
        state[14],
        state[15],
        state[16],
    )


def canonicalize_task2_v4_state(
    state: tuple,
) -> tuple[tuple, FeatureActionTransform]:
    """Return the lexicographically smallest D4 image and its transform."""
    candidates = tuple(
        (
            transform_task2_v4_state(state, transform),
            transform_index,
            transform,
        )
        for transform_index, transform in enumerate(
            D4_FEATURE_ACTION_TRANSFORMS
        )
    )
    canonical_state, _, selected_transform = min(
        candidates,
        key=lambda candidate: (candidate[0], candidate[1]),
    )

    return canonical_state, selected_transform


def transform_task2_v5_state(
    state: tuple,
    transform: FeatureActionTransform,
) -> tuple:
    """Apply one D4 transform to a raw 15-entry V5 state."""
    if len(state) != 15:
        raise ValueError(
            "A Task 2 V5 state must contain exactly 15 entries."
        )

    safety_mask = transform_direction_mask(state[0:4], transform)
    coin_mask = transform_direction_mask(state[5:9], transform)
    bomb_direction_mask = transform_direction_mask(
        state[9:13],
        transform,
    )

    return (
        *safety_mask,
        state[4],
        *coin_mask,
        *bomb_direction_mask,
        state[13],
        state[14],
    )


def canonicalize_task2_v5_state(
    state: tuple,
) -> tuple[tuple, FeatureActionTransform]:
    """Return the lexicographically smallest V5 D4 image."""
    candidates = tuple(
        (
            transform_task2_v5_state(state, transform),
            transform_index,
            transform,
        )
        for transform_index, transform in enumerate(
            D4_FEATURE_ACTION_TRANSFORMS
        )
    )
    canonical_state, _, selected_transform = min(
        candidates,
        key=lambda candidate: (candidate[0], candidate[1]),
    )

    return canonical_state, selected_transform


def baseline_features_v1(
    game_state: dict | None,
) -> FeatureState:
    """
    Convert the complete game state into a small, discrete state
    representation for tabular SARSA.

    The state contains:
        - whether each neighboring tile is free
        - the horizontal and vertical direction of the nearest coin

    :param game_state: A dictionary describing the current game board.
    :return: A hashable tuple representing the state.

    This implementation is kept unchanged so that models trained with
    task1_baseline_v1 remain compatible.
    """
    if game_state is None:
        return None

    field = game_state["field"]

    # game_state["self"] has the structure:
    # (name, score, bomb_available, (x, y))
    x, y = game_state["self"][3]

    # Direction order: UP, RIGHT, DOWN, LEFT.
    neighbor_directions = [
        (0, -1),
        (1, 0),
        (0, 1),
        (-1, 0),
    ]

    free_neighbors = tuple(
        int(field[x + dx, y + dy] == 0)
        for dx, dy in neighbor_directions
    )

    coins = game_state["coins"]

    if coins:
        nearest_coin = min(
            coins,
            key=lambda coin: abs(coin[0] - x) + abs(coin[1] - y),
        )

        coin_x, coin_y = nearest_coin
        coin_direction_x = int(np.sign(coin_x - x))
        coin_direction_y = int(np.sign(coin_y - y))

    else:
        coin_direction_x = 0
        coin_direction_y = 0

    return (
        *free_neighbors,
        coin_direction_x,
        coin_direction_y,
    )


def _is_free_tile(
    field: np.ndarray,
    position: tuple[int, int],
) -> bool:
    """Return whether a board position exists and is walkable."""
    x, y = position

    return (
        0 <= x < field.shape[0]
        and 0 <= y < field.shape[1]
        and field[x, y] == 0
    )


def _navigation_data(
    x: int,
    y: int,
    field: np.ndarray,
) -> NavigationData:
    """Return distances and shortest first-step masks for all free tiles."""
    start_position = (x, y)
    distances = {
        start_position: 0,
    }
    first_step_masks = {
        start_position: 0,
    }
    queue = deque([start_position])

    while queue:
        current_position = queue.popleft()
        current_distance = distances[current_position]
        current_x, current_y = current_position

        for direction_index, (dx, dy) in enumerate(
            NEIGHBOR_DIRECTIONS
        ):
            next_position = (
                current_x + dx,
                current_y + dy,
            )

            if not _is_free_tile(field, next_position):
                continue

            next_distance = current_distance + 1

            if current_distance == 0:
                propagated_mask = 1 << direction_index
            else:
                propagated_mask = first_step_masks[
                    current_position
                ]

            if next_position not in distances:
                distances[next_position] = next_distance
                first_step_masks[next_position] = (
                    propagated_mask
                )
                queue.append(next_position)

            elif distances[next_position] == next_distance:
                merged_mask = (
                    first_step_masks[next_position]
                    | propagated_mask
                )

                if merged_mask != first_step_masks[next_position]:
                    first_step_masks[next_position] = merged_mask
                    queue.append(next_position)

    return distances, first_step_masks


def _shortest_path_mask_from_navigation_data(
    start_position: tuple[int, int],
    coins: list[tuple[int, int]],
    navigation_data: NavigationData,
) -> tuple[int, int, int, int]:
    """Derive the nearest-coin direction mask from one complete BFS."""
    coin_positions = {
        tuple(coin)
        for coin in coins
    }

    if (
        not coin_positions
        or start_position in coin_positions
    ):
        return (0, 0, 0, 0)

    distances, first_step_masks = navigation_data
    reachable_coins = [
        coin
        for coin in coin_positions
        if coin in distances
    ]

    if not reachable_coins:
        return (0, 0, 0, 0)

    nearest_coin_distance = min(
        distances[coin]
        for coin in reachable_coins
    )
    shortest_direction_mask = 0

    for coin in reachable_coins:
        if distances[coin] == nearest_coin_distance:
            shortest_direction_mask |= first_step_masks[coin]

    return tuple(
        int(
            shortest_direction_mask
            & (1 << direction_index)
            != 0
        )
        for direction_index in range(
            len(NEIGHBOR_DIRECTIONS)
        )
    )


def _shortest_path_direction_mask(
    x: int,
    y: int,
    field: np.ndarray,
    coins: list[tuple[int, int]],
) -> tuple[int, int, int, int]:
    """
    Return all first movements belonging to a shortest path to a coin.

    The returned mask uses the direction order:
        UP, RIGHT, DOWN, LEFT

    Multiple entries can be one when several first movements lead to
    an equally short path. This avoids selecting one shortest path
    through an arbitrary direction priority.

    A zero mask is returned when:
        - there are no coins
        - the agent currently stands on a coin
        - no coin is reachable
    """
    start_position = (x, y)
    coin_positions = {
        tuple(coin)
        for coin in coins
    }

    if (
        not coin_positions
        or start_position in coin_positions
    ):
        return (0, 0, 0, 0)

    distances = {
        start_position: 0,
    }

    # Each bit records one possible first movement:
    # bit 0 = UP, bit 1 = RIGHT,
    # bit 2 = DOWN, bit 3 = LEFT.
    first_step_masks = {
        start_position: 0,
    }

    queue = deque([start_position])

    nearest_coin_distance = None
    shortest_direction_mask = 0

    while queue:
        current_position = queue.popleft()
        current_distance = distances[current_position]

        if (
            nearest_coin_distance is not None
            and current_distance > nearest_coin_distance
        ):
            continue

        if current_position in coin_positions:
            if nearest_coin_distance is None:
                nearest_coin_distance = current_distance

            if current_distance == nearest_coin_distance:
                shortest_direction_mask |= first_step_masks[
                    current_position
                ]

            continue

        if (
            nearest_coin_distance is not None
            and current_distance >= nearest_coin_distance
        ):
            continue

        current_x, current_y = current_position

        for direction_index, (dx, dy) in enumerate(
            NEIGHBOR_DIRECTIONS
        ):
            next_position = (
                current_x + dx,
                current_y + dy,
            )

            if not _is_free_tile(field, next_position):
                continue

            next_distance = current_distance + 1

            if current_distance == 0:
                propagated_mask = 1 << direction_index
            else:
                propagated_mask = first_step_masks[
                    current_position
                ]

            if next_position not in distances:
                distances[next_position] = next_distance
                first_step_masks[next_position] = (
                    propagated_mask
                )
                queue.append(next_position)

            elif distances[next_position] == next_distance:
                merged_mask = (
                    first_step_masks[next_position]
                    | propagated_mask
                )

                if merged_mask != first_step_masks[next_position]:
                    first_step_masks[next_position] = merged_mask
                    queue.append(next_position)

    return tuple(
        int(
            shortest_direction_mask
            & (1 << direction_index)
            != 0
        )
        for direction_index in range(
            len(NEIGHBOR_DIRECTIONS)
        )
    )


def bfs_mask_features_v1(
    game_state: dict | None,
) -> FeatureState:
    """
    Build a path-aware state representation for Task 1.

    The state contains:
        - whether each neighboring tile is free
        - which first movements belong to a shortest path to the
          nearest reachable coin

    Both groups use the direction order:
        UP, RIGHT, DOWN, LEFT

    The resulting state therefore has the structure:
        (
            free_up,
            free_right,
            free_down,
            free_left,
            shortest_up,
            shortest_right,
            shortest_down,
            shortest_left,
        )

    Multiple shortest-path entries can be one simultaneously.
    The feature extractor therefore does not impose an arbitrary
    priority between equally short paths.
    """
    if game_state is None:
        return None

    field = game_state["field"]

    # game_state["self"] has the structure:
    # (name, score, bomb_available, (x, y))
    x, y = game_state["self"][3]

    free_neighbors = tuple(
        int(field[x + dx, y + dy] == 0)
        for dx, dy in NEIGHBOR_DIRECTIONS
    )

    shortest_path_mask = _shortest_path_direction_mask(
        x=x,
        y=y,
        field=field,
        coins=game_state["coins"],
    )

    return (
        *free_neighbors,
        *shortest_path_mask,
    )


def _coin_direction_potentials(
    x: int,
    y: int,
    field: np.ndarray,
    coins: list[tuple[int, int]],
    decay: float = COIN_POTENTIAL_DECAY,
    navigation_data: NavigationData | None = None,
) -> tuple[
    float | None,
    float | None,
    float | None,
    float | None,
]:
    """
    Return the discounted coin potential of every movement.

    For a valid first movement ``a``, the potential is

        sum(decay ** distance(successor(a), coin))

    over all reachable coins. Distances are shortest-path distances
    through free tiles, so walls are respected. Invalid movements are
    represented by ``None``.

    One breadth-first search from the current position is sufficient.
    In the orthogonal grid, moving to a neighbor changes the distance
    to every reachable coin by exactly one. A direction contained in
    a shortest path reduces that distance; every other valid direction
    increases it.
    """
    coin_positions = {
        tuple(coin)
        for coin in coins
    }

    if not coin_positions:
        return (None, None, None, None)

    if navigation_data is None:
        navigation_data = _navigation_data(
            x=x,
            y=y,
            field=field,
        )

    distances, first_step_masks = navigation_data

    reachable_coins = [
        coin
        for coin in coin_positions
        if coin in distances
    ]

    if not reachable_coins:
        return (None, None, None, None)

    direction_potentials: list[float | None] = [
        0.0
        if _is_free_tile(
            field,
            (x + dx, y + dy),
        )
        else None
        for dx, dy in NEIGHBOR_DIRECTIONS
    ]

    for coin in reachable_coins:
        coin_distance = distances[coin]
        shortest_first_steps = first_step_masks[coin]

        for direction_index, potential in enumerate(
            direction_potentials
        ):
            if potential is None:
                continue

            if coin_distance == 0:
                successor_distance = 1
            elif (
                shortest_first_steps
                & (1 << direction_index)
            ):
                successor_distance = coin_distance - 1
            else:
                successor_distance = coin_distance + 1

            direction_potentials[direction_index] += (
                decay ** successor_distance
            )

    valid_potentials = [
        potential
        for potential in direction_potentials
        if potential is not None
    ]

    if not valid_potentials:
        return (None, None, None, None)

    return tuple(direction_potentials)


def _coin_potential_direction_mask(
    x: int,
    y: int,
    field: np.ndarray,
    coins: list[tuple[int, int]],
    decay: float = COIN_POTENTIAL_DECAY,
    navigation_data: NavigationData | None = None,
) -> tuple[int, int, int, int]:
    """
    Return all movements with maximal discounted coin potential.

    This preserves the binary mask used by
    ``task1_bfs_potential_mask_v1`` while sharing the potential
    calculation with newer feature versions.
    """
    direction_potentials = _coin_direction_potentials(
        x=x,
        y=y,
        field=field,
        coins=coins,
        decay=decay,
        navigation_data=navigation_data,
    )
    valid_potentials = [
        potential
        for potential in direction_potentials
        if potential is not None
    ]

    if not valid_potentials:
        return (0, 0, 0, 0)

    best_potential = max(valid_potentials)

    return tuple(
        int(
            potential is not None
            and np.isclose(
                potential,
                best_potential,
                rtol=1e-12,
                atol=1e-12,
            )
        )
        for potential in direction_potentials
    )


def _potential_confidence_bucket(
    best_potential: float,
    alternative_potential: float,
) -> int:
    """Discretize the relative advantage over a relevant alternative."""
    relative_advantage = max(
        0.0,
        (
            best_potential
            - alternative_potential
        )
        / max(abs(best_potential), 1e-12),
    )

    for bucket, threshold in enumerate(
        POTENTIAL_CONFIDENCE_THRESHOLDS,
        start=1,
    ):
        if relative_advantage <= threshold:
            return bucket

    return len(POTENTIAL_CONFIDENCE_THRESHOLDS) + 1


def _potential_priority_features(
    shortest_path_mask: tuple[int, int, int, int],
    direction_potentials: tuple[
        float | None,
        float | None,
        float | None,
        float | None,
    ],
) -> tuple[int, int, int]:
    """
    Relate the global potential recommendation to the BFS guidance.

    The returned categorical features are:
        - the potential-recommended direction
        - a close BFS-compatible fallback direction
        - a confidence bucket for the recommendation

    Redundant recommendations are suppressed. A direction is encoded
    only when it breaks a BFS tie or recommends an override of BFS.
    Exact potential ties produce no recommendation, avoiding a fixed
    absolute direction priority.
    """
    valid_directions = [
        direction_index
        for direction_index, potential in enumerate(
            direction_potentials
        )
        if potential is not None
    ]
    bfs_directions = [
        direction_index
        for direction_index, is_shortest in enumerate(
            shortest_path_mask
        )
        if (
            is_shortest
            and direction_potentials[direction_index]
            is not None
        )
    ]

    if not valid_directions or not bfs_directions:
        return (NO_DIRECTION, NO_DIRECTION, 0)

    best_potential = max(
        direction_potentials[direction_index]
        for direction_index in valid_directions
    )
    best_directions = [
        direction_index
        for direction_index in valid_directions
        if np.isclose(
            direction_potentials[direction_index],
            best_potential,
            rtol=1e-12,
            atol=1e-12,
        )
    ]

    if len(best_directions) != 1:
        return (NO_DIRECTION, NO_DIRECTION, 0)

    best_direction = best_directions[0]
    best_direction_code = best_direction + 1

    best_bfs_potential = max(
        direction_potentials[direction_index]
        for direction_index in bfs_directions
    )
    best_bfs_directions = [
        direction_index
        for direction_index in bfs_directions
        if np.isclose(
            direction_potentials[direction_index],
            best_bfs_potential,
            rtol=1e-12,
            atol=1e-12,
        )
    ]

    if best_direction in bfs_directions:
        if len(bfs_directions) == 1:
            return (NO_DIRECTION, NO_DIRECTION, 0)

        alternative_potential = max(
            direction_potentials[direction_index]
            for direction_index in bfs_directions
            if direction_index != best_direction
        )
        confidence_bucket = _potential_confidence_bucket(
            best_potential=best_potential,
            alternative_potential=alternative_potential,
        )

        return (
            best_direction_code,
            NO_DIRECTION,
            confidence_bucket,
        )

    confidence_bucket = _potential_confidence_bucket(
        best_potential=best_potential,
        alternative_potential=best_bfs_potential,
    )
    fallback_direction = NO_DIRECTION

    if (
        len(bfs_directions) > 1
        and len(best_bfs_directions) == 1
        and confidence_bucket <= 3
    ):
        fallback_direction = best_bfs_directions[0] + 1

    return (
        best_direction_code,
        fallback_direction,
        confidence_bucket,
    )


def _current_coin_potential(
    coins: list[tuple[int, int]],
    navigation_data: NavigationData,
    decay: float = COIN_POTENTIAL_DECAY,
) -> float | None:
    """Return the discounted coin potential at the current position."""
    distances, _ = navigation_data
    reachable_distances = [
        distances[tuple(coin)]
        for coin in coins
        if tuple(coin) in distances
    ]

    if not reachable_distances:
        return None

    return sum(
        decay ** distance
        for distance in reachable_distances
    )


def _unique_best_direction(
    direction_indices: list[int],
    direction_potentials: tuple[
        float | None,
        float | None,
        float | None,
        float | None,
    ],
) -> int | None:
    """Return a unique potential maximum without absolute tie-breaking."""
    if not direction_indices:
        return None

    best_potential = max(
        direction_potentials[direction_index]
        for direction_index in direction_indices
    )
    best_directions = [
        direction_index
        for direction_index in direction_indices
        if np.isclose(
            direction_potentials[direction_index],
            best_potential,
            rtol=1e-12,
            atol=1e-12,
        )
    ]

    if len(best_directions) != 1:
        return None

    return best_directions[0]


def _potential_tie_break_direction(
    shortest_path_mask: tuple[int, int, int, int],
    direction_potentials: tuple[
        float | None,
        float | None,
        float | None,
        float | None,
    ],
) -> int:
    """Return a unique potential maximum inside a genuine BFS tie.

    Zero means that no tie-breaking information is available. The helper
    never recommends a direction outside the shortest-path mask and never
    changes a single-direction BFS recommendation.
    """
    bfs_directions = [
        direction_index
        for direction_index, is_shortest in enumerate(
            shortest_path_mask
        )
        if (
            is_shortest
            and direction_potentials[direction_index] is not None
        )
    ]

    if len(bfs_directions) <= 1:
        return NO_DIRECTION

    best_bfs_direction = _unique_best_direction(
        bfs_directions,
        direction_potentials,
    )

    if best_bfs_direction is None:
        return NO_DIRECTION

    return best_bfs_direction + 1


def _strong_potential_direction_v2(
    position: tuple[int, int],
    previous_position: tuple[int, int] | None,
    shortest_path_mask: tuple[int, int, int, int],
    direction_potentials: tuple[
        float | None,
        float | None,
        float | None,
        float | None,
    ],
    current_potential: float | None,
) -> int:
    """Return one optional potential hint while leaving policy choice learned.

    A unique potential maximum inside a non-trivial BFS mask is exposed as
    a tie-breaking hint. A direction outside the BFS mask is exposed only
    when it improves both the current potential and the best BFS successor
    by a relative margin. A direct return to the preceding tile is
    suppressed. The returned code never restricts or executes an action.
    """
    valid_directions = [
        direction_index
        for direction_index, potential in enumerate(
            direction_potentials
        )
        if potential is not None
    ]
    bfs_directions = [
        direction_index
        for direction_index, is_shortest in enumerate(
            shortest_path_mask
        )
        if (
            is_shortest
            and direction_potentials[direction_index] is not None
        )
    ]

    if not valid_directions or not bfs_directions:
        return NO_DIRECTION

    tie_break_hint = _potential_tie_break_direction(
        shortest_path_mask=shortest_path_mask,
        direction_potentials=direction_potentials,
    )

    best_direction = _unique_best_direction(
        valid_directions,
        direction_potentials,
    )

    if best_direction is None or best_direction in bfs_directions:
        return tie_break_hint

    candidate_potential = direction_potentials[best_direction]
    best_bfs_potential = max(
        direction_potentials[direction_index]
        for direction_index in bfs_directions
    )

    if (
        current_potential is None
        or candidate_potential <= current_potential
    ):
        return tie_break_hint

    relative_bfs_advantage = (
        candidate_potential - best_bfs_potential
    ) / max(abs(best_bfs_potential), 1e-12)

    if (
        relative_bfs_advantage
        < POTENTIAL_OVERRIDE_RELATIVE_MARGIN
    ):
        return tie_break_hint

    dx, dy = NEIGHBOR_DIRECTIONS[best_direction]
    candidate_position = (
        position[0] + dx,
        position[1] + dy,
    )

    if (
        previous_position is not None
        and candidate_position == tuple(previous_position)
    ):
        return tie_break_hint

    return best_direction + 1


def bfs_potential_mask_features_v1(
    game_state: dict | None,
) -> FeatureState:
    """
    Add global coin-distribution information to the BFS features.

    The state contains three four-direction masks, each ordered as
    UP, RIGHT, DOWN, LEFT:

        - free neighboring movements
        - movements on a shortest path to the nearest reachable coin
        - movements with maximal discounted potential over all coins

    The resulting state has twelve binary entries. Ties in either
    path-based mask remain represented explicitly.
    """
    if game_state is None:
        return None

    field = game_state["field"]
    x, y = game_state["self"][3]

    free_neighbors = tuple(
        int(field[x + dx, y + dy] == 0)
        for dx, dy in NEIGHBOR_DIRECTIONS
    )
    navigation_data = _navigation_data(
        x=x,
        y=y,
        field=field,
    )
    shortest_path_mask = _shortest_path_mask_from_navigation_data(
        start_position=(x, y),
        coins=game_state["coins"],
        navigation_data=navigation_data,
    )
    potential_mask = _coin_potential_direction_mask(
        x=x,
        y=y,
        field=field,
        coins=game_state["coins"],
        navigation_data=navigation_data,
    )

    return (
        *free_neighbors,
        *shortest_path_mask,
        *potential_mask,
    )


def bfs_potential_priority_features_v1(
    game_state: dict | None,
) -> FeatureState:
    """
    Add a compact, confidence-aware global potential recommendation.

    The first eight entries are unchanged from
    ``task1_bfs_mask_v1``. Three categorical entries are appended:

        - potential direction: 0 = none, 1..4 = UP/RIGHT/DOWN/LEFT
        - close BFS fallback: 0 = none, 1..4 = UP/RIGHT/DOWN/LEFT
        - confidence: 0 = none, 1..4 = increasing relative advantage

    The potential direction can break a tie between shortest BFS
    movements or recommend a globally favorable override. A fallback
    is included only for a close override when BFS offers multiple
    shortest movements. Redundant recommendations and exact ties are
    suppressed to limit unnecessary state-space fragmentation.
    """
    if game_state is None:
        return None

    field = game_state["field"]
    x, y = game_state["self"][3]

    free_neighbors = tuple(
        int(field[x + dx, y + dy] == 0)
        for dx, dy in NEIGHBOR_DIRECTIONS
    )
    navigation_data = _navigation_data(
        x=x,
        y=y,
        field=field,
    )
    shortest_path_mask = _shortest_path_mask_from_navigation_data(
        start_position=(x, y),
        coins=game_state["coins"],
        navigation_data=navigation_data,
    )
    direction_potentials = _coin_direction_potentials(
        x=x,
        y=y,
        field=field,
        coins=game_state["coins"],
        navigation_data=navigation_data,
    )
    (
        potential_direction,
        fallback_direction,
        confidence_bucket,
    ) = _potential_priority_features(
        shortest_path_mask=shortest_path_mask,
        direction_potentials=direction_potentials,
    )

    return (
        *free_neighbors,
        *shortest_path_mask,
        potential_direction,
        fallback_direction,
        confidence_bucket,
    )


def bfs_potential_priority_features_v2(
    game_state: dict | None,
) -> FeatureState:
    """Add one gated global-potential hint to the BFS representation.

    The first eight entries remain identical to ``task1_bfs_mask_v1``.
    The ninth entry is a categorical direction code:

        - 0: no sufficiently informative potential hint
        - 1..4: UP, RIGHT, DOWN, LEFT

    Within a multi-direction BFS mask, the hint identifies a unique
    potential maximum. A direction outside the BFS mask is exposed only
    when it raises the potential relative to the current tile, exceeds the
    best BFS successor by the configured margin, and does not immediately
    return to the preceding tile. This value is state information only;
    the Q-policy retains all actions and learns whether to follow it.
    """
    if game_state is None:
        return None

    field = game_state["field"]
    x, y = game_state["self"][3]
    previous_position = game_state.get(
        PREVIOUS_AGENT_POSITION_KEY
    )

    free_neighbors = tuple(
        int(field[x + dx, y + dy] == 0)
        for dx, dy in NEIGHBOR_DIRECTIONS
    )
    navigation_data = _navigation_data(
        x=x,
        y=y,
        field=field,
    )
    shortest_path_mask = _shortest_path_mask_from_navigation_data(
        start_position=(x, y),
        coins=game_state["coins"],
        navigation_data=navigation_data,
    )
    direction_potentials = _coin_direction_potentials(
        x=x,
        y=y,
        field=field,
        coins=game_state["coins"],
        navigation_data=navigation_data,
    )
    current_potential = _current_coin_potential(
        coins=game_state["coins"],
        navigation_data=navigation_data,
    )
    strong_potential_direction = _strong_potential_direction_v2(
        position=(x, y),
        previous_position=previous_position,
        shortest_path_mask=shortest_path_mask,
        direction_potentials=direction_potentials,
        current_potential=current_potential,
    )

    if POTENTIAL_GUIDANCE_CONTEXT_KEY in game_state:
        game_state[POTENTIAL_GUIDANCE_CONTEXT_KEY] = (
            PotentialGuidanceContext(
                hint_code=strong_potential_direction,
                current_potential=current_potential,
                direction_potentials=direction_potentials,
                shortest_path_mask=shortest_path_mask,
            )
        )

    return (
        *free_neighbors,
        *shortest_path_mask,
        strong_potential_direction,
    )


def bfs_potential_tie_break_features_v1(
    game_state: dict | None,
) -> FeatureState:
    """Add a potential recommendation only for genuine BFS ties.

    The first eight entries are identical to ``task1_bfs_mask_v1``.
    The ninth entry is categorical:

        - 0: no unique tie-break recommendation
        - 1..4: UP, RIGHT, DOWN, LEFT

    A non-zero value is possible only when at least two movements lie on
    shortest paths to the nearest reachable coin and exactly one of those
    movements maximizes the discounted all-coin potential. Directions
    outside the BFS mask are never exposed by this feature.
    """
    if game_state is None:
        return None

    field = game_state["field"]
    x, y = game_state["self"][3]

    free_neighbors = tuple(
        int(field[x + dx, y + dy] == 0)
        for dx, dy in NEIGHBOR_DIRECTIONS
    )
    navigation_data = _navigation_data(
        x=x,
        y=y,
        field=field,
    )
    shortest_path_mask = _shortest_path_mask_from_navigation_data(
        start_position=(x, y),
        coins=game_state["coins"],
        navigation_data=navigation_data,
    )
    direction_potentials = _coin_direction_potentials(
        x=x,
        y=y,
        field=field,
        coins=game_state["coins"],
        navigation_data=navigation_data,
    )
    tie_break_direction = _potential_tie_break_direction(
        shortest_path_mask=shortest_path_mask,
        direction_potentials=direction_potentials,
    )

    if POTENTIAL_GUIDANCE_CONTEXT_KEY in game_state:
        game_state[POTENTIAL_GUIDANCE_CONTEXT_KEY] = (
            PotentialGuidanceContext(
                hint_code=tie_break_direction,
                current_potential=_current_coin_potential(
                    coins=game_state["coins"],
                    navigation_data=navigation_data,
                ),
                direction_potentials=direction_potentials,
                shortest_path_mask=shortest_path_mask,
            )
        )

    return (
        *free_neighbors,
        *shortest_path_mask,
        tie_break_direction,
    )


def task2_baseline_features_v1(
    game_state: dict | None,
) -> FeatureState:
    """Return a compact bomb-aware state for Task 2.

    The ten categorical entries are:

        - four neighboring-tile states (blocked, dangerous, safe),
        - four shortest first-step bits for the active objective,
        - objective mode (none, escape, coin, crate),
        - local bomb status (unavailable, useless, unsafe, useful).

    Danger takes priority. Outside danger, visible reachable coins are
    targeted first; otherwise the mask points to the nearest reachable
    position from which a bomb can destroy a crate and still be escaped.
    The feature provides information rather than a deterministic action:
    multiple movements may be recommended and the Q-table remains free
    to learn which action is best in each categorical state.
    """
    if game_state is None:
        return None

    field = game_state["field"]
    position = tuple(game_state["self"][3])
    neighbor_statuses = movement_neighbor_statuses(
        game_state,
        position,
    )
    blocked_positions = occupied_positions(game_state)
    navigation = task2_navigation_data(
        field,
        position,
        blocked_positions,
    )

    if position_is_threatened(game_state, position):
        objective_mode = TASK2_ESCAPE_OBJECTIVE
        objective_mask = safe_escape_first_step_mask(
            game_state,
            start_position=position,
        )

    else:
        objective_mask, target_found = (
            nearest_target_direction_mask(
                game_state["coins"],
                navigation,
            )
        )

        if target_found:
            objective_mode = TASK2_COIN_OBJECTIVE
        else:
            objective_mask, target_found = (
                nearest_safe_bomb_position_mask(
                    game_state,
                    navigation,
                )
            )
            objective_mode = (
                TASK2_CRATE_OBJECTIVE
                if target_found
                else TASK2_NO_OBJECTIVE
            )

    return (
        *neighbor_statuses,
        *objective_mask,
        objective_mode,
        bomb_placement_status(game_state),
    )


def task2_parallel_masks_features_v2(
    game_state: dict | None,
) -> FeatureState:
    """Return the final compact parallel-mask state for Task 2.

    The 17 entries are five binary action-survivability values, four
    nearest-coin direction bits, five scored bomb-target bits including
    the current tile, relative target distance, threat urgency, and
    current bomb availability. All board mechanics stay in
    ``task2_mechanics.py`` so this extractor only assembles the state.
    """
    if game_state is None:
        return None

    feature_data = task2_v2_feature_data(game_state)

    return (
        *feature_data.safety_mask,
        *feature_data.coin_mask,
        *feature_data.bomb_target_mask,
        feature_data.relative_target_distance,
        feature_data.threat_urgency,
        feature_data.bomb_available,
    )


def task2_independent_targets_features_v3(
    game_state: dict | None,
) -> FeatureState:
    """Return parallel masks with independent target-quality evidence.

    The 18 entries preserve V2's five safety bits, four nearest-coin
    direction bits, and five exact-best bomb-target bits. V2's relative
    target-distance recommendation is replaced by two independent ordinal
    observations: coin distance from far (0) to very near (3), and best bomb
    quality from weak (0) to strong (2). Threat urgency and bomb availability
    remain the final two entries.

    Empty objective masks encode target absence. Therefore class value zero
    remains a real active far/weak category and is also the harmless canonical
    filler when the corresponding target does not exist.
    """
    if game_state is None:
        return None

    feature_data = task2_v3_feature_data(game_state)

    return (
        *feature_data.safety_mask,
        *feature_data.coin_mask,
        *feature_data.bomb_target_mask,
        feature_data.coin_distance_class,
        feature_data.bomb_quality_class,
        feature_data.threat_urgency,
        feature_data.bomb_available,
    )


def task2_compact_symmetric_features_v4(
    game_state: dict | None,
) -> FeatureState:
    """Return the compact, D4-canonical Task 2 V4 representation.

    V4 retains the three action masks and bomb availability from V3. Threat
    urgency is removed from the learned state but exposed through a private
    diagnostic context. Coin-distance and bomb-quality classes are active
    only when safety-filtered coin and bomb directions form a true conflict;
    all other cases use dedicated out-of-range inactive values.

    The resulting raw 17-entry tuple is transformed under all eight square
    symmetries. The lexicographically smallest image is the Q-table key. The
    selected transform is returned through callback-only context so actions
    can be mapped consistently between canonical model space and the board.
    """
    if game_state is None:
        return None

    feature_data = task2_v3_feature_data(game_state)
    local_bomb_opportunity = bool(
        feature_data.bomb_target_mask[4]
        and feature_data.bomb_available
    )
    target_case = classify_safe_target_navigation(
        safety_mask=feature_data.safety_mask,
        coin_mask=feature_data.coin_mask,
        bomb_target_mask=feature_data.bomb_target_mask,
        local_bomb_opportunity=local_bomb_opportunity,
    )

    if target_case.name == TASK2_TARGET_CASE_TRUE_CONFLICT:
        coin_distance_class = feature_data.coin_distance_class
        bomb_quality_class = feature_data.bomb_quality_class
    else:
        coin_distance_class = TASK2_V4_INACTIVE_COIN_CLASS
        bomb_quality_class = TASK2_V4_INACTIVE_BOMB_CLASS

    raw_state = (
        *feature_data.safety_mask,
        *feature_data.coin_mask,
        *feature_data.bomb_target_mask,
        coin_distance_class,
        bomb_quality_class,
        feature_data.bomb_available,
    )
    canonical_state, selected_transform = (
        canonicalize_task2_v4_state(raw_state)
    )

    if FEATURE_ACTION_TRANSFORM_CONTEXT_KEY in game_state:
        game_state[FEATURE_ACTION_TRANSFORM_CONTEXT_KEY] = (
            selected_transform
        )

    if THREAT_URGENCY_DIAGNOSTIC_CONTEXT_KEY in game_state:
        game_state[THREAT_URGENCY_DIAGNOSTIC_CONTEXT_KEY] = (
            feature_data.threat_urgency
        )

    return canonical_state


def task2_compact_symmetric_features_v5(
    game_state: dict | None,
) -> FeatureState:
    """Return the class-free, D4-canonical Task 2 V5 state.

    V5 keeps V4's safety, coin-target, and bomb-target masks plus bomb
    availability, but removes coin-distance and bomb-quality classes from
    the Q-table key. Their original values remain available through private
    evaluation context, so the true-conflict heatmap stays comparable while
    class-only observations no longer fragment learning states.
    """
    if game_state is None:
        return None

    feature_data = task2_v3_feature_data(game_state)
    raw_state = (
        *feature_data.safety_mask,
        *feature_data.coin_mask,
        *feature_data.bomb_target_mask,
        feature_data.bomb_available,
    )
    canonical_state, selected_transform = (
        canonicalize_task2_v5_state(raw_state)
    )

    if FEATURE_ACTION_TRANSFORM_CONTEXT_KEY in game_state:
        game_state[FEATURE_ACTION_TRANSFORM_CONTEXT_KEY] = (
            selected_transform
        )

    if THREAT_URGENCY_DIAGNOSTIC_CONTEXT_KEY in game_state:
        game_state[THREAT_URGENCY_DIAGNOSTIC_CONTEXT_KEY] = (
            feature_data.threat_urgency
        )

    if TARGET_CLASS_DIAGNOSTIC_CONTEXT_KEY in game_state:
        game_state[TARGET_CLASS_DIAGNOSTIC_CONTEXT_KEY] = (
            feature_data.coin_distance_class,
            feature_data.bomb_quality_class,
        )

    return canonical_state


def task2_coin_route_bomb_features_v6(
    game_state: dict | None,
) -> FeatureState:
    """Return V5's compact state with one compatible bomb alternative.

    V6 retains the exact 15-entry V5 layout and D4 canonicalization. Its
    only learned-state change is the bomb-target mask: the global optimum
    is augmented by the best safe productive bomb tile lying strictly on
    a shortest route to a nearest coin, provided a time-feasible escape can
    continue through that coin. The alternative must reach at least 50% of
    the primary target's score.
    """
    if game_state is None:
        return None

    feature_data = task2_v6_feature_data(game_state)
    raw_state = (
        *feature_data.safety_mask,
        *feature_data.coin_mask,
        *feature_data.bomb_target_mask,
        feature_data.bomb_available,
    )
    canonical_state, selected_transform = (
        canonicalize_task2_v5_state(raw_state)
    )

    if FEATURE_ACTION_TRANSFORM_CONTEXT_KEY in game_state:
        game_state[FEATURE_ACTION_TRANSFORM_CONTEXT_KEY] = (
            selected_transform
        )

    if THREAT_URGENCY_DIAGNOSTIC_CONTEXT_KEY in game_state:
        game_state[THREAT_URGENCY_DIAGNOSTIC_CONTEXT_KEY] = (
            feature_data.threat_urgency
        )

    if TARGET_CLASS_DIAGNOSTIC_CONTEXT_KEY in game_state:
        game_state[TARGET_CLASS_DIAGNOSTIC_CONTEXT_KEY] = (
            feature_data.coin_distance_class,
            feature_data.bomb_quality_class,
        )

    if COIN_ROUTE_BOMB_DIAGNOSTIC_CONTEXT_KEY in game_state:
        transformed_primary_mask = transform_direction_mask(
            feature_data.primary_bomb_target_mask[:4],
            selected_transform,
        )
        transformed_alternative_mask = transform_direction_mask(
            feature_data.coin_route_alternative_mask[:4],
            selected_transform,
        )
        game_state[COIN_ROUTE_BOMB_DIAGNOSTIC_CONTEXT_KEY] = (
            CoinRouteBombDiagnosticContext(
                primary_mask=(
                    *transformed_primary_mask,
                    feature_data.primary_bomb_target_mask[4],
                ),
                alternative_mask=(
                    *transformed_alternative_mask,
                    feature_data.coin_route_alternative_mask[4],
                ),
                alternative_score_ratio_milli=(
                    feature_data.coin_route_alternative_score_ratio_milli
                ),
            )
        )

    return canonical_state


def task3_pressure_bomb_features_v7(game_state: dict | None) -> FeatureState:
    """V6's 15 binary entries with exclusive pressure/crate bomb targets.

    The five movement-safety bits, coin mask and action canonicalization stay
    unchanged. All richer pressure/blockade information is private telemetry.
    """
    if game_state is None:
        return None
    data, diagnostic = task3_v7_feature_data(
        game_state, diagnostic_context=game_state.get(EVALUATION_DIAGNOSTIC_CONTEXT_KEY))
    return _task3_canonical_feature_state(game_state, data, diagnostic)


def task3_robust_safety_features_v8(game_state: dict | None) -> FeatureState:
    """V7 targets and 15-bit D4 layout with body-robust movement safety."""
    if game_state is None:
        return None
    data, diagnostic = task3_v8_feature_data(
        game_state, diagnostic_context=game_state.get(EVALUATION_DIAGNOSTIC_CONTEXT_KEY))
    return _task3_canonical_feature_state(game_state, data, diagnostic)


def task3_tactical_bomb_features_v9(game_state: dict | None) -> FeatureState:
    """V8 safety with snapshot pressure and safe, timed body-block targets.

    Keep 15 entries, but entry 13 is ternary: 0 approach/none, 1 BOMB,
    2 hold with WAIT. The action-at-target value is invariant under D4.
    Existing V7/V8 Q-tables keep their own version and are not reinterpreted.
    """
    if game_state is None:
        return None
    data, diagnostic = task3_v9_feature_data(
        game_state, diagnostic_context=game_state.get(EVALUATION_DIAGNOSTIC_CONTEXT_KEY))
    return _task3_canonical_feature_state(game_state, data, diagnostic)


def task3_tactical_bomb_features_v10(game_state: dict | None) -> FeatureState:
    """V10 keeps 15 entries and pools availability only for confirmed HOLD.

    Training and action selection use this same encoder. Game state, legality
    and diagnostic context retain the true availability; no action is masked.
    """
    if game_state is None:
        return None
    data, diagnostic = task3_v10_feature_data(
        game_state, planner_context=game_state.get(TACTICAL_PLANNER_CONTEXT_KEY),
        diagnostic_context=game_state.get(EVALUATION_DIAGNOSTIC_CONTEXT_KEY))
    return _task3_canonical_feature_state(game_state, data, diagnostic, pool_hold=True)


def task3_tactical_bomb_features_v11_1(game_state: dict | None) -> FeatureState:
    """V11 independent block safety and priority; pressure threshold 0.25."""
    return _task3_v11_features(game_state, pressure_threshold=0.25)


def task3_tactical_bomb_features_v11_2(game_state: dict | None) -> FeatureState:
    """Identical V11 planner and encoding; pressure threshold 0.50."""
    return _task3_v11_features(game_state, pressure_threshold=0.50)


def task3_tactical_bomb_features_v11_3(game_state: dict | None) -> FeatureState:
    """Identical V11 planner and HOLD pooling; pressure threshold 0.30."""
    return _task3_v11_features(game_state, pressure_threshold=0.30)


def task3_tactical_bomb_features_v12(game_state: dict | None) -> FeatureState:
    """V11 HOLD pooling/arrival targets; corrected WAIT and no retry memory."""
    if game_state is None:
        return None
    data, diagnostic = task3_v12_feature_data(
        game_state, diagnostic_context=game_state.get(EVALUATION_DIAGNOSTIC_CONTEXT_KEY))
    return _task3_canonical_feature_state(game_state, data, diagnostic, pool_hold=True)


def _task3_v11_features(game_state, *, pressure_threshold):
    if game_state is None:
        return None
    data, diagnostic = task3_v11_feature_data(
        game_state, pressure_threshold=pressure_threshold,
        planner_context=game_state.get(TACTICAL_PLANNER_CONTEXT_KEY),
        diagnostic_context=game_state.get(EVALUATION_DIAGNOSTIC_CONTEXT_KEY))
    return _task3_canonical_feature_state(game_state, data, diagnostic, pool_hold=True)


def _task3_canonical_feature_state(game_state, data, diagnostic, *, pool_hold=False):
    raw = (*data.safety_mask, *data.coin_mask, *data.bomb_target_mask,
           1 if pool_hold and data.bomb_target_mask[4] == 2 else data.bomb_available)
    canonical, transform = canonicalize_task2_v5_state(raw)
    context = {
        FEATURE_ACTION_TRANSFORM_CONTEXT_KEY: transform,
        THREAT_URGENCY_DIAGNOSTIC_CONTEXT_KEY: data.threat_urgency,
        TARGET_CLASS_DIAGNOSTIC_CONTEXT_KEY: (
            data.coin_distance_class, data.bomb_quality_class),
        COIN_ROUTE_BOMB_DIAGNOSTIC_CONTEXT_KEY: PressureBombDiagnosticContext(
            primary_mask=(*transform_direction_mask(data.primary_bomb_target_mask[:4], transform),
                          data.primary_bomb_target_mask[4]),
            alternative_mask=(*transform_direction_mask(data.coin_route_alternative_mask[:4], transform),
                              data.coin_route_alternative_mask[4]),
            alternative_score_ratio_milli=data.coin_route_alternative_score_ratio_milli,
            pressure=diagnostic, evaluation=game_state.get(EVALUATION_DIAGNOSTIC_CONTEXT_KEY),
            bomb_available=bool(data.bomb_available)),
    }
    for key, value in context.items():
        if key in game_state:
            game_state[key] = value
    return canonical


FEATURE_EXTRACTORS: dict[str, FeatureExtractor] = {
    TASK3_TACTICAL_BOMB_V12_FEATURE_VERSION: task3_tactical_bomb_features_v12,
    TASK3_TACTICAL_BOMB_V11_3_FEATURE_VERSION: task3_tactical_bomb_features_v11_3,
    TASK3_TACTICAL_BOMB_V11_1_FEATURE_VERSION: task3_tactical_bomb_features_v11_1,
    TASK3_TACTICAL_BOMB_V11_2_FEATURE_VERSION: task3_tactical_bomb_features_v11_2,
    TASK3_TACTICAL_BOMB_V10_FEATURE_VERSION: task3_tactical_bomb_features_v10,
    TASK3_TACTICAL_BOMB_V9_FEATURE_VERSION: task3_tactical_bomb_features_v9,
    TASK3_ROBUST_SAFETY_V8_FEATURE_VERSION: task3_robust_safety_features_v8,
    TASK3_PRESSURE_BOMB_V7_FEATURE_VERSION: task3_pressure_bomb_features_v7,
    BASELINE_FEATURE_VERSION: baseline_features_v1,
    BFS_MASK_FEATURE_VERSION: bfs_mask_features_v1,
    BFS_POTENTIAL_MASK_FEATURE_VERSION: (
        bfs_potential_mask_features_v1
    ),
    BFS_POTENTIAL_PRIORITY_FEATURE_VERSION: (
        bfs_potential_priority_features_v1
    ),
    BFS_POTENTIAL_PRIORITY_V2_FEATURE_VERSION: (
        bfs_potential_priority_features_v2
    ),
    BFS_POTENTIAL_TIE_BREAK_FEATURE_VERSION: (
        bfs_potential_tie_break_features_v1
    ),
    TASK2_BASELINE_FEATURE_VERSION: task2_baseline_features_v1,
    TASK2_PARALLEL_MASKS_FEATURE_VERSION: (
        task2_parallel_masks_features_v2
    ),
    TASK2_INDEPENDENT_TARGETS_FEATURE_VERSION: (
        task2_independent_targets_features_v3
    ),
    TASK2_COMPACT_SYMMETRIC_FEATURE_VERSION: (
        task2_compact_symmetric_features_v4
    ),
    TASK2_COMPACT_SYMMETRIC_V5_FEATURE_VERSION: (
        task2_compact_symmetric_features_v5
    ),
    TASK2_COIN_ROUTE_BOMB_V6_FEATURE_VERSION: (
        task2_coin_route_bomb_features_v6
    ),
}


def get_feature_extractor(
    feature_version: str,
) -> FeatureExtractor:
    """
    Return the feature extractor registered for one feature version.
    """
    try:
        return FEATURE_EXTRACTORS[feature_version]

    except KeyError as error:
        available_versions = ", ".join(
            sorted(FEATURE_EXTRACTORS)
        )

        raise ValueError(
            f"Unknown feature version {feature_version!r}. "
            f"Available versions: {available_versions}."
        ) from error
