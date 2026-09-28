"""Shared board geometry, shortest paths and nominal bomb-route forecasts.

The local robust safety proof lives in bomb_mechanics. These forecasts retain
static opponents and crates for remote target checks, and advance known hazards
by the arrival time before testing a hypothetical placement.
"""

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
import math

import numpy as np

import settings as s

Position = tuple[int, int]
DirectionMask = tuple[int, int, int, int]
ActionSafetyMask = tuple[int, int, int, int, int]
BombTargetMask = tuple[int, int, int, int, int]
NavigationData = tuple[
    dict[Position, int],
    dict[Position, int],
]


TARGET_CASE_NONE = "none"
TARGET_CASE_LOCAL_BOMB = "local_bomb"
TARGET_CASE_COIN_ONLY = "coin_only"
TARGET_CASE_BOMB_ONLY = "bomb_only"
TARGET_CASE_JOINT = "joint"
TARGET_CASE_TRUE_CONFLICT = "true_conflict"

# UP, RIGHT, DOWN, LEFT: shared by all masks and canonical actions.
DIRECTION_DELTAS = (
    (0, -1),
    (1, 0),
    (0, 1),
    (-1, 0),
)


ZERO_DIRECTION_MASK: DirectionMask = (0, 0, 0, 0)
ZERO_ACTION_SAFETY_MASK: ActionSafetyMask = (0, 0, 0, 0, 0)
ZERO_BOMB_TARGET_MASK: BombTargetMask = (0, 0, 0, 0, 0)
WAIT_DELTA = (0, 0)
SURVIVAL_ACTION_DELTAS = (*DIRECTION_DELTAS, WAIT_DELTA)

# Target ranking stays fixed independently of the Q-learning discount.
BOMB_TARGET_DISCOUNT = 0.95
COIN_ROUTE_BOMB_MIN_SCORE_RATIO = 0.50

# Retained for target diagnostics; these classes do not enter the Q-table key.
COIN_DISTANCE_CLASS_BOUNDARIES = (2, 4, 6)
BOMB_QUALITY_CLASS_THRESHOLDS = (3.5, 5.0)


@dataclass(frozen=True)
class BombTargetSelection:
    """Best safe productive bomb target for one visible game state."""

    mask: BombTargetMask
    distance: int | None
    crate_yield: int
    score: float | None


@dataclass(frozen=True)
class BombTargetCandidate:
    """One reachable, productive, and generally escapable bomb tile."""

    position: Position
    distance: int
    crate_yield: int
    score: float


@dataclass(frozen=True)
class CoinRouteBombTargetSelection:
    """Global bomb target enriched by at most one coin-route alternative."""

    mask: BombTargetMask
    global_target: BombTargetSelection
    alternative_mask: BombTargetMask
    alternative_score_ratio_milli: int | None


@dataclass(frozen=True)
class CoinRouteBombDiagnosticContext:
    """Canonical primary/alternative masks retained for evaluation."""

    primary_mask: BombTargetMask
    alternative_mask: BombTargetMask
    alternative_score_ratio_milli: int | None


@dataclass(frozen=True)
class FeatureData:
    """Three masks plus observation metadata used by encoding and diagnostics."""

    safety_mask: ActionSafetyMask
    coin_mask: DirectionMask
    bomb_target_mask: BombTargetMask
    coin_distance_class: int
    bomb_quality_class: int
    threat_urgency: int
    bomb_available: int
    primary_bomb_target_mask: BombTargetMask
    coin_route_alternative_mask: BombTargetMask
    coin_route_alternative_score_ratio_milli: int | None


@dataclass(frozen=True)
class SafeTargetNavigationCase:
    """One safety-filtered, mutually exclusive target situation."""

    name: str
    safe_coin_mask: DirectionMask
    safe_bomb_mask: DirectionMask
    safe_joint_mask: DirectionMask


def classify_safe_target_navigation(
    safety_mask: ActionSafetyMask,
    coin_mask: DirectionMask,
    bomb_target_mask: BombTargetMask,
    *,
    local_bomb_opportunity: bool = False,
) -> SafeTargetNavigationCase:
    """Classify the target choice after applying movement safety.

    This definition is shared by reward shaping and policy diagnostics.
    A usable bomb target on the current tile takes precedence because
    navigation is suppressed in that situation. Otherwise joint directions
    take precedence over a true conflict, followed by the two single-objective
    cases.
    """
    safe_coin_mask = tuple(
        int(bool(safety_mask[index] and coin_mask[index]))
        for index in range(4)
    )
    safe_bomb_mask = tuple(
        int(
            bool(
                safety_mask[index]
                and bomb_target_mask[index]
            )
        )
        for index in range(4)
    )
    safe_joint_mask = tuple(
        int(
            bool(
                safe_coin_mask[index]
                and safe_bomb_mask[index]
            )
        )
        for index in range(4)
    )

    has_safe_coin = any(safe_coin_mask)
    has_safe_bomb = any(safe_bomb_mask)

    if local_bomb_opportunity:
        case_name = TARGET_CASE_LOCAL_BOMB
    elif any(safe_joint_mask):
        case_name = TARGET_CASE_JOINT
    elif has_safe_coin and has_safe_bomb:
        case_name = TARGET_CASE_TRUE_CONFLICT
    elif has_safe_coin:
        case_name = TARGET_CASE_COIN_ONLY
    elif has_safe_bomb:
        case_name = TARGET_CASE_BOMB_ONLY
    else:
        case_name = TARGET_CASE_NONE

    return SafeTargetNavigationCase(
        name=case_name,
        safe_coin_mask=safe_coin_mask,
        safe_bomb_mask=safe_bomb_mask,
        safe_joint_mask=safe_joint_mask,
    )


def position_in_bounds(
    field: np.ndarray,
    position: Position,
) -> bool:
    """Return whether a board position lies inside the field."""
    x, y = position

    return (
        0 <= x < field.shape[0]
        and 0 <= y < field.shape[1]
    )


def occupied_positions(game_state: dict) -> set[Position]:
    """Return currently occupied bomb and opponent positions."""
    occupied = {
        tuple(position)
        for position, _ in game_state.get("bombs", ())
    }
    occupied.update(
        tuple(other[3])
        for other in game_state.get("others", ())
    )

    return occupied


def is_walkable(
    field: np.ndarray,
    position: Position,
    blocked_positions: Iterable[Position] = (),
) -> bool:
    """Return whether a position can currently be entered."""
    return (
        position_in_bounds(field, position)
        and field[position] == 0
        and position not in blocked_positions
    )


def blast_coordinates(
    field: np.ndarray,
    bomb_position: Position,
    power: int = s.BOMB_POWER,
) -> tuple[Position, ...]:
    """Return the blast tiles produced by the actual game engine.

    The supplied framework stops a blast at walls, but not at crates.
    Keeping that detail here prevents features from planning against a
    different transition model than the environment executes.
    """
    blast_tiles = [bomb_position]
    bomb_x, bomb_y = bomb_position

    for dx, dy in DIRECTION_DELTAS:
        for distance in range(1, power + 1):
            position = (
                bomb_x + dx * distance,
                bomb_y + dy * distance,
            )

            if not position_in_bounds(field, position):
                break

            if field[position] == -1:
                break

            blast_tiles.append(position)

    return tuple(blast_tiles)


def crates_hit_by_bomb(
    field: np.ndarray,
    bomb_position: Position,
) -> int:
    """Return how many current crates one bomb would destroy."""
    return sum(
        int(field[position] == 1)
        for position in blast_coordinates(field, bomb_position)
    )


def danger_schedule(
    game_state: dict,
    extra_bomb: tuple[Position, int] | None = None,
) -> tuple[dict[Position, set[int]], dict[Position, int]]:
    """Return dangerous future steps and bomb blocking times.

    Steps are counted from the next action. A bomb visible with timer
    ``t`` explodes after action ``t + 1`` in the supplied environment.
    Its explosion is dangerous in that step and the following step.
    """
    field = game_state["field"]
    dangerous_steps: dict[Position, set[int]] = {}
    bomb_blocked_until: dict[Position, int] = {}

    explosion_map = np.asarray(game_state["explosion_map"])

    # explosion_map is not a yes or no map, rather it counts how much is left for fire to turn off
    # ceil rounds this number upwards, so danger is never underestimated

    for x, y in np.argwhere(explosion_map > 0):
        remaining_steps = max(
            1,
            int(math.ceil(float(explosion_map[x, y]))),
        )

        for step in range(1, remaining_steps + 1):
            dangerous_steps.setdefault(
                (int(x), int(y)),
                set(),
            ).add(step)

    bombs = [
        (tuple(position), int(timer))
        for position, timer in game_state.get("bombs", ())
    ]

    if extra_bomb is not None:
        bombs.append((tuple(extra_bomb[0]), int(extra_bomb[1])))

    for position, timer in bombs:
        explosion_step = max(1, timer + 1)
        bomb_blocked_until[position] = max(
            bomb_blocked_until.get(position, 0),
            explosion_step,
        )

        for blast_position in blast_coordinates(field, position):
            dangerous_steps.setdefault(
                blast_position,
                set(),
            ).update(
                (explosion_step, explosion_step + 1)
            )

    return dangerous_steps, bomb_blocked_until


def _survivable_action_mask_from_schedule(
    field: np.ndarray,
    start_position: Position,
    dangerous_steps: dict[Position, set[int]],
    bomb_blocked_until: dict[Position, int],
    opponent_positions: set[Position],
) -> ActionSafetyMask:
    """Return initial actions with a complete known-horizon survival path.

    A single time-layer search propagates a bit for each of
    ``UP, RIGHT, DOWN, LEFT, WAIT``. Merging equal position/time states
    avoids counting loops or multiple paths to the same future state.
    """
    final_step = (
        max(
            step
            for steps in dangerous_steps.values()
            for step in steps
        )
        + 1
        if dangerous_steps
        else 1
    )
    reachable = {start_position: 0}

    for step in range(1, final_step + 1):
        next_reachable: dict[Position, int] = {}

        for current_position, initial_action_bits in reachable.items():
            current_x, current_y = current_position

            for action_index, (dx, dy) in enumerate(
                SURVIVAL_ACTION_DELTAS
            ):
                candidate = (
                    current_x + dx,
                    current_y + dy,
                )

                if not position_in_bounds(field, candidate):
                    continue

                if field[candidate] != 0:
                    continue

                if candidate in opponent_positions:
                    continue

                blocked_until = bomb_blocked_until.get(candidate)
                if (
                    blocked_until is not None
                    and step < blocked_until
                    and candidate != current_position
                ):
                    continue

                if step in dangerous_steps.get(candidate, ()):
                    continue

                # time steps BFS: 1 bit per action to know which start action brought agent to that position alive
                # in the 1st step bit is planted, after it is only dragged

                candidate_bits = initial_action_bits

                if step == 1:
                    candidate_bits = 1 << action_index

                next_reachable[candidate] = (
                    next_reachable.get(candidate, 0)
                    | candidate_bits
                )

        reachable = next_reachable

        if not reachable:
            return ZERO_ACTION_SAFETY_MASK

    surviving_initial_actions = 0

    for initial_action_bits in reachable.values():
        surviving_initial_actions |= initial_action_bits

    return tuple(
        int(bool(surviving_initial_actions & (1 << index)))
        for index in range(len(SURVIVAL_ACTION_DELTAS))
    )


def _shift_danger_schedule(
    dangerous_steps: dict[Position, set[int]],
    bomb_blocked_until: dict[Position, int],
    elapsed_steps: int,
) -> tuple[dict[Position, set[int]], dict[Position, int]]:
    """Move a known danger schedule to a later decision point."""
    shifted_danger = {
        position: {
            step - elapsed_steps
            for step in steps
            if step > elapsed_steps
        }
        for position, steps in dangerous_steps.items()
    }
    shifted_danger = {
        position: steps
        for position, steps in shifted_danger.items()
        if steps
    }
    shifted_blocking = {
        position: blocked_until - elapsed_steps
        for position, blocked_until in bomb_blocked_until.items()
        if blocked_until > elapsed_steps
    }

    return shifted_danger, shifted_blocking


def _add_hypothetical_bomb(
    field: np.ndarray,
    bomb_position: Position,
    dangerous_steps: dict[Position, set[int]],
    bomb_blocked_until: dict[Position, int],
) -> None:
    """Add a newly placed bomb to mutable relative-time schedules."""
    explosion_step = s.BOMB_TIMER
    bomb_blocked_until[bomb_position] = max(
        bomb_blocked_until.get(bomb_position, 0),
        explosion_step,
    )

    for blast_position in blast_coordinates(field, bomb_position):
        dangerous_steps.setdefault(blast_position, set()).update(
            (explosion_step, explosion_step + 1)
        )


def _bomb_at_arrival_schedule(
    game_state: dict,
    bomb_position: Position,
    arrival_steps: int,
    dangerous_steps: dict[Position, set[int]] | None = None,
    bomb_blocked_until: dict[Position, int] | None = None,
) -> tuple[
    dict[Position, set[int]],
    dict[Position, int],
] | None:
    """Return the post-placement schedule for a reachable candidate.

    The helper centralizes the arrival and placement checks shared by the
    unrestricted survival test and the coin-route-constrained test. Returned steps are relative to the first action after BOMB.
    """
    if dangerous_steps is None or bomb_blocked_until is None:
        dangerous_steps, bomb_blocked_until = danger_schedule(
            game_state
        )

    placement_step = arrival_steps + 1

    if (
        arrival_steps > 0
        and arrival_steps
        in dangerous_steps.get(bomb_position, ())
    ):
        return None

    if placement_step in dangerous_steps.get(bomb_position, ()):
        return None

    blocked_until = bomb_blocked_until.get(bomb_position)
    if (
        blocked_until is not None
        and placement_step < blocked_until
    ):
        return None

    shifted_danger, shifted_blocking = _shift_danger_schedule(
        dangerous_steps,
        bomb_blocked_until,
        elapsed_steps=placement_step,
    )
    _add_hypothetical_bomb(
        field=game_state["field"],
        bomb_position=bomb_position,
        dangerous_steps=shifted_danger,
        bomb_blocked_until=shifted_blocking,
    )

    return shifted_danger, shifted_blocking


def can_survive_bomb_at_arrival(
    game_state: dict,
    bomb_position: Position,
    arrival_steps: int,
    dangerous_steps: dict[Position, set[int]] | None = None,
    bomb_blocked_until: dict[Position, int] | None = None,
) -> bool:
    """Return whether placing a bomb after reaching a candidate is safe.

    Existing hazards are advanced by the BFS travel time and the bomb
    placement action before the hypothetical bomb is added. The field is
    deliberately kept static; openings created by earlier explosions are
    therefore treated conservatively and will be discovered by replanning.
    """
    schedule = _bomb_at_arrival_schedule(
        game_state=game_state,
        bomb_position=bomb_position,
        arrival_steps=arrival_steps,
        dangerous_steps=dangerous_steps,
        bomb_blocked_until=bomb_blocked_until,
    )

    if schedule is None:
        return False

    shifted_danger, shifted_blocking = schedule
    opponent_positions = {
        tuple(other[3])
        for other in game_state.get("others", ())
    }
    escape_mask = _survivable_action_mask_from_schedule(
        field=game_state["field"],
        start_position=bomb_position,
        dangerous_steps=shifted_danger,
        bomb_blocked_until=shifted_blocking,
        opponent_positions=opponent_positions,
    )

    return any(escape_mask)


def can_survive_bomb_via_coin_at_arrival(
    game_state: dict,
    bomb_position: Position,
    arrival_steps: int,
    coin_position: Position,
    coin_distances: dict[Position, int],
    dangerous_steps: dict[Position, set[int]] | None = None,
    bomb_blocked_until: dict[Position, int] | None = None,
) -> bool:
    """Return whether a bomb escape can continue through one coin.

    Before collecting the coin, movement may wait in place or traverse an
    edge that reduces the remaining static BFS distance by exactly one.
    Consequently the spatial route never detours from a shortest path to
    the coin. If the new bomb reaches the coin, the coin must be entered
    before detonation and left through a different edge than the one used
    to enter it. The complete route must survive every known danger step.
    """
    route_distance = coin_distances.get(bomb_position)

    if route_distance is None or route_distance <= 0:
        return False

    schedule = _bomb_at_arrival_schedule(
        game_state=game_state,
        bomb_position=bomb_position,
        arrival_steps=arrival_steps,
        dangerous_steps=dangerous_steps,
        bomb_blocked_until=bomb_blocked_until,
    )

    if schedule is None:
        return False

    shifted_danger, shifted_blocking = schedule
    field = game_state["field"]
    opponent_positions = {
        tuple(other[3])
        for other in game_state.get("others", ())
    }
    coin_in_new_blast = coin_position in blast_coordinates(
        field,
        bomb_position,
    )
    last_danger_step = max(
        step
        for steps in shifted_danger.values()
        for step in steps
    )
    # Waiting is allowed because it does not leave the shortest spatial route. 
    # After all hazards have passed, route_distance additional steps are sufficient to finish that route.
    final_step = last_danger_step + route_distance + 1

    # check if escape route passes through coin, not only passing and leaving
    # before_coin: agent did not catch it yet. on_blast_coin: in radius of new bomb, cross it, no backtrack. passed_coin: caught it already.

    before_coin = 0
    on_blast_coin = 1
    passed_coin = 2
    reachable: set[tuple[Position, int, Position | None]] = {
        (bomb_position, before_coin, None)
    }

    for step in range(1, final_step + 1):
        next_reachable: set[
            tuple[Position, int, Position | None]
        ] = set()

        for current_position, phase, entry_position in reachable:
            current_x, current_y = current_position

            for dx, dy in SURVIVAL_ACTION_DELTAS:
                candidate = (current_x + dx, current_y + dy)
                waiting = candidate == current_position

                if not position_in_bounds(field, candidate):
                    continue

                if field[candidate] != 0:
                    continue

                if candidate in opponent_positions:
                    continue

                blocked_until = shifted_blocking.get(candidate)
                if (
                    blocked_until is not None
                    and step < blocked_until
                    and candidate != current_position
                ):
                    continue

                if step in shifted_danger.get(candidate, ()):
                    continue

                next_phase = phase
                next_entry_position = entry_position

                if phase == before_coin:
                    if not waiting:
                        current_distance = coin_distances.get(
                            current_position
                        )
                        candidate_distance = coin_distances.get(
                            candidate
                        )

                        if (
                            current_distance is None
                            or candidate_distance
                            != current_distance - 1
                        ):
                            continue

                        if candidate == coin_position:
                            if coin_in_new_blast:
                                if step >= s.BOMB_TIMER:
                                    continue

                                next_phase = on_blast_coin
                                next_entry_position = current_position
                            else:
                                next_phase = passed_coin
                                next_entry_position = None

                elif phase == on_blast_coin:
                    # A blast-exposed coin must be crossed rather than entered and immediately left along the same edge.
                    if waiting or candidate == entry_position:
                        continue

                    next_phase = passed_coin
                    next_entry_position = None

                next_reachable.add(
                    (
                        candidate,
                        next_phase,
                        next_entry_position,
                    )
                )

        reachable = next_reachable

        if not reachable:
            return False

    return any(
        phase == passed_coin
        for _, phase, _ in reachable
    )


def navigation_data(
    field: np.ndarray,
    start_position: Position,
    blocked_positions: Iterable[Position] = (),
) -> NavigationData:
    """Return distances and shortest first-step bits on free tiles."""
    blocked = set(blocked_positions)
    distances = {start_position: 0}
    first_step_masks = {start_position: 0}
    queue = deque([start_position])

    while queue:
        current_position = queue.popleft()
        current_distance = distances[current_position]
        current_x, current_y = current_position

        for direction_index, (dx, dy) in enumerate(
            DIRECTION_DELTAS
        ):
            next_position = (
                current_x + dx,
                current_y + dy,
            )

            if not is_walkable(
                field,
                next_position,
                blocked,
            ):
                continue

            next_distance = current_distance + 1
            propagated_mask = (
                1 << direction_index
                if current_distance == 0
                else first_step_masks[current_position]
            )

            if next_position not in distances:
                distances[next_position] = next_distance
                first_step_masks[next_position] = propagated_mask
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


def nearest_target_mask_and_distance(
    targets: Iterable[Position],
    navigation: NavigationData,
) -> tuple[DirectionMask, int | None]:
    """Return nearest-target directions together with BFS distance."""
    target_positions = {
        tuple(position)
        for position in targets
    }
    distances, first_step_masks = navigation
    reachable_targets = [
        target
        for target in target_positions
        if target in distances
    ]

    if not reachable_targets:
        return ZERO_DIRECTION_MASK, None

    nearest_distance = min(
        distances[target]
        for target in reachable_targets
    )
    direction_bits = 0

    for target in reachable_targets:
        if distances[target] == nearest_distance:
            direction_bits |= first_step_masks[target]

    return (
        tuple(
            int(bool(direction_bits & (1 << index)))
            for index in range(len(DIRECTION_DELTAS))
        ),
        nearest_distance,
    )


def _safe_productive_bomb_candidates(
    game_state: dict,
    navigation: NavigationData,
    discount: float = BOMB_TARGET_DISCOUNT,
    dangerous_steps: dict[Position, set[int]] | None = None,
    bomb_blocked_until: dict[Position, int] | None = None,
) -> tuple[BombTargetCandidate, ...]:
    """Return all reachable productive targets with a generic escape."""
    if dangerous_steps is None or bomb_blocked_until is None:
        dangerous_steps, bomb_blocked_until = danger_schedule(
            game_state
        )

    field = game_state["field"]
    distances, _ = navigation
    safe_candidates = []

    for position, distance in distances.items():
        crate_yield = crates_hit_by_bomb(field, position)

        if crate_yield <= 0:
            continue

        if not can_survive_bomb_at_arrival(
            game_state=game_state,
            bomb_position=position,
            arrival_steps=distance,
            dangerous_steps=dangerous_steps,
            bomb_blocked_until=bomb_blocked_until,
        ):
            continue

        score = crate_yield * (discount ** distance)
        safe_candidates.append(
            BombTargetCandidate(
                position=position,
                distance=distance,
                crate_yield=crate_yield,
                score=score,
            )
        )

    return tuple(safe_candidates)


def _highest_ranked_bomb_candidates(
    candidates: Iterable[BombTargetCandidate],
) -> tuple[BombTargetCandidate, ...]:
    """Return all exact score-and-distance ties without spatial tie-breaks."""
    candidates = tuple(candidates)

    if not candidates:
        return ()

    best_score = max(candidate.score for candidate in candidates)
    highest_scoring = tuple(
        candidate
        for candidate in candidates
        if math.isclose(
            candidate.score,
            best_score,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    )
    best_distance = min(
        candidate.distance
        for candidate in highest_scoring
    )

    return tuple(
        candidate
        for candidate in highest_scoring
        if candidate.distance == best_distance
    )


def _bomb_target_selection_from_candidates(
    candidates: Iterable[BombTargetCandidate],
    navigation: NavigationData,
    start_position: Position,
) -> BombTargetSelection:
    """Build one mask while preserving every invariant exact tie."""
    candidates = tuple(candidates)

    if not candidates:
        return BombTargetSelection(
            mask=ZERO_BOMB_TARGET_MASK,
            distance=None,
            crate_yield=0,
            score=None,
        )

    _, first_step_masks = navigation
    direction_bits = 0
    target_here = False

    for candidate in candidates:
        direction_bits |= first_step_masks[candidate.position]
        target_here = (
            target_here
            or candidate.position == start_position
        )

    direction_mask = tuple(
        int(bool(direction_bits & (1 << index)))
        for index in range(len(DIRECTION_DELTAS))
    )

    return BombTargetSelection(
        mask=(*direction_mask, int(target_here)),
        distance=candidates[0].distance,
        crate_yield=candidates[0].crate_yield,
        score=candidates[0].score,
    )


def bomb_target_with_coin_route_alternative(
    game_state: dict,
    navigation: NavigationData,
    discount: float = BOMB_TARGET_DISCOUNT,
    dangerous_steps: dict[Position, set[int]] | None = None,
    bomb_blocked_until: dict[Position, int] | None = None,
    *,
    candidates: tuple[BombTargetCandidate, ...] | None = None,
    local_coin_survival=None,
) -> CoinRouteBombTargetSelection:
    """Add the best safe bomb target lying on a nearest-coin route.

    The ordinary global optimum is always retained. At most one additional
    score-and-distance rank is admitted, with all of that rank's exact ties
    preserved for D4 symmetry. A compatible alternative must retain at least
    50 percent of the global optimum's score, in addition to satisfying the
    route, productivity, and constrained-survival requirements.
    The planner supplies one complete robust local coin-route proof; only
    distance zero uses it. Remote candidates use the nominal forecast.
    """
    if dangerous_steps is None or bomb_blocked_until is None:
        dangerous_steps, bomb_blocked_until = danger_schedule(
            game_state
        )

    start_position = tuple(game_state["self"][3])
    distances, _ = navigation
    # Reuse the filtered candidates after a rejected local placement.
    safe_candidates = candidates
    if safe_candidates is None:
        safe_candidates = _safe_productive_bomb_candidates(
            game_state=game_state,
            navigation=navigation,
            discount=discount,
            dangerous_steps=dangerous_steps,
            bomb_blocked_until=bomb_blocked_until,
        )
    global_candidates = _highest_ranked_bomb_candidates(
        safe_candidates
    )
    global_target = _bomb_target_selection_from_candidates(
        candidates=global_candidates,
        navigation=navigation,
        start_position=start_position,
    )

    reachable_coins = tuple(
        tuple(coin)
        for coin in game_state.get("coins", ())
        if tuple(coin) in distances
    )

    if not reachable_coins or not safe_candidates:
        return CoinRouteBombTargetSelection(
            mask=global_target.mask,
            global_target=global_target,
            alternative_mask=ZERO_BOMB_TARGET_MASK,
            alternative_score_ratio_milli=None,
        )

    nearest_coin_distance = min(
        distances[coin]
        for coin in reachable_coins
    )
    nearest_coins = tuple(
        coin
        for coin in reachable_coins
        if distances[coin] == nearest_coin_distance
    )
    blocked_positions = occupied_positions(game_state)
    coin_distance_maps = {
        coin: navigation_data(
            field=game_state["field"],
            start_position=coin,
            blocked_positions=blocked_positions,
        )[0]
        for coin in nearest_coins
    }
    compatible_candidates = []

    for candidate in safe_candidates:
        for coin in nearest_coins:
            coin_distances = coin_distance_maps[coin]
            remaining_distance = coin_distances.get(
                candidate.position
            )

            # Equality proves that the candidate lies strictly before this coin on at least one globally shortest current coin route.
            if (
                remaining_distance is None
                or remaining_distance <= 0
                or candidate.distance + remaining_distance
                != nearest_coin_distance
            ):
                continue

            if candidate.distance == 0 and local_coin_survival is not None:
                survives = local_coin_survival(coin, coin_distances)
            else:
                survives = can_survive_bomb_via_coin_at_arrival(
                    game_state=game_state,
                    bomb_position=candidate.position,
                    arrival_steps=candidate.distance,
                    coin_position=coin,
                    coin_distances=coin_distances,
                    dangerous_steps=dangerous_steps,
                    bomb_blocked_until=bomb_blocked_until,
                )
            if survives:
                compatible_candidates.append(candidate)
                break

    minimum_alternative_score = (
        COIN_ROUTE_BOMB_MIN_SCORE_RATIO
        * global_candidates[0].score
    )
    quality_compatible_candidates = tuple(
        candidate
        for candidate in compatible_candidates
        if candidate.score >= minimum_alternative_score
    )
    selected_compatible = _highest_ranked_bomb_candidates(
        quality_compatible_candidates
    )
    global_positions = {
        candidate.position
        for candidate in global_candidates
    }
    alternative_candidates = tuple(
        candidate
        for candidate in selected_compatible
        if candidate.position not in global_positions
    )

    if not alternative_candidates:
        return CoinRouteBombTargetSelection(
            mask=global_target.mask,
            global_target=global_target,
            alternative_mask=ZERO_BOMB_TARGET_MASK,
            alternative_score_ratio_milli=None,
        )

    alternative_target = _bomb_target_selection_from_candidates(
        candidates=alternative_candidates,
        navigation=navigation,
        start_position=start_position,
    )
    combined_mask = tuple(
        int(bool(global_bit or alternative_bit))
        for global_bit, alternative_bit in zip(
            global_target.mask,
            alternative_target.mask,
        )
    )
    score_ratio = (
        alternative_candidates[0].score
        / global_candidates[0].score
    )

    return CoinRouteBombTargetSelection(
        mask=combined_mask,
        global_target=global_target,
        alternative_mask=alternative_target.mask,
        # saves ratio * 1000 for diagnostics, not actual state of Q-table
        alternative_score_ratio_milli=int(
            round(1000 * min(1.0, score_ratio))
        ),
    )


def coin_distance_class(coin_distance: int | None) -> int:
    """Return the data-derived coin-distance class from far to near.

    Codes:
        0: no reachable coin or distance at least seven
        1: distance five or six
        2: distance three or four
        3: distance one or two

    An empty coin mask distinguishes the canonical no-target filler ``0``
    from an available far-away target. The probe-derived active bins contain
    approximately 16%, 26%, 32%, and 25% of unique observed coin states.
    """
    if coin_distance is None:
        return 0

    if coin_distance <= COIN_DISTANCE_CLASS_BOUNDARIES[0]:
        return 3

    if coin_distance <= COIN_DISTANCE_CLASS_BOUNDARIES[1]:
        return 2

    if coin_distance <= COIN_DISTANCE_CLASS_BOUNDARIES[2]:
        return 1

    return 0


def bomb_quality_class(bomb_score: float | None) -> int:
    """Return the data-derived best-bomb quality class.

    Codes:
        0: no productive target or best score below 3.5
        1: best score from 3.5 (inclusive) to 5.0 (exclusive)
        2: best score at least 5.0

    As for coins, an empty bomb-target mask disambiguates the no-target
    filler ``0``. The two thresholds divide unique observed bomb states into
    approximately equal thirds.
    """
    if bomb_score is None:
        return 0

    medium_threshold, strong_threshold = (
        BOMB_QUALITY_CLASS_THRESHOLDS
    )

    if bomb_score < medium_threshold:
        return 0

    if bomb_score < strong_threshold:
        return 1

    return 2


def threat_urgency_from_schedule(
    dangerous_steps: dict[Position, set[int]],
    position: Position,
) -> int:
    """Return urgency from the earliest danger on the current tile.

    Codes:
        0: no known future danger
        1: earliest danger in at least three steps
        2: earliest danger in two steps
        3: danger in the next step
    """
    position_danger = dangerous_steps.get(position)

    if not position_danger:
        return 0

    earliest_danger = min(position_danger)

    if earliest_danger <= 1:
        return 3

    if earliest_danger == 2:
        return 2

    return 1
