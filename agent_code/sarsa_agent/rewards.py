from collections.abc import Callable
from dataclasses import dataclass
import math

import events as e

from .features import (
    FeatureState,
    NO_DIRECTION,
    PotentialGuidanceContext,
)
from .task2_mechanics import (
    TASK2_TARGET_CASE_BOMB_ONLY,
    TASK2_TARGET_CASE_COIN_ONLY,
    TASK2_TARGET_CASE_JOINT,
    TASK2_TARGET_CASE_LOCAL_BOMB,
    TASK2_TARGET_CASE_NONE,
    TASK2_TARGET_CASE_TRUE_CONFLICT,
    classify_safe_target_navigation,
    task2_v2_feature_data,
    task2_v3_feature_data,
)


@dataclass(frozen=True)
class RewardResult:
    """Reward and its individual event contributions."""

    total: float
    contributions: tuple[tuple[str, float], ...]


RewardFunction = Callable[
    [
        dict | None,
        str | None,
        dict | None,
        list[str],
        PotentialGuidanceContext | None,
        float,
        FeatureState,
    ],
    RewardResult,
]

BASELINE_REWARD_VERSION = "task1_baseline_v1"
WAIT_REWARD_VERSION = "task1_wait_v1"
POTENTIAL_GUIDANCE_REWARD_VERSION = (
    "task1_potential_guidance_v1"
)
BFS_POTENTIAL_GUIDANCE_REWARD_VERSION = (
    "task1_bfs_potential_guidance_v2"
)
TASK2_BASELINE_REWARD_VERSION = "task2_baseline_v1"
TASK2_BOMB_CONTROL_REWARD_VERSION = "task2_bomb_control_v2"
TASK2_HIERARCHICAL_REWARD_VERSION = (
    "task2_hierarchical_navigation_v3"
)
TASK2_HIERARCHICAL_SAFETY_REWARD_VERSION = (
    "task2_hierarchical_navigation_safety_v3_1"
)
TASK2_HIERARCHICAL_URGENCY_REWARD_VERSION = (
    "task2_hierarchical_navigation_urgency_v3_1_1"
)
TASK2_INDEPENDENT_TARGET_REWARD_VERSION = (
    "task2_hierarchical_independent_targets_v3_2"
)
TASK2_INDEPENDENT_TARGET_SCALE_4_REWARD_VERSION = (
    "task2_hierarchical_independent_targets_v3_3"
)
TASK2_INDEPENDENT_TARGET_CONTRASTIVE_REWARD_VERSION = (
    "task2_hierarchical_independent_targets_v3_4"
)
TASK2_INDEPENDENT_TARGET_HIERARCHY_REWARD_VERSION = (
    "task2_hierarchical_independent_targets_v3_5"
)
TASK2_INDEPENDENT_TARGET_CONSISTENT_REWARD_VERSION = (
    "task2_hierarchical_independent_targets_v3_6"
)
TASK2_INDEPENDENT_TARGET_STRONGER_REWARD_VERSION = (
    "task2_hierarchical_independent_targets_v3_7"
)
TASK2_INDEPENDENT_TARGET_BOMB_EMPHASIS_REWARD_VERSION = (
    "task2_hierarchical_independent_targets_v3_8"
)
TASK3_PRESSURE_KILL_REWARD_VERSION = "task3_hierarchical_pressure_kills_v3_9"
TASK3_TACTICAL_HOLD_REWARD_VERSION = "task3_hierarchical_tactical_hold_v3_10"
TASK3_TACTICAL_REWARD_V4_VERSION = "task3_hierarchical_tactical_v4"
TASK3_TACTICAL_REWARD_V5_VERSION = "task3_hierarchical_tactical_v5"
KILLED_BY_OPPONENT = "KILLED_BY_OPPONENT"
WAIT_AT_BLOCK_TARGET = "WAIT_AT_BLOCK_TARGET"
SKIPPED_BLOCK_TARGET = "SKIPPED_BLOCK_TARGET"
DEFAULT_REWARD_VERSION = WAIT_REWARD_VERSION

FOLLOWED_POTENTIAL_HINT = "FOLLOWED_POTENTIAL_HINT"
MOVED_AGAINST_POTENTIAL = "MOVED_AGAINST_POTENTIAL"
FOLLOWED_BFS_DIRECTION = "FOLLOWED_BFS_DIRECTION"
MOVED_OUTSIDE_GUIDANCE = "MOVED_OUTSIDE_GUIDANCE"
FOLLOWED_POTENTIAL_TIE_BREAK = (
    "FOLLOWED_POTENTIAL_TIE_BREAK"
)
FOLLOWED_POTENTIAL_OVERRIDE = "FOLLOWED_POTENTIAL_OVERRIDE"
BOMB_AT_SELECTED_TARGET = "BOMB_AT_SELECTED_TARGET"
BOMB_OUTSIDE_SELECTED_TARGET = "BOMB_OUTSIDE_SELECTED_TARGET"
SKIPPED_BOMB_TARGET = "SKIPPED_BOMB_TARGET"
TASK2_TARGET_NAVIGATION_SCORE = "TASK2_TARGET_NAVIGATION_SCORE"
UNSAFE_ACTION_WITH_SAFE_ALTERNATIVE = (
    "UNSAFE_ACTION_WITH_SAFE_ALTERNATIVE"
)
UNSAFE_ACTION_URGENCY_0 = "UNSAFE_ACTION_URGENCY_0"
UNSAFE_ACTION_URGENCY_1 = "UNSAFE_ACTION_URGENCY_1"
UNSAFE_ACTION_URGENCY_2 = "UNSAFE_ACTION_URGENCY_2"
UNSAFE_ACTION_URGENCY_3 = "UNSAFE_ACTION_URGENCY_3"

POTENTIAL_GUIDANCE_REWARD = 0.01
BFS_GUIDANCE_REWARD = 0.005
POTENTIAL_TIE_BREAK_REWARD = 0.005
POTENTIAL_OVERRIDE_REWARD = 0.05

MOVEMENT_DIRECTION_INDICES = {
    "UP": 0,
    "RIGHT": 1,
    "DOWN": 2,
    "LEFT": 3,
}
TASK2_NON_BOMB_ACTION_INDICES = {
    **MOVEMENT_DIRECTION_INDICES,
    "WAIT": 4,
}

TASK2_TARGET_REWARD_SCALE = 0.005
TASK2_INDEPENDENT_TARGET_REWARD_SCALE_V3_3 = 0.020
TASK2_INDEPENDENT_TARGET_REWARD_SCALE_V3_5 = 0.040
TASK2_INDEPENDENT_TARGET_REWARD_SCALE_V3_6 = 0.040
TASK2_INDEPENDENT_TARGET_REWARD_SCALE_V3_8 = (
    1.5 * TASK2_INDEPENDENT_TARGET_REWARD_SCALE_V3_6
)
TASK2_AVOIDABLE_UNSAFE_ACTION_PENALTY_V3_5 = -0.50
TASK2_TARGET_WEIGHTS = {
    0: (0.8, 0.2),
    1: (0.6, 0.4),
    2: (0.5, 0.5),
    3: (0.4, 0.6),
    4: (0.2, 0.8),
}
TASK2_COIN_DISTANCE_CLASS_STRENGTHS = (
    0.25,
    0.50,
    0.75,
    1.00,
)
TASK2_BOMB_QUALITY_CLASS_STRENGTHS = (
    1.0 / 3.0,
    2.0 / 3.0,
    1.0,
)
TASK2_UNSAFE_ACTION_PENALTIES = {
    0: (UNSAFE_ACTION_URGENCY_0, -0.02),
    1: (UNSAFE_ACTION_URGENCY_1, -0.05),
    2: (UNSAFE_ACTION_URGENCY_2, -0.10),
    3: (UNSAFE_ACTION_URGENCY_3, -0.20),
}
TASK2_SAFETY_V3_1_UNSAFE_ACTION_PENALTIES = {
    0: (UNSAFE_ACTION_URGENCY_0, -0.02),
    1: (UNSAFE_ACTION_URGENCY_1, -0.10),
    2: (UNSAFE_ACTION_URGENCY_2, -0.30),
    3: (UNSAFE_ACTION_URGENCY_3, -0.60),
}

TASK1_BASELINE_EVENT_REWARDS = {
    e.COIN_COLLECTED: 1.0,
    e.INVALID_ACTION: -0.2,
    e.MOVED_UP: -0.01,
    e.MOVED_RIGHT: -0.01,
    e.MOVED_DOWN: -0.01,
    e.MOVED_LEFT: -0.01,
}
TASK1_WAIT_EVENT_REWARDS = {
    **TASK1_BASELINE_EVENT_REWARDS,
    e.WAITED: -0.02,
}
TASK1_POTENTIAL_GUIDANCE_EVENT_REWARDS = {
    **TASK1_WAIT_EVENT_REWARDS,
    FOLLOWED_POTENTIAL_HINT: POTENTIAL_GUIDANCE_REWARD,
    MOVED_AGAINST_POTENTIAL: -POTENTIAL_GUIDANCE_REWARD,
}
TASK1_BFS_POTENTIAL_GUIDANCE_EVENT_REWARDS = {
    **TASK1_WAIT_EVENT_REWARDS,
    FOLLOWED_BFS_DIRECTION: BFS_GUIDANCE_REWARD,
    MOVED_OUTSIDE_GUIDANCE: -BFS_GUIDANCE_REWARD,
    FOLLOWED_POTENTIAL_TIE_BREAK: POTENTIAL_TIE_BREAK_REWARD,
    FOLLOWED_POTENTIAL_OVERRIDE: POTENTIAL_OVERRIDE_REWARD,
    MOVED_AGAINST_POTENTIAL: -POTENTIAL_GUIDANCE_REWARD,
}
TASK2_BASELINE_EVENT_REWARDS = {
    e.COIN_COLLECTED: 1.0,
    e.CRATE_DESTROYED: 0.2,
    e.KILLED_SELF: -1.0,
    e.INVALID_ACTION: -0.2,
    e.BOMB_DROPPED: -0.02,
    e.MOVED_UP: -0.01,
    e.MOVED_RIGHT: -0.01,
    e.MOVED_DOWN: -0.01,
    e.MOVED_LEFT: -0.01,
    e.WAITED: -0.02,
}
TASK2_BOMB_CONTROL_EVENT_REWARDS = {
    **TASK2_BASELINE_EVENT_REWARDS,
    e.CRATE_DESTROYED: 0.1,
    e.COIN_FOUND: 0.2,
    e.KILLED_SELF: -2.0,
    BOMB_AT_SELECTED_TARGET: 0.05,
    BOMB_OUTSIDE_SELECTED_TARGET: -0.4,
}
TASK2_HIERARCHICAL_EVENT_REWARDS = {
    **TASK2_BOMB_CONTROL_EVENT_REWARDS,
    e.CRATE_DESTROYED: 0.05,
}
TASK2_HIERARCHICAL_SAFETY_EVENT_REWARDS = {
    **TASK2_HIERARCHICAL_EVENT_REWARDS,
    e.KILLED_SELF: -5.0,
}
TASK2_INDEPENDENT_TARGET_HIERARCHY_EVENT_REWARDS = {
    **TASK2_HIERARCHICAL_EVENT_REWARDS,
    e.KILLED_SELF: -1.0,
    BOMB_AT_SELECTED_TARGET: 0.10,
    SKIPPED_BOMB_TARGET: -0.20,
}
TASK2_INDEPENDENT_TARGET_STRONGER_EVENT_REWARDS = {
    **TASK2_INDEPENDENT_TARGET_HIERARCHY_EVENT_REWARDS,
    e.INVALID_ACTION: -0.70,
    BOMB_AT_SELECTED_TARGET: 0.20,
}
TASK2_INDEPENDENT_TARGET_BOMB_EMPHASIS_EVENT_REWARDS = {
    **TASK2_INDEPENDENT_TARGET_STRONGER_EVENT_REWARDS,
    BOMB_AT_SELECTED_TARGET: 0.30,
    SKIPPED_BOMB_TARGET: -0.30,
}


def score_events(
    events: list[str],
    reward_values: dict[str, float],
) -> RewardResult:
    """Score all event occurrences and preserve their contributions."""
    contributions = tuple(
        (event, float(reward_values.get(event, 0.0)))
        for event in events
    )

    return RewardResult(
        total=float(sum(value for _, value in contributions)),
        contributions=contributions,
    )


def add_reward_contributions(
    reward_result: RewardResult,
    contributions: tuple[tuple[str, float], ...],
) -> RewardResult:
    """Append dynamically valued shaping contributions."""
    return RewardResult(
        total=float(
            reward_result.total
            + sum(value for _, value in contributions)
        ),
        contributions=(
            *reward_result.contributions,
            *contributions,
        ),
    )


def _task2_v2_reward_state(
    old_game_state: dict | None,
    feature_state: FeatureState,
) -> FeatureState:
    """Return the V2 feature tuple visible when the action was chosen.

    Training passes the tuple cached by ``act``. The fallback keeps direct
    reward-function calls usable without forcing normal training to repeat
    the expensive safety and target searches.
    """
    if feature_state is not None and len(feature_state) == 17:
        return feature_state

    if old_game_state is None:
        return None

    feature_data = task2_v2_feature_data(old_game_state)

    return (
        *feature_data.safety_mask,
        *feature_data.coin_mask,
        *feature_data.bomb_target_mask,
        feature_data.relative_target_distance,
        feature_data.threat_urgency,
        feature_data.bomb_available,
    )


def _task2_v3_reward_state(
    old_game_state: dict | None,
    feature_state: FeatureState,
) -> FeatureState:
    """Return the independent-target tuple visible for one action."""
    if feature_state is not None and len(feature_state) in {15, 17, 18}:
        return feature_state

    if old_game_state is None:
        return None

    feature_data = task2_v3_feature_data(old_game_state)

    return (
        *feature_data.safety_mask,
        *feature_data.coin_mask,
        *feature_data.bomb_target_mask,
        feature_data.coin_distance_class,
        feature_data.bomb_quality_class,
        feature_data.threat_urgency,
        feature_data.bomb_available,
    )


def task1_baseline_rewards_v1(
    old_game_state: dict | None,
    action: str | None,
    new_game_state: dict | None,
    events: list[str],
    guidance_context: PotentialGuidanceContext | None = None,
    discount_factor: float = 0.95,
    feature_state: FeatureState = None,
) -> RewardResult:
    """Return the original Task 1 baseline reward."""
    reward_events = list(events)

    return score_events(
        reward_events,
        TASK1_BASELINE_EVENT_REWARDS,
    )

def task1_wait_rewards_v1(
    old_game_state: dict | None,
    action: str | None,
    new_game_state: dict | None,
    events: list[str],
    guidance_context: PotentialGuidanceContext | None = None,
    discount_factor: float = 0.95,
    feature_state: FeatureState = None,
) -> RewardResult:
    """Return the Task 1 reward including a small waiting cost."""
    return score_events(
        list(events),
        TASK1_WAIT_EVENT_REWARDS,
    )


def _potential_guidance_event(
    action: str | None,
    events: list[str],
    guidance_context: PotentialGuidanceContext | None,
) -> str | None:
    """Return one balanced shaping event for a V2 potential hint.

    The cached context contains the exact hint and potential values
    that was visible when the action was selected. Following it earns a
    reward equal to one movement cost. An alternative movement is only
    penalized when it lowers the old state's coin potential; alternative
    movements that preserve or improve the potential remain neutral.

    The old coin set is used for both positions. Consequently, collecting
    a coin cannot look like a potential regression merely because the coin
    is absent from the successor state.
    """
    if (
        action not in MOVEMENT_DIRECTION_INDICES
        or e.INVALID_ACTION in events
        or guidance_context is None
    ):
        return None

    hint_code = guidance_context.hint_code

    if hint_code == NO_DIRECTION:
        return None

    direction_index = MOVEMENT_DIRECTION_INDICES[action]

    if direction_index == hint_code - 1:
        return FOLLOWED_POTENTIAL_HINT

    current_potential = guidance_context.current_potential
    chosen_potential = guidance_context.direction_potentials[
        direction_index
    ]

    if (
        current_potential is not None
        and chosen_potential is not None
        and chosen_potential < current_potential
        and not math.isclose(
            chosen_potential,
            current_potential,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    ):
        return MOVED_AGAINST_POTENTIAL

    return None


def task1_potential_guidance_rewards_v1(
    old_game_state: dict | None,
    action: str | None,
    new_game_state: dict | None,
    events: list[str],
    guidance_context: PotentialGuidanceContext | None = None,
    discount_factor: float = 0.95,
    feature_state: FeatureState = None,
) -> RewardResult:
    """Add small, balanced guidance rewards to ``task1_wait_v1``."""
    reward_events = list(events)
    guidance_event = _potential_guidance_event(
        action=action,
        events=reward_events,
        guidance_context=guidance_context,
    )

    if guidance_event is not None:
        reward_events.append(guidance_event)

    return score_events(
        reward_events,
        TASK1_POTENTIAL_GUIDANCE_EVENT_REWARDS,
    )


def _hierarchical_guidance_events(
    action: str | None,
    events: list[str],
    guidance_context: PotentialGuidanceContext | None,
) -> tuple[str, ...]:
    """Return balanced BFS and potential events for one movement.

    Every valid shortest-path movement receives a small dense reward.
    A valid movement outside the BFS mask is penalized unless it is the
    explicitly gated V2 override. Following a potential tie-break keeps
    the same total shaping reward as V1, while following an outside-BFS
    override receives a stronger, separately measurable signal.

    BFS movements are never classified as potential regressions. This
    prevents the global heuristic from contradicting the reliable local
    shortest-path guidance. An unapproved outside-BFS movement can receive
    both the navigation penalty and the potential-regression penalty.
    """
    if (
        action not in MOVEMENT_DIRECTION_INDICES
        or e.INVALID_ACTION in events
        or guidance_context is None
    ):
        return ()

    direction_index = MOVEMENT_DIRECTION_INDICES[action]
    shortest_path_mask = guidance_context.shortest_path_mask

    if not any(shortest_path_mask):
        return ()

    hint_code = guidance_context.hint_code
    hint_index = (
        hint_code - 1
        if hint_code != NO_DIRECTION
        else None
    )
    follows_hint = direction_index == hint_index
    hint_is_bfs = (
        hint_index is not None
        and bool(shortest_path_mask[hint_index])
    )
    follows_override = follows_hint and not hint_is_bfs
    follows_bfs = bool(shortest_path_mask[direction_index])

    guidance_events: list[str] = []

    if follows_bfs:
        guidance_events.append(FOLLOWED_BFS_DIRECTION)

        if follows_hint and hint_is_bfs:
            guidance_events.append(
                FOLLOWED_POTENTIAL_TIE_BREAK
            )

        return tuple(guidance_events)

    if follows_override:
        guidance_events.append(FOLLOWED_POTENTIAL_OVERRIDE)
        return tuple(guidance_events)

    guidance_events.append(MOVED_OUTSIDE_GUIDANCE)

    current_potential = guidance_context.current_potential
    chosen_potential = guidance_context.direction_potentials[
        direction_index
    ]

    if (
        current_potential is not None
        and chosen_potential is not None
        and chosen_potential < current_potential
        and not math.isclose(
            chosen_potential,
            current_potential,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    ):
        guidance_events.append(MOVED_AGAINST_POTENTIAL)

    return tuple(guidance_events)


def task1_bfs_potential_guidance_rewards_v2(
    old_game_state: dict | None,
    action: str | None,
    new_game_state: dict | None,
    events: list[str],
    guidance_context: PotentialGuidanceContext | None = None,
    discount_factor: float = 0.95,
    feature_state: FeatureState = None,
) -> RewardResult:
    """Combine dense BFS guidance with gated potential recommendations."""
    reward_events = list(events)
    reward_events.extend(
        _hierarchical_guidance_events(
            action=action,
            events=reward_events,
            guidance_context=guidance_context,
        )
    )

    return score_events(
        reward_events,
        TASK1_BFS_POTENTIAL_GUIDANCE_EVENT_REWARDS,
    )


def task2_baseline_rewards_v1(
    old_game_state: dict | None,
    action: str | None,
    new_game_state: dict | None,
    events: list[str],
    guidance_context: PotentialGuidanceContext | None = None,
    discount_factor: float = 0.95,
    feature_state: FeatureState = None,
) -> RewardResult:
    """Return the sparse, event-based Task 2 baseline reward.

    Bomb placement has a small action cost but no intrinsic positive
    reward. Productive bombs must earn their value through later crate
    destruction and coin collection. The engine emits both
    ``KILLED_SELF`` and ``GOT_KILLED`` for a suicide; only the former is
    scored here so one death is not penalized twice.
    """
    return score_events(
        list(events),
        TASK2_BASELINE_EVENT_REWARDS,
    )


def _task2_bomb_placement_event(
    old_game_state: dict | None,
    action: str | None,
    events: list[str],
    feature_state: FeatureState = None,
) -> str | None:
    """Classify one successful bomb placement against the V2 target.

    ``BOMB_DROPPED`` is the framework's confirmation that the action was
    valid. Consequently, an unavailable BOMB action keeps only its normal
    invalid-action penalty instead of also receiving an off-target penalty.
    The target is recomputed from the state in which the action was chosen,
    using exactly the same mechanics as the V2 bomb-target feature.
    """
    if (
        (old_game_state is None and feature_state is None)
        or action != "BOMB"
        or e.BOMB_DROPPED not in events
    ):
        return None

    if feature_state is not None and len(feature_state) in (15, 17, 18):
        reward_state = feature_state
    else:
        reward_state = _task2_v2_reward_state(
            old_game_state=old_game_state,
            feature_state=feature_state,
        )

    if reward_state is None:
        return None

    bomb_target_here = reward_state[13] == 1

    if bomb_target_here:
        return BOMB_AT_SELECTED_TARGET

    return BOMB_OUTSIDE_SELECTED_TARGET


def task2_bomb_control_rewards_v2(
    old_game_state: dict | None,
    action: str | None,
    new_game_state: dict | None,
    events: list[str],
    guidance_context: PotentialGuidanceContext | None = None,
    discount_factor: float = 0.95,
    feature_state: FeatureState = None,
) -> RewardResult:
    """Prioritize selected bomb sites, hidden coins, and survival.

    Relative to ``task2_baseline_v1``, this reward halves the crate value,
    rewards revealing a hidden coin, doubles the suicide penalty, and adds
    an immediate classification of every successful bomb placement. The
    existing bomb action cost remains active in both placement cases.
    """
    reward_events = list(events)
    bomb_placement_event = _task2_bomb_placement_event(
        old_game_state=old_game_state,
        action=action,
        events=reward_events,
        feature_state=feature_state,
    )

    if bomb_placement_event is not None:
        reward_events.append(bomb_placement_event)

    return score_events(
        reward_events,
        TASK2_BOMB_CONTROL_EVENT_REWARDS,
    )


def _task2_safety_penalty_contribution(
    action: str | None,
    events: list[str],
    feature_state: FeatureState,
    unsafe_action_penalties: dict[
        int,
        tuple[str, float],
    ] = TASK2_UNSAFE_ACTION_PENALTIES,
    threat_urgency_index: int = 15,
) -> tuple[str, float] | None:
    """Penalize a legal unsafe action only when safety was attainable."""
    if (
        feature_state is None
        or action not in TASK2_NON_BOMB_ACTION_INDICES
        or e.INVALID_ACTION in events
    ):
        return None

    safety_mask = feature_state[0:5]

    if not any(safety_mask):
        return None

    action_index = TASK2_NON_BOMB_ACTION_INDICES[action]

    if safety_mask[action_index]:
        return None

    threat_urgency = int(feature_state[threat_urgency_index])
    return unsafe_action_penalties[threat_urgency]


def _task2_local_bomb_opportunity(
    feature_state: FeatureState,
) -> bool:
    """Return whether the V3 state calls for an immediate bomb drop."""
    return bool(
        feature_state is not None
        and len(feature_state) in {15, 17, 18}
        and feature_state[13] == 1
        and feature_state[-1]
    )


def _task2_skipped_bomb_target_event(
    action: str | None,
    feature_state: FeatureState,
) -> str | None:
    """Classify a non-bomb action at a currently usable bomb target."""
    if (
        action in TASK2_NON_BOMB_ACTION_INDICES
        and _task2_local_bomb_opportunity(feature_state)
    ):
        return SKIPPED_BOMB_TARGET

    return None


def _task2_constant_safety_penalty_contribution(
    action: str | None,
    feature_state: FeatureState,
) -> tuple[str, float] | None:
    """Apply one urgency-independent penalty to an avoidable unsafe action."""
    if (
        feature_state is None
        or action not in TASK2_NON_BOMB_ACTION_INDICES
    ):
        return None

    safety_mask = feature_state[0:5]

    if not any(safety_mask):
        return None

    action_index = TASK2_NON_BOMB_ACTION_INDICES[action]

    if safety_mask[action_index]:
        return None

    return (
        UNSAFE_ACTION_WITH_SAFE_ALTERNATIVE,
        TASK2_AVOIDABLE_UNSAFE_ACTION_PENALTY_V3_5,
    )


def _task2_target_score_contribution(
    action: str | None,
    events: list[str],
    feature_state: FeatureState,
) -> tuple[str, float] | None:
    """Score one safe movement against weighted coin and bomb masks.

    The reward is bounded by ``TASK2_TARGET_REWARD_SCALE``. Waiting,
    bombing, invalid actions, unsafe movements, and states without a safe
    target direction receive no target contribution.
    """
    if (
        feature_state is None
        or action not in MOVEMENT_DIRECTION_INDICES
        or e.INVALID_ACTION in events
    ):
        return None

    action_index = MOVEMENT_DIRECTION_INDICES[action]
    safety_mask = feature_state[0:5]

    if not safety_mask[action_index]:
        return None

    coin_mask = feature_state[5:9]
    bomb_direction_mask = feature_state[9:13]
    safe_coin_mask = tuple(
        int(bool(safety_mask[index] and coin_mask[index]))
        for index in range(4)
    )
    safe_bomb_mask = tuple(
        int(
            bool(
                safety_mask[index]
                and bomb_direction_mask[index]
            )
        )
        for index in range(4)
    )
    has_safe_coin = any(safe_coin_mask)
    has_safe_bomb = any(safe_bomb_mask)

    if not has_safe_coin and not has_safe_bomb:
        return None

    if has_safe_coin and has_safe_bomb:
        relative_target_distance = int(feature_state[14])
        coin_weight, bomb_weight = TASK2_TARGET_WEIGHTS[
            relative_target_distance
        ]
    elif has_safe_coin:
        coin_weight, bomb_weight = 1.0, 0.0
    else:
        coin_weight, bomb_weight = 0.0, 1.0

    target_score = (
        coin_weight * safe_coin_mask[action_index]
        + bomb_weight * safe_bomb_mask[action_index]
    )
    target_reward = TASK2_TARGET_REWARD_SCALE * (
        2.0 * target_score - 1.0
    )

    return (
        TASK2_TARGET_NAVIGATION_SCORE,
        float(target_reward),
    )


def _task2_independent_target_score_contribution(
    action: str | None,
    events: list[str],
    feature_state: FeatureState,
    target_reward_scale: float = TASK2_TARGET_REWARD_SCALE,
    contrastive_target_adherence: bool = False,
) -> tuple[str, float] | None:
    """Score class-conditioned adherence to safely available targets.

    The original mode preserves V3.2/V3.3 exactly: when both objectives are
    available, their mean strength controls only the magnitude, and either
    side of a true conflict receives the same positive reward. Contrastive
    mode independently signs each available objective by whether the selected
    movement follows its safe mask, then averages those contributions. Thus a
    joint movement receives the mean positive strength, a movement following
    only one target receives half the strength difference, and a movement
    following neither receives the negative mean strength. The complete term
    remains inside the configured signed reward scale in both modes.
    """
    if (
        feature_state is None
        or action not in MOVEMENT_DIRECTION_INDICES
        or e.INVALID_ACTION in events
    ):
        return None

    action_index = MOVEMENT_DIRECTION_INDICES[action]
    safety_mask = feature_state[0:5]

    if not safety_mask[action_index]:
        return None

    coin_mask = feature_state[5:9]
    bomb_direction_mask = feature_state[9:13]
    safe_coin_mask = tuple(
        int(bool(safety_mask[index] and coin_mask[index]))
        for index in range(4)
    )
    safe_bomb_mask = tuple(
        int(
            bool(
                safety_mask[index]
                and bomb_direction_mask[index]
            )
        )
        for index in range(4)
    )
    has_safe_coin = any(safe_coin_mask)
    has_safe_bomb = any(safe_bomb_mask)

    if not has_safe_coin and not has_safe_bomb:
        return None

    coin_strength = None
    bomb_strength = None

    if has_safe_coin:
        coin_distance_class = int(feature_state[14])
        coin_strength = TASK2_COIN_DISTANCE_CLASS_STRENGTHS[
            coin_distance_class
        ]

    if has_safe_bomb:
        bomb_quality_class = int(feature_state[15])
        bomb_strength = TASK2_BOMB_QUALITY_CLASS_STRENGTHS[
            bomb_quality_class
        ]

    follows_coin = bool(safe_coin_mask[action_index])
    follows_bomb = bool(safe_bomb_mask[action_index])

    if contrastive_target_adherence:
        signed_target_strengths = []

        if has_safe_coin:
            signed_target_strengths.append(
                coin_strength
                if follows_coin
                else -coin_strength
            )

        if has_safe_bomb:
            signed_target_strengths.append(
                bomb_strength
                if follows_bomb
                else -bomb_strength
            )

        target_reward = target_reward_scale * (
            sum(signed_target_strengths)
            / len(signed_target_strengths)
        )
    elif has_safe_coin and has_safe_bomb:
        target_strength = (coin_strength + bomb_strength) / 2.0
        has_safe_joint = any(
            coin_bit and bomb_bit
            for coin_bit, bomb_bit in zip(
                safe_coin_mask,
                safe_bomb_mask,
            )
        )

        if has_safe_joint:
            if follows_coin and follows_bomb:
                adherence_score = 1.0
            elif follows_coin or follows_bomb:
                adherence_score = 0.0
            else:
                adherence_score = -1.0
        else:
            # This is a true conflict: reward using a safe target, but let
            # long-term environment returns learn which objective wins.
            adherence_score = (
                1.0
                if follows_coin or follows_bomb
                else -1.0
            )
    elif has_safe_coin:
        target_strength = coin_strength
        adherence_score = 1.0 if follows_coin else -1.0
    else:
        target_strength = bomb_strength
        adherence_score = 1.0 if follows_bomb else -1.0

    if not contrastive_target_adherence:
        target_reward = (
            target_reward_scale
            * target_strength
            * adherence_score
        )

    return (
        TASK2_TARGET_NAVIGATION_SCORE,
        float(target_reward),
    )


def _task2_hierarchical_target_score_contribution_v3_5(
    action: str | None,
    events: list[str],
    feature_state: FeatureState,
) -> tuple[str, float] | None:
    """Score safe V3 target navigation with an explicit case hierarchy.

    Uncontested single-objective and joint-target states use fixed feedback
    because no coin-versus-bomb decision has to be learned there. Independent
    class strengths are used only in a true conflict, where they define the
    relative evidence for the two mutually exclusive objectives. A currently
    usable local bomb target has precedence over navigation and is scored by
    the bomb-opportunity events instead.
    """
    if (
        feature_state is None
        or action not in MOVEMENT_DIRECTION_INDICES
        or e.INVALID_ACTION in events
        or _task2_local_bomb_opportunity(feature_state)
    ):
        return None

    action_index = MOVEMENT_DIRECTION_INDICES[action]
    safety_mask = feature_state[0:5]

    if not safety_mask[action_index]:
        return None

    target_case = classify_safe_target_navigation(
        safety_mask=safety_mask,
        coin_mask=feature_state[5:9],
        bomb_target_mask=feature_state[9:14],
        local_bomb_opportunity=(
            _task2_local_bomb_opportunity(feature_state)
        ),
    )

    if target_case.name in {
        TASK2_TARGET_CASE_NONE,
        TASK2_TARGET_CASE_LOCAL_BOMB,
    }:
        return None

    safe_coin_mask = target_case.safe_coin_mask
    safe_bomb_mask = target_case.safe_bomb_mask

    follows_coin = bool(safe_coin_mask[action_index])
    follows_bomb = bool(safe_bomb_mask[action_index])
    target_scale = TASK2_INDEPENDENT_TARGET_REWARD_SCALE_V3_5

    if target_case.name == TASK2_TARGET_CASE_JOINT:
        if follows_coin and follows_bomb:
            target_reward = target_scale
        elif follows_coin or follows_bomb:
            target_reward = 0.0
        else:
            target_reward = -target_scale
    elif target_case.name == TASK2_TARGET_CASE_TRUE_CONFLICT:
        coin_distance_class = int(feature_state[14])
        bomb_quality_class = int(feature_state[15])
        coin_strength = TASK2_COIN_DISTANCE_CLASS_STRENGTHS[
            coin_distance_class
        ]
        bomb_strength = TASK2_BOMB_QUALITY_CLASS_STRENGTHS[
            bomb_quality_class
        ]
        signed_coin_strength = (
            coin_strength
            if follows_coin
            else -coin_strength
        )
        signed_bomb_strength = (
            bomb_strength
            if follows_bomb
            else -bomb_strength
        )
        target_reward = 0.5 * target_scale * (
            signed_coin_strength + signed_bomb_strength
        )
    elif target_case.name == TASK2_TARGET_CASE_COIN_ONLY:
        target_reward = (
            target_scale
            if follows_coin
            else -target_scale
        )
    elif target_case.name == TASK2_TARGET_CASE_BOMB_ONLY:
        target_reward = (
            target_scale
            if follows_bomb
            else -target_scale
        )
    else:
        raise RuntimeError(
            f"Unsupported target case {target_case.name!r}."
        )

    return (
        TASK2_TARGET_NAVIGATION_SCORE,
        float(target_reward),
    )


def _task2_hierarchical_target_score_contribution_v3_6(
    action: str | None,
    events: list[str],
    feature_state: FeatureState,
    single_joint_target_reward_scale: float = (
        TASK2_INDEPENDENT_TARGET_REWARD_SCALE_V3_6
    ),
) -> tuple[str, float] | None:
    """Score one safe navigation action without choosing a conflict side.

    Single-objective and joint cases use the supplied fixed three-level target
    scale. In a true conflict, following either disjoint target is neutral and
    only ignoring both targets is penalized at V3.6's unchanged base scale.
    Coin-distance and bomb-quality classes therefore remain available to the
    Q-state without being treated as commensurable immediate utilities. WAIT
    participates as a safe action that follows neither directional target.
    """
    if (
        feature_state is None
        or action not in TASK2_NON_BOMB_ACTION_INDICES
        or e.INVALID_ACTION in events
        or _task2_local_bomb_opportunity(feature_state)
    ):
        return None

    action_index = TASK2_NON_BOMB_ACTION_INDICES[action]
    safety_mask = feature_state[0:5]

    if not safety_mask[action_index]:
        return None

    target_case = classify_safe_target_navigation(
        safety_mask=safety_mask,
        coin_mask=feature_state[5:9],
        bomb_target_mask=feature_state[9:14],
        local_bomb_opportunity=False,
    )

    if target_case.name == TASK2_TARGET_CASE_NONE:
        return None

    follows_coin = bool(
        action_index < 4
        and target_case.safe_coin_mask[action_index]
    )
    follows_bomb = bool(
        action_index < 4
        and target_case.safe_bomb_mask[action_index]
    )
    target_scale = TASK2_INDEPENDENT_TARGET_REWARD_SCALE_V3_6

    if target_case.name == TASK2_TARGET_CASE_JOINT:
        if follows_coin and follows_bomb:
            target_reward = single_joint_target_reward_scale
        elif follows_coin or follows_bomb:
            target_reward = 0.0
        else:
            target_reward = -single_joint_target_reward_scale
    elif target_case.name == TASK2_TARGET_CASE_TRUE_CONFLICT:
        target_reward = (
            0.0
            if follows_coin or follows_bomb
            else -target_scale
        )
    elif target_case.name == TASK2_TARGET_CASE_COIN_ONLY:
        target_reward = (
            single_joint_target_reward_scale
            if follows_coin
            else -single_joint_target_reward_scale
        )
    elif target_case.name == TASK2_TARGET_CASE_BOMB_ONLY:
        target_reward = (
            single_joint_target_reward_scale
            if follows_bomb
            else -single_joint_target_reward_scale
        )
    else:
        raise RuntimeError(
            f"Unsupported target case {target_case.name!r}."
        )

    return (
        TASK2_TARGET_NAVIGATION_SCORE,
        float(target_reward),
    )


def task2_hierarchical_navigation_rewards_v3(
    old_game_state: dict | None,
    action: str | None,
    new_game_state: dict | None,
    events: list[str],
    guidance_context: PotentialGuidanceContext | None = None,
    discount_factor: float = 0.95,
    feature_state: FeatureState = None,
    *,
    event_rewards: dict[
        str,
        float,
    ] = TASK2_HIERARCHICAL_EVENT_REWARDS,
    unsafe_action_penalties: dict[
        int,
        tuple[str, float],
    ] = TASK2_UNSAFE_ACTION_PENALTIES,
) -> RewardResult:
    """Gate compact target shaping through action survivability.

    Reward V3 preserves V2 bomb-site control, halves the crate reward once
    more, penalizes avoidable unsafe actions by threat urgency, and adds a
    normalized coin/bomb navigation score for safe movements. Joint target
    movements obtain the maximum target reward automatically.
    """
    reward_events = list(events)
    reward_state = _task2_v2_reward_state(
        old_game_state=old_game_state,
        feature_state=feature_state,
    )
    bomb_placement_event = _task2_bomb_placement_event(
        old_game_state=old_game_state,
        action=action,
        events=reward_events,
        feature_state=reward_state,
    )

    if bomb_placement_event is not None:
        reward_events.append(bomb_placement_event)

    dynamic_contributions = tuple(
        contribution
        for contribution in (
            _task2_safety_penalty_contribution(
                action=action,
                events=reward_events,
                feature_state=reward_state,
                unsafe_action_penalties=(
                    unsafe_action_penalties
                ),
            ),
            _task2_target_score_contribution(
                action=action,
                events=reward_events,
                feature_state=reward_state,
            ),
        )
        if contribution is not None
    )
    event_result = score_events(
        reward_events,
        event_rewards,
    )

    return add_reward_contributions(
        event_result,
        dynamic_contributions,
    )


def task2_hierarchical_navigation_safety_rewards_v3_1(
    old_game_state: dict | None,
    action: str | None,
    new_game_state: dict | None,
    events: list[str],
    guidance_context: PotentialGuidanceContext | None = None,
    discount_factor: float = 0.95,
    feature_state: FeatureState = None,
) -> RewardResult:
    """Strengthen only V3's avoidable-danger and suicide costs.

    V3.1 is a controlled safety ablation. It preserves V3's crate, coin,
    bomb-site, movement, waiting, and target-navigation rewards exactly.
    Only the terminal self-kill penalty and urgency-conditioned penalties
    for legal unsafe Movement/WAIT actions are increased.
    """
    return task2_hierarchical_navigation_rewards_v3(
        old_game_state=old_game_state,
        action=action,
        new_game_state=new_game_state,
        events=events,
        guidance_context=guidance_context,
        discount_factor=discount_factor,
        feature_state=feature_state,
        event_rewards=(
            TASK2_HIERARCHICAL_SAFETY_EVENT_REWARDS
        ),
        unsafe_action_penalties=(
            TASK2_SAFETY_V3_1_UNSAFE_ACTION_PENALTIES
        ),
    )


def task2_hierarchical_navigation_urgency_rewards_v3_1_1(
    old_game_state: dict | None,
    action: str | None,
    new_game_state: dict | None,
    events: list[str],
    guidance_context: PotentialGuidanceContext | None = None,
    discount_factor: float = 0.95,
    feature_state: FeatureState = None,
) -> RewardResult:
    """Test stronger urgency costs without V3.1's suicide increase.

    V3.1.1 restores V3's ``KILLED_SELF`` penalty of -2.0 while retaining
    V3.1's urgency-conditioned penalties. All other rewards are identical
    to V3 and V3.1.
    """
    return task2_hierarchical_navigation_rewards_v3(
        old_game_state=old_game_state,
        action=action,
        new_game_state=new_game_state,
        events=events,
        guidance_context=guidance_context,
        discount_factor=discount_factor,
        feature_state=feature_state,
        event_rewards=TASK2_HIERARCHICAL_EVENT_REWARDS,
        unsafe_action_penalties=(
            TASK2_SAFETY_V3_1_UNSAFE_ACTION_PENALTIES
        ),
    )


def task2_hierarchical_independent_target_rewards_v3_2(
    old_game_state: dict | None,
    action: str | None,
    new_game_state: dict | None,
    events: list[str],
    guidance_context: PotentialGuidanceContext | None = None,
    discount_factor: float = 0.95,
    feature_state: FeatureState = None,
    *,
    target_reward_scale: float = TASK2_TARGET_REWARD_SCALE,
    contrastive_target_adherence: bool = False,
) -> RewardResult:
    """Replace only V3.1.1's relative target-navigation shaping.

    Outcome rewards, bomb-site control, movement costs, the -2 self-kill
    penalty, and V3.1's stronger urgency penalties are unchanged. The sole
    conceptual replacement is the target term: independent data-derived
    coin-distance and bomb-quality classes now scale mask adherence without
    constructing a relative priority feature or preferring one side of a true
    conflict.
    """
    reward_events = list(events)
    reward_state = _task2_v3_reward_state(
        old_game_state=old_game_state,
        feature_state=feature_state,
    )
    bomb_placement_event = _task2_bomb_placement_event(
        old_game_state=old_game_state,
        action=action,
        events=reward_events,
        feature_state=reward_state,
    )

    if bomb_placement_event is not None:
        reward_events.append(bomb_placement_event)

    dynamic_contributions = tuple(
        contribution
        for contribution in (
            _task2_safety_penalty_contribution(
                action=action,
                events=reward_events,
                feature_state=reward_state,
                unsafe_action_penalties=(
                    TASK2_SAFETY_V3_1_UNSAFE_ACTION_PENALTIES
                ),
                threat_urgency_index=16,
            ),
            _task2_independent_target_score_contribution(
                action=action,
                events=reward_events,
                feature_state=reward_state,
                target_reward_scale=target_reward_scale,
                contrastive_target_adherence=(
                    contrastive_target_adherence
                ),
            ),
        )
        if contribution is not None
    )
    event_result = score_events(
        reward_events,
        TASK2_HIERARCHICAL_EVENT_REWARDS,
    )

    return add_reward_contributions(
        event_result,
        dynamic_contributions,
    )


def task2_hierarchical_independent_target_rewards_v3_3(
    old_game_state: dict | None,
    action: str | None,
    new_game_state: dict | None,
    events: list[str],
    guidance_context: PotentialGuidanceContext | None = None,
    discount_factor: float = 0.95,
    feature_state: FeatureState = None,
) -> RewardResult:
    """Strengthen only V3.2's independent target-adherence term.

    V3.3 raises the target scale from 0.005 to 0.020. Class strengths,
    true-conflict neutrality, outcome and bomb-site rewards, movement costs,
    the self-kill penalty, and urgency-conditioned safety penalties remain
    identical to V3.2.
    """
    return task2_hierarchical_independent_target_rewards_v3_2(
        old_game_state=old_game_state,
        action=action,
        new_game_state=new_game_state,
        events=events,
        guidance_context=guidance_context,
        discount_factor=discount_factor,
        feature_state=feature_state,
        target_reward_scale=(
            TASK2_INDEPENDENT_TARGET_REWARD_SCALE_V3_3
        ),
    )


def task2_hierarchical_independent_target_rewards_v3_4(
    old_game_state: dict | None,
    action: str | None,
    new_game_state: dict | None,
    events: list[str],
    guidance_context: PotentialGuidanceContext | None = None,
    discount_factor: float = 0.95,
    feature_state: FeatureState = None,
) -> RewardResult:
    """Correct V3.3's aggregation of two available target objectives.

    V3.4 keeps V3.3's 0.020 scale and every non-target reward unchanged.
    Each safely available objective now contributes its own signed ordinal
    strength, and the available contributions are averaged. Consequently,
    joint and true-conflict states distinguish coin-only from bomb-only
    movements by their independent class strengths without increasing the
    target term's signed bound.
    """
    return task2_hierarchical_independent_target_rewards_v3_2(
        old_game_state=old_game_state,
        action=action,
        new_game_state=new_game_state,
        events=events,
        guidance_context=guidance_context,
        discount_factor=discount_factor,
        feature_state=feature_state,
        target_reward_scale=(
            TASK2_INDEPENDENT_TARGET_REWARD_SCALE_V3_3
        ),
        contrastive_target_adherence=True,
    )


def task2_hierarchical_independent_target_rewards_v3_5(
    old_game_state: dict | None,
    action: str | None,
    new_game_state: dict | None,
    events: list[str],
    guidance_context: PotentialGuidanceContext | None = None,
    discount_factor: float = 0.95,
    feature_state: FeatureState = None,
) -> RewardResult:
    """Separate safety, local bombing, and target-choice supervision.

    V3.5 preserves V3.4's feature representation and all ordinary outcome
    and action rewards. It uses fixed +/-0.040 target feedback whenever no
    objective choice is necessary, retaining class-conditioned contrast only
    for true conflicts. Local usable bomb targets take precedence: dropping
    there earns +0.10 while a non-bomb action earns -0.20. Avoidable unsafe
    Movement/WAIT actions receive a constant -0.50 irrespective of urgency,
    and the terminal self-kill penalty is reduced to -1.0.
    """
    reward_events = list(events)
    reward_state = _task2_v3_reward_state(
        old_game_state=old_game_state,
        feature_state=feature_state,
    )
    bomb_placement_event = _task2_bomb_placement_event(
        old_game_state=old_game_state,
        action=action,
        events=reward_events,
        feature_state=reward_state,
    )

    if bomb_placement_event is not None:
        reward_events.append(bomb_placement_event)

    skipped_bomb_event = _task2_skipped_bomb_target_event(
        action=action,
        feature_state=reward_state,
    )

    if skipped_bomb_event is not None:
        reward_events.append(skipped_bomb_event)

    dynamic_contributions = tuple(
        contribution
        for contribution in (
            _task2_constant_safety_penalty_contribution(
                action=action,
                feature_state=reward_state,
            ),
            _task2_hierarchical_target_score_contribution_v3_5(
                action=action,
                events=reward_events,
                feature_state=reward_state,
            ),
        )
        if contribution is not None
    )
    event_result = score_events(
        reward_events,
        TASK2_INDEPENDENT_TARGET_HIERARCHY_EVENT_REWARDS,
    )

    return add_reward_contributions(
        event_result,
        dynamic_contributions,
    )


def _task2_hierarchical_independent_target_rewards_consistent(
    old_game_state: dict | None,
    action: str | None,
    events: list[str],
    feature_state: FeatureState,
    event_rewards: dict[str, float],
    single_joint_target_reward_scale: float = (
        TASK2_INDEPENDENT_TARGET_REWARD_SCALE_V3_6
    ),
) -> RewardResult:
    """Apply the shared mutually exclusive V3.6--V3.8 hierarchy."""
    reward_events = list(events)
    reward_state = _task2_v3_reward_state(
        old_game_state=old_game_state,
        feature_state=feature_state,
    )
    dynamic_contributions = []

    if e.INVALID_ACTION not in reward_events:
        safety_contribution = (
            _task2_constant_safety_penalty_contribution(
                action=action,
                feature_state=reward_state,
            )
        )

        if safety_contribution is not None:
            dynamic_contributions.append(safety_contribution)
        else:
            local_bomb_opportunity = (
                _task2_local_bomb_opportunity(reward_state)
            )
            bomb_placement_event = _task2_bomb_placement_event(
                old_game_state=old_game_state,
                action=action,
                events=reward_events,
                feature_state=reward_state,
            )

            if local_bomb_opportunity:
                if bomb_placement_event == BOMB_AT_SELECTED_TARGET:
                    reward_events.append(bomb_placement_event)
                elif action in TASK2_NON_BOMB_ACTION_INDICES:
                    reward_events.append(SKIPPED_BOMB_TARGET)
            elif bomb_placement_event is not None:
                reward_events.append(bomb_placement_event)
            else:
                target_contribution = (
                    _task2_hierarchical_target_score_contribution_v3_6(
                        action=action,
                        events=reward_events,
                        feature_state=reward_state,
                        single_joint_target_reward_scale=(
                            single_joint_target_reward_scale
                        ),
                    )
                )

                if target_contribution is not None:
                    dynamic_contributions.append(
                        target_contribution
                    )

    event_result = score_events(
        reward_events,
        event_rewards,
    )

    return add_reward_contributions(
        event_result,
        tuple(dynamic_contributions),
    )


def task2_hierarchical_independent_target_rewards_v3_6(
    old_game_state: dict | None,
    action: str | None,
    new_game_state: dict | None,
    events: list[str],
    guidance_context: PotentialGuidanceContext | None = None,
    discount_factor: float = 0.95,
    feature_state: FeatureState = None,
) -> RewardResult:
    """Apply V3.6's mutually exclusive decision-shaping hierarchy."""
    return _task2_hierarchical_independent_target_rewards_consistent(
        old_game_state=old_game_state,
        action=action,
        events=events,
        feature_state=feature_state,
        event_rewards=(
            TASK2_INDEPENDENT_TARGET_HIERARCHY_EVENT_REWARDS
        ),
    )


def task2_hierarchical_independent_target_rewards_v3_7(
    old_game_state: dict | None,
    action: str | None,
    new_game_state: dict | None,
    events: list[str],
    guidance_context: PotentialGuidanceContext | None = None,
    discount_factor: float = 0.95,
    feature_state: FeatureState = None,
) -> RewardResult:
    """Strengthen only invalid-action and correct bomb-site feedback.

    V3.7 preserves V3.6's hierarchy and all other reward values. A single
    invalid action now receives -0.70 directly, without auxiliary penalties,
    and a successful bomb at the selected local BFS target receives +0.20.
    """
    return _task2_hierarchical_independent_target_rewards_consistent(
        old_game_state=old_game_state,
        action=action,
        events=events,
        feature_state=feature_state,
        event_rewards=TASK2_INDEPENDENT_TARGET_STRONGER_EVENT_REWARDS,
    )


def task2_hierarchical_independent_target_rewards_v3_8(
    old_game_state: dict | None,
    action: str | None,
    new_game_state: dict | None,
    events: list[str],
    guidance_context: PotentialGuidanceContext | None = None,
    discount_factor: float = 0.95,
    feature_state: FeatureState = None,
) -> RewardResult:
    """Strengthen bomb decisions and non-conflict target navigation.

    V3.8 preserves V3.7's hierarchy. Dropping a bomb at a selected local target
    receives +0.30, while skipping such an opportunity receives -0.30. Fixed
    single-objective and joint navigation feedback is multiplied by 1.5 from
    +/-0.040 to +/-0.060; true-conflict feedback remains unchanged.
    """
    return _task2_hierarchical_independent_target_rewards_consistent(
        old_game_state=old_game_state,
        action=action,
        events=events,
        feature_state=feature_state,
        event_rewards=(
            TASK2_INDEPENDENT_TARGET_BOMB_EMPHASIS_EVENT_REWARDS
        ),
        single_joint_target_reward_scale=(
            TASK2_INDEPENDENT_TARGET_REWARD_SCALE_V3_8
        ),
    )


def task3_pressure_kill_rewards_v3_9(
    old_game_state: dict | None,
    action: str | None,
    new_game_state: dict | None,
    events: list[str],
    guidance_context: PotentialGuidanceContext | None = None,
    discount_factor: float = 0.95,
    feature_state: FeatureState = None,
) -> RewardResult:
    """V3.8 shaping plus +5 for each engine-attributed own-bomb opponent kill.

    Kills are delayed outcomes, so credit is independent of the action taken
    on the kill step. KILLED_SELF, GOT_KILLED and OPPONENT_ELIMINATED are not
    credited as kills. The engine supplies KILLED_OPPONENT to the bomb owner.
    """
    if feature_state is None and old_game_state is not None:
        raise ValueError("Reward V3.9 requires the actual feature_state used for the action.")
    return _task2_hierarchical_independent_target_rewards_consistent(
        old_game_state=old_game_state, action=action, events=events,
        feature_state=feature_state,
        event_rewards={**TASK2_INDEPENDENT_TARGET_BOMB_EMPHASIS_EVENT_REWARDS,
                       e.KILLED_OPPONENT: 5.0},
        single_joint_target_reward_scale=TASK2_INDEPENDENT_TARGET_REWARD_SCALE_V3_8,
    )


def task3_tactical_hold_rewards_v3_10(
    old_game_state: dict | None,
    action: str | None,
    new_game_state: dict | None,
    events: list[str],
    guidance_context: PotentialGuidanceContext | None = None,
    discount_factor: float = 0.95,
    feature_state: FeatureState = None,
) -> RewardResult:
    """V3.9 plus +/-0.30 for the confirmed local WAIT target in V9.

    Invalid actions and avoidable unsafe moves retain their existing priority.
    At a safe HOLD target, WAIT replaces local bomb/navigation shaping; any
    other valid, safe action skips that target. Ordinary action costs and
    outcomes (including delayed +5 own-bomb kills) remain unchanged. A target
    value of 2 never means that BOMB was recommended.
    """
    if feature_state is None or feature_state[13] != 2:
        return task3_pressure_kill_rewards_v3_9(
            old_game_state, action, new_game_state, events,
            guidance_context, discount_factor, feature_state)
    reward_events = list(events)
    contributions = ()
    if e.INVALID_ACTION not in reward_events:
        safety = _task2_constant_safety_penalty_contribution(action, feature_state)
        if safety is not None:
            contributions = (safety,)
        elif feature_state[4]:
            if action == "WAIT" and e.WAITED in reward_events:
                reward_events.append(WAIT_AT_BLOCK_TARGET)
            elif action in (*TASK2_NON_BOMB_ACTION_INDICES, "BOMB") and action != "WAIT":
                reward_events.append(SKIPPED_BLOCK_TARGET)
    result = score_events(reward_events, {
        **TASK2_INDEPENDENT_TARGET_BOMB_EMPHASIS_EVENT_REWARDS,
        e.KILLED_OPPONENT: 5.0,
        WAIT_AT_BLOCK_TARGET: 0.30,
        SKIPPED_BLOCK_TARGET: -0.30,
    })
    return add_reward_contributions(result, contributions)


def task3_tactical_rewards_v4(
    old_game_state: dict | None,
    action: str | None,
    new_game_state: dict | None,
    events: list[str],
    guidance_context: PotentialGuidanceContext | None = None,
    discount_factor: float = 0.95,
    feature_state: FeatureState = None,
) -> RewardResult:
    """V3.10 shaping and kills, plus exactly one -1 penalty per own death.

    GOT_KILLED also accompanies KILLED_SELF. Simultaneous own/foreign hits
    must therefore not add two death penalties. Engine events stay untouched;
    the derived event uses the ordinary reward-contribution recording path.
    """
    reward_events = list(events)
    if e.KILLED_SELF in reward_events:
        reward_events = [event for event in reward_events if event != e.KILLED_SELF]
        reward_events.append(e.KILLED_SELF)
    result = task3_tactical_hold_rewards_v3_10(
        old_game_state, action, new_game_state, reward_events,
        guidance_context, discount_factor, feature_state)
    if e.GOT_KILLED in events and e.KILLED_SELF not in events:
        death = score_events([KILLED_BY_OPPONENT], {KILLED_BY_OPPONENT: -1.0})
        result = add_reward_contributions(result, death.contributions)
    return result


def task3_tactical_rewards_v5(
    old_game_state: dict | None,
    action: str | None,
    new_game_state: dict | None,
    events: list[str],
    guidance_context: PotentialGuidanceContext | None = None,
    discount_factor: float = 0.95,
    feature_state: FeatureState = None,
) -> RewardResult:
    """V4 with +/-1 feedback only at a currently certified safe HOLD.

    Successful WAIT receives +1; every non-WAIT action, including BOMB,
    receives -1. Unsafe/invalid actions retain their additional V4 penalties.
    Replace the old HOLD contribution instead of double-counting it. All
    outcome rewards, death handling and non-HOLD behavior remain exactly V4.
    The action's actual feature key supplies the proof; no replanning occurs.
    """
    result = task3_tactical_rewards_v4(
        old_game_state, action, new_game_state, events,
        guidance_context, discount_factor, feature_state)
    if feature_state is None or feature_state[13] != 2 or not feature_state[4]:
        return result

    contributions = tuple(
        (name, amount) for name, amount in result.contributions
        if name not in (WAIT_AT_BLOCK_TARGET, SKIPPED_BLOCK_TARGET))
    if action == "WAIT":
        if e.WAITED in events and e.INVALID_ACTION not in events:
            contributions += ((WAIT_AT_BLOCK_TARGET, 1.0),)
    elif action in (*TASK2_NON_BOMB_ACTION_INDICES, "BOMB"):
        contributions += ((SKIPPED_BLOCK_TARGET, -1.0),)
    return RewardResult(
        total=float(sum(amount for _, amount in contributions)),
        contributions=contributions)


REWARD_FUNCTIONS: dict[str, RewardFunction] = {
    TASK3_TACTICAL_REWARD_V5_VERSION: task3_tactical_rewards_v5,
    TASK3_TACTICAL_REWARD_V4_VERSION: task3_tactical_rewards_v4,
    TASK3_TACTICAL_HOLD_REWARD_VERSION: task3_tactical_hold_rewards_v3_10,
    TASK3_PRESSURE_KILL_REWARD_VERSION: task3_pressure_kill_rewards_v3_9,
    BASELINE_REWARD_VERSION: task1_baseline_rewards_v1,
    WAIT_REWARD_VERSION: task1_wait_rewards_v1,
    POTENTIAL_GUIDANCE_REWARD_VERSION: (
        task1_potential_guidance_rewards_v1
    ),
    BFS_POTENTIAL_GUIDANCE_REWARD_VERSION: (
        task1_bfs_potential_guidance_rewards_v2
    ),
    TASK2_BOMB_CONTROL_REWARD_VERSION: (
        task2_bomb_control_rewards_v2
    ),
    TASK2_HIERARCHICAL_REWARD_VERSION: (
        task2_hierarchical_navigation_rewards_v3
    ),
    TASK2_HIERARCHICAL_SAFETY_REWARD_VERSION: (
        task2_hierarchical_navigation_safety_rewards_v3_1
    ),
    TASK2_HIERARCHICAL_URGENCY_REWARD_VERSION: (
        task2_hierarchical_navigation_urgency_rewards_v3_1_1
    ),
    TASK2_INDEPENDENT_TARGET_REWARD_VERSION: (
        task2_hierarchical_independent_target_rewards_v3_2
    ),
    TASK2_INDEPENDENT_TARGET_SCALE_4_REWARD_VERSION: (
        task2_hierarchical_independent_target_rewards_v3_3
    ),
    TASK2_INDEPENDENT_TARGET_CONTRASTIVE_REWARD_VERSION: (
        task2_hierarchical_independent_target_rewards_v3_4
    ),
    TASK2_INDEPENDENT_TARGET_HIERARCHY_REWARD_VERSION: (
        task2_hierarchical_independent_target_rewards_v3_5
    ),
    TASK2_INDEPENDENT_TARGET_CONSISTENT_REWARD_VERSION: (
        task2_hierarchical_independent_target_rewards_v3_6
    ),
    TASK2_INDEPENDENT_TARGET_STRONGER_REWARD_VERSION: (
        task2_hierarchical_independent_target_rewards_v3_7
    ),
    TASK2_INDEPENDENT_TARGET_BOMB_EMPHASIS_REWARD_VERSION: (
        task2_hierarchical_independent_target_rewards_v3_8
    ),
    TASK2_BASELINE_REWARD_VERSION: task2_baseline_rewards_v1,
}


def get_reward_function(
    reward_version: str,
) -> RewardFunction:
    """Return the reward function registered for one version."""
    try:
        return REWARD_FUNCTIONS[reward_version]

    except KeyError as error:
        available_versions = ", ".join(
            sorted(REWARD_FUNCTIONS)
        )

        raise ValueError(
            f"Unknown reward version {reward_version!r}. "
            f"Available versions: {available_versions}."
        ) from error
