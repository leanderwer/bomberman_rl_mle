"""Reward V5 for the final agent, evaluated from the action's actual features.

The safety, local bomb and navigation signals are mutually exclusive. A
certified HOLD has its own stronger signal, in addition to costs for invalid
or unsafe actions. Outcome events always retain their ordinary contributions.
"""

from dataclasses import dataclass

import events as e

from .features import FeatureState
from .navigation import (
    TARGET_CASE_BOMB_ONLY,
    TARGET_CASE_COIN_ONLY,
    TARGET_CASE_JOINT,
    TARGET_CASE_NONE,
    TARGET_CASE_TRUE_CONFLICT,
    classify_safe_target_navigation,
)


REWARD_VERSION = "task3_hierarchical_tactical_v5"
WAIT_AT_BLOCK_TARGET = "WAIT_AT_BLOCK_TARGET"
SKIPPED_BLOCK_TARGET = "SKIPPED_BLOCK_TARGET"
KILLED_BY_OPPONENT = "KILLED_BY_OPPONENT"
BOMB_AT_SELECTED_TARGET = "BOMB_AT_SELECTED_TARGET"
BOMB_OUTSIDE_SELECTED_TARGET = "BOMB_OUTSIDE_SELECTED_TARGET"
SKIPPED_BOMB_TARGET = "SKIPPED_BOMB_TARGET"
# Retain event identifiers so the existing reward decomposition stays readable.
TASK2_TARGET_NAVIGATION_SCORE = "TASK2_TARGET_NAVIGATION_SCORE"
UNSAFE_ACTION_WITH_SAFE_ALTERNATIVE = "UNSAFE_ACTION_WITH_SAFE_ALTERNATIVE"

NON_BOMB_ACTION_INDICES = {
    "UP": 0,
    "RIGHT": 1,
    "DOWN": 2,
    "LEFT": 3,
    "WAIT": 4,
}
EVENT_REWARDS = {
    e.COIN_COLLECTED: 1.0,
    e.CRATE_DESTROYED: 0.05,
    e.COIN_FOUND: 0.2,
    e.KILLED_OPPONENT: 5.0,
    e.KILLED_SELF: -1.0,
    e.INVALID_ACTION: -0.7,
    e.BOMB_DROPPED: -0.02,
    e.MOVED_UP: -0.01,
    e.MOVED_RIGHT: -0.01,
    e.MOVED_DOWN: -0.01,
    e.MOVED_LEFT: -0.01,
    e.WAITED: -0.02,
    BOMB_AT_SELECTED_TARGET: 0.3,
    BOMB_OUTSIDE_SELECTED_TARGET: -0.4,
    SKIPPED_BOMB_TARGET: -0.3,
}


@dataclass(frozen=True)
class RewardResult:
    """One transition reward and its ordered event contributions."""

    total: float
    contributions: tuple[tuple[str, float], ...]


def _unsafe_action_penalty(action: str | None, state: FeatureState):
    """Penalize an unsafe movement/WAIT only when a safe alternative exists."""
    if state is None or action not in NON_BOMB_ACTION_INDICES:
        return None
    safety_mask = state[:5]
    if any(safety_mask) and not safety_mask[NON_BOMB_ACTION_INDICES[action]]:
        return UNSAFE_ACTION_WITH_SAFE_ALTERNATIVE, -0.5
    return None


def _navigation_contribution(action: str | None, state: FeatureState):
    """Reward safe target following; remain neutral between disjoint targets."""
    if state is None or action not in NON_BOMB_ACTION_INDICES:
        return None
    action_index = NON_BOMB_ACTION_INDICES[action]
    if not state[action_index]:
        return None

    target = classify_safe_target_navigation(
        safety_mask=state[:5],
        coin_mask=state[5:9],
        bomb_target_mask=state[9:14],
        local_bomb_opportunity=False,
    )
    if target.name == TARGET_CASE_NONE:
        return None

    follows_coin = action_index < 4 and target.safe_coin_mask[action_index]
    follows_bomb = action_index < 4 and target.safe_bomb_mask[action_index]
    if target.name == TARGET_CASE_JOINT:
        amount = 0.06 if follows_coin and follows_bomb else (
            0.0 if follows_coin or follows_bomb else -0.06
        )
    elif target.name == TARGET_CASE_TRUE_CONFLICT:
        amount = 0.0 if follows_coin or follows_bomb else -0.04
    elif target.name == TARGET_CASE_COIN_ONLY:
        amount = 0.06 if follows_coin else -0.06
    elif target.name == TARGET_CASE_BOMB_ONLY:
        amount = 0.06 if follows_bomb else -0.06
    else:
        raise RuntimeError(f"Unsupported target case {target.name!r}.")
    return TASK2_TARGET_NAVIGATION_SCORE, float(amount)


def calculate_reward(
    old_game_state: dict | None,
    action: str | None,
    new_game_state: dict | None,
    events: list[str],
    guidance_context=None,
    discount_factor: float = 0.95,
    feature_state: FeatureState = None,
) -> RewardResult:
    """Calculate V5 without recomputing the features or changing engine events.

    ``action`` and ``feature_state`` use the same canonical orientation. The
    game-state/context arguments retain the training callback interface; reward
    shaping depends only on the features that actually selected this action.
    """
    if feature_state is None and old_game_state is not None:
        raise ValueError("Reward V5 requires the actual feature_state used for the action.")

    reward_events = list(events)
    # The engine can report overlapping own/foreign explosions. Charge exactly one death and retain the established position of this contribution.
    if e.KILLED_SELF in reward_events:
        reward_events = [event for event in reward_events if event != e.KILLED_SELF]
        reward_events.append(e.KILLED_SELF)

    hold_target = feature_state is not None and feature_state[13] == 2
    certified_hold = hold_target and bool(feature_state[4])
    dynamic = []
    if e.INVALID_ACTION not in reward_events:
        unsafe = _unsafe_action_penalty(action, feature_state)
        if unsafe is not None:
            dynamic.append(unsafe)
        elif not hold_target:
            local_bomb = bool(
                feature_state is not None
                and feature_state[13] == 1
                and feature_state[-1]
            )
            placement = None
            if (feature_state is not None and action == "BOMB"
                    and e.BOMB_DROPPED in reward_events):
                placement = (BOMB_AT_SELECTED_TARGET if feature_state[13] == 1
                             else BOMB_OUTSIDE_SELECTED_TARGET)
            if local_bomb:
                if placement == BOMB_AT_SELECTED_TARGET:
                    reward_events.append(placement)
                elif action in NON_BOMB_ACTION_INDICES:
                    reward_events.append(SKIPPED_BOMB_TARGET)
            elif placement is not None:
                reward_events.append(placement)
            else:
                navigation = _navigation_contribution(action, feature_state)
                if navigation is not None:
                    dynamic.append(navigation)

    values = EVENT_REWARDS
    if hold_target:
        # Usually these two events are derived below, not passed by the engine.
        # Preserve V5's behavior even for explicitly supplied shaping events.
        values = {**values, WAIT_AT_BLOCK_TARGET: 0.3, SKIPPED_BLOCK_TARGET: -0.3}
    contributions = tuple((event, float(values.get(event, 0.0))) for event in reward_events)
    total = float(sum(amount for _, amount in contributions))
    total = float(total + sum(amount for _, amount in dynamic))
    contributions += tuple(dynamic)
    if e.GOT_KILLED in events and e.KILLED_SELF not in events:
        contributions += ((KILLED_BY_OPPONENT, -1.0),)
        total = float(total + -1.0)

    if certified_hold:
        # HOLD replaces bomb/navigation shaping, but retains the independent invalid/unsafe penalties and delayed outcomes, including own kills.
        contributions = tuple(
            (name, amount) for name, amount in contributions
            if name not in (WAIT_AT_BLOCK_TARGET, SKIPPED_BLOCK_TARGET)
        )
        if action == "WAIT":
            if e.WAITED in events and e.INVALID_ACTION not in events:
                contributions += ((WAIT_AT_BLOCK_TARGET, 1.0),)
        elif action in (*NON_BOMB_ACTION_INDICES, "BOMB"):
            contributions += ((SKIPPED_BLOCK_TARGET, -1.0),)
        total = float(sum(amount for _, amount in contributions))

    return RewardResult(total=total, contributions=contributions)
