import os
from collections import OrderedDict
from functools import partial

from .action_spaces import (
    ACTION_ORDER,
    DEFAULT_ACTION_SPACE_VERSION,
    get_learning_action_indices,
    get_learning_actions,
)
from .features import (
    BFS_POTENTIAL_PRIORITY_V2_FEATURE_VERSION,
    BFS_POTENTIAL_TIE_BREAK_FEATURE_VERSION,
    COIN_ROUTE_BOMB_DIAGNOSTIC_CONTEXT_KEY,
    CoinPriorityDiagnosticContext,
    EVALUATION_DIAGNOSTIC_CONTEXT_KEY,
    DEFAULT_FEATURE_VERSION,
    FEATURE_ACTION_TRANSFORM_CONTEXT_KEY,
    FeatureState,
    IDENTITY_FEATURE_ACTION_TRANSFORM,
    POTENTIAL_GUIDANCE_CONTEXT_KEY,
    PREVIOUS_AGENT_POSITION_KEY,
    TASK2_BASELINE_FEATURE_VERSION,
    TASK2_COMPACT_SYMMETRIC_FEATURE_VERSION,
    TASK2_COMPACT_SYMMETRIC_V5_FEATURE_VERSION,
    TASK2_COIN_ROUTE_BOMB_V6_FEATURE_VERSION,
    TASK2_REPORT_V6_FEATURE_VERSIONS,
    TASK2_REPORT_V6_COIN_PRIORITY_FEATURE_VERSION,
    TASK3_PRESSURE_BOMB_V7_FEATURE_VERSION,
    TASK3_ROBUST_SAFETY_V8_FEATURE_VERSION,
    TASK3_TACTICAL_BOMB_V9_FEATURE_VERSION,
    TASK3_TACTICAL_BOMB_V12_FEATURE_VERSION,
    TASK3_ARRIVAL_TACTICAL_FEATURE_VERSIONS,
    TASK3_RETRY_TACTICAL_FEATURE_VERSIONS,
    TACTICAL_PLANNER_CONTEXT_KEY,
    PressureBombDiagnosticContext,
    TASK2_COIN_OBJECTIVE,
    TASK2_CRATE_OBJECTIVE,
    TASK2_ESCAPE_OBJECTIVE,
    TASK2_INDEPENDENT_TARGETS_FEATURE_VERSION,
    TASK2_PARALLEL_MASKS_FEATURE_VERSION,
    TARGET_CLASS_DIAGNOSTIC_CONTEXT_KEY,
    THREAT_URGENCY_DIAGNOSTIC_CONTEXT_KEY,
    get_feature_extractor,
    task3_tactical_bomb_features_v12,
)
from .task3_tactical_mechanics import validate_pressure_parameters
from .model_io import load_checkpoint
from .exploration_schedules import softmax_probabilities
from .task2_mechanics import (
    CoinRouteBombDiagnosticContext,
    TASK2_TARGET_CASE_BOMB_ONLY,
    TASK2_TARGET_CASE_COIN_ONLY,
    TASK2_TARGET_CASE_JOINT,
    TASK2_TARGET_CASE_LOCAL_BOMB,
    TASK2_TARGET_CASE_NONE,
    TASK2_TARGET_CASE_TRUE_CONFLICT,
    classify_safe_target_navigation,
)
import random

import numpy as np


ACTIONS = list(ACTION_ORDER)

FEATURE_CACHE_SIZE = 3
_FEATURE_CACHE_MISS = object()

TASK2_OBJECTIVE_DIAGNOSTIC_NAMES = {
    TASK2_COIN_OBJECTIVE: "coin_objective",
    TASK2_CRATE_OBJECTIVE: "crate_objective",
    TASK2_ESCAPE_OBJECTIVE: "escape_objective",
}


def _mask_followed(
    mask: tuple,
    chosen_action: str,
    actions: tuple[str, ...],
) -> bool:
    """Return whether the chosen action follows a positional mask."""
    if chosen_action not in actions:
        return False

    return bool(mask[actions.index(chosen_action)])


def _task2_parallel_mask_statistics(
    state: FeatureState,
    chosen_action: str,
    independent_target_classes: bool = False,
    compact_symmetric_features: bool = False,
    threat_urgency_override: int | None = None,
    target_class_override: tuple[int, int] | None = None,
    coin_route_bomb_override: (
        CoinRouteBombDiagnosticContext | None
    ) = None,
) -> dict[str, int]:
    """Return feature-conditioned parallel-mask diagnostics."""
    safety_mask = state[0:5]
    coin_mask = state[5:9]
    bomb_direction_mask = state[9:13]
    bomb_target_here = state[13] == 1

    if independent_target_classes:
        if target_class_override is not None:
            coin_distance_class = int(target_class_override[0])
            bomb_quality_class = int(target_class_override[1])
        elif len(state) >= 16:
            coin_distance_class = int(state[14])
            bomb_quality_class = int(state[15])
        else:
            coin_distance_class = None
            bomb_quality_class = None

        if compact_symmetric_features:
            threat_urgency = threat_urgency_override
            bomb_available = bool(state[-1])
        else:
            threat_urgency = state[16]
            bomb_available = bool(state[17])
        relative_target_distance = None
    else:
        relative_target_distance = state[14]
        threat_urgency = state[15]
        bomb_available = bool(state[16])
        coin_distance_class = None
        bomb_quality_class = None
    statistics = {
        # This marker lets the framework distinguish new V2 telemetry
        # from older result files when a self-kill is classified.
        "self_kill_diagnostic_observations": 1,
    }

    if independent_target_classes:
        statistics[
            "independent_target_diagnostic_observations"
        ] = 1
    movement_actions = tuple(ACTIONS[:4])
    non_bomb_actions = tuple(ACTIONS[:5])
    chose_non_bomb_action = chosen_action in non_bomb_actions

    if any(safety_mask) and chose_non_bomb_action:
        followed_safety = _mask_followed(
            safety_mask,
            chosen_action,
            non_bomb_actions,
        )
        statistics["survivable_action_opportunities"] = 1
        statistics["survivable_action_followed"] = int(
            followed_safety
        )
        statistics[
            "unsafe_action_with_safe_alternative"
        ] = int(not followed_safety)
    elif not any(safety_mask):
        statistics["no_survivable_action_available"] = 1

    if (
        threat_urgency in (0, 1, 2, 3)
        and any(safety_mask)
        and chose_non_bomb_action
    ):
        metric_prefix = f"threat_urgency_{threat_urgency}"
        statistics[f"{metric_prefix}_opportunities"] = 1
        statistics[f"{metric_prefix}_followed"] = int(
            _mask_followed(
                safety_mask,
                chosen_action,
                non_bomb_actions,
            )
        )

    local_bomb_opportunity = (
        bomb_target_here and bomb_available
    )
    target_case = classify_safe_target_navigation(
        safety_mask=safety_mask,
        coin_mask=coin_mask,
        bomb_target_mask=(
            *bomb_direction_mask,
            int(bomb_target_here),
        ),
        local_bomb_opportunity=local_bomb_opportunity,
    )
    safe_coin_mask = target_case.safe_coin_mask
    safe_bomb_direction_mask = target_case.safe_bomb_mask

    # The shared classifier guarantees that target navigation is counted
    # exactly once. Local BOMB opportunities are measured separately below.
    if target_case.name != TASK2_TARGET_CASE_LOCAL_BOMB:
        if target_case.name == TASK2_TARGET_CASE_JOINT:
            statistics["safe_joint_target_opportunities"] = 1
            statistics["safe_joint_target_followed"] = int(
                _mask_followed(
                    target_case.safe_joint_mask,
                    chosen_action,
                    movement_actions,
                )
            )
        elif target_case.name == TASK2_TARGET_CASE_TRUE_CONFLICT:
            if independent_target_classes:
                if (
                    coin_distance_class is None
                    or bomb_quality_class is None
                ):
                    raise ValueError(
                        "Independent-target diagnostics require "
                        "coin and bomb classes in a true conflict."
                    )

                conflict_prefix = (
                    "target_conflict_"
                    f"coin_class_{coin_distance_class}_"
                    f"bomb_class_{bomb_quality_class}"
                )
                statistics[
                    f"{conflict_prefix}_opportunities"
                ] = 1
                statistics[
                    f"{conflict_prefix}_coin_followed"
                ] = int(
                    _mask_followed(
                        safe_coin_mask,
                        chosen_action,
                        movement_actions,
                    )
                )
                statistics[
                    f"{conflict_prefix}_bomb_followed"
                ] = int(
                    _mask_followed(
                        safe_bomb_direction_mask,
                        chosen_action,
                        movement_actions,
                    )
                )
            else:
                conflict_prefixes = {
                    0: "target_conflict_coin_strong",
                    1: "target_conflict_coin_weak",
                    3: "target_conflict_bomb_weak",
                    4: "target_conflict_bomb_strong",
                }
                conflict_prefix = conflict_prefixes.get(
                    relative_target_distance
                )

                if conflict_prefix is None:
                    # Category 2 means equal target distance and does not
                    # prescribe either side of the true conflict.
                    statistics[
                        "target_conflict_equal_opportunities"
                    ] = 1
                else:
                    prioritized_mask = (
                        safe_coin_mask
                        if relative_target_distance in (0, 1)
                        else safe_bomb_direction_mask
                    )
                    statistics[
                        f"{conflict_prefix}_opportunities"
                    ] = 1
                    statistics[
                        f"{conflict_prefix}_followed"
                    ] = int(
                        _mask_followed(
                            prioritized_mask,
                            chosen_action,
                            movement_actions,
                        )
                    )
        elif target_case.name == TASK2_TARGET_CASE_COIN_ONLY:
            statistics["safe_coin_only_target_opportunities"] = 1
            statistics["safe_coin_only_target_followed"] = int(
                _mask_followed(
                    safe_coin_mask,
                    chosen_action,
                    movement_actions,
                )
            )
        elif target_case.name == TASK2_TARGET_CASE_BOMB_ONLY:
            statistics["safe_bomb_only_target_opportunities"] = 1
            statistics["safe_bomb_only_target_followed"] = int(
                _mask_followed(
                    safe_bomb_direction_mask,
                    chosen_action,
                    movement_actions,
                )
            )
        elif target_case.name == TASK2_TARGET_CASE_NONE:
            statistics["no_safe_target_opportunities"] = 1
        else:
            raise RuntimeError(
                f"Unsupported target case {target_case.name!r}."
            )

    primary_bomb_target_mask = (
        coin_route_bomb_override.primary_mask
        if coin_route_bomb_override is not None
        else (*bomb_direction_mask, int(bomb_target_here))
    )
    primary_bomb_target_here = bool(primary_bomb_target_mask[4])
    valid_bomb_drop = chosen_action == "BOMB" and bomb_available
    primary_bomb_target_opportunity = (
        primary_bomb_target_here and bomb_available
    )
    bomb_dropped_at_any_target = (
        valid_bomb_drop and bomb_target_here
    )
    bomb_dropped_at_primary_target = (
        valid_bomb_drop and primary_bomb_target_here
    )

    if valid_bomb_drop:
        statistics["valid_bomb_drops"] = 1

    if primary_bomb_target_opportunity:
        statistics["bomb_target_opportunities"] = 1

    # Precision accepts either V6 target, whereas the legacy response
    # metric deliberately remains restricted to the primary BFS target.
    if bomb_dropped_at_any_target:
        statistics["bomb_drops_at_target"] = 1

    if bomb_dropped_at_primary_target:
        statistics["bomb_drops_at_primary_target"] = 1

    if coin_route_bomb_override is not None:
        statistics["coin_route_bomb_diagnostic_observations"] = 1
        alternative_mask = coin_route_bomb_override.alternative_mask
        score_ratio_milli = (
            coin_route_bomb_override.alternative_score_ratio_milli
        )

        if score_ratio_milli is not None:
            statistics[
                "coin_route_bomb_alternative_opportunities"
            ] = 1
            statistics[
                "coin_route_bomb_alternative_score_ratio_milli_sum"
            ] = int(score_ratio_milli)
            safe_primary_joint_mask = tuple(
                int(
                    bool(
                        safety_mask[index]
                        and coin_mask[index]
                        and primary_bomb_target_mask[index]
                    )
                )
                for index in range(4)
            )
            safe_alternative_joint_mask = tuple(
                int(
                    bool(
                        safety_mask[index]
                        and coin_mask[index]
                        and alternative_mask[index]
                    )
                )
                for index in range(4)
            )

            # Count only genuine conversions: the primary target offered
            # no safe joint direction, while the alternative creates one
            # and the shared classifier therefore exposes a joint case.
            if (
                target_case.name == TASK2_TARGET_CASE_JOINT
                and not any(safe_primary_joint_mask)
                and any(safe_alternative_joint_mask)
            ):
                statistics[
                    "coin_route_bomb_joint_conversion_opportunities"
                ] = 1
                statistics[
                    "coin_route_bomb_joint_conversion_followed"
                ] = int(
                    _mask_followed(
                        safe_alternative_joint_mask,
                        chosen_action,
                        movement_actions,
                    )
                )

            if alternative_mask[4] and bomb_available:
                statistics[
                    "coin_route_bomb_alternative_target_opportunities"
                ] = 1
                statistics[
                    "coin_route_bomb_alternative_target_followed"
                ] = int(chosen_action == "BOMB")

    return statistics


def clear_feature_cache(self) -> None:
    """Clear all feature tuples retained for the current round."""
    self.feature_cache = OrderedDict()
    self.feature_cache_round = None


def clear_feature_context(self) -> None:
    """Clear cached features and position history at a round boundary."""
    clear_feature_cache(self)
    self.feature_context_round = None
    self.last_observed_position = None
    self.last_action_features = None
    self.latest_potential_guidance_context = None
    self.last_action_potential_guidance_context = None
    self.latest_feature_action_transform = (
        IDENTITY_FEATURE_ACTION_TRANSFORM
    )
    self.last_action_feature_action_transform = (
        IDENTITY_FEATURE_ACTION_TRANSFORM
    )
    self.latest_threat_urgency_diagnostic = None
    self.last_action_threat_urgency_diagnostic = None
    self.latest_target_class_diagnostic = None
    self.last_action_target_class_diagnostic = None
    self.latest_coin_route_bomb_diagnostic = None
    self.last_action_coin_route_bomb_diagnostic = None
    self.tactical_rejections = {}


def block_action_statistics(state, chosen_action, context):
    """Cheap V9 counters shared by training and evaluation, including zeros."""
    active = (state is not None and isinstance(context, PressureBombDiagnosticContext)
              and context.pressure.target_family == "block")
    local_hold = bool(active and state[13] == 2)
    followed = bool(active and (
        chosen_action == "WAIT" if local_hold else
        _mask_followed(state[9:13], chosen_action, ACTIONS[:4])))
    return {
        "tactical_block_observations": 1,
        "block_target_opportunities": int(active),
        "block_target_followed": int(followed),
        "block_local_wait_opportunities": int(local_hold),
        "block_local_wait_followed": int(local_hold and chosen_action == "WAIT"),
    }


def task2_action_statistics(
    feature_version: str,
    state: FeatureState,
    chosen_action: str,
    threat_urgency_override: int | None = None,
    target_class_override: tuple[int, int] | None = None,
    coin_route_bomb_override: (
        CoinRouteBombDiagnosticContext | None
    ) = None,
) -> dict[str, int]:
    """Return aggregate adherence counters for Task 2 features.

    Every mask is treated as a set of equally valid first steps. For the
    baseline, a safe productive bomb opportunity is bomb status 3.
    """
    if state is None:
        return {}

    if feature_version == TASK2_REPORT_V6_COIN_PRIORITY_FEATURE_VERSION:
        if not isinstance(coin_route_bomb_override, CoinPriorityDiagnosticContext):
            raise ValueError("Coin-priority diagnostics require full V6 context.")
        # Count opportunities against the same full target evidence as V6;
        # the Q-state intentionally hides the lower-priority navigation mask.
        state = coin_route_bomb_override.full_feature_state

    if feature_version == TASK2_PARALLEL_MASKS_FEATURE_VERSION:
        return _task2_parallel_mask_statistics(
            state=state,
            chosen_action=chosen_action,
        )

    if feature_version == TASK2_INDEPENDENT_TARGETS_FEATURE_VERSION:
        return _task2_parallel_mask_statistics(
            state=state,
            chosen_action=chosen_action,
            independent_target_classes=True,
        )

    if feature_version == TASK2_COMPACT_SYMMETRIC_FEATURE_VERSION:
        return _task2_parallel_mask_statistics(
            state=state,
            chosen_action=chosen_action,
            independent_target_classes=True,
            compact_symmetric_features=True,
            threat_urgency_override=threat_urgency_override,
        )

    if feature_version == TASK2_COMPACT_SYMMETRIC_V5_FEATURE_VERSION:
        return _task2_parallel_mask_statistics(
            state=state,
            chosen_action=chosen_action,
            independent_target_classes=True,
            compact_symmetric_features=True,
            threat_urgency_override=threat_urgency_override,
            target_class_override=target_class_override,
        )

    if feature_version in (TASK2_COIN_ROUTE_BOMB_V6_FEATURE_VERSION,
                           TASK2_REPORT_V6_COIN_PRIORITY_FEATURE_VERSION,
                           *TASK2_REPORT_V6_FEATURE_VERSIONS,
                           TASK3_PRESSURE_BOMB_V7_FEATURE_VERSION,
                           TASK3_ROBUST_SAFETY_V8_FEATURE_VERSION,
                           TASK3_TACTICAL_BOMB_V9_FEATURE_VERSION,
                           *TASK3_ARRIVAL_TACTICAL_FEATURE_VERSIONS):
        if coin_route_bomb_override is None:
            raise ValueError(
                "V6/V7/V8/V9/V10/V11 diagnostics require primary and alternative "
                "bomb-target context."
            )

        if feature_version in TASK3_ARRIVAL_TACTICAL_FEATURE_VERSIONS:
            if (not isinstance(coin_route_bomb_override, PressureBombDiagnosticContext)
                    or coin_route_bomb_override.bomb_available is None):
                raise ValueError("V10/V11 diagnostics require real bomb availability.")
            state = (*state[:-1], int(coin_route_bomb_override.bomb_available))
        block_active = (isinstance(coin_route_bomb_override, PressureBombDiagnosticContext)
                        and coin_route_bomb_override.pressure.target_family == "block")
        # A shared tactical direction is not a crate/pressure BOMB direction.
        # Retain safety and actual bomb-use counters without mislabelling HOLD
        # as BOMB or adding block navigation to the existing crate heatmaps.
        statistics_state = (*state[:5], *(0,) * 9, state[-1]) if block_active else state
        statistics = _task2_parallel_mask_statistics(
            state=statistics_state,
            chosen_action=chosen_action,
            independent_target_classes=True,
            compact_symmetric_features=True,
            threat_urgency_override=threat_urgency_override,
            target_class_override=target_class_override,
            coin_route_bomb_override=coin_route_bomb_override,
        )
        if block_active:
            statistics.pop("no_safe_target_opportunities", None)
        if feature_version in (TASK3_TACTICAL_BOMB_V9_FEATURE_VERSION,
                               *TASK3_ARRIVAL_TACTICAL_FEATURE_VERSIONS):
            statistics.update(block_action_statistics(
                state, chosen_action, coin_route_bomb_override))
        if isinstance(coin_route_bomb_override, PressureBombDiagnosticContext):
            diagnostic = coin_route_bomb_override.pressure
            pressure_active = diagnostic.target_family == "pressure"
            if pressure_active:
                # Crate-yield class thresholds have no meaning for pressure.
                # Preserve the old heatmap for actual coin/crate conflicts.
                statistics = {k: v for k, v in statistics.items()
                              if not k.startswith("target_conflict_coin_class_")}
            local = bool(state[13] == 1 and state[14] and pressure_active)
            followed = (chosen_action == "BOMB" if local else
                        _mask_followed(state[9:13], chosen_action, ACTIONS[:4]))
            statistics.update({
                "pressure_target_opportunities": int(pressure_active),
                "pressure_target_followed": int(pressure_active and followed),
                "pressure_local_bomb_opportunities": int(local),
                "pressure_local_bomb_followed": int(local and chosen_action == "BOMB"),
                "bomb_blockade_check_opportunities": int(diagnostic.blockade_checked),
                "bomb_blockade_check_followed": int(diagnostic.blockade_passed),
            })
        return statistics

    if feature_version != TASK2_BASELINE_FEATURE_VERSION:
        return {}

    objective_mask = state[4:8]
    objective_mode = state[8]
    bomb_status = state[9]
    statistics = {}

    objective_name = TASK2_OBJECTIVE_DIAGNOSTIC_NAMES.get(
        objective_mode
    )

    if objective_name is not None and any(objective_mask):
        statistics[f"{objective_name}_opportunities"] = 1

        direction_index = (
            ACTIONS.index(chosen_action)
            if chosen_action in ACTIONS[:4]
            else None
        )
        followed = (
            direction_index is not None
            and bool(objective_mask[direction_index])
        )
        statistics[f"{objective_name}_followed"] = int(
            followed
        )

    if (
        objective_mode == TASK2_CRATE_OBJECTIVE
        and bomb_status == 3
    ):
        statistics["safe_productive_bomb_opportunities"] = 1
        statistics["safe_productive_bomb_followed"] = int(
            chosen_action == "BOMB"
        )

    return statistics


def _ensure_feature_round(self, game_state: dict) -> None:
    """Initialize feature context for the state's round when necessary."""
    round_number = game_state.get("round")

    if getattr(self, "feature_context_round", None) == round_number:
        return

    clear_feature_context(self)
    self.feature_context_round = round_number
    self.feature_cache_round = round_number


def observe_feature_state(
    self,
    game_state: dict,
) -> tuple[int, int] | None:
    """Return the preceding tile and retain the current visible position."""
    _ensure_feature_round(self, game_state)
    previous_position = self.last_observed_position
    self.last_observed_position = tuple(
        game_state["self"][3]
    )

    return previous_position


def _freeze_cache_value(value):
    """Convert visible game-state data into an exact hashable snapshot."""
    if isinstance(value, np.ndarray):
        return (
            "numpy",
            value.shape,
            value.dtype.str,
            value.tobytes(),
        )

    if isinstance(value, dict):
        return tuple(
            (
                key,
                _freeze_cache_value(item),
            )
            for key, item in sorted(value.items())
        )

    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_cache_value(item)
            for item in value
        )

    if isinstance(value, np.generic):
        return value.item()

    return value


def _feature_cache_key(
    game_state: dict,
    previous_position: tuple[int, int] | None,
) -> tuple:
    """Describe feature inputs while ignoring time labels.

    The explicit previous-position context keeps history-sensitive v2
    states distinct. Excluding round and step still permits safe reuse
    of a post-action state as the next pre-action state.
    """
    visible_snapshot = tuple(
        (key, _freeze_cache_value(value))
        for key, value in sorted(game_state.items())
        if (
            key not in {"round", "step"}
            and key
            not in {
                PREVIOUS_AGENT_POSITION_KEY,
                POTENTIAL_GUIDANCE_CONTEXT_KEY,
                FEATURE_ACTION_TRANSFORM_CONTEXT_KEY,
                THREAT_URGENCY_DIAGNOSTIC_CONTEXT_KEY,
                TARGET_CLASS_DIAGNOSTIC_CONTEXT_KEY,
                COIN_ROUTE_BOMB_DIAGNOSTIC_CONTEXT_KEY,
                EVALUATION_DIAGNOSTIC_CONTEXT_KEY,
                TACTICAL_PLANNER_CONTEXT_KEY,
            }
        )
    )

    return (
        visible_snapshot,
        previous_position,
    )


def extract_features(
    self,
    game_state: dict | None,
    previous_position: tuple[int, int] | None = None,
    *,
    after_action: bool = False,
) -> FeatureState:
    """Return an immutable feature tuple, reusing repeated state snapshots."""
    if game_state is None:
        return None

    try:
        round_number = game_state["round"]
    except KeyError:
        return self.feature_extractor(game_state)

    if not hasattr(self, "feature_cache"):
        clear_feature_context(self)

    if self.feature_cache_round != round_number:
        _ensure_feature_round(self, game_state)

    cache_key = _feature_cache_key(
        game_state,
        previous_position,
    )
    uses_retry = getattr(self, "feature_version", None) in TASK3_RETRY_TACTICAL_FEATURE_VERSIONS
    if uses_retry:
        # A repeated post-action/pre-action observation must reuse the SAME
        # history-sensitive features. An identical board at a later step may
        # have a different just-rejected target and must be recomputed.
        # The engine labels its post-action state with the step just executed.
        # It is the same decision state that act receives as step+1 next time.
        decision_step = int(game_state["step"]) + int(after_action)
        cache_key = (cache_key, decision_step)
    cached_entry = self.feature_cache.pop(
        cache_key,
        _FEATURE_CACHE_MISS,
    )

    if cached_entry is not _FEATURE_CACHE_MISS:
        (
            cached_features,
            cached_guidance_context,
            cached_action_transform,
            cached_threat_urgency,
            cached_target_classes,
            cached_coin_route_bomb_diagnostic,
        ) = cached_entry
        self.feature_cache[cache_key] = cached_entry
        self.latest_potential_guidance_context = (
            cached_guidance_context
        )
        self.latest_feature_action_transform = (
            cached_action_transform
        )
        self.latest_threat_urgency_diagnostic = (
            cached_threat_urgency
        )
        self.latest_target_class_diagnostic = (
            cached_target_classes
        )
        self.latest_coin_route_bomb_diagnostic = (
            cached_coin_route_bomb_diagnostic
        )
        return cached_features

    contextual_game_state = dict(game_state)
    if uses_retry:
        contextual_game_state[TACTICAL_PLANNER_CONTEXT_KEY] = {
            "previous_rejection": self.tactical_rejections.get(decision_step - 1)}
    contextual_game_state[
        PREVIOUS_AGENT_POSITION_KEY
    ] = previous_position
    contextual_game_state[POTENTIAL_GUIDANCE_CONTEXT_KEY] = None
    contextual_game_state[
        FEATURE_ACTION_TRANSFORM_CONTEXT_KEY
    ] = None
    contextual_game_state[
        THREAT_URGENCY_DIAGNOSTIC_CONTEXT_KEY
    ] = None
    contextual_game_state[
        TARGET_CLASS_DIAGNOSTIC_CONTEXT_KEY
    ] = None
    contextual_game_state[
        COIN_ROUTE_BOMB_DIAGNOSTIC_CONTEXT_KEY
    ] = None
    if getattr(self, "evaluation_diagnostics_enabled", False):
        contextual_game_state[EVALUATION_DIAGNOSTIC_CONTEXT_KEY] = {}
    features = self.feature_extractor(contextual_game_state)
    if uses_retry:
        self.tactical_rejections[decision_step] = contextual_game_state[
            TACTICAL_PLANNER_CONTEXT_KEY].get("rejected_position")
        self.tactical_rejections = {t: p for t, p in self.tactical_rejections.items()
                                    if t >= decision_step - FEATURE_CACHE_SIZE}
    guidance_context = contextual_game_state[
        POTENTIAL_GUIDANCE_CONTEXT_KEY
    ]
    self.latest_potential_guidance_context = guidance_context
    action_transform = contextual_game_state[
        FEATURE_ACTION_TRANSFORM_CONTEXT_KEY
    ]
    if action_transform is None:
        action_transform = IDENTITY_FEATURE_ACTION_TRANSFORM
    threat_urgency = contextual_game_state[
        THREAT_URGENCY_DIAGNOSTIC_CONTEXT_KEY
    ]
    target_classes = contextual_game_state[
        TARGET_CLASS_DIAGNOSTIC_CONTEXT_KEY
    ]
    coin_route_bomb_diagnostic = contextual_game_state[
        COIN_ROUTE_BOMB_DIAGNOSTIC_CONTEXT_KEY
    ]
    self.latest_feature_action_transform = action_transform
    self.latest_threat_urgency_diagnostic = threat_urgency
    self.latest_target_class_diagnostic = target_classes
    self.latest_coin_route_bomb_diagnostic = (
        coin_route_bomb_diagnostic
    )
    self.feature_cache[cache_key] = (
        features,
        guidance_context,
        action_transform,
        threat_urgency,
        target_classes,
        coin_route_bomb_diagnostic,
    )

    while len(self.feature_cache) > FEATURE_CACHE_SIZE:
        self.feature_cache.popitem(last=False)

    return features


def _read_evaluation_pressure_parameters(training):
    """Read once; bind values to one agent instead of changing module defaults."""
    names = ("BOMBERMAN_EVAL_PRESSURE_THRESHOLD",
             "BOMBERMAN_EVAL_PRESSURE_DISTANCE_DISCOUNT")
    if not any(name in os.environ for name in names):
        return None
    if training:
        raise ValueError("Pressure parameter overrides are only allowed during evaluation.")
    values = []
    for name, default in zip(names, (0.25, 0.95)):
        try:
            values.append(float(os.environ.get(name, default)))
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} must contain a finite numeric value.") from error
    return validate_pressure_parameters(*values)


def setup(self):
    """
    Load the Q-table, versioned settings and agent-local random generator.

    The framework calls setup_training afterwards in training mode.
    """

    self.evaluation_pressure_parameters = _read_evaluation_pressure_parameters(self.train)
    self.evaluation_diagnostics_enabled = (
        not self.train and os.environ.get("BOMBERMAN_TASK3_DIAGNOSTICS") == "1"
    )

    # Allow experiments to use a separate model file
    # for every independent training run.
    self.model_file = os.environ.get(
        "BOMBERMAN_MODEL_FILE",
        "my-saved-model.pt",
    )

    self.logger.info(
        f"Using model file '{self.model_file}'."
    )

    # Use a separate reproducible random-number generator for the agent.
    self.agent_seed = int(
        os.environ.get("BOMBERMAN_AGENT_SEED", "0")
    )
    self.random_generator = random.Random(self.agent_seed)

    self.logger.info(
        f"Using agent random seed {self.agent_seed}."
    )

    configured_feature_version = os.environ.get(
        "BOMBERMAN_FEATURE_VERSION"
    )
    configured_action_space_version = os.environ.get(
        "BOMBERMAN_ACTION_SPACE_VERSION"
    )

    if self.train:
        # Training uses the explicitly configured feature version,
        # or the default version if none was provided.
        self.feature_version = (
            configured_feature_version
            or DEFAULT_FEATURE_VERSION
        )
        self.feature_extractor = get_feature_extractor(
            self.feature_version
        )

        self.logger.info(
            f"Training with feature version "
            f"{self.feature_version!r}."
        )
        self.logger.info("Setting up Q-table from scratch.")

        self.model = {}
        self.checkpoint_metadata = {}
        self.action_space_version = (
            configured_action_space_version
            or DEFAULT_ACTION_SPACE_VERSION
        )

    elif os.path.isfile(self.model_file):
        self.logger.info("Loading Q-table from saved state.")

        (
            self.model,
            self.checkpoint_metadata,
            self.feature_version,
            legacy_checkpoint,
        ) = load_checkpoint(
            file_path=self.model_file,
            expected_action_order=tuple(ACTIONS),
            expected_feature_version=(
                configured_feature_version
            ),
            legacy_feature_version=(
                DEFAULT_FEATURE_VERSION
            ),
        )

        self.feature_extractor = get_feature_extractor(
            self.feature_version
        )

        saved_action_space_version = self.checkpoint_metadata.get(
            "action_space_version",
            DEFAULT_ACTION_SPACE_VERSION,
        )

        if (
            configured_action_space_version is not None
            and configured_action_space_version
            != saved_action_space_version
        ):
            raise ValueError(
                "Action-space-version mismatch: "
                f"configured {configured_action_space_version!r}, "
                f"checkpoint {saved_action_space_version!r}."
            )

        self.action_space_version = saved_action_space_version

        if legacy_checkpoint:
            self.logger.warning(
                "Loaded a legacy Q-table and assigned feature "
                f"version {self.feature_version!r}."
            )
        else:
            self.logger.info(
                f"Loaded checkpoint with feature version "
                f"{self.feature_version!r}."
            )

    else:
        raise FileNotFoundError(
            f"Cannot evaluate agent because model file "
            f"'{self.model_file}' does not exist."
        )

    if self.evaluation_pressure_parameters is not None:
        if self.feature_version != TASK3_TACTICAL_BOMB_V12_FEATURE_VERSION:
            raise ValueError("Pressure parameter overrides require the V12 feature version.")
        threshold, distance_discount = self.evaluation_pressure_parameters
        self.feature_extractor = partial(
            task3_tactical_bomb_features_v12,
            pressure_threshold=threshold, pressure_distance_discount=distance_discount)
        self.logger.info(
            "Evaluating pressure threshold %s and distance discount %s.",
            threshold, distance_discount)

    self.learning_actions = get_learning_actions(
        self.action_space_version
    )
    self.learning_action_indices = get_learning_action_indices(
        self.action_space_version
    )
    self.logger.info(
        "Using action-space version "
        f"{self.action_space_version!r}: "
        f"{', '.join(self.learning_actions)}."
    )

    clear_feature_context(self)

def act(self, game_state: dict) -> str:
    """
    Select an action using the configured exploration or greedy policy.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # Evaluation-only telemetry is reset first so an exception cannot
    # expose counters retained from the preceding action. Training keeps
    # this value at None and therefore avoids computing or transporting
    # policy-adherence diagnostics that do not affect learning.
    self.last_action_statistics = None
    self.last_evaluation_diagnostic = None

    previous_position = observe_feature_state(self, game_state)
    state = extract_features(
        self,
        game_state,
        previous_position=previous_position,
    )
    self.last_action_features = state
    self.last_action_potential_guidance_context = (
        self.latest_potential_guidance_context
    )
    self.last_action_feature_action_transform = (
        self.latest_feature_action_transform
    )
    self.last_action_threat_urgency_diagnostic = (
        self.latest_threat_urgency_diagnostic
    )
    self.last_action_target_class_diagnostic = (
        self.latest_target_class_diagnostic
    )
    self.last_action_coin_route_bomb_diagnostic = (
        self.latest_coin_route_bomb_diagnostic
    )

    # Add previously unseen states to the Q-table.
    if state not in self.model:
        self.model[state] = np.zeros(len(ACTIONS), dtype=float)

    # Preserve the historical epsilon-greedy draw order. Evaluation is greedy
    # regardless of which behavior policy was used during training.
    schedule = getattr(self, "exploration_parameters", {}).get(
        "schedule", "multiplicative_epsilon_decay"
    )
    exploring = (
        self.train
        and schedule == "multiplicative_epsilon_decay"
        and self.random_generator.random() < self.epsilon
    )

    if exploring:
        self.logger.debug("Choosing a random learning action.")
        chosen_model_action = self.random_generator.choice(
            self.learning_actions
        )
    elif self.train and schedule == "multiplicative_temperature_decay":
        exploring = True
        probabilities = softmax_probabilities(
            self.model[state][list(self.learning_action_indices)],
            self.temperature,
        )
        chosen_model_action = self.random_generator.choices(
            self.learning_actions, weights=probabilities, k=1,
        )[0]
    else:
        # Exploitation: query the current Q-values of this state.
        self.logger.debug("Querying Q-table for action.")
        q_values = self.model[state]

        learning_q_values = q_values[
            list(self.learning_action_indices)
        ]
        maximum_q_value = np.max(learning_q_values)

        # Resolve equal maximum Q-values randomly.
        best_local_indices = np.flatnonzero(
            np.isclose(learning_q_values, maximum_q_value)
        )
        chosen_local_index = self.random_generator.choice(
            best_local_indices.tolist()
        )
        chosen_model_action = self.learning_actions[
            chosen_local_index
        ]

    chosen_action = (
        self.last_action_feature_action_transform.to_board_action(
            chosen_model_action
        )
    )

    if self.train and hasattr(self, "round_action_counts"):
        # Training update counts use the action stored in the Q-table.
        # Keep selection counts in the same orientation so both telemetry
        # series remain directly comparable under V4 canonicalization.
        self.round_action_counts[chosen_model_action] += 1

    if (self.train and self.feature_version in (TASK3_TACTICAL_BOMB_V9_FEATURE_VERSION,
                                               *TASK3_ARRIVAL_TACTICAL_FEATURE_VERSIONS)
            and hasattr(self, "round_block_diagnostics")):
        for name, count in block_action_statistics(
                state, chosen_model_action,
                self.last_action_coin_route_bomb_diagnostic).items():
            self.round_block_diagnostics[name] = (
                self.round_block_diagnostics.get(name, 0) + count)

    if (
        self.train
        and self.feature_version
        in {
            BFS_POTENTIAL_PRIORITY_V2_FEATURE_VERSION,
            BFS_POTENTIAL_TIE_BREAK_FEATURE_VERSION,
        }
        and state is not None
        and hasattr(self, "round_potential_hint_diagnostics")
    ):
        hint_code = state[-1]

        if hint_code != 0:
            diagnostics = self.round_potential_hint_diagnostics
            followed_hint = (
                chosen_model_action == ACTIONS[hint_code - 1]
            )
            hint_is_bfs_tie_break = bool(
                state[4 + hint_code - 1]
            )
            hint_kind = (
                "tie_break"
                if hint_is_bfs_tie_break
                else "outside_bfs"
            )

            diagnostics["available"] += 1
            diagnostics[f"{hint_kind}_available"] += 1

            if followed_hint:
                diagnostics["followed"] += 1
                diagnostics[f"{hint_kind}_followed"] += 1

            if not exploring:
                diagnostics["exploitation_available"] += 1
                diagnostics[
                    f"{hint_kind}_exploitation_available"
                ] += 1

                if followed_hint:
                    diagnostics["exploitation_followed"] += 1
                    diagnostics[
                        f"{hint_kind}_exploitation_followed"
                    ] += 1

    if not self.train:
        self.last_action_statistics = task2_action_statistics(
            feature_version=self.feature_version,
            state=state,
            chosen_action=chosen_model_action,
            threat_urgency_override=(
                self.last_action_threat_urgency_diagnostic
            ),
            target_class_override=(
                self.last_action_target_class_diagnostic
            ),
            coin_route_bomb_override=(
                self.last_action_coin_route_bomb_diagnostic
            ),
        )

    context = getattr(self.last_action_coin_route_bomb_diagnostic, "evaluation", None)
    if not self.train and context is not None:
        self.last_evaluation_diagnostic = {
            **context, "schema_version": 1, "feature_version": self.feature_version,
            "round": int(game_state["round"]), "step": int(game_state["step"]),
        }

    return chosen_action


def model_action_for_last_feature_state(
    self,
    board_action: str | None,
) -> str | None:
    """Map an executed board action into the last state's Q orientation."""
    transform = getattr(
        self,
        "last_action_feature_action_transform",
        IDENTITY_FEATURE_ACTION_TRANSFORM,
    )
    return transform.to_canonical_action(board_action)
