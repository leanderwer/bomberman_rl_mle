"""Evaluation counters derived from the exact features used for an action.

These observations retain the existing result schema. They do not add
rewards, choose actions, or change the feature state.
"""

from .features import ACTIONS
from .bomb_mechanics import PressureBombDiagnosticContext
from .navigation import (
    TARGET_CASE_NONE, TARGET_CASE_LOCAL_BOMB, TARGET_CASE_COIN_ONLY,
    TARGET_CASE_BOMB_ONLY, TARGET_CASE_JOINT, TARGET_CASE_TRUE_CONFLICT,
    classify_safe_target_navigation,
)


def _mask_followed(mask, action, actions):
    return action in actions and bool(mask[actions.index(action)])


def block_action_statistics(state, chosen_action, context):
    """Count selected block targets and HOLD separately from bomb placement."""
    active = (
        state is not None and isinstance(context, PressureBombDiagnosticContext)
        and context.pressure.target_family == "block"
    )
    local_hold = bool(active and state[13] == 2)
    followed = bool(active and (
        chosen_action == "WAIT" if local_hold
        else _mask_followed(state[9:13], chosen_action, ACTIONS[:4])
    ))
    return {
        "tactical_block_observations": 1,
        "block_target_opportunities": int(active),
        "block_target_followed": int(followed),
        "block_local_wait_opportunities": int(local_hold),
        "block_local_wait_followed": int(local_hold and chosen_action == "WAIT"),
    }


def _navigation_statistics(state, action, urgency, classes, context):
    safety, coins, bomb_directions = state[:5], state[5:9], state[9:13]
    bomb_here = state[13] == 1
    bomb_available = bool(state[-1])
    statistics = {
        "self_kill_diagnostic_observations": 1,
        "independent_target_diagnostic_observations": 1,
    }
    if any(safety) and action in ACTIONS[:5]:
        safe = _mask_followed(safety, action, ACTIONS[:5])
        statistics.update(
            survivable_action_opportunities=1,
            survivable_action_followed=int(safe),
            unsafe_action_with_safe_alternative=int(not safe),
        )
        if urgency in (0, 1, 2, 3):
            statistics[f"threat_urgency_{urgency}_opportunities"] = 1
            statistics[f"threat_urgency_{urgency}_followed"] = int(safe)
    elif not any(safety):
        statistics["no_survivable_action_available"] = 1

    target_case = classify_safe_target_navigation(
        safety_mask=safety,
        coin_mask=coins,
        bomb_target_mask=(*bomb_directions, int(bomb_here)),
        local_bomb_opportunity=bomb_here and bomb_available,
    )
    if target_case.name == TARGET_CASE_TRUE_CONFLICT:
        if classes is None:
            raise ValueError("Target diagnostics require both target classes in a conflict.")
        coin_class, bomb_class = map(int, classes)
        prefix = f"target_conflict_coin_class_{coin_class}_bomb_class_{bomb_class}"
        statistics[f"{prefix}_opportunities"] = 1
        statistics[f"{prefix}_coin_followed"] = int(
            _mask_followed(target_case.safe_coin_mask, action, ACTIONS[:4]))
        statistics[f"{prefix}_bomb_followed"] = int(
            _mask_followed(target_case.safe_bomb_mask, action, ACTIONS[:4]))
    elif target_case.name == TARGET_CASE_NONE:
        statistics["no_safe_target_opportunities"] = 1
    elif target_case.name != TARGET_CASE_LOCAL_BOMB:
        prefix, mask = {
            TARGET_CASE_JOINT: ("safe_joint_target", target_case.safe_joint_mask),
            TARGET_CASE_COIN_ONLY: ("safe_coin_only_target", target_case.safe_coin_mask),
            TARGET_CASE_BOMB_ONLY: ("safe_bomb_only_target", target_case.safe_bomb_mask),
        }[target_case.name]
        statistics[f"{prefix}_opportunities"] = 1
        statistics[f"{prefix}_followed"] = int(_mask_followed(mask, action, ACTIONS[:4]))

    primary = context.primary_mask
    valid_bomb = action == "BOMB" and bomb_available
    if valid_bomb:
        statistics["valid_bomb_drops"] = 1
    if primary[4] and bomb_available:
        statistics["bomb_target_opportunities"] = 1
    if valid_bomb and bomb_here:
        statistics["bomb_drops_at_target"] = 1
    if valid_bomb and primary[4]:
        statistics["bomb_drops_at_primary_target"] = 1

    statistics["coin_route_bomb_diagnostic_observations"] = 1
    alternative = context.alternative_mask
    ratio = context.alternative_score_ratio_milli
    if ratio is not None:
        statistics["coin_route_bomb_alternative_opportunities"] = 1
        statistics["coin_route_bomb_alternative_score_ratio_milli_sum"] = int(ratio)
        primary_joint = tuple(bool(safety[i] and coins[i] and primary[i]) for i in range(4))
        alternative_joint = tuple(
            bool(safety[i] and coins[i] and alternative[i]) for i in range(4)
        )
        if (target_case.name == TARGET_CASE_JOINT
                and not any(primary_joint) and any(alternative_joint)):
            statistics["coin_route_bomb_joint_conversion_opportunities"] = 1
            statistics["coin_route_bomb_joint_conversion_followed"] = int(
                _mask_followed(alternative_joint, action, ACTIONS[:4]))
        if alternative[4] and bomb_available:
            statistics["coin_route_bomb_alternative_target_opportunities"] = 1
            statistics["coin_route_bomb_alternative_target_followed"] = int(action == "BOMB")
    return statistics


def action_statistics(
    state, chosen_action, threat_urgency_override=None,
    target_class_override=None, coin_route_bomb_override=None,
):
    """Publish V12 adherence counters using physical bomb availability.

    HOLD pools the last Q-key entry, so that entry must not be used as a
    legality signal. Selected block directions are also kept out of the
    coin/crate navigation heatmaps.
    """
    if state is None:
        return {}
    context = coin_route_bomb_override
    if not isinstance(context, PressureBombDiagnosticContext) or context.bomb_available is None:
        raise ValueError("V12 diagnostics require target context and real bomb availability.")
    physical_state = (*state[:-1], int(context.bomb_available))
    diagnostic = context.pressure
    block_active = diagnostic.target_family == "block"
    statistics_state = (
        (*physical_state[:5], *(0,) * 9, physical_state[-1])
        if block_active else physical_state
    )
    statistics = _navigation_statistics(
        statistics_state, chosen_action, threat_urgency_override, target_class_override, context
    )
    if block_active:
        statistics.pop("no_safe_target_opportunities", None)
    statistics.update(block_action_statistics(physical_state, chosen_action, context))

    pressure_active = diagnostic.target_family == "pressure"
    if pressure_active:
        # Crate-quality categories cannot rank an attack score.
        statistics = {
            key: value for key, value in statistics.items()
            if not key.startswith("target_conflict_coin_class_")
        }
    local_pressure = bool(physical_state[13] == 1 and physical_state[14] and pressure_active)
    followed = (
        chosen_action == "BOMB" if local_pressure
        else _mask_followed(physical_state[9:13], chosen_action, ACTIONS[:4])
    )
    statistics.update({
        "pressure_target_opportunities": int(pressure_active),
        "pressure_target_followed": int(pressure_active and followed),
        "pressure_local_bomb_opportunities": int(local_pressure),
        "pressure_local_bomb_followed": int(local_pressure and chosen_action == "BOMB"),
        "bomb_blockade_check_opportunities": int(diagnostic.blockade_checked),
        "bomb_blockade_check_followed": int(diagnostic.blockade_passed),
    })
    return statistics
