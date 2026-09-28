"""Evaluation diagnostics from live decisions and authoritative engine events.

No second feature extraction, replay simulation or per-round files. V9's world
observer also checks local BOMB/hold certificates after action selection, outside
the policy timer. Analysis only aggregates the existing evaluation result JSONs.
"""

from collections import deque
from copy import deepcopy
import json
import math
from pathlib import Path

VERSIONS = ("task3_parallel_masks_pressure_bomb_v7",
            "task3_parallel_masks_robust_safety_v8",
            "task3_parallel_masks_tactical_bombs_v9",
            "task3_parallel_masks_tactical_bombs_v10",
            "task3_parallel_masks_tactical_bombs_v11_1",
            "task3_parallel_masks_tactical_bombs_v11_2",
            "task3_parallel_masks_tactical_bombs_v11_3",
            "task3_parallel_masks_tactical_bombs_v12",
            "task3_report_v12_no_hold_pooling_v1",
            "task3_report_v12_nominal_opponent_safety_v1",
            "task3_report_v12_no_coin_route_targets_v1",
            "task3_report_v12_no_block_targets_v1")
ACTIONS = ("UP", "RIGHT", "DOWN", "LEFT", "WAIT")
CATEGORIES = ("unsafe_despite_safe", "lost_safety_despite_following", "other")
SCHEMA = 1
BLOCK_SCENARIOS = ("own_preplanned", "own_later", "opponent", "mixed_unknown")
TARGET_MODES = ("pressure", "crates", "block", "none")
BLOCK_COUNTS = ("target_steps", "target_followed", "wait_steps", "wait_followed")
HOLD_EXECUTION_COUNTS = ("observed_steps", "stayed_steps", "left_steps",
                         "invalid_steps", "non_wait_stayed_steps")
CERTIFIED_COUNTS = (*BLOCK_COUNTS, "selected_target_steps", "selected_wait_steps",
                    "approach_observed_steps", "approach_executed_steps")
TACTICAL_SCHEMA = 2
BLOCK_TRACE_LIMIT = 32
NEW_BOMB = "hypothetical_own_bomb"


def snapshot_hazards(bombs, explosions, arena):
    """Owner/ID metadata at observation time, never exposed to the policy."""
    import settings

    result = [{"id": getattr(b, "diagnostic_id", None), "owner": b.owner.name,
               "start": int(b.timer) + 1,
               "end": int(b.timer) + settings.EXPLOSION_TIMER,
               "cells": tuple(b.get_blast_coords(arena))} for b in bombs]
    result.extend({"id": getattr(e, "diagnostic_id", None), "owner": e.owner.name,
                   "start": 1, "end": int(e.timer) - 1, "cells": tuple(e.blast_coords)}
                  for e in explosions if e.is_dangerous() and e.timer > 1)
    return result


def plan_evidence(analysis, timeline, plan, hazards, *, placing=False):
    """Identify hazards hitting possible victim paths in the certified hold.

    Prefer a bomb whose fire alone makes this held gate fatal, then a sufficient
    single-owner group. Other bombs' physical obstacles/crate openings stay as
    observed. If several owners suffice or only their union suffices, ownership
    stays mixed. This is certificate attribution, not an actual-killer forecast.
    """
    gate = analysis.bit(plan.position)
    reached = analysis.opponents[plan.opponent]
    sources, missing = {}, False
    encoded = [(h, analysis.encode(h["cells"])) for h in hazards]
    for t in range(1, timeline.horizon + 1):
        blocked = analysis.me if placing and t == 1 else 0
        if max(1, plan.arrival) <= t <= plan.hold_until:
            blocked |= gate
        enterable = timeline.before[t] & ~(timeline.bombs[t] | blocked)
        possible = reached | (analysis.neighbours(reached) & enterable)
        lethal = possible & timeline.danger[t]
        accounted = 0
        for hazard, cells in encoded:
            if hazard["start"] <= t <= hazard["end"] and lethal & cells:
                accounted |= cells
                if hazard["id"] is None:
                    missing = True
                else:
                    sources[hazard["id"]] = hazard["owner"]
        missing |= bool(lethal & ~accounted)
        reached = possible & ~timeline.danger[t]
    if reached:
        raise ValueError("Block diagnostic received a non-complete trap")
    if not missing and sources:
        from agent_code.q_learning_agent.task3_bomb_mechanics import Timeline

        def sufficient(ids):
            danger = [0] * (timeline.horizon + 1)
            for hazard, cells in encoded:
                if hazard["id"] in ids:
                    for t in range(max(1, hazard["start"]),
                                   min(hazard["end"], timeline.horizon) + 1):
                        danger[t] |= cells & timeline.danger[t]
            restricted = Timeline(timeline.before, timeline.after, tuple(danger), timeline.bombs)
            return not analysis._enemy_reach(
                analysis.opponents[plan.opponent], restricted,
                analysis.me if placing else 0, gate,
                max(1, plan.arrival), plan.hold_until)[-1]

        single = {bomb for bomb in sources if sufficient({bomb})}
        if single:
            sources = {bomb: owner for bomb, owner in sources.items() if bomb in single}
        else:
            sufficient_owners = {owner for owner in sources.values()
                                 if sufficient({b for b, o in sources.items() if o == owner})}
            if sufficient_owners:
                sources = {bomb: owner for bomb, owner in sources.items() if owner in sufficient_owners}
    return {"victim": analysis.state["others"][plan.opponent][0],
            "sources": sources, "unknown_source": bool(missing or not sources),
            "position": list(plan.position), "arrival": plan.arrival,
            "hold_until": plan.hold_until,
            # Survival includes every known hazard in the certified escape,
            # not just the explosions that can kill the blocked opponent.
            "hazard_end": max((h["end"] for h in hazards), default=0)}


def local_combo_evidence(analysis, hazards, *, robust_local=False):
    """Local, available and safe raw-pressure<1 -> complete-body-trap proof.

    This runs only in evaluation and independently of the selected target/action.
    A pressure of 1 from the bomb alone is never counted as a body combination.
    """
    import settings

    state = analysis.state
    position = tuple(state["self"][3])
    if (not state["self"][2] or not analysis.opponents
            or not (analysis.local_placement_legal() if robust_local
                    else analysis.nominal_survival(position, 0))):
        return []
    raw = analysis.pressure(position, 0)
    if raw >= 1.0:
        return []
    base, scenario = analysis.scenario(position, 0)
    hypothetical = {"id": NEW_BOMB, "owner": state["self"][0],
                    "start": settings.BOMB_TIMER + 1,
                    "end": settings.BOMB_TIMER + settings.EXPLOSION_TIMER,
                    "cells": [divmod(i, analysis.height) for i in range(
                        analysis.width * analysis.height)
                        if analysis.blast(analysis.me) & (1 << i)]}
    result = []
    for i, enemy in enumerate(analysis.opponents):
        if not analysis._enemy_reach(enemy, base, analysis.me)[-1]:
            continue
        plans = analysis.block_plans(scenario, analysis.me, placing=True,
                                    first_only=True, eligible_opponents=(i,))
        if plans:
            evidence = plan_evidence(analysis, scenario, plans[0],
                                     [*hazards, hypothetical], placing=True)
            result.append({**evidence, "raw_pressure": raw, "pressure_with_block": 1.0})
    return result


class CertifiedOpportunityTracker:
    """Available certificates, independent of the planner's selected objective.

    Action-following counts describe the chosen action. Engine observations
    separately distinguish physical HOLD/movement from invalid or blocked moves.
    Missing older packets remain unavailable, never zero opportunities.
    """

    def __init__(self):
        self.observations = self.missing = 0
        self.pending = None
        self.scenarios = {s: {**dict.fromkeys(CERTIFIED_COUNTS, 0),
                             "hold_execution": dict.fromkeys(HOLD_EXECUTION_COUNTS, 0)}
                          for s in BLOCK_SCENARIOS}

    def observe(self, step, action, frame, state, analysis, hazards, categorize):
        from agent_code.q_learning_agent.task3_tactical_mechanics import BlockPlan

        self.observations += 1
        self.pending = None
        detail = frame.get("tactical") if frame else None
        plans = detail.get("certified_plans") if isinstance(detail, dict) else None
        mask = detail.get("certified_bomb_mask") if isinstance(detail, dict) else None
        valid = (state is not None and isinstance(plans, (list, tuple))
                 and isinstance(mask, (list, tuple)) and len(mask) == 5
                 and all(bit in (0, 1) for bit in mask[:4]) and mask[4] in (0, 2)
                 and bool(plans) == bool(any(mask)))
        if not valid:
            self.missing += 1
            return
        if not plans:
            return
        if analysis is None:
            raise ValueError("Certified plans require their tactical analysis")
        hold = mask[4] == 2
        compatible = [p for p in plans if p["arrival"] == 0] if hold else plans
        if not compatible:
            raise ValueError("Certified HOLD has no local plan")
        records = [plan_evidence(analysis, analysis.base,
                                BlockPlan(tuple(p["position"]), p["arrival"], p["hold_until"],
                                          tuple(p["first_steps"]), p["opponent"]), hazards)
                   for p in compatible]
        category = categorize(records)
        counts = self.scenarios[category]
        followed = (action == "WAIT" if hold else
                    action in ACTIONS[:4] and bool(mask[ACTIONS.index(action)]))
        selected = frame["mode"] == "block"
        values = {"target_steps": 1, "target_followed": int(followed),
                  "wait_steps": int(hold), "wait_followed": int(hold and followed),
                  "selected_target_steps": int(selected),
                  "selected_wait_steps": int(hold and selected and detail["bomb_mask"][4] == 2)}
        for key, value in values.items():
            counts[key] += value
        self.pending = {"step": step, "category": category, "hold": hold,
                        "position_before": tuple(state["self"][3]),
                        "action": action, "followed": bool(followed)}

    def action_result(self, step, position, events):
        import events as engine_events

        pending = self.pending
        if pending is None or pending["step"] != step:
            return
        counts = self.scenarios[pending["category"]]
        stayed = tuple(position) == pending["position_before"]
        invalid = engine_events.INVALID_ACTION in events
        if pending["hold"]:
            execution = counts["hold_execution"]
            execution["observed_steps"] += 1
            execution["stayed_steps"] += int(stayed)
            execution["left_steps"] += int(not stayed)
            execution["invalid_steps"] += int(invalid)
            execution["non_wait_stayed_steps"] += int(stayed and pending["action"] != "WAIT")
        else:
            counts["approach_observed_steps"] += 1
            counts["approach_executed_steps"] += int(pending["followed"] and not stayed and not invalid)
        self.pending = None

    def finish(self):
        return {"schema_version": 1, "observations": self.observations,
                "missing_observations": self.missing,
                "complete": self.observations > 0 and self.missing == 0,
                "scenarios": self.scenarios}


class TacticalTracker:
    """Per-step recommendations and per-attempt outcomes, evaluation only."""

    def __init__(self, agent, *, arrival_aware=False):
        self.agent = agent
        self.arrival_aware = arrival_aware
        self.modes = dict.fromkeys(TARGET_MODES, 0)
        self.counts = {s: dict.fromkeys(BLOCK_COUNTS, 0) for s in BLOCK_SCENARIOS}
        self.observations = self.missing = 0
        self.local_opportunities = self.local_selected = self.local_placements = 0
        self.pending_combos = []
        self.preplanned = set()
        self.episodes = []
        self.death_step = None
        self.action_context = None
        self.pending_actions = []
        self.certified = CertifiedOpportunityTracker()

    def category(self, records):
        # Plan origin is independent of contributing owners and eventual kills.
        if any((bomb, r["victim"]) in self.preplanned
               for r in records for bomb in r["sources"]):
            return "own_preplanned"
        owners = {owner for r in records for owner in r["sources"].values()}
        if self.agent in owners:
            return "own_later"
        if any(r["unknown_source"] for r in records) or not owners:
            return "mixed_unknown"
        return "opponent"

    def episode(self, step, records, category, *, new_plan=False):
        victims = sorted({r["victim"] for r in records})
        bomb_ids = sorted({bomb for r in records for bomb in r["sources"]})
        deadline = step + max(r["hazard_end"] for r in records) - 1
        # Replanning/re-entry or several WAITs during the same bomb phase must
        # not manufacture multiple kills/attempts. Bomb IDs survive explosions.
        episode = None if new_plan else next((e for e in reversed(self.episodes)
                        if e["own_survived_phase"] is None and e["victims"] == victims
                        and set(e["bomb_ids"]) & set(bomb_ids)), None)
        if episode is None:
            episode = {"id": len(self.episodes) + 1, "category": category,
                       "start_step": step, "last_block_step": None,
                       "hazard_end_step": deadline, "victims": victims, "bomb_ids": bomb_ids,
                       "initiated_by_bomb": False, **dict.fromkeys(BLOCK_COUNTS, 0),
                       "enemy_died": False, "related_enemy_death": False,
                       "own_kill": False, "opponent_self_kill": False,
                       "related_own_kill": False, "related_opponent_self_kill": False,
                       "mixed_attribution": False, "own_survived_phase": None,
                       "phase_complete": False}
            episode.update(source_owners=[], sources={}, unknown_source=False,
                           hold_execution=dict.fromkeys(HOLD_EXECUTION_COUNTS, 0),
                           actions=[], trace_dropped_steps=0)
            self.episodes.append(episode)
        else:
            episode["bomb_ids"] = sorted(set(episode["bomb_ids"]) | set(bomb_ids))
            episode["hazard_end_step"] = max(episode["hazard_end_step"], deadline)
            # Keep the origin even when sufficient hazards change. Step
            # counters and episode outcomes must use this same stable cohort.
        for record in records:
            episode["sources"].update(record["sources"])
            episode["unknown_source"] |= record["unknown_source"]
        episode["source_owners"] = sorted(set(episode["sources"].values()))
        return episode

    def record_action(self, episode, phase):
        if self.action_context is None:
            return
        row = {**self.action_context, "phase": phase, "position_after": None,
               "invalid": None, "events": None}
        episode["actions"].append(row)
        if len(episode["actions"]) > BLOCK_TRACE_LIMIT:
            # Retain both the beginning and end of unusually long episodes.
            del episode["actions"][BLOCK_TRACE_LIMIT // 2]
            episode["trace_dropped_steps"] += 1
        self.pending_actions.append((episode, row))

    def action_result(self, step, position, events):
        """Authoritative position/events AFTER the engine applied the action."""
        import events as engine_events

        self.certified.action_result(step, position, events)

        for episode, row in self.pending_actions:
            if row["step"] != step or row["position_after"] is not None:
                continue
            row["position_after"] = [int(p) for p in position]
            row["events"] = list(events)
            row["invalid"] = engine_events.INVALID_ACTION in events
            if row["phase"] == "hold":
                stayed = row["position_before"] == row["position_after"]
                counts = episode["hold_execution"]
                counts["observed_steps"] += 1
                counts["stayed_steps"] += int(stayed)
                counts["left_steps"] += int(not stayed)
                counts["invalid_steps"] += int(row["invalid"])
                counts["non_wait_stayed_steps"] += int(stayed and row["action"] != "WAIT")

    def observe(self, step, action, frame, state, hazards):
        self.pending_combos = []
        self.pending_actions = []
        self.action_context = None if state is None else {
            "step": step, "action": action,
            "position_before": [int(p) for p in state["self"][3]],
            "bomb_available": bool(state["self"][2]),
            "safety_mask": list(frame["mask"]) if frame else None,
            "bomb_mask": None,
        }
        self.observations += 1
        detail = frame.get("tactical") if frame else None
        mask = detail.get("bomb_mask") if isinstance(detail, dict) else None
        valid = (state is not None and isinstance(detail, dict)
                 and detail.get("schema_version") == 1
                 and isinstance(mask, (list, tuple)) and len(mask) == 5
                 and all(b in (0, 1) for b in mask[:4]) and mask[4] in (0, 1, 2)
                 and isinstance(detail.get("plans"), list)
                 and bool(detail["plans"]) == (frame["mode"] == "block"))
        if not valid:
            self.missing += 1
            self.certified.observe(step, action, None, state, None, hazards, self.category)
            for episode in self.episodes:
                if episode["own_survived_phase"] is None:
                    self.record_action(episode, "unobserved")
            return
        self.action_context["bomb_mask"] = list(mask)
        mode = frame["mode"] if any(mask) else "none"
        self.modes[mode] += 1
        # Full features have already been computed. Only this cheap local
        # analysis/provenance work is additional, after the policy has acted.
        from agent_code.q_learning_agent.task3_tactical_mechanics import (
            TacticalAnalysis, ArrivalTacticalAnalysis, BlockPlan)
        analysis = None
        if (detail["plans"] or detail.get("certified_plans")
                or (state["self"][2] and state.get("others"))):
            analysis = (ArrivalTacticalAnalysis if self.arrival_aware else TacticalAnalysis)(state)
        self.certified.observe(step, action, frame, state, analysis, hazards, self.category)
        if analysis is not None:
            self.pending_combos = local_combo_evidence(
                analysis, hazards, robust_local=self.arrival_aware)
        if self.pending_combos:
            self.local_opportunities += 1
            self.local_selected += int(mask[4] == 1)
        if mode != "block":
            for episode in self.episodes:
                if episode["own_survived_phase"] is None:
                    self.record_action(episode, "other")
            return
        plans = [BlockPlan(tuple(p["position"]), p["arrival"], p["hold_until"],
                           tuple(p["first_steps"]), p["opponent"]) for p in detail["plans"]]
        hold = mask[4] == 2
        followed = action == "WAIT" if hold else action in ACTIONS[:4] and bool(mask[ACTIONS.index(action)])
        # An opportunity's category must not depend on the chosen action.
        # Ownership is metadata; certified own-plan provenance takes precedence.
        compatible = [p for p in plans if p.arrival == 0] if hold else plans
        records = [plan_evidence(analysis, analysis.base, p, hazards) for p in compatible]
        category = self.category(records)
        episode = self.episode(step, records, category)
        category = episode["category"]
        episode["last_block_step"] = step
        values = {"target_steps": 1, "target_followed": int(followed),
                  "wait_steps": int(hold), "wait_followed": int(hold and followed)}
        for name, value in values.items():
            self.counts[category][name] += value
            episode[name] += value
        for active in self.episodes:
            if active["own_survived_phase"] is None:
                self.record_action(active, ("hold" if hold else "approach")
                                   if active is episode else "other")

    def placed(self, bomb_id, step):
        if not self.pending_combos:
            return
        self.local_placements += 1
        records = []
        for proof in self.pending_combos:
            self.preplanned.add((bomb_id, proof["victim"]))
            sources = {(bomb_id if key == NEW_BOMB else key): owner
                       for key, owner in proof["sources"].items()}
            records.append({**proof, "sources": sources})
        # A new accepted combination must not retroactively relabel an earlier
        # opponent-bomb episode that shares one of its hazard sources.
        episode = self.episode(step, records, "own_preplanned", new_plan=True)
        episode["initiated_by_bomb"] = True
        episode["origin_bomb_id"] = bomb_id
        episode["placement_certificates"] = records
        self.record_action(episode, "placement")
        self.pending_combos = []

    def resolve_hits(self, step, hits):
        if any(victim == self.agent for _, _, victim in hits):
            self.death_step = step
        for episode in self.episodes:
            if episode["phase_complete"]:
                continue
            relevant = [(bomb, owner, victim) for bomb, owner, victim in hits
                        if victim in episode["victims"]]
            episode["enemy_died"] |= bool(relevant)
            linked = [(bomb, owner, victim) for bomb, owner, victim in relevant
                      if bomb in episode["bomb_ids"]]
            episode["related_enemy_death"] |= bool(linked)
            episode["own_kill"] |= any(owner == self.agent for _, owner, _ in relevant)
            episode["opponent_self_kill"] |= any(owner == victim for _, owner, victim in relevant)
            episode["related_own_kill"] |= any(owner == self.agent for _, owner, _ in linked)
            episode["related_opponent_self_kill"] |= any(owner == victim for _, owner, victim in linked)
            episode["mixed_attribution"] |= any(
                len({owner for _, owner, target in relevant if target == victim}) > 1
                for victim in episode["victims"])
            if self.death_step is not None or step >= episode["hazard_end_step"]:
                episode["own_survived_phase"] = self.death_step is None
            episode["phase_complete"] = step >= episode["hazard_end_step"]

    def finish(self, step):
        self.resolve_hits(step, [])
        return {"schema_version": TACTICAL_SCHEMA, "agent": self.agent,
                "observations": self.observations,
                "missing_observations": self.missing,
                "complete": self.observations > 0 and self.missing == 0,
                "mode_steps": self.modes, "inactive_steps": max(0, step - self.observations),
                "certified_opportunities": self.certified.finish(),
                "scenarios": self.counts, "local_combos": {
                    "opportunities": self.local_opportunities, "selected_opportunities": self.local_selected,
                    "placements": self.local_placements}, "episodes": self.episodes}


def classify_self_kill(rows, fatal_bombs, death_step, *, mixed_owners=False):
    """Classify one self-kill, not explosion-hit multiplicity or only the last action.

    A/B require one identified fatal bomb marked safe when actually placed.
    Ambiguous, unmarked and otherwise unclassified deaths remain in Other.
    The labels report adherence, not a counterfactual proof of causation.
    """
    result = {
        "death_step": death_step, "category": "other", "reason": None,
        "fatal_bomb_ids": [b["id"] for b in fatal_bombs],
        "first_unsafe_step": None, "first_empty_step": None, "trajectory": [],
    }
    if mixed_owners or len(fatal_bombs) != 1:
        result["reason"] = "ambiguous_explosion_attribution"
        return result
    bomb = fatal_bombs[0]
    window = [r for r in rows if bomb["placed_step"] <= r["step"] <= death_step]
    result["trajectory"] = window
    if not bomb.get("marked_safe"):
        result["reason"] = "placement_not_marked_safe"
        return result
    if (not window or window[0]["step"] != bomb["placed_step"]
            or window[-1]["step"] != death_step or window[0]["action"] != "BOMB"
            or [r["step"] for r in window] != list(range(bomb["placed_step"], death_step + 1))
            or any(len(r.get("mask") or ()) != 5 for r in window)):
        result["reason"] = "incomplete_escape_trace"
        return result
    escape = window[1:]
    violations = [r for r in escape if r["action"] in ACTIONS and any(r["mask"])
                  and not r["mask"][ACTIONS.index(r["action"])]]
    empty = [r for r in escape if not any(r["mask"])]
    result["first_unsafe_step"] = violations[0]["step"] if violations else None
    result["first_empty_step"] = empty[0]["step"] if empty else None
    if violations:
        result.update(category="unsafe_despite_safe", reason="zero_chosen_while_one_available")
    elif escape:
        # Only a continuous prefix of following ones can certify group B.
        # BOMB has no movement safety bit; additional BOMB/ERROR decisions are
        # not silently treated as following WAIT or as a safety violation.
        prefix = escape[:escape.index(empty[0])] if empty else escape
        if all(r["action"] in ACTIONS and
               r["mask"][ACTIONS.index(r["action"])] for r in prefix):
            result.update(category="lost_safety_despite_following",
                          reason="mask_became_empty" if empty else "death_after_safe_action")
        else:
            result["reason"] = "escape_action_without_movement_safety_bit"
    else:
        result["reason"] = "death_during_placement"
    return result


class ChainTracker:
    """Maximal contiguous pressure-mode intervals; bomb ownership fixes success."""

    def __init__(self):
        self.chains = []
        self.bombs = []
        self.current = None
        self.by_bomb = {}

    def observe(self, step, mode, score, crate_target):
        if mode != "pressure":
            self.close(step - 1, "mode_exit")
            return
        if self.current is None:
            self.current = {
                "id": len(self.chains) + 1, "start_step": step, "end_step": step,
                "end_reason": None, "start_score": float(score),
                "eligible": bool(crate_target), "scores": [], "bomb_ids": [],
                "success": False, "kill_steps": [],
            }
            self.chains.append(self.current)
        self.current["scores"].append(float(score))
        self.current["end_step"] = step

    def close(self, step, reason):
        if self.current is not None:
            self.current["end_step"] = step
            self.current["end_reason"] = reason
            self.current = None

    def place(self, bomb_id, step, position, score):
        if self.current is None:
            return
        bomb = {"id": bomb_id, "chain_id": self.current["id"], "placed_step": step,
                "position": [int(p) for p in position], "score": float(score), "kill_steps": [],
                "killed_opponent": False}
        self.bombs.append(bomb)
        self.by_bomb[bomb_id] = bomb
        self.current["bomb_ids"].append(bomb_id)

    def killed(self, bomb_id, step):
        bomb = self.by_bomb.get(bomb_id)
        if bomb is None:
            return
        bomb["killed_opponent"] = True
        bomb["kill_steps"].append(step)
        chain = self.chains[bomb["chain_id"] - 1]
        chain["success"] = True
        chain["kill_steps"].append(step)


class RoundDiagnostics:
    """Small observer owned by the evaluation world, never by the policy."""

    def __init__(self, agent, feature_version, round_number, history_limit):
        self.agent = agent
        self.feature_version = feature_version
        self.round = round_number
        self.history = deque(maxlen=history_limit)
        self.tracker = ChainTracker()
        self.origins = {}
        self.frame = None
        self.self_kill = None
        self.missing_observations = 0
        self.pressure_threshold = None
        self.observations = 0
        self.tactical = (TacticalTracker(agent, arrival_aware=feature_version in VERSIONS[3:])
                         if feature_version in VERSIONS[2:] else None)

    def observe(self, step, action, packet, *, state=None, hazards=()):
        # Round/step prevent a timeout, cache hit or skipped callback from
        # accidentally attributing the preceding decision's mask to this one.
        valid = (isinstance(packet, dict) and packet.get("schema_version") == SCHEMA
                 and packet.get("feature_version") == self.feature_version
                 and packet.get("round") == self.round and packet.get("step") == step
                 and len(packet.get("mask") or ()) == 5
                 and all(bit in (0, 1) for bit in packet["mask"])
                 and packet.get("mode") in ("pressure", "crates", "none", "block")
                 and all(isinstance(packet.get(key), (int, float))
                         and math.isfinite(packet[key]) and 0 <= packet[key] <= 1
                         for key in ("score", "local_pressure", "pressure_threshold"))
                 and isinstance(packet.get("crate_target"), bool)
                 and isinstance(packet.get("bomb_safe"), bool))
        self.frame = packet if valid else None
        self.observations += 1
        self.history.append({"step": step, "action": action,
                             "mask": list(packet["mask"]) if valid else None})
        if self.tactical is not None:
            self.tactical.observe(step, action, self.frame, state, hazards)
        if not valid:
            self.missing_observations += 1
            self.tracker.close(step - 1, "missing_observation")
            return
        self.pressure_threshold = packet["pressure_threshold"]
        self.tracker.observe(step, packet["mode"], packet["score"], packet["crate_target"])

    def placed(self, bomb_id, step, position):
        """Called only AFTER the engine accepted a BOMB action."""
        self.origins[bomb_id] = {"id": bomb_id, "placed_step": step,
                                 "marked_safe": bool(self.frame and self.frame["bomb_safe"])}
        if self.frame is not None:
            self.tracker.place(bomb_id, step, position, self.frame["local_pressure"])
        if self.tactical is not None:
            self.tactical.placed(bomb_id, step)

    def action_result(self, step, position, events):
        if self.tactical is not None:
            self.tactical.action_result(step, position, events)

    def resolve_hits(self, step, hits):
        """Observe all simultaneous (bomb_id, owner, victim) explosion hits."""
        own_fatal = {}
        if self.tactical is not None:
            self.tactical.resolve_hits(step, hits)
        victim_owners = set()
        for bomb_id, owner, victim in hits:
            if owner == self.agent and victim != self.agent:
                self.tracker.killed(bomb_id, step)
            if victim == self.agent:
                victim_owners.add(owner)
                if owner == self.agent:
                    own_fatal[bomb_id] = self.origins.get(
                        bomb_id, {"id": bomb_id, "placed_step": step, "marked_safe": False})
        if own_fatal and self.self_kill is None:
            self.self_kill = classify_self_kill(
                list(self.history), list(own_fatal.values()), step,
                mixed_owners=len(victim_owners) > 1)
        if victim_owners:
            self.tracker.close(step, "death")

    def finish(self, step):
        self.tracker.close(step, "round_end")
        return {"schema_version": SCHEMA, "feature_version": self.feature_version,
                "round": self.round, "steps": step, "observations": self.observations,
                "pressure_threshold": self.pressure_threshold,
                "missing_observations": self.missing_observations,
                "pressure_complete": self.missing_observations == 0 and self.observations > 0,
                "self_kill": self.self_kill, "chains": self.tracker.chains,
                "pressure_bombs": self.tracker.bombs,
                **({"tactical": self.tactical.finish(step)} if self.tactical is not None else {})}


def score_bin(score):
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError(f"Invalid pressure score: {score}")
    return min(9, int(math.floor(score * 10 + 1e-12)))


def normalize_tactical_record(tactical):
    """Recover proven legacy origins without inventing missing action traces.

    V1 counted steps before possibly relabeling an episode mixed. Reassign
    counters only if its original episode buckets reconcile exactly with ALL
    per-step buckets. Otherwise retain the unresolved legacy attribution.
    """
    version = tactical.get("schema_version")
    if version not in (1, TACTICAL_SCHEMA):
        raise ValueError("Unsupported tactical diagnostic schema")
    if version == TACTICAL_SCHEMA:
        return tactical
    result = deepcopy(tactical)
    reconciled = all(
        counts[key] == sum(e[key] for e in result["episodes"] if e["category"] == category)
        for category, counts in result["scenarios"].items() for key in BLOCK_COUNTS)
    result["legacy_attribution_unresolved"] = not reconciled
    result["legacy_corrected_episodes"] = 0
    if reconciled:
        for episode in result["episodes"]:
            proofs = episode.get("placement_certificates", [])
            if (episode["initiated_by_bomb"] and proofs
                    and all(p.get("raw_pressure", 1) < 1
                            and p.get("pressure_with_block") == 1 for p in proofs)):
                result["legacy_corrected_episodes"] += int(episode["category"] != "own_preplanned")
                episode["category"] = "own_preplanned"
                episode["sources"] = {bomb: owner for p in proofs for bomb, owner in p["sources"].items()}
                episode["source_owners"] = sorted(set(episode["sources"].values()))
                episode["unknown_source"] = any(p["unknown_source"] for p in proofs)
        result["scenarios"] = {category: {
            key: sum(e[key] for e in result["episodes"] if e["category"] == category)
            for key in BLOCK_COUNTS} for category in BLOCK_SCENARIOS}
    return result


def aggregate_hold_execution(episodes, opportunities):
    """Missing legacy engine observations are unavailable, never zero holds."""
    rows = [e["hold_execution"] for e in episodes if "hold_execution" in e]
    counts = {key: sum(row[key] for row in rows) for key in HOLD_EXECUTION_COUNTS}
    n = counts["observed_steps"]
    return {**counts, "opportunities": opportunities,
            "status": ("no_opportunities" if not opportunities else
                       "unavailable" if not n else "complete" if n == opportunities else "partial"),
            "stay_rate": counts["stayed_steps"] / n if n else None}


def aggregate_certified_opportunities(rounds):
    """Pool certified opportunity steps without conditioning on selected targets."""
    recorded, complete = [], []
    for record in rounds:
        tactical = record.get("tactical")
        value = tactical.get("certified_opportunities") if isinstance(tactical, dict) else None
        if not isinstance(value, dict):
            continue
        recorded.append(value)
        if value.get("schema_version") != 1:
            raise ValueError("Unsupported certified-opportunity schema")
        observations, missing = value["observations"], value["missing_observations"]
        if (type(observations) is not int or type(missing) is not int
                or not 0 <= missing <= observations):
            raise ValueError("Invalid certified-opportunity observation counts")
        if not value["complete"]:
            continue
        if observations <= 0 or missing or observations != tactical["observations"]:
            raise ValueError("Inconsistent complete certified-opportunity coverage")
        if set(value["scenarios"]) != set(BLOCK_SCENARIOS):
            raise ValueError("Missing certified-opportunity scenarios")
        for counts in value["scenarios"].values():
            execution = counts["hold_execution"]
            approach = counts["target_steps"] - counts["wait_steps"]
            approach_followed = counts["target_followed"] - counts["wait_followed"]
            if (any(type(counts[key]) is not int or counts[key] < 0 for key in CERTIFIED_COUNTS)
                    or not 0 <= counts["wait_followed"] <= counts["wait_steps"] <= counts["target_steps"]
                    or not counts["wait_followed"] <= counts["target_followed"] <= counts["target_steps"]
                    or approach_followed > approach
                    or counts["selected_target_steps"] > counts["target_steps"]
                    or counts["selected_wait_steps"] > min(counts["wait_steps"], counts["selected_target_steps"])
                    or counts["approach_observed_steps"] > approach
                    or counts["approach_executed_steps"] > min(approach_followed, counts["approach_observed_steps"])
                    or any(type(execution[key]) is not int or execution[key] < 0 for key in HOLD_EXECUTION_COUNTS)
                    or execution["observed_steps"] > counts["wait_steps"]
                    or execution["stayed_steps"] + execution["left_steps"] != execution["observed_steps"]
                    or execution["invalid_steps"] > execution["observed_steps"]
                    or execution["non_wait_stayed_steps"] > execution["stayed_steps"]):
                raise ValueError("Inconsistent certified-opportunity counts")
        if sum(c["target_steps"] for c in value["scenarios"].values()) > observations:
            raise ValueError("More certified opportunity steps than observations")
        complete.append(value)

    def rate(numerator, denominator):
        return numerator / denominator if denominator else None

    scenarios = {}
    for scenario in BLOCK_SCENARIOS:
        rows = [value["scenarios"][scenario] for value in complete]
        counts = {key: sum(row[key] for row in rows) for key in CERTIFIED_COUNTS}
        execution = {key: sum(row["hold_execution"][key] for row in rows)
                     for key in HOLD_EXECUTION_COUNTS}
        observed = execution["observed_steps"]
        holds = counts["wait_steps"]
        approach = counts["target_steps"] - holds
        approach_followed = counts["target_followed"] - counts["wait_followed"]
        scenarios[scenario] = {
            **counts, "approach_steps": approach, "approach_followed": approach_followed,
            "target_follow_rate": rate(counts["target_followed"], counts["target_steps"]),
            "target_selection_rate": rate(counts["selected_target_steps"], counts["target_steps"]),
            "wait_follow_rate": rate(counts["wait_followed"], holds),
            "approach_follow_rate": rate(approach_followed, approach),
            "approach_execution_rate": rate(counts["approach_executed_steps"], counts["approach_observed_steps"]),
            "target_steps_per_round": rate(counts["target_steps"], len(complete)),
            "wait_steps_per_round": rate(holds, len(complete)),
            "hold_execution": {**execution, "opportunities": holds,
                "status": ("no_opportunities" if not holds else "unavailable" if not observed
                           else "complete" if observed == holds else "partial"),
                "stay_rate": rate(execution["stayed_steps"], observed)},
        }
    observed = sum(value["observations"] - value["missing_observations"] for value in recorded)
    return {"schema_version": 1,
            "status": ("unavailable" if not observed else
                       "complete" if len(complete) == len(rounds) else "partial"),
            "recorded_rounds": len(recorded), "complete_rounds": len(complete),
            "missing_rounds": len(rounds) - len(recorded),
            "incomplete_rounds": len(recorded) - len(complete),
            "observed_steps": observed,
            "missing_observations": sum(value["missing_observations"] for value in recorded),
            "scenarios": scenarios}


def aggregate_tactical_rounds(rounds):
    """Equal round weights for time shares; pooled event counts for adherence."""
    import numpy as np

    recorded = [r for r in rounds if isinstance(r.get("tactical"), dict)]
    complete = []
    for r in recorded:
        t = normalize_tactical_record(r["tactical"])
        if not t["complete"]:
            continue
        steps = r["steps"]
        if (steps <= 0 or t["missing_observations"] != 0
                or sum(t["mode_steps"].values()) + t["inactive_steps"] != steps
                or sum(c["target_steps"] for c in t["scenarios"].values()) != t["mode_steps"]["block"]):
            raise ValueError("Inconsistent tactical diagnostic denominators")
        for counts in t["scenarios"].values():
            if (any(type(v) is not int or v < 0 for v in counts.values())
                    or not 0 <= counts["wait_followed"] <= counts["wait_steps"] <= counts["target_steps"]
                    or not counts["wait_followed"] <= counts["target_followed"] <= counts["target_steps"]):
                raise ValueError("Inconsistent tactical opportunity/follow counts")
        for episode in t["episodes"]:
            execution = episode.get("hold_execution")
            if execution is not None:
                if (any(type(v) is not int or v < 0 for v in execution.values())
                        or execution["observed_steps"] > episode["wait_steps"]
                        or execution["stayed_steps"] + execution["left_steps"] != execution["observed_steps"]
                        or execution["invalid_steps"] > execution["observed_steps"]
                        or execution["non_wait_stayed_steps"] > execution["stayed_steps"]):
                    raise ValueError("Inconsistent HOLD execution counts")
        if t["schema_version"] == TACTICAL_SCHEMA and any(
                t["scenarios"][category][key] != sum(e[key] for e in t["episodes"] if e["category"] == category)
                for category in BLOCK_SCENARIOS for key in BLOCK_COUNTS):
            raise ValueError("Inconsistent stable episode attribution")
        complete.append({**r, "tactical": t})

    def moments(values):
        return {"mean": float(np.mean(values)) if values else None,
                "sd": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                "observations": len(values)}

    scenarios = {}
    for scenario in BLOCK_SCENARIOS:
        counts = {key: sum(r["tactical"]["scenarios"][scenario][key] for r in complete)
                  for key in BLOCK_COUNTS}
        episodes = [e for r in complete for e in r["tactical"]["episodes"]
                    if e["category"] == scenario]
        attempts = [e for e in episodes if e["initiated_by_bomb"] or e["target_followed"] > 0]
        outcomes = {key: sum(bool(e[key]) for e in attempts) for key in (
            "enemy_died", "related_enemy_death", "own_kill", "opponent_self_kill",
            "related_own_kill", "related_opponent_self_kill", "mixed_attribution")}
        resolved = sum(e["phase_complete"] for e in attempts)
        survival_known = sum(e["own_survived_phase"] is not None for e in attempts)
        survived = sum(e["own_survived_phase"] is True for e in attempts)
        scenarios[scenario] = {
            **counts,
            "approach_steps": counts["target_steps"] - counts["wait_steps"],
            "approach_followed": counts["target_followed"] - counts["wait_followed"],
            "hold_execution": aggregate_hold_execution(episodes, counts["wait_steps"]),
            "target_steps_per_round": moments([r["tactical"]["scenarios"][scenario]["target_steps"]
                                               for r in complete]),
            "wait_steps_per_round": moments([r["tactical"]["scenarios"][scenario]["wait_steps"]
                                             for r in complete]),
            "target_follow_rate": counts["target_followed"] / counts["target_steps"] if counts["target_steps"] else None,
            "wait_follow_rate": counts["wait_followed"] / counts["wait_steps"] if counts["wait_steps"] else None,
            "outcomes": {"episodes": len(episodes), "attempts": len(attempts),
                         "attempts_with_block_following": sum(e["target_followed"] > 0 for e in attempts),
                         "attempts_with_wait": sum(e["wait_followed"] > 0 for e in attempts),
                         "placements_without_block_following": sum(
                             e["initiated_by_bomb"] and not e["target_followed"] for e in attempts),
                         **outcomes, "phase_complete_attempts": resolved,
                         "censored_attempts": len(attempts) - resolved,
                         "own_survival_known": survival_known, "own_survived_phase": survived,
                         "own_survival_rate": survived / survival_known if survival_known else None},
        }
    return {"schema_version": TACTICAL_SCHEMA,
            "status": "unavailable" if not recorded else
                      "complete" if len(complete) == len(rounds) else "partial",
            "recorded_rounds": len(recorded), "complete_rounds": len(complete),
            "missing_rounds": len(rounds) - len(recorded),
            "incomplete_rounds": len(recorded) - len(complete),
            "legacy_corrected_episodes": sum(r["tactical"].get("legacy_corrected_episodes", 0) for r in complete),
            "legacy_unresolved_rounds": sum(r["tactical"].get("legacy_attribution_unresolved", False) for r in complete),
            "mode_shares": {mode: moments([
                r["tactical"]["mode_steps"][mode] / r["steps"] for r in complete]) for mode in TARGET_MODES},
            "inactive_share": moments([r["tactical"]["inactive_steps"] / r["steps"] for r in complete]),
            "scenarios": scenarios,
            "certified_opportunities": aggregate_certified_opportunities(rounds),
            "local_combos": {key: sum(r["tactical"]["local_combos"][key] for r in complete)
                             for key in ("opportunities", "selected_opportunities", "placements")}}


def aggregate_rounds(rounds):
    import numpy as np

    counts = dict.fromkeys(CATEGORIES, 0)
    groups = [[] for _ in range(10)]
    for result in rounds:
        if result["self_kill"] is not None:
            counts[result["self_kill"]["category"]] += 1
        if not result.get("pressure_complete", True):
            continue
        for chain in result["chains"]:
            if chain["eligible"]:
                groups[score_bin(chain["start_score"])].append(chain)
    bins, curves = [], []
    for i, chains in enumerate(groups):
        n = len(chains)
        wins = sum(c["success"] for c in chains)
        bins.append({"lower": i / 10, "upper": (i + 1) / 10, "count": n,
                     "successes": wins, "success_rate": wins / n if n else None})
        curve = {"lower": i / 10, "upper": (i + 1) / 10, "chain_count": n,
                 "mean": [], "sd": [], "count": []}
        for t in range(max((len(c["scores"]) for c in chains), default=0)):
            values = [c["scores"][t] for c in chains if t < len(c["scores"])]
            curve["mean"].append(float(np.mean(values)))
            curve["sd"].append(float(np.std(values, ddof=1)) if len(values) > 1 else None)
            curve["count"].append(len(values))
        curves.append(curve)
    return {"evaluation_rounds": len(rounds),
            "self_kills": {"total": sum(counts.values()), **counts},
            "pressure": {"eligible_chains": sum(len(g) for g in groups),
                         "bins": bins, "curves": curves},
            "tactical": aggregate_tactical_rounds(rounds)}


def analyze_evaluation_diagnostics(experiment_file, index):
    """Read only the existing per-seed JSONs; missing records stay unavailable."""
    configuration = index["configuration"]
    version, agent = configuration["feature_version"], configuration["agent"]
    if version not in VERSIONS:
        return {"schema_version": SCHEMA, "status": "unsupported_feature"}
    root = Path(experiment_file).resolve().parent
    models, all_rounds, thresholds = [], [], set()
    collected = missing = incomplete = 0
    for run in index["runs"]:
        model_rounds = []
        for evaluation in run["evaluations"]:
            path = (root / evaluation["results"]).resolve()
            if not path.is_relative_to(root):
                raise ValueError("Evaluation result path escapes experiment directory")
            statistics = json.loads(path.read_text(encoding="utf-8"))
            for round_id, expected in statistics["by_round"].items():
                is_self = any(k["killer"] == k["victim"] == agent
                              for k in expected.get("kill_events", []))
                result = expected["by_agent"][agent].get("diagnostics")
                if result is None:
                    missing += 1
                    result = {"self_kill": {"category": "other", "reason": "not_recorded"}
                              if is_self else None, "chains": [], "pressure_complete": False}
                else:
                    if (result.get("schema_version") != SCHEMA
                            or result.get("feature_version") != version
                            or result.get("steps") != expected["steps"]
                            or (result.get("self_kill") is not None) != is_self):
                        raise ValueError(f"Inconsistent evaluation diagnostics: {path.name}, {round_id}")
                    collected += 1
                    incomplete += int(not result["pressure_complete"])
                    if result["pressure_threshold"] is not None:
                        thresholds.add(result["pressure_threshold"])
                model_rounds.append(result)
        all_rounds.extend(model_rounds)
        models.append({"training_world_seed": run["training_world_seed"],
                       "training_agent_seed": run["training_agent_seed"],
                       **aggregate_rounds(model_rounds)})
    if len(thresholds) > 1:
        raise ValueError("Mixed pressure thresholds in one experiment")
    return {"schema_version": SCHEMA, "source": "evaluation", "feature_version": version,
            "pressure_threshold": next(iter(thresholds), None),
            "status": "unavailable" if not collected else
                      ("partial" if missing or incomplete else "complete"),
            "collected_rounds": collected, "missing_diagnostic_rounds": missing,
            "incomplete_pressure_rounds": incomplete, "models": models,
            **aggregate_rounds(all_rounds)}
