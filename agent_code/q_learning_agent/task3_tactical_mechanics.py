"""V9-V12 bomb pressure and executable, complete body traps.

All counts start at observed opponent positions. Approach distance discounts
the opportunity, while actual bomb-target survival still uses arrival time.
A body trap is one fixed held cell: a timed approach, a sufficient hold, and
a robust surviving continuation. Partial body restrictions receive no bonus.
Only known bombs are simulated. This module neither selects policy actions
nor changes the V6/V7/V8 mechanics or the six-action Q-learning action space.
V10 adds arrival-aware reference positions, targeted retry memory and pooled
HOLD keys (encoded in features.py), while preserving the V9 entry point.
V11 retains V10 forecasts and HOLD pooling, restores existing-block priority,
and derives certified movement/WAIT safety before selecting a target.
V12 conditions WAIT safety on our occupied cell and drops retry memory.
"""

from copy import copy
from dataclasses import dataclass
from math import isfinite
from numbers import Real

from . import task2_mechanics as v6
from .task3_bomb_mechanics import (
    BOMB_TARGET_DISCOUNT, PRESSURE_THRESHOLD, ZERO_BOMB_MASK,
    BombAnalysis, BombDiagnostics, PressureCandidate, Timeline, _select_targets,
)


@dataclass(frozen=True)
class BlockPlan:
    position: tuple[int, int]
    arrival: int
    hold_until: int
    first_steps: tuple[int, int, int, int]
    opponent: int


class TacticalAnalysis(BombAnalysis):
    """Reuse V8 timelines and robustness; add one-cell timed trap searches."""

    def _contested(self, timeline, placement_cell=0, gate=0, start=1, end=0):
        # Opponents may move during our BOMB action. Only its occupied cell
        # is unavailable to them for that first action in the snapshot.
        contested = [0] * (timeline.horizon + 1)
        reached = list(self.opponents)
        for t in range(1, timeline.horizon + 1):
            enterable = timeline.before[t] & ~timeline.bombs[t]
            if t == 1:
                enterable &= ~placement_cell
            if start <= t <= end:
                enterable &= ~gate
            for i, old in enumerate(reached):
                possible = old | (self.neighbours(old) & enterable)
                contested[t] |= old | possible
                reached[i] = possible & ~timeline.danger[t]
        return contested

    def _enemy_reach(self, origin, timeline, placement_cell=0,
                     gate=0, start=1, end=0):
        reached = [origin]
        for t in range(1, timeline.horizon + 1):
            blocked = placement_cell if t == 1 else 0
            if start <= t <= end:
                blocked |= gate
            reached.append(self.advance(reached[-1], timeline, t, blocked))
        return reached

    def _approach_layers(self, origin, timeline, contested, placing):
        """Movement-only approaches, with BOMB as a mandatory initial stay.

        A directed approach never relies on an unrepresented intermediate
        WAIT. Holding at the destination is tested separately. Excluding all
        potentially occupied destinations guarantees arrival despite action
        order; a race to the gate is not accepted as an executable trap.
        """
        layers = [origin]
        for t in range(1, timeline.horizon + 1):
            if placing and t == 1:
                reached = origin
            else:
                enterable = timeline.before[t] & ~(timeline.bombs[t] | contested[t])
                reached = self.neighbours(layers[-1]) & enterable
            layers.append(reached & ~timeline.danger[t])
        return layers

    def _first_steps(self, origin, gate, arrival, timeline, contested):
        """All first moves of safe, uncontested approaches to this timed gate."""
        if arrival == 0:
            return (0, 0, 0, 0)
        good = gate
        for t in range(arrival, 1, -1):
            targets = good & timeline.before[t]
            targets &= ~(timeline.bombs[t] | contested[t])
            good = self.neighbours(targets) & timeline.after[t - 1]
            good &= ~timeline.danger[t - 1]
        good &= timeline.before[1] & ~(timeline.bombs[1] | contested[1])
        x, y = divmod(origin.bit_length() - 1, self.height)
        return tuple(int(bool(self.bit((x + dx, y + dy)) & good))
                     if 0 <= x + dx < self.width and 0 <= y + dy < self.height else 0
                     for dx, dy in v6.DIRECTION_DELTAS)

    def _witness_cells(self, origin, timeline, good, placement_cell):
        """Entered cells of one complete surviving path (not a path count).

        Any single held cell that kills ALL surviving paths must intercept
        this witness too. This exact necessary filter cuts the gate search
        to at most horizon cells, regardless of the size of the open board.
        Different witness tie choices cannot remove any valid complete trap.
        """
        position, entered = origin, 0
        for t in range(1, timeline.horizon + 1):
            if position & good[t]:
                continue  # Prefer WAIT; it introduces no fictitious entry.
            targets = self.neighbours(position) & timeline.before[t] & good[t]
            targets &= ~timeline.bombs[t]
            if t == 1:
                targets &= ~placement_cell
            if not targets:
                return 0
            position = targets & -targets
            entered |= position
        return entered

    def block_plans(self, timeline, origin, *, placing=False, first_only=False,
                    eligible_opponents=None, required_opponents=()):
        """Find complete traps caused by a concretely feasible held cell.

        The optimistic permanent-gate test is only a necessary prefilter.
        Each accepted plan is rechecked with its actual arrival and release
        times. We give no helpful blocking credit to intermediate moves.
        The body may leave immediately after hold_until, so that leaving
        action is deliberately unblocked in the opponent simulation.
        """
        h = timeline.horizon
        if not self.opponents or not any(timeline.danger[1:]):
            return ()
        danger_union = 0
        for danger in timeline.danger[1:]:
            danger_union |= danger
        eligible_opponents = tuple(
            i for i, enemy in enumerate(self.opponents)
            if enemy & danger_union
            and (eligible_opponents is None or i in eligible_opponents))
        if not eligible_opponents:
            # An enemy can always keep WAITing on its never-dangerous origin;
            # body blocking cannot kill it under these known hazards.
            return ()
        placement_cell = origin if placing else 0
        contested = self._contested(timeline, placement_cell)
        layers = self._approach_layers(origin, timeline, contested, placing)
        enemy_good = self.viable(timeline)
        first_arrival = 1 if placing else 0
        reachable = 0
        for layer in layers[first_arrival:h]:
            reachable |= layer
        plans = []
        best_rank = None
        # The escape and the opponent cut must describe the SAME held body.
        # An unconditional forecast lets an enemy pass through our gate and
        # falsely occupy the escape we can actually take after releasing it.
        escape_cache = {}
        for opponent, enemy in enumerate(self.opponents):
            if eligible_opponents is not None and opponent not in eligible_opponents:
                continue
            baseline = self._enemy_reach(enemy, timeline, placement_cell)
            if not baseline[-1]:
                continue  # No credit for a trap that already exists.
            candidates = self._witness_cells(enemy, timeline, enemy_good,
                                             placement_cell) & reachable
            while candidates:
                gate = candidates & -candidates
                candidates ^= gate
                if self._enemy_reach(enemy, timeline, placement_cell,
                                     gate, 1, h - 1)[-1]:
                    continue  # Even an immediate permanent hold cannot trap.
                for arrival in range(first_arrival, h):
                    if best_rank is not None and arrival > best_rank[0]:
                        break
                    if not layers[arrival] & gate:
                        continue
                    start = max(1, arrival)
                    end = None
                    # Releasing earlier cannot cut more enemy paths. Search
                    # the shortest sufficient, safe hold; stop before fire.
                    for release in range(start, h):
                        if gate & timeline.danger[release]:
                            break
                        if self._enemy_reach(enemy, timeline, placement_cell,
                                             gate, start, release)[-1]:
                            continue
                        if any(self._enemy_reach(
                                self.opponents[i], timeline, placement_cell,
                                gate, start, release)[-1] for i in required_opponents):
                            continue
                        key = gate, start, release
                        if key not in escape_cache:
                            held_contested = self._contested(
                                timeline, placement_cell, gate, start, release)
                            escape_cache[key] = self.viable(
                                timeline, contested=held_contested)
                        if gate & escape_cache[key][release]:
                            end = release
                            break
                    if end is None:
                        continue
                    rank = (arrival, end)
                    if best_rank is not None and rank > best_rank:
                        continue
                    mask = ((0, 0, 0, 0) if placing else
                            self._first_steps(origin, gate, arrival, timeline, contested))
                    if not placing and arrival > 0 and not any(mask):
                        continue
                    plan = BlockPlan(divmod(gate.bit_length() - 1, self.height),
                                     arrival, end, mask, opponent)
                    if first_only:
                        return (plan,)
                    if best_rank is None or rank < best_rank:
                        plans.clear()
                        best_rank = rank
                    plans.append(plan)
        return tuple(plans)

    def local_blockade_survival(self):
        """A verified BOMB/approach/hold/escape plan also proves BOMB safety."""
        if self.blockade_checks:
            return self.blockade_passed
        if super().local_blockade_survival():
            return True
        base, scenario = self.scenario(tuple(self.state["self"][3]), 0)
        if (not self.state["self"][2] or self.me & base.bombs[1]
                or self.me & scenario.danger[1]):
            return False
        self.blockade_passed = bool(self.block_plans(
            scenario, self.me, placing=True, first_only=True))
        return self.blockade_passed

    def pressure_now(self, position):
        """Present-position pressure; a complete feasible body trap raises it to 1.

        Both worlds include the stationary placement action and opponent
        movement during it. Distance never changes their initial positions.
        """
        pressure = self.pressure(position, 0)
        if pressure >= 1.0 or not self.opponents:
            return pressure
        base, scenario = self.scenario(position, 0)
        q = self.bit(position)
        eligible = tuple(i for i, enemy in enumerate(self.opponents)
                         if self._enemy_reach(enemy, base, q)[-1])
        if eligible and self.block_plans(scenario, q, placing=True, first_only=True,
                                        eligible_opponents=eligible):
            pressure = 1.0
        return pressure


class ArrivalTacticalAnalysis(TacticalAnalysis):
    """V10: advance known hazards and only the opponents' necessary escapes.

    Reference positions are uniformly weighted DISTINCT positions on complete
    safe paths with the fewest movement actions (WAIT costs zero). They are a
    ranking heuristic, not a distribution over an opponent policy. Robust local
    safety always starts from the real, observed opponents instead.
    """

    def __init__(self, state, max_arrival=0, *, forecast_opponent_blocks=True):
        super().__init__(state, max_arrival,
                         forecast_opponent_blocks=forecast_opponent_blocks)
        self.escape_references = None
        self.arrival_pressure = {}

    def _minimum_move_escape(self, origin):
        h = self.known_end + 1
        if not any(origin & danger for danger in self.base.danger[1:h + 1]):
            return (origin,) * (h + 1)
        layers = [{origin: 0}]
        for t in range(1, h + 1):
            following = {}
            for cell, cost in layers[-1].items():
                targets = self.advance(cell, self.base, t)
                while targets:
                    target = targets & -targets
                    targets ^= target
                    value = cost + int(target != cell)
                    following[target] = min(following.get(target, h + 1), value)
            layers.append(following)
        remaining = [None] * (h + 1)
        remaining[h] = {cell: 0 for cell in layers[h]}
        for t in range(h - 1, -1, -1):
            remaining[t] = {}
            for cell in layers[t]:
                targets = self.advance(cell, self.base, t + 1)
                costs = []
                while targets:
                    target = targets & -targets
                    targets ^= target
                    if target in remaining[t + 1]:
                        costs.append(int(target != cell) + remaining[t + 1][target])
                if costs:
                    remaining[t][cell] = min(costs)
        minimum = remaining[0].get(origin)
        if minimum is None:
            return (0,) * (h + 1)  # Already doomed under known hazards.
        return tuple(sum(cell for cell, cost in layer.items()
                         if cost + remaining[t].get(cell, h + 1) == minimum)
                     for t, layer in enumerate(layers))

    def reference_positions(self, arrival):
        if self.escape_references is None:
            self.escape_references = tuple(self._minimum_move_escape(enemy)
                                           for enemy in self.opponents)
        return tuple(path[min(arrival, len(path) - 1)]
                     for path in self.escape_references)

    def _witness_cells(self, origin, timeline, good, placement_cell):
        # A forecast may contain several equally good starting positions.
        # Any complete trap must intercept this one surviving witness too.
        while origin:
            cell = origin & -origin
            origin ^= cell
            if self.advance(cell, timeline, 1, placement_cell) & good[1]:
                return super()._witness_cells(cell, timeline, good, placement_cell)
        return 0

    def pressure_at_arrival(self, position, arrival):
        key = position, arrival
        if key in self.arrival_pressure:
            return self.arrival_pressure[key]
        self.pressure_evaluations += 1
        base, scenario = self.scenario(position, arrival)
        q = self.bit(position)
        if arrival not in self.base_viability:
            self.base_viability[arrival] = self.viable(base)
        baseline_good = self.base_viability[arrival]
        candidate_good = self.viable(scenario)
        # At time zero use the actual observation, including doomed enemies
        # as physical bodies. At a remote time use only necessary evasions.
        references = self.opponents if arrival == 0 else self.reference_positions(arrival)
        references = tuple(enemy & ~q for enemy in references)
        pressures, eligible = [], []
        for i, starts in enumerate(references):
            count, total, all_survive = starts.bit_count(), 0.0, bool(starts)
            while starts:
                origin = starts & -starts
                starts ^= origin
                n0 = self.surviving_volume(origin, base, q, good=baseline_good)
                if not n0:
                    all_survive = False
                    continue
                nq = self.surviving_volume(origin, scenario, q, good=candidate_good)
                total += max(0.0, 1.0 - nq / n0)
            pressures.append(total / count if count else 0.0)
            if all_survive:
                eligible.append(i)
        pressure = max(pressures, default=0.0)
        if pressure < 1.0 and eligible:
            projected = copy(self)
            projected.opponents = references
            # One common gate must trap ALL reference positions of the chosen
            # enemy, with all forecast opponent bodies included in the escape.
            if projected.block_plans(scenario, q, placing=True, first_only=True,
                                     eligible_opponents=eligible):
                pressure = 1.0
        self.arrival_pressure[key] = pressure
        return pressure

    def local_placement_legal(self):
        base, scenario = self.scenario(tuple(self.state["self"][3]), 0)
        return bool(self.state["self"][2] and self.me & base.before[1]
                    and not self.me & (base.bombs[1] | scenario.danger[1]))

    def structural_local_rejection(self):
        """Remember body-safety failures, not cooldowns or impossible escapes."""
        if not self.local_placement_legal() or self.local_blockade_survival():
            return False
        _, scenario = self.scenario(tuple(self.state["self"][3]), 0)
        return bool(self.me & self.viable(scenario)[1])

    def bomb_preserves_blocks(self, plans):
        """A BOMB-now priority may not sacrifice a currently certified victim.

        Every required victim must die in ONE safe continuation. Certificates
        with different gates cannot be combined into an imaginary joint plan.
        No bomb-owner metadata is needed or read by this policy heuristic.
        """
        if not self.local_placement_legal():
            return False
        position = tuple(self.state["self"][3])
        self.local_pressure = self.pressure_at_arrival(position, 0)
        if self.local_pressure < 1.0 or not self.local_blockade_survival():
            return False
        _, scenario = self.scenario(position, 0)
        required = tuple(sorted({p.opponent for p in plans}))
        if all(not self._enemy_reach(self.opponents[i], scenario, self.me)[-1]
               for i in required):
            return True
        return bool(self.block_plans(scenario, self.me, placing=True, first_only=True,
                                     required_opponents=required))


def _v10_pressure_candidates(analysis, navigation, *, pressure_distance_discount=BOMB_TARGET_DISCOUNT):
    result = []
    if not analysis.opponents:
        return ()
    for position, distance in navigation[0].items():
        if distance == 0:
            if not analysis.local_placement_legal():
                continue
        elif not analysis.nominal_survival(position, distance):
            continue
        pressure = analysis.pressure_at_arrival(position, distance)
        if distance == 0:
            analysis.local_pressure = pressure
        if pressure > 0:
            result.append(PressureCandidate(position, distance, pressure,
                                            pressure * pressure_distance_discount ** distance))
    return tuple(result)


def _v10_select_targets(analysis, navigation, crates, pressure, danger, blocked_until,
                        *, pressure_threshold=PRESSURE_THRESHOLD,
                        include_coin_route_targets=True):
    ranked = v6._highest_ranked_bomb_candidates(pressure)
    if ranked:
        # Only exact primary score/distance ties may use crate yield. Weak
        # threats against several opponents never add up to a stronger score.
        yields = {c.position: (analysis.blast(analysis.bit(c.position))
                               & analysis.nonwall & ~analysis.base.after[c.distance]).bit_count()
                  for c in ranked}
        best_yield = max(yields.values())
        ranked = tuple(c for c in ranked if yields[c.position] == best_yield)
    return _select_targets(
        analysis.state, navigation, crates, ranked, danger, blocked_until,
        local_coin_survival=analysis.local_coin_route_survival,
        robust_local_alternative=True, pressure_threshold=pressure_threshold,
        include_coin_route_targets=include_coin_route_targets)


def task3_v10_feature_data(state, *, planner_context=None, diagnostic_context=None):
    """Historical V10 behavior, including its local-BOMB priority."""
    return _arrival_tactical_feature_data(
        state, planner_context=planner_context, diagnostic_context=diagnostic_context)


def task3_v11_feature_data(state, *, pressure_threshold,
                          planner_context=None, diagnostic_context=None):
    """V10 with independent block safety and priority for existing blocks.

    Registered V11 encoders pass their respective threshold to this SAME planner.
    Thresholds govern discounted pressure versus a valid crate alternative;
    as in V10, pressure remains eligible when no crate target exists.
    """
    return _arrival_tactical_feature_data(
        state, planner_context=planner_context, diagnostic_context=diagnostic_context,
        pressure_threshold=pressure_threshold, block_priority=True,
        independent_block_safety=True)


def validate_pressure_parameters(pressure_threshold, pressure_distance_discount):
    """Validate explicit V12 pressure controls without coercing booleans or strings."""
    for name, value, lower_inclusive in (
        ("pressure_threshold", pressure_threshold, True),
        ("pressure_distance_discount", pressure_distance_discount, False),
    ):
        if (isinstance(value, bool) or not isinstance(value, Real)
                or value > 1
                or (value < 0 if lower_inclusive else value <= 0)
                or not isfinite(value)):
            interval = "[0, 1]" if lower_inclusive else "(0, 1]"
            raise ValueError(f"{name} must be a finite number in {interval}.")
    return float(pressure_threshold), float(pressure_distance_discount)


def task3_v12_feature_data(state, *, diagnostic_context=None,
                          pressure_threshold=0.25, pressure_distance_discount=0.95,
                          forecast_opponent_blocks=True,
                          include_coin_route_targets=True, include_block_targets=True):
    """V12 safety/HOLD and the isolated report body-forecast ablation.

    Defaults reproduce the trained representation. Callbacks restrict overrides
    to evaluation; neither parameter changes crate scoring or learning gamma.
    Disabling forecast_opponent_blocks affects ordinary movement and local
    BOMB/coin-route survival only. Complete offensive block certificates keep
    their robust approach/hold/escape proof and independent safety credit.
    The two target ablations omit either the extra coin-route bomb rank or
    the explicit approach/HOLD objective. All safety and Pressure certificates
    remain available; independent block opportunities remain in diagnostics.
    """
    for name, value in (("include_coin_route_targets", include_coin_route_targets),
                        ("include_block_targets", include_block_targets)):
        if not isinstance(value, bool):
            raise ValueError(f"{name} must be a boolean.")
    pressure_threshold, pressure_distance_discount = validate_pressure_parameters(
        pressure_threshold, pressure_distance_discount)
    return _arrival_tactical_feature_data(
        state, diagnostic_context=diagnostic_context,
        pressure_threshold=pressure_threshold,
        pressure_distance_discount=pressure_distance_discount, block_priority=True,
        independent_block_safety=True, condition_wait=True,
        forecast_opponent_blocks=forecast_opponent_blocks,
        include_coin_route_targets=include_coin_route_targets,
        include_block_targets=include_block_targets)


def _block_target_mask(plans):
    if any(p.arrival == 0 for p in plans):
        return (0, 0, 0, 0, 2)
    return (*tuple(int(any(p.first_steps[i] for p in plans)) for i in range(4)), 0)


def _include_block_safety(safety, plans):
    """Add only first actions backed by complete, executable block plans.

    A local plan certifies WAIT; an approach certifies its first moves. The
    target family does not invalidate these proofs and is deliberately absent.
    """
    mask = _block_target_mask(plans)
    return tuple(int(bit or (mask[4] == 2 if i == 4 else mask[i]))
                 for i, bit in enumerate(safety))


def _arrival_tactical_feature_data(state, *, planner_context=None, diagnostic_context=None,
                                   pressure_threshold=PRESSURE_THRESHOLD,
                                   pressure_distance_discount=BOMB_TARGET_DISCOUNT,
                                   block_priority=False, independent_block_safety=False,
                                   condition_wait=False, forecast_opponent_blocks=True,
                                   include_coin_route_targets=True, include_block_targets=True):
    """Arrival-aware targets and robust local acceptance; optional retry memory.

    planner_context is private per-agent state managed by callbacks. It stores
    only the just-rejected site, not a permanent taboo or a new Q-key category.
    Returned bomb_available is REAL; only the feature encoder pools HOLD keys.
    """
    position = tuple(state["self"][3])
    danger, blocked_until = v6.danger_schedule(state)
    navigation = v6.navigation_data(state["field"], position, v6.occupied_positions(state))
    coin, coin_distance = v6.nearest_target_mask_and_distance(state["coins"], navigation)
    analysis = ArrivalTacticalAnalysis(
        state, max(navigation[0].values(), default=0),
        forecast_opponent_blocks=forecast_opponent_blocks)
    safety = analysis.movement_survivability_mask(condition_wait=condition_wait)
    # Local candidates reach the full robust proof below without a nominal
    # veto. All remote crate candidates keep V6's arrival-aware nominal test.
    remote_navigation = ({p: d for p, d in navigation[0].items() if d}, navigation[1])
    crates = v6._safe_productive_bomb_candidates(
        state, remote_navigation, dangerous_steps=danger, bomb_blocked_until=blocked_until)
    local_yield = v6.crates_hit_by_bomb(state["field"], position)
    if local_yield and analysis.local_placement_legal():
        crates += (v6.BombTargetCandidate(position, 0, local_yield, float(local_yield)),)
    stop = analysis.known_end + 2
    current = Timeline(*(getattr(analysis.base, name)[:stop]
                         for name in ("before", "after", "danger", "bombs")))
    plans = analysis.block_plans(current, analysis.me)
    certified_plans = plans
    # Safety is evidence about executable actions, independent of which
    # objective wins. Retain the certificates even if selection is changed.
    if independent_block_safety:
        safety = _include_block_safety(safety, plans)
    prioritize_bomb = bool(plans and include_block_targets and not block_priority
                           and analysis.bomb_preserves_blocks(plans))
    if planner_context is not None:
        planner_context["rejected_position"] = None
    if plans and include_block_targets and not prioritize_bomb:
        mask = _block_target_mask(plans)
        if not independent_block_safety:
            safety = _include_block_safety(safety, plans)
        primary_mask, alternative_mask, ratio = ZERO_BOMB_MASK, ZERO_BOMB_MASK, None
        family, pressure_score, quality = "block", 0.0, 0
    else:
        plans = ()  # Telemetry describes the selected target family only.
        if prioritize_bomb:
            pressure = (PressureCandidate(position, 0, 1.0, 1.0),)
        else:
            pressure = _v10_pressure_candidates(
                analysis, navigation, pressure_distance_discount=pressure_distance_discount)
            rejected = planner_context.get("previous_rejection") if planner_context else None
            # Recheck just the previously rejected neighbouring site using
            # today's observed bodies/hazards at that site. This deliberately
            # does not forecast every possible enemy approach to remote sites.
            if (rejected is not None and navigation[0].get(rejected) == 1
                    and state["self"][2]
                    and any(c.position == rejected for c in (*pressure, *crates))):
                probe_state = dict(state)
                probe_state["self"] = (*state["self"][:3], rejected)
                if not TacticalAnalysis(probe_state).local_blockade_survival():
                    pressure = tuple(c for c in pressure if c.position != rejected)
                    crates = tuple(c for c in crates if c.position != rejected)
        family, targets, pressure_score = _v10_select_targets(
            analysis, navigation, crates, pressure, danger, blocked_until,
            pressure_threshold=pressure_threshold,
            include_coin_route_targets=include_coin_route_targets)
        if targets.mask[4] and not analysis.local_blockade_survival():
            if planner_context is not None and analysis.structural_local_rejection():
                planner_context["rejected_position"] = position
            crates = tuple(c for c in crates if c.position != position)
            pressure = tuple(c for c in pressure if c.position != position)
            family, targets, pressure_score = _v10_select_targets(
                analysis, navigation, crates, pressure, danger, blocked_until,
                pressure_threshold=pressure_threshold,
                include_coin_route_targets=include_coin_route_targets)
        mask = (0, 0, 0, 0, 1) if targets.mask[4] else targets.mask
        primary_mask, alternative_mask = targets.global_target.mask, targets.alternative_mask
        ratio = targets.alternative_score_ratio_milli
        quality = v6.bomb_quality_class(targets.global_target.score)
    data = v6.Task2V6FeatureData(
        safety, coin, mask, v6.coin_distance_class(coin_distance), quality,
        v6.threat_urgency_from_schedule(danger, position), int(bool(state["self"][2])),
        primary_mask, alternative_mask, ratio)
    local_passed = analysis.blockade_passed and (
        not alternative_mask[4] or any(analysis.local_coin_cache.values()))
    diagnostic = BombDiagnostics(family, pressure_score, analysis.pressure_evaluations,
                                 bool(analysis.blockade_checks), local_passed)
    if diagnostic_context is not None:
        diagnostic_context.update(
            mask=tuple(safety), mode=family, score=float(pressure_score),
            crate_target=bool(crates), local_pressure=float(analysis.local_pressure),
            bomb_safe=bool(mask[4] == 1 and diagnostic.blockade_passed),
            pressure_threshold=pressure_threshold,
            tactical={"schema_version": 1, "bomb_mask": tuple(mask),
                      # These certificates describe available opportunities,
                      # independently of whether the policy receives a target.
                      "certified_bomb_mask": _block_target_mask(certified_plans),
                      "certified_plans": [
                          {"position": p.position, "arrival": p.arrival,
                           "hold_until": p.hold_until, "first_steps": p.first_steps,
                           "opponent": p.opponent} for p in certified_plans],
                      "plans": [{"position": p.position, "arrival": p.arrival,
                                 "hold_until": p.hold_until, "first_steps": p.first_steps,
                                 "opponent": p.opponent} for p in plans]})
    return data, diagnostic


def _v9_pressure_candidates(analysis, navigation):
    """Rank every nominally surviving site with current-position pressure."""
    result = []
    if not analysis.opponents:
        return ()
    for position, distance in navigation[0].items():
        if not analysis.nominal_survival(position, distance):
            continue
        pressure = analysis.pressure_now(position)
        if distance == 0:
            analysis.local_pressure = pressure
        if pressure > 0:
            result.append(PressureCandidate(position, distance, pressure,
                                            pressure * BOMB_TARGET_DISCOUNT ** distance))
    return tuple(result)


def task3_v9_feature_data(state, *, diagnostic_context=None):
    """V8 safety/coins, current-position pressure, and shared BOMB/HOLD targets.

    Existing-bomb complete traps take precedence over prospective new bombs;
    they expire with the current opportunity. Every observation is replanned.
    target_here is 0 (approach/none), 1 (BOMB) or 2 (hold via WAIT).
    """
    position = tuple(state["self"][3])
    danger, blocked_until = v6.danger_schedule(state)
    navigation = v6.navigation_data(state["field"], position, v6.occupied_positions(state))
    coin, coin_distance = v6.nearest_target_mask_and_distance(state["coins"], navigation)
    analysis = TacticalAnalysis(state, max(navigation[0].values(), default=0))
    safety = analysis.movement_survivability_mask()
    crates = v6._safe_productive_bomb_candidates(
        state, navigation, dangerous_steps=danger, bomb_blocked_until=blocked_until)
    stop = analysis.known_end + 2
    current = Timeline(*(getattr(analysis.base, name)[:stop]
                         for name in ("before", "after", "danger", "bombs")))
    plans = analysis.block_plans(current, analysis.me)
    if plans:
        here = any(p.arrival == 0 for p in plans)
        mask = ((0, 0, 0, 0, 2) if here else
                (*tuple(int(any(p.first_steps[i] for p in plans)) for i in range(4)), 0))
        # Only the first actions certified by an executable plan gain a bit.
        # Other actions must not borrow a hold that they would abandon.
        safety = tuple(int(bit or (here if i == 4 else mask[i]))
                       for i, bit in enumerate(safety))
        primary_mask, alternative_mask, ratio = ZERO_BOMB_MASK, ZERO_BOMB_MASK, None
        family, pressure_score, quality = "block", 0.0, 0
    else:
        pressure = _v9_pressure_candidates(analysis, navigation)
        family, targets, pressure_score = _select_targets(
            state, navigation, crates, pressure, danger, blocked_until,
            local_coin_survival=analysis.local_coin_route_survival)
        if targets.mask[4] and not analysis.local_blockade_survival():
            crates = tuple(c for c in crates if c.position != position)
            pressure = tuple(c for c in pressure if c.position != position)
            family, targets, pressure_score = _select_targets(
                state, navigation, crates, pressure, danger, blocked_until,
                local_coin_survival=analysis.local_coin_route_survival)
        mask = (0, 0, 0, 0, 1) if targets.mask[4] else targets.mask
        primary_mask = targets.global_target.mask
        alternative_mask = targets.alternative_mask
        ratio = targets.alternative_score_ratio_milli
        quality = v6.bomb_quality_class(targets.global_target.score)
    data = v6.Task2V6FeatureData(
        safety, coin, mask, v6.coin_distance_class(coin_distance), quality,
        v6.threat_urgency_from_schedule(danger, position), int(bool(state["self"][2])),
        primary_mask, alternative_mask, ratio)
    diagnostics = BombDiagnostics(family, pressure_score, analysis.pressure_evaluations,
                                  bool(analysis.blockade_checks), analysis.local_checks_passed)
    if diagnostic_context is not None:
        diagnostic_context.update(
            mask=tuple(int(bit) for bit in safety), mode=family,
            score=float(pressure_score), crate_target=bool(crates),
            local_pressure=float(analysis.local_pressure),
            bomb_safe=bool(mask[4] == 1 and diagnostics.blockade_passed),
            pressure_threshold=PRESSURE_THRESHOLD,
            # Only copy existing results here. Ownership, additional local
            # certificates and outcomes are evaluated by the world observer,
            # outside the policy's feature key and decision-time budget.
            tactical={
                "schema_version": 1,
                "bomb_mask": tuple(int(bit) for bit in mask),
                "plans": [{"position": p.position, "arrival": p.arrival,
                           "hold_until": p.hold_until, "first_steps": p.first_steps,
                           "opponent": p.opponent} for p in plans],
            },
        )
    return data, diagnostics
