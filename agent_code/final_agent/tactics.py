"""Arrival-aware bomb pressure and complete, executable body traps.

A block plan describes a timed approach to one held cell, a sufficient hold,
and a robust surviving continuation. Partial body restrictions receive no
bonus. Existing complete traps take priority over prospective bomb targets.
Only known bombs are simulated; no ownership metadata enters the features.
"""

from copy import copy
from dataclasses import dataclass

from . import navigation as nav
from .bomb_mechanics import (
    BOMB_TARGET_DISCOUNT, PRESSURE_THRESHOLD, ZERO_BOMB_MASK,
    BombAnalysis, BombDiagnostics, PressureCandidate, Timeline,
)


@dataclass(frozen=True)
class BlockPlan:
    position: tuple[int, int]
    arrival: int
    hold_until: int
    first_steps: tuple[int, int, int, int]
    opponent: int


class TacticalAnalysis(BombAnalysis):
    """Add one-cell timed trap searches to known-bomb safety analysis."""

    def _contested(self, timeline, placement_cell=0, gate=0, start=1, end=0):
        # Opponents may move during our BOMB action. Only its occupied cell is unavailable to them for that first action in the snapshot.
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
                     for dx, dy in nav.DIRECTION_DELTAS)

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
            # each position in game is a bit from a whole number
            # x & -x isolates the lowest turned on bit, to take away any position from the set
            position = targets & -targets
            entered |= position
        return entered

    def block_plans(self, timeline, origin, *, placing=False, first_only=False,
                    eligible_opponents=None):
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
            # An enemy can always keep WAITing on its never-dangerous origin.
            # Body blocking cannot kill it under these known hazards.
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
        # An unconditional forecast lets an enemy pass through our gate and falsely occupy the escape we can actually take after releasing it.
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
                    # Releasing earlier cannot cut more enemy paths. Search the shortest sufficient, safe hold; stop before fire.
                    for release in range(start, h):
                        if gate & timeline.danger[release]:
                            break
                        if self._enemy_reach(enemy, timeline, placement_cell,
                                             gate, start, release)[-1]:
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


class ArrivalTacticalAnalysis(TacticalAnalysis):
    """Advance known hazards and only the opponents' necessary escapes.

    Reference positions are uniformly weighted DISTINCT positions on complete
    safe paths with the fewest movement actions (WAIT costs zero). They are a
    ranking heuristic, not a distribution over an opponent policy. Robust local
    safety always starts from the real, observed opponents instead.
    """

    def __init__(self, state, max_arrival=0):
        super().__init__(state, max_arrival)
        self.escape_references = None
        self.arrival_pressure = {}

    # calculate cheapest escape route, the one with less steps (WAIT does not count)
    # first pass: how much does going to any position cost from starting position
    # second pass: how much is left from every position to finish position
    # optimal route = first pass + second pass = total minimum

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
        # for every posible enemy position, compare the surviving cell-time volume with (nq) vs without bomb (n0), left with worst case (more pressure) between opponents
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
        # if pressure with estimate under 1.0, but guaranteed trap, force 1.0
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


def _pressure_candidates(analysis, navigation):
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
                                            pressure * BOMB_TARGET_DISCOUNT ** distance))
    return tuple(result)


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


def _select_bomb_targets(analysis, navigation, crates, pressure, danger, blocked_until):
    """Choose pressure or crates, including a jointly safe coin-route alternative."""
    ranked = nav._highest_ranked_bomb_candidates(pressure)
    if ranked:
        # Only exact primary score/distance ties may use crate yield. Weak threats against several opponents never add up to a stronger score.
        yields = {c.position: (analysis.blast(analysis.bit(c.position))
                               & analysis.nonwall & ~analysis.base.after[c.distance]).bit_count()
                  for c in ranked}
        best_yield = max(yields.values())
        ranked = tuple(c for c in ranked if yields[c.position] == best_yield)
    # Keep the same final ranking step as the shared target selector: close scores use its established tolerance and distance tie-breaking rules.
    pressure_best = nav._highest_ranked_bomb_candidates(ranked)
    pressure_score = pressure_best[0].score if pressure_best else 0.0
    if pressure_best and (pressure_score >= PRESSURE_THRESHOLD or not crates):
        primary = nav._bomb_target_selection_from_candidates(
            tuple(nav.BombTargetCandidate(c.position, c.distance, 0, c.score)
                  for c in pressure_best), navigation, tuple(analysis.state["self"][3]))
        return "pressure", nav.CoinRouteBombTargetSelection(
            primary.mask, primary, ZERO_BOMB_MASK, None), pressure_score
    targets = nav.bomb_target_with_coin_route_alternative(
        analysis.state, navigation, dangerous_steps=danger,
        bomb_blocked_until=blocked_until, candidates=crates,
        local_coin_survival=analysis.local_coin_route_survival)
    return ("crates" if crates else "none"), targets, pressure_score


def feature_data(state, *, diagnostic_context=None):
    """Assemble V12 masks with independent safety proofs and block priority.

    Bomb availability is the observed value. Only the feature encoder pools
    HOLD states; diagnostics and reward calculation retain the real value.
    Every observation is replanned without rejected-target memory.
    """
    position = tuple(state["self"][3])
    danger, blocked_until = nav.danger_schedule(state)
    navigation = nav.navigation_data(state["field"], position, nav.occupied_positions(state))
    coin, coin_distance = nav.nearest_target_mask_and_distance(state["coins"], navigation)
    analysis = ArrivalTacticalAnalysis(state, max(navigation[0].values(), default=0))
    safety = analysis.movement_survivability_mask()
    # Local candidates reach the full robust proof below without a nominal veto. Remote crate candidates keep their arrival-aware nominal test.
    remote_navigation = ({p: d for p, d in navigation[0].items() if d}, navigation[1])
    crates = nav._safe_productive_bomb_candidates(
        state, remote_navigation, dangerous_steps=danger, bomb_blocked_until=blocked_until)
    local_yield = nav.crates_hit_by_bomb(state["field"], position)
    if local_yield and analysis.local_placement_legal():
        crates += (nav.BombTargetCandidate(position, 0, local_yield, float(local_yield)),)
    stop = analysis.known_end + 2
    current = Timeline(*(getattr(analysis.base, name)[:stop]
                         for name in ("before", "after", "danger", "bombs")))
    plans = analysis.block_plans(current, analysis.me)
    # Safety describes executable actions independently of target selection.
    safety = _include_block_safety(safety, plans)
    if plans:
        mask = _block_target_mask(plans)
        primary_mask, alternative_mask, ratio = ZERO_BOMB_MASK, ZERO_BOMB_MASK, None
        family, pressure_score, quality = "block", 0.0, 0
    else:
        pressure = _pressure_candidates(analysis, navigation)
        family, targets, pressure_score = _select_bomb_targets(
            analysis, navigation, crates, pressure, danger, blocked_until)
        if targets.mask[4] and not analysis.local_blockade_survival():
            crates = tuple(c for c in crates if c.position != position)
            pressure = tuple(c for c in pressure if c.position != position)
            family, targets, pressure_score = _select_bomb_targets(
                analysis, navigation, crates, pressure, danger, blocked_until)
        mask = (0, 0, 0, 0, 1) if targets.mask[4] else targets.mask
        primary_mask, alternative_mask = targets.global_target.mask, targets.alternative_mask
        ratio = targets.alternative_score_ratio_milli
        quality = nav.bomb_quality_class(targets.global_target.score)
    data = nav.FeatureData(
        safety, coin, mask, nav.coin_distance_class(coin_distance), quality,
        nav.threat_urgency_from_schedule(danger, position), int(bool(state["self"][2])),
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
            pressure_threshold=PRESSURE_THRESHOLD,
            tactical={"schema_version": 1, "bomb_mask": tuple(mask),
                      "plans": [{"position": p.position, "arrival": p.arrival,
                                 "hold_until": p.hold_until, "first_steps": p.first_steps,
                                 "opponent": p.opponent} for p in plans]})
    return data, diagnostic
