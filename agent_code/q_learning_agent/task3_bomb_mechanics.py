"""V7/V8 bomb targets and shared time-space body-blocking mechanics.

V7 retains V6 movement safety. V8 uses the same body-robustness calculation
for movement under existing hazards, without inserting a hypothetical bomb.
Coin navigation and nominal remote bomb-target forecasts remain unchanged.
Unplaced enemy bombs are not enumerated.

Pressure counts distinct (cell, time) pairs on fully surviving opponent paths,
over the same placement-to-hazard-end window with and without our bomb. It is
a restriction score, not a kill probability. Arrival forecasts use reachable
opponent positions; they are recalculated from each new observation.

Bomb-target body robustness is computed only at our current position; V8
also applies it to movement under existing hazards. Opponent reachability
overapproximates their movement. A contestable move must survive success AND
forced staying. This proves existence within that finite approximation; it
neither forces Q-learning to follow the escape nor hard-filters any action.
"""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import settings

from . import task2_mechanics as v6


PRESSURE_THRESHOLD = 0.25
BOMB_TARGET_DISCOUNT = v6.TASK2_BOMB_TARGET_DISCOUNT
ZERO_BOMB_MASK = (0, 0, 0, 0, 0)


@dataclass(frozen=True)
class PressureCandidate:
    position: tuple[int, int]
    distance: int
    pressure: float
    score: float


@dataclass(frozen=True)
class BombDiagnostics:
    target_family: str
    pressure_score: float
    pressure_candidates_evaluated: int
    blockade_checked: bool
    blockade_passed: bool


@dataclass(frozen=True)
class PressureBombDiagnosticContext(v6.CoinRouteBombDiagnosticContext):
    """Private telemetry only; none of these fields enters the Q-table key."""

    pressure: BombDiagnostics
    evaluation: dict | None = None
    bomb_available: bool | None = None


@dataclass(frozen=True)
class Timeline:
    before: tuple[int, ...]
    after: tuple[int, ...]
    danger: tuple[int, ...]
    bombs: tuple[int, ...]

    @property
    def horizon(self):
        return len(self.danger) - 1


class BombAnalysis:
    """Per-observation bitset geometry, schedules and opponent forecasts."""

    def __init__(self, state, max_arrival=0, *, forecast_opponent_blocks=True):
        if not isinstance(forecast_opponent_blocks, bool):
            raise ValueError("forecast_opponent_blocks must be a boolean.")
        self.forecast_opponent_blocks = forecast_opponent_blocks
        field = np.asarray(state["field"])
        self.state = state
        self.width, self.height = field.shape
        self.all = (1 << field.size) - 1
        self.top = sum(1 << (x * self.height) for x in range(self.width))
        self.bottom = self.top << (self.height - 1)
        self.floor = self.encode(np.argwhere(field == 0))
        self.nonwall = self.encode(np.argwhere(field != -1))
        self.me = self.bit(state["self"][3])
        self.opponents = tuple(self.bit(o[3]) for o in state.get("others", ()))
        self.occupied = self.encode(o[3] for o in state.get("others", ()))
        self.blast_cache = {}
        self.scenarios = {}
        self.base_windows = {}
        self.base_viability = {}
        self.nominal_cache = {}
        self.local_coin_cache = {}
        self.pressure_evaluations = 0
        self.local_pressure = 0.0
        self.blockade_checks = 0
        self.blockade_passed = False
        original = [(self.bit(p), int(k) + 1) for p, k in state["bombs"]]
        explosion = np.asarray(state["explosion_map"])
        self.known_end = max(
            [int(explosion.max()), 0]
            + [t + settings.EXPLOSION_TIMER - 1 for _, t in original]
        )
        horizon = max(
            self.known_end + 1,
            max_arrival + 1 + settings.BOMB_TIMER + settings.EXPLOSION_TIMER,
        )
        danger = [0] * (horizon + 1)
        bombs = [0] * (horizon + 1)
        opens = [0] * (horizon + 1)
        for p in np.argwhere(explosion > 0):
            for t in range(1, int(explosion[tuple(p)]) + 1):
                danger[t] |= self.bit(p)
        for p, detonation in original:
            blast = self.blast(p)
            for t in range(1, detonation + 1):
                bombs[t] |= p
            for t in range(detonation, detonation + settings.EXPLOSION_TIMER):
                danger[t] |= blast
            opens[detonation] |= blast & self.nonwall
        before = [self.floor] * (horizon + 1)
        after = [self.floor] * (horizon + 1)
        opened = self.floor
        for t in range(1, horizon + 1):
            before[t] = opened
            opened |= opens[t]
            after[t] = opened
        self.base = Timeline(tuple(before), tuple(after), tuple(danger), tuple(bombs))
        # The same prefix is reused by every candidate with that arrival time.
        self.enemy_prefixes = []
        for origin in self.opponents:
            reach = [origin]
            for t in range(1, max_arrival + 1):
                reach.append(self.advance(reach[-1], self.base, t))
            self.enemy_prefixes.append(reach)

    def bit(self, position):
        return 1 << (int(position[0]) * self.height + int(position[1]))

    def encode(self, positions):
        bits = 0
        for p in positions:
            bits |= self.bit(p)
        return bits

    def neighbours(self, bits):
        return (
            ((bits & ~self.top) >> 1) | ((bits & ~self.bottom) << 1)
            | (bits >> self.height) | (bits << self.height)
        ) & self.all

    def blast(self, bit):
        if bit not in self.blast_cache:
            x, y = divmod(bit.bit_length() - 1, self.height)
            result = bit
            for dx, dy in v6.DIRECTION_DELTAS:
                for distance in range(1, settings.BOMB_POWER + 1):
                    px, py = x + dx * distance, y + dy * distance
                    if not (0 <= px < self.width and 0 <= py < self.height):
                        break
                    cell = self.bit((px, py))
                    if not cell & self.nonwall:
                        break
                    result |= cell  # This engine's blasts pass THROUGH crates.
            self.blast_cache[bit] = result
        return self.blast_cache[bit]

    def scenario(self, position, arrival):
        """Return equal-horizon baseline/own-bomb timelines, relative to arrival.

        Placement is relative action 1, detonation 1+BOMB_TIMER; the bomb blocks
        entry through the detonation action. Fire lasts EXPLOSION_TIMER actions.
        Crates open for movement on the following action. There are no chains.
        """
        key = position, arrival
        if key in self.scenarios:
            return self.scenarios[key]
        detonation = 1 + settings.BOMB_TIMER
        horizon = max(self.known_end - arrival + 1,
                      detonation + settings.EXPLOSION_TIMER)
        stop = arrival + horizon + 1
        if arrival not in self.base_windows:
            self.base_windows[arrival] = Timeline(*(
                tuple(getattr(self.base, f)[arrival:stop])
                for f in ("before", "after", "danger", "bombs")))
        base = self.base_windows[arrival]
        before, after, danger, bombs = (list(getattr(base, f))
                                      for f in ("before", "after", "danger", "bombs"))
        q = self.bit(position)
        blast = self.blast(q)
        for t in range(1, detonation + 1):
            bombs[t] |= q
        for t in range(detonation, detonation + settings.EXPLOSION_TIMER):
            danger[t] |= blast
        for t in range(detonation, horizon + 1):
            after[t] |= blast & self.nonwall
            if t > detonation:
                before[t] |= blast & self.nonwall
        hypothetical = Timeline(tuple(before), tuple(after), tuple(danger), tuple(bombs))
        self.scenarios[key] = base, hypothetical
        return base, hypothetical

    def advance(self, reached, timeline, t, extra_blocked=0):
        enterable = timeline.before[t] & ~(timeline.bombs[t] | extra_blocked)
        return (reached | (self.neighbours(reached) & enterable)) & ~timeline.danger[t]

    def viable(self, timeline, *, contested=None, fixed_bodies=0):
        """Backward survival sets at each decision time, including WAIT.

        For a contestable x->y, both y AND x must be in V[t+1]. For existence
        of some viable action this adds nothing beyond WAIT at x. Thus WAIT
        plus uncontested moves is the equivalent compact recurrence below.
        """
        h = timeline.horizon
        good = [0] * (h + 1)
        good[h] = timeline.after[h] & ~(timeline.danger[h] | fixed_bodies)
        for t in range(h - 1, -1, -1):
            blocked = fixed_bodies | (contested[t + 1] if contested is not None else 0)
            targets = good[t + 1] & timeline.before[t + 1]
            targets &= ~(timeline.bombs[t + 1] | blocked)
            good[t] = (good[t + 1] | self.neighbours(targets))
            good[t] &= timeline.after[t] & ~(timeline.danger[t] | fixed_bodies)
        return good

    def nominal_survival(self, position, arrival):
        """Placement and subsequent escape under known hazards; no body forecast.

        Navigation uses the V6 current-board shortest distance. As in V6 this
        is a target forecast, not a promise to execute the entire approach.
        The current movement-safety mask still governs its own known hazards.
        """
        key = position, arrival
        if key not in self.nominal_cache:
            base, scenario = self.scenario(position, arrival)
            q = self.bit(position)
            available_tile = (base.before[1] & base.after[0])
            available_tile &= ~(base.danger[0] | base.danger[1] | base.bombs[1])
            good = self.viable(scenario, fixed_bodies=self.occupied)
            self.nominal_cache[key] = bool(q & available_tile & good[1])
        return self.nominal_cache[key]

    def surviving_volume(self, start, timeline, placement_cell, *, good=None):
        """Count (cell,time) pairs belonging to complete surviving paths.

        Our body occupies the placement cell during the placement action in
        BOTH worlds. Afterwards we do not assume a helpful body-blocking plan.
        Enemy movement paths are not assigned probabilities.
        """
        reaches = [start]
        for t in range(1, timeline.horizon + 1):
            reaches.append(self.advance(reaches[-1], timeline, t,
                                        placement_cell if t == 1 else 0))
        if good is None:
            good = self.viable(timeline)
        # The final bookkeeping layer is after the fire; count only the bomb
        # phase, through the last dangerous action, in both worlds.
        return sum((reaches[t] & good[t]).bit_count()
                   for t in range(1, timeline.horizon))

    def pressure(self, position, arrival):
        self.pressure_evaluations += 1
        base, scenario = self.scenario(position, arrival)
        q = self.bit(position)
        if arrival not in self.base_viability:
            self.base_viability[arrival] = self.viable(base)
        baseline_good = self.base_viability[arrival]
        candidate_good = self.viable(scenario)
        best = 0.0
        for prefix in self.enemy_prefixes:
            # Condition the arrival forecast on our actually occupying q.
            start = prefix[arrival] & ~q
            n0 = self.surviving_volume(start, base, q, good=baseline_good)
            if not n0:
                continue  # An already doomed opponent gives no extra credit.
            nq = self.surviving_volume(start, scenario, q, good=candidate_good)
            best = max(best, max(0.0, 1.0 - nq / n0))
        return best

    def local_blockade_survival(self):
        """One check for BOMB now, shared by all target families in this state."""
        if self.blockade_checks:
            return self.blockade_passed
        self.blockade_checks += 1
        position = tuple(self.state["self"][3])
        base, scenario = self.scenario(position, 0)
        if (not self.state["self"][2] or self.me & base.bombs[1]
                or self.me & scenario.danger[1]):
            return False
        contested = self.opponent_contested_cells(scenario, stationary_until=1)
        good = self.viable(scenario, contested=contested)
        self.blockade_passed = bool(self.me & good[1])
        return self.blockade_passed

    def opponent_contested_cells(self, timeline, *, stationary_until=0):
        """Cells bodies can occupy before/after moves, before deaths this tick.

        Reachability is a conservative union, not an enumeration of joint
        opponent actions. Our cell cannot be entered during a proved initial
        stationary prefix: one action for BOMB, or a checked sequence of WAITs.
        The cell is released on the next action. Movement certificates must
        not borrow the stationary assumption.

        The report safety ablation removes only these future body forecasts.
        Immediate movement legality still rejects currently occupied cells;
        known hazards and the offensive planner's separate trap certificates
        remain unchanged. This is not the older static-opponent target check.
        """
        contested = [0] * (timeline.horizon + 1)
        if not self.forecast_opponent_blocks:
            return contested
        reached = list(self.opponents)
        for t in range(1, timeline.horizon + 1):
            enterable = timeline.before[t] & ~timeline.bombs[t]
            if t <= stationary_until:
                enterable &= ~self.me
            for i, old in enumerate(reached):
                possible = old | (self.neighbours(old) & enterable)
                # Bodies can block BEFORE deaths are evaluated this action.
                contested[t] |= old | possible
                reached[i] = possible & ~timeline.danger[t]
        return contested

    def local_coin_route_survival(self, coin_position, coin_distances):
        """One local escape must satisfy BOTH coin routing and body safety.

        Placement is action 1; a blast-exposed coin must be entered before
        action 1+BOMB_TIMER and crossed through another edge immediately.
        Before the coin, only WAIT and shortest-route moves are admissible.
        Contested moves need a safe failed-move continuation too; for the
        existence test WAIT already represents that option. After crossing
        the coin the ordinary robust survival sets apply. An unexposed coin
        can be collected along the remaining static route after all fire.
        V8/V9 also check the original nominal complete coin route; V10 uses
        this joint robust proof directly for a selected local alternative.
        """
        coin_position = tuple(coin_position)
        if coin_position in self.local_coin_cache:
            return self.local_coin_cache[coin_position]
        position = tuple(self.state["self"][3])
        if coin_distances.get(position, 0) <= 0 or not self.local_blockade_survival():
            self.local_coin_cache[coin_position] = False
            return False
        _, timeline = self.scenario(position, 0)
        contested = self.opponent_contested_cells(timeline, stationary_until=1)
        good = self.viable(timeline, contested=contested)
        coin = self.bit(coin_position)
        exposed = bool(coin & self.blast(self.me))
        detonation = 1 + settings.BOMB_TIMER
        distances = {self.bit(p): d for p, d in coin_distances.items()}

        @lru_cache(maxsize=None)
        def before_coin(cell, t):
            if cell & timeline.danger[t] or not cell & timeline.after[t]:
                return False
            if t == timeline.horizon:
                # There is no remaining lethal hazard. Waiting out a body
                # contest before continuing to an unexposed coin is safe.
                return not exposed and cell in distances
            if before_coin(cell, t + 1):
                return True
            targets = self.neighbours(cell) & timeline.before[t + 1]
            targets &= ~(timeline.bombs[t + 1] | contested[t + 1]
                         | timeline.danger[t + 1])
            while targets:
                target = targets & -targets
                targets ^= target
                if distances.get(target) != distances.get(cell, 0) - 1:
                    continue
                if target != coin:
                    if before_coin(target, t + 1):
                        return True
                elif not exposed:
                    if coin & good[t + 1]:
                        return True
                elif t + 1 < detonation and t + 2 <= timeline.horizon:
                    # At a blast coin, a blocked exit would force forbidden
                    # waiting. Require an uncontested exit other than entry.
                    exits = self.neighbours(coin) & ~cell
                    exits &= timeline.before[t + 2] & good[t + 2]
                    exits &= ~(timeline.bombs[t + 2] | contested[t + 2])
                    if exits:
                        return True
            return False

        result = before_coin(self.me, 1)
        self.local_coin_cache[coin_position] = result
        return result

    @property
    def local_checks_passed(self):
        """Acceptance of the selected local target, including its coin promise."""
        return self.blockade_passed and (
            not self.local_coin_cache or any(self.local_coin_cache.values()))

    def movement_survivability_mask(self, *, condition_wait=False):
        """V8's five binary action features under known hazards and body blocks.

        One backward search provides all first-action values. A legal move
        must survive reaching its destination; if contestable it must also
        survive staying at its origin. WAIT is tested directly. Existing
        bombs keep their observed timers and block entry during detonation.
        This is feature data, never a hard restriction on Q-learning actions.

        V12 additionally proves a safe WAIT prefix with our body present for
        its entire duration, followed by a robust escape after releasing it.
        The complete remaining hazard horizon still has to be survivable.
        Older feature versions keep their original unconditional forecast.
        """
        position = tuple(self.state["self"][3])
        destinations = []
        for dx, dy in v6.SURVIVAL_ACTION_DELTAS:
            x, y = int(position[0]) + dx, int(position[1]) + dy
            if not (0 <= x < self.width and 0 <= y < self.height):
                destinations.append(0)
                continue
            cell = self.bit((x, y))
            legal = bool(cell & self.floor) and not bool(cell & self.occupied)
            if (dx or dy) and cell & self.base.bombs[1]:
                legal = False
            destinations.append(cell if legal else 0)

        if not self.known_end:
            # Without known hazards a failed legal move cannot cause a death.
            return tuple(int(bool(cell)) for cell in destinations)

        # The full base also covers distant bomb-target forecasts. Only the
        # actual danger horizon plus one fire-free layer is needed here.
        stop = self.known_end + 2
        timeline = Timeline(*(getattr(self.base, name)[:stop]
                              for name in ("before", "after", "danger", "bombs")))
        contested = self.opponent_contested_cells(timeline)
        good = self.viable(timeline, contested=contested)
        can_stay = bool(self.me & good[1])
        mask = tuple(int(bool(cell & good[1]) and
                         (i == 4 or not cell & contested[1] or can_stay))
                     for i, cell in enumerate(destinations))
        if condition_wait and destinations[4] and not mask[4]:
            for wait_until in range(1, timeline.horizon + 1):
                if self.me & timeline.danger[wait_until]:
                    break  # Every action of the WAIT prefix must survive.
                if not self.me & contested[wait_until]:
                    continue  # Occupancy cannot change this prefix's forecast.
                wait_contested = self.opponent_contested_cells(
                    timeline, stationary_until=wait_until)
                wait_good = self.viable(timeline, contested=wait_contested)
                # The continuation starts AFTER all required WAITs. Starting
                # it at action 1 could leave a cell we still claim to occupy.
                if self.me & wait_good[wait_until]:
                    # Moves and failed-move fallbacks keep their own proofs.
                    return (*mask[:4], 1)
        return mask


def _pressure_candidates(analysis, navigation):
    candidates = []
    if not analysis.opponents:
        return ()
    # Score all reachable sites. Check nominal survival in score order below;
    # inferior remote sites cannot win and do not need another escape search.
    for position, distance in navigation[0].items():
        pressure = analysis.pressure(position, distance)
        if distance == 0:
            analysis.local_pressure = pressure
        if pressure <= 0:
            continue
        candidates.append(PressureCandidate(position, distance, pressure,
                                             pressure * BOMB_TARGET_DISCOUNT ** distance))
    selected = []
    while candidates:
        rank = v6._highest_ranked_bomb_candidates(candidates)
        surviving = [c for c in rank if analysis.nominal_survival(c.position, c.distance)]
        selected.extend(surviving)
        if any(c.distance > 0 for c in surviving):
            break
        # Keep one additional remote rank if the best site is here: it is the
        # exact fallback if the subsequent LOCAL blockade check rejects here.
        removed = {c.position for c in rank}
        candidates = [c for c in candidates if c.position not in removed]
    return tuple(selected)


def _select_targets(state, navigation, crate_candidates, pressure_candidates,
                    dangerous_steps, bomb_blocked_until, *, local_coin_survival=None,
                    robust_local_alternative=False, pressure_threshold=PRESSURE_THRESHOLD,
                    include_coin_route_targets=True):
    pressure_best = v6._highest_ranked_bomb_candidates(pressure_candidates)
    pressure_score = pressure_best[0].score if pressure_best else 0.0
    if pressure_best and (pressure_score >= pressure_threshold or not crate_candidates):
        primary = v6._bomb_target_selection_from_candidates(
            tuple(v6.BombTargetCandidate(c.position, c.distance, 0, c.score)
                  for c in pressure_best), navigation, tuple(state["self"][3]))
        return "pressure", v6.CoinRouteBombTargetSelection(
            primary.mask, primary, ZERO_BOMB_MASK, None), pressure_score
    if not include_coin_route_targets:
        # Keep the same productive candidates and exact primary ties. Only
        # the extra nearest-coin-route rank is omitted; its constrained route
        # proof must neither be required for nor veto an ordinary primary.
        primary = v6._bomb_target_selection_from_candidates(
            v6._highest_ranked_bomb_candidates(crate_candidates), navigation,
            tuple(state["self"][3]))
        return ("crates" if crate_candidates else "none"), v6.CoinRouteBombTargetSelection(
            primary.mask, primary, ZERO_BOMB_MASK, None), pressure_score
    crates = v6.bomb_target_with_coin_route_alternative(
        state, navigation, dangerous_steps=dangerous_steps,
        bomb_blocked_until=bomb_blocked_until, candidates=crate_candidates,
        local_coin_survival=local_coin_survival if robust_local_alternative else None)
    if (crates.alternative_mask[4] and local_coin_survival is not None
            and not robust_local_alternative):
        # Check only an actually selected LOCAL alternative. A primary site
        # need not promise a coin route, and remote sites stay nominal.
        position = tuple(state["self"][3])
        reachable_coins = [tuple(c) for c in state["coins"] if tuple(c) in navigation[0]]
        nearest = min(navigation[0][c] for c in reachable_coins)
        accepted = False
        for coin in reachable_coins:
            if navigation[0][coin] != nearest:
                continue
            distances = v6.navigation_data(
                state["field"], coin, v6.occupied_positions(state))[0]
            if (v6.can_survive_bomb_via_coin_at_arrival(
                    state, position, 0, coin, distances,
                    dangerous_steps, bomb_blocked_until)
                    and local_coin_survival(coin, distances)):
                accepted = True
                break
        if not accepted:
            # The rejected position is not a primary optimum. Removing it
            # retains that optimum and reranks the remaining alternatives.
            crates = v6.bomb_target_with_coin_route_alternative(
                state, navigation, dangerous_steps=dangerous_steps,
                bomb_blocked_until=bomb_blocked_until,
                candidates=tuple(c for c in crate_candidates if c.position != position))
    return ("crates" if crate_candidates else "none"), crates, pressure_score


def task3_v7_feature_data(state, *, diagnostic_context=None):
    """Assemble V6-compatible data with mutually exclusive bomb target families."""
    return _task3_feature_data(state, robust_movement=False,
                              diagnostic_context=diagnostic_context)


def task3_v8_feature_data(state, *, diagnostic_context=None):
    """Keep V7 targets, replacing only movement safety with body robustness."""
    return _task3_feature_data(state, robust_movement=True,
                              diagnostic_context=diagnostic_context)


def _task3_feature_data(state, *, robust_movement, diagnostic_context=None):
    position = tuple(state["self"][3])
    danger, blocked_until = v6.danger_schedule(state)
    navigation = v6.navigation_data(state["field"], position, v6.occupied_positions(state))
    coin, coin_distance = v6.nearest_target_mask_and_distance(state["coins"], navigation)
    analysis = BombAnalysis(state, max(navigation[0].values(), default=0))
    if robust_movement:
        safety = analysis.movement_survivability_mask()
    else:
        opponents = {tuple(o[3]) for o in state.get("others", ())}
        safety = v6._survivable_action_mask_from_schedule(
            state["field"], position, danger, blocked_until, opponents)
    crates = v6._safe_productive_bomb_candidates(
        state, navigation, dangerous_steps=danger, bomb_blocked_until=blocked_until)
    pressure = _pressure_candidates(analysis, navigation)
    family, targets, pressure_score = _select_targets(
        state, navigation, crates, pressure, danger, blocked_until,
        local_coin_survival=analysis.local_coin_route_survival if robust_movement else None)

    if targets.mask[4]:
        allowed = bool(state["self"][2]) and analysis.local_blockade_survival()
        if not allowed:
            # Only the CURRENT selected site is removed. Remote targets never
            # trigger a blockade forecast. Rerank both families in this state.
            crates = tuple(c for c in crates if c.position != position)
            pressure = tuple(c for c in pressure if c.position != position)
            family, targets, pressure_score = _select_targets(
                state, navigation, crates, pressure, danger, blocked_until,
                local_coin_survival=analysis.local_coin_route_survival if robust_movement else None)

    # A local alternative has the same navigation representation as a local
    # primary target. The original separate masks survive in private telemetry.
    mask = (0, 0, 0, 0, 1) if targets.mask[4] else targets.mask
    data = v6.Task2V6FeatureData(
        safety, coin, mask, v6.coin_distance_class(coin_distance),
        v6.bomb_quality_class(targets.global_target.score),
        v6.threat_urgency_from_schedule(danger, position), int(bool(state["self"][2])),
        targets.global_target.mask, targets.alternative_mask,
        targets.alternative_score_ratio_milli)
    diagnostics = BombDiagnostics(family, pressure_score, analysis.pressure_evaluations,
                                  bool(analysis.blockade_checks), analysis.local_checks_passed)
    if diagnostic_context is not None:
        # Copy already computed values; no additional search, including for
        # bombs placed away from the selected target. Movement bits retain
        # BOARD orientation so they describe the actual executed action.
        diagnostic_context.update(
            mask=tuple(int(bit) for bit in safety), mode=family,
            score=float(pressure_score), crate_target=bool(crates),
            local_pressure=float(analysis.local_pressure),
            bomb_safe=bool(mask[4] and diagnostics.blockade_checked
                           and diagnostics.blockade_passed),
            pressure_threshold=PRESSURE_THRESHOLD,
        )
    return data, diagnostics
