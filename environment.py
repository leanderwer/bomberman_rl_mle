import json
import logging
import pickle
import os
import subprocess
from collections import namedtuple
from datetime import datetime
from pathlib import Path
from threading import Event
from time import perf_counter, time
from typing import List, Tuple, Dict

import numpy as np

import events as e
import settings as s
from agents import (
    ACTION_DIAGNOSTIC_STAT_PREFIX,
    Agent,
    SequentialAgentBackend,
    agent_configurations,
)
from fallbacks import pygame
from items import Coin, Explosion, Bomb

WorldArgs = namedtuple("WorldArgs",
                       ["no_gui", "fps", "turn_based", "update_interval", "save_replay", "replay", "make_video", "continue_without_training", "log_dir", "save_stats", "match_name", "seed", "silence_errors", "scenario"])


class Trophy:
    coin_trophy = pygame.transform.smoothscale(pygame.image.load(s.ASSET_DIR / 'coin.png'), (15, 15))
    suicide_trophy = pygame.transform.smoothscale(pygame.image.load(s.ASSET_DIR / 'explosion_0.png'), (15, 15))
    time_trophy = pygame.image.load(s.ASSET_DIR / 'hourglass.png')


class GenericWorld:
    logger: logging.Logger

    running: bool = False
    step: int
    replay: Dict
    round_statistics: Dict

    agents: List[Agent]
    active_agents: List[Agent]
    arena: np.ndarray
    coins: List[Coin]
    bombs: List[Bomb]
    explosions: List[Explosion]

    round_id: str

    def __init__(self, args: WorldArgs):
        self.args = args
        self.setup_logging()

        self.colors = list(s.AGENT_COLORS)

        self.round = 0
        self.round_statistics = {}
        self.training_statistics = {}
        self.run_started_at = perf_counter()

        self.running = False

    def setup_logging(self):
        self.logger = logging.getLogger('BombeRLeWorld')
        self.logger.setLevel(s.LOG_GAME)
        handler = logging.FileHandler(f'{self.args.log_dir}/game.log', mode="w")
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s [%(name)s] %(levelname)s: %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.info('Initializing game world')

    def new_round(self):
        if self.running:
            self.logger.warning('New round requested while still running')
            self.end_round()

        new_round = self.round + 1
        self.logger.info(f'STARTING ROUND #{new_round}')

        # Bookkeeping
        self.step = 0
        self.bombs = []
        self.explosions = []

        if self.args.match_name is not None:
            match_prefix = f"{self.args.match_name} | "
        else:
            match_prefix = ""
        self.round_id = f'{match_prefix}Round {new_round:02d} ({datetime.now().strftime("%Y-%m-%d %H-%M-%S")})'

        # Arena with wall and crate layout
        self.arena, self.coins, self.active_agents = self.build_arena()
        self.initial_crates = int((self.arena == 1).sum())
        self.initial_coins = len(self.coins)

        for agent in self.active_agents:
            agent.start_round()

        self.evaluation_diagnostics = {}
        if (os.environ.get("BOMBERMAN_TASK3_DIAGNOSTICS") == "1"
                and not any(agent.train for agent in self.agents)):
            from evaluation.task3.diagnostics import RoundDiagnostics, VERSIONS
            version = os.environ.get("BOMBERMAN_FEATURE_VERSION")
            subject = os.environ.get("BOMBERMAN_DIAGNOSTIC_AGENT", "q_learning_agent")
            if version in VERSIONS:
                self.evaluation_diagnostics = {
                    agent.name: RoundDiagnostics(
                        agent.name, version, new_round,
                        history_limit=s.BOMB_TIMER + s.EXPLOSION_TIMER + 3)
                    for agent in self.agents if agent.name == subject
                }

        # Store the step in which each agent collects a coin.
        self.coin_collection_steps = {
            agent.name: []
            for agent in self.agents
        }
        self.kill_events = []

        self.replay = {
            'round': new_round,
            'arena': np.array(self.arena),
            'coins': [c.get_state() for c in self.coins],
            'agents': [a.get_state() for a in self.agents],
            'actions': dict([(a.name, []) for a in self.agents]),
            'permutations': []
        }

        self.round = new_round
        self.running = True

    def build_arena(self) -> Tuple[np.array, List[Coin], List[Agent]]:
        raise NotImplementedError()

    def add_agent(self, agent_dir, name, train=False):
        assert len(self.agents) < s.MAX_AGENTS

        # if self.args.single_process:
        backend = SequentialAgentBackend(train, name, agent_dir)
        # else:
        # backend = ProcessAgentBackend(train, name, agent_dir)
        backend.start()

        color = self.colors.pop()
        agent = Agent(name, agent_dir, name, train, backend, color, color)
        self.agents.append(agent)

    def tile_is_free(self, x, y):
        is_free = (self.arena[x, y] == 0)
        if is_free:
            for obstacle in self.bombs + self.active_agents:
                is_free = is_free and (obstacle.x != x or obstacle.y != y)
        return is_free

    def perform_agent_action(self, agent: Agent, action: str):
        # Perform the specified action if possible, wait otherwise
        if action == 'UP' and self.tile_is_free(agent.x, agent.y - 1):
            agent.y -= 1
            agent.add_event(e.MOVED_UP)
        elif action == 'DOWN' and self.tile_is_free(agent.x, agent.y + 1):
            agent.y += 1
            agent.add_event(e.MOVED_DOWN)
        elif action == 'LEFT' and self.tile_is_free(agent.x - 1, agent.y):
            agent.x -= 1
            agent.add_event(e.MOVED_LEFT)
        elif action == 'RIGHT' and self.tile_is_free(agent.x + 1, agent.y):
            agent.x += 1
            agent.add_event(e.MOVED_RIGHT)
        elif action == 'BOMB' and agent.bombs_left:
            self.logger.info(f'Agent <{agent.name}> drops bomb at {(agent.x, agent.y)}')
            self.bombs.append(Bomb((agent.x, agent.y), agent, s.BOMB_TIMER, s.BOMB_POWER, agent.bomb_sprite))
            observers = getattr(self, "evaluation_diagnostics", {})
            if observers:
                bomb_id = f"{agent.name}:{self.step}"
                self.bombs[-1].diagnostic_id = bomb_id
                if agent.name in observers:
                    observers[agent.name].placed(bomb_id, self.step, (agent.x, agent.y))
            agent.bombs_left = False
            agent.add_event(e.BOMB_DROPPED)
        elif action == 'WAIT':
            agent.add_event(e.WAITED)
        else:
            agent.add_event(e.INVALID_ACTION)

    def poll_and_run_agents(self):
        raise NotImplementedError()

    def send_game_events(self):
        pass

    def do_step(self, user_input='WAIT'):
        assert self.running

        self.step += 1
        self.logger.info(f'STARTING STEP {self.step}')

        self.user_input = user_input
        self.logger.debug(f'User input: {self.user_input}')

        self.poll_and_run_agents()

        # Progress world elements based
        self.collect_coins()
        self.record_coin_collection_steps()
        self.update_explosions()
        self.update_bombs()
        self.evaluate_explosions()

        if self.time_to_stop():
            self.end_round()
        else:
            self.send_game_events()

    def collect_coins(self):
        for coin in self.coins:
            if coin.collectable:
                for a in self.active_agents:
                    if a.x == coin.x and a.y == coin.y:
                        coin.collectable = False
                        self.logger.info(f'Agent <{a.name}> picked up coin at {(a.x, a.y)} and receives 1 point')
                        a.update_score(s.REWARD_COIN)
                        a.add_event(e.COIN_COLLECTED)
                        a.trophies.append(Trophy.coin_trophy)

    def record_coin_collection_steps(self):
        """
        Record the current step for every newly collected coin.

        The framework already counts collected coins in
        agent.statistics. We only add the step numbers at which
        this count increased.
        """
        for agent in self.agents:
            recorded_coins = len(
                self.coin_collection_steps[agent.name]
            )

            collected_coins = agent.statistics.get(
                "coins",
                0,
            )

            newly_collected_coins = (
                collected_coins - recorded_coins
            )

            if newly_collected_coins > 0:
                self.coin_collection_steps[agent.name].extend(
                    [self.step] * newly_collected_coins
                )

    def update_explosions(self):
        # Progress explosions
        remaining_explosions = []
        for explosion in self.explosions:
            explosion.timer -= 1
            if explosion.timer <= 0:
                explosion.next_stage()
                if explosion.stage == 1:
                    explosion.owner.bombs_left = True
            if explosion.stage is not None:
                remaining_explosions.append(explosion)
        self.explosions = remaining_explosions

    def update_bombs(self):
        """
        Count down bombs placed
        Explode bombs at zero timer.

        :return:
        """
        for bomb in self.bombs:
            if bomb.timer <= 0:
                # Explode when timer is finished
                self.logger.info(f'Agent <{bomb.owner.name}>\'s bomb at {(bomb.x, bomb.y)} explodes')
                bomb.owner.add_event(e.BOMB_EXPLODED)
                blast_coords = bomb.get_blast_coords(self.arena)

                # Clear crates
                for (x, y) in blast_coords:
                    if self.arena[x, y] == 1:
                        self.arena[x, y] = 0
                        bomb.owner.add_event(e.CRATE_DESTROYED)
                        # Maybe reveal a coin
                        for c in self.coins:
                            if (c.x, c.y) == (x, y):
                                c.collectable = True
                                self.logger.info(f'Coin found at {(x, y)}')
                                bomb.owner.add_event(e.COIN_FOUND)

                # Create explosion
                screen_coords = [(s.GRID_OFFSET[0] + s.GRID_SIZE * x, s.GRID_OFFSET[1] + s.GRID_SIZE * y) for (x, y) in
                                 blast_coords]
                self.explosions.append(Explosion(blast_coords, screen_coords, bomb.owner, s.EXPLOSION_TIMER))
                if getattr(self, "evaluation_diagnostics", {}):
                    self.explosions[-1].diagnostic_id = getattr(bomb, "diagnostic_id", None)
                bomb.active = False
            else:
                # Progress countdown
                bomb.timer -= 1
        self.bombs = [b for b in self.bombs if b.active]

    def evaluate_explosions(self):
        # Explosions
        agents_hit = set()
        observers = getattr(self, "evaluation_diagnostics", {})
        diagnostic_hits = []
        for explosion in self.explosions:
            # Kill agents
            if explosion.is_dangerous():
                for a in self.active_agents:
                    if (not a.dead) and (a.x, a.y) in explosion.blast_coords:
                        agents_hit.add(a)
                        if observers:
                            diagnostic_hits.append((getattr(explosion, "diagnostic_id", None),
                                                    explosion.owner.name, a.name))
                        self.kill_events.append(
                            {
                                "step": self.step,
                                "killer": explosion.owner.name,
                                "victim": a.name,
                            }
                        )
                        # Note who killed whom, adjust scores
                        if a is explosion.owner:
                            self.logger.info(f'Agent <{a.name}> blown up by own bomb')
                            a.note_self_kill_diagnostic()
                            a.add_event(e.KILLED_SELF)
                            explosion.owner.trophies.append(Trophy.suicide_trophy)
                        else:
                            self.logger.info(f'Agent <{a.name}> blown up by agent <{explosion.owner.name}>\'s bomb')
                            self.logger.info(f'Agent <{explosion.owner.name}> receives 1 point')
                            explosion.owner.update_score(s.REWARD_KILL)
                            explosion.owner.add_event(e.KILLED_OPPONENT)
                            explosion.owner.trophies.append(pygame.transform.smoothscale(a.avatar, (15, 15)))

        for observer in observers.values():
            observer.resolve_hits(self.step, diagnostic_hits)

        # Remove hit agents
        for a in agents_hit:
            a.dead = True
            self.active_agents.remove(a)
            a.add_event(e.GOT_KILLED)
            for aa in self.active_agents:
                if aa is not a:
                    aa.add_event(e.OPPONENT_ELIMINATED)

    def end_round(self):
        if not self.running:
            raise ValueError('End-of-round requested while no round was running')
        # Wait in case there is still a game step running
        self.running = False

        for a in self.agents:
            a.note_stat("score", a.score)
            a.note_stat("rounds")
        self.round_statistics[self.round_id] = {
            "steps": self.step,
            "initial_crates": self.initial_crates,
            "remaining_crates": int((self.arena == 1).sum()),
            "initial_coins": self.initial_coins,
            "kill_events": list(self.kill_events),

            **{
                key: sum(
                    agent.statistics[key]
                    for agent in self.agents
                )
                for key in [
                    "coins",
                    "kills",
                    "suicides",
                ]
            },

            "by_agent": {
                agent.name: {
                    "score": agent.score,
                    "coins": agent.statistics.get("coins", 0),
                    "kills": agent.statistics.get("kills", 0),
                    "suicides": agent.statistics.get(
                        "suicides",
                        0,
                    ),
                    "crates": agent.statistics.get("crates", 0),
                    "bombs": agent.statistics.get("bombs", 0),
                    "bombs_exploded": agent.statistics.get(
                        "bombs_exploded",
                        0,
                    ),
                    "coins_found": agent.statistics.get(
                        "coins_found",
                        0,
                    ),
                    "waits": agent.statistics.get("waits", 0),
                    "deaths": agent.statistics.get("deaths", 0),
                    "moves": agent.statistics.get("moves", 0),
                    "invalid": agent.statistics.get(
                        "invalid",
                        0,
                    ),
                    "steps": agent.statistics.get("steps", 0),
                    "time": agent.statistics.get("time", 0.0),
                    "decision_times": list(
                        agent.decision_times
                    ),
                    "survived": not agent.dead,
                    "coin_collection_steps": list(
                        self.coin_collection_steps[agent.name]
                    ),
                    **(
                        {
                            "action_diagnostics": {
                                name[
                                    len(
                                        ACTION_DIAGNOSTIC_STAT_PREFIX
                                    ):
                                ]: value
                                for name, value
                                in agent.statistics.items()
                                if name.startswith(
                                    ACTION_DIAGNOSTIC_STAT_PREFIX
                                )
                            }
                        }
                        if not agent.train
                        else {}
                    ),
                }
                for agent in self.agents
            },
        }
        for name, observer in getattr(self, "evaluation_diagnostics", {}).items():
            self.round_statistics[self.round_id]["by_agent"][name]["diagnostics"] = (
                observer.finish(self.step)
            )

    def time_to_stop(self):
        # Check round stopping criteria
        if len(self.active_agents) == 0:
            self.logger.info(f'No agent left alive, wrap up round')
            return True

        if (len(self.active_agents) == 1
                and (self.arena == 1).sum() == 0
                and all([not c.collectable for c in self.coins])
                and len(self.bombs) + len(self.explosions) == 0):
            self.logger.info(f'One agent left alive with nothing to do, wrap up round')
            return True

        if any(a.train for a in self.agents) and not self.args.continue_without_training:
            if not any([a.train for a in self.active_agents]):
                self.logger.info('No training agent left alive, wrap up round')
                return True

        if self.step >= s.MAX_STEPS:
            self.logger.info('Maximum number of steps reached, wrap up round')
            return True

        return False

    def end(self):
        if self.running:
            self.end_round()

        results = {'by_agent': {a.name: a.lifetime_statistics for a in self.agents}}
        for a in self.agents:
            results['by_agent'][a.name]['score'] = a.total_score
        results['by_round'] = self.round_statistics
        if self.training_statistics:
            results["training_metadata"] = {
                "mode": "training",
                "world_seed": self.args.seed,
                "agent_seed": int(
                    os.environ.get(
                        "BOMBERMAN_AGENT_SEED",
                        "0",
                    )
                ),
                "scenario": self.args.scenario,
                "feature_version": os.environ.get(
                    "BOMBERMAN_FEATURE_VERSION"
                ),
                "reward_version": os.environ.get(
                    "BOMBERMAN_REWARD_VERSION"
                ),
                "action_space_version": os.environ.get(
                    "BOMBERMAN_ACTION_SPACE_VERSION"
                ),
                "learning_rate_version": os.environ.get(
                    "BOMBERMAN_LEARNING_RATE_VERSION"
                ),
                "exploration_version": os.environ.get(
                    "BOMBERMAN_EXPLORATION_VERSION"
                ),
                "discount_factor_version": os.environ.get(
                    "BOMBERMAN_DISCOUNT_FACTOR_VERSION"
                ),
                "td_target_version": os.environ.get(
                    "BOMBERMAN_TD_TARGET_VERSION"
                ),
                "training_wall_time_seconds": float(
                    perf_counter() - self.run_started_at
                ),
            }

            results["training_by_agent"] = (
                self.training_statistics
            )
            if os.environ.get("BOMBERMAN_AGENT_CONFIGS") is not None:
                results["training_metadata_by_agent"] = {}
                version_keys = (
                    "feature_version", "reward_version", "action_space_version",
                    "learning_rate_version", "exploration_version",
                    "discount_factor_version", "td_target_version",
                )
                for agent in self.agents:
                    if agent.name not in self.training_statistics:
                        continue
                    metadata = dict(results["training_metadata"])
                    overrides = agent.backend.environment_overrides
                    metadata["agent_seed"] = int(overrides.get(
                        "BOMBERMAN_AGENT_SEED", metadata["agent_seed"]
                    ))
                    for key in version_keys:
                        metadata[key] = overrides.get(
                            "BOMBERMAN_" + key.upper(), metadata[key]
                        )
                    results["training_metadata_by_agent"][agent.name] = metadata

        if self.args.save_stats is not False:
            if self.args.save_stats is not True:
                file_name = self.args.save_stats
            elif self.args.match_name is not None:
                file_name = f'results/{self.args.match_name}.json'
            else:
                file_name = f'results/{datetime.now().strftime("%Y-%m-%d %H-%M-%S")}.json'

            name = Path(file_name)
            if not name.parent.exists():
                name.parent.mkdir(parents=True)
            with open(name, "w") as file:
                compact_statistics = (
                    (bool(self.training_statistics) and os.environ.get(
                        "BOMBERMAN_COMPACT_TRAINING_STATS"
                    ) == "1")
                    or bool(getattr(self, "evaluation_diagnostics", {}))
                )

                if compact_statistics:
                    json.dump(
                        results,
                        file,
                        separators=(",", ":"),
                    )
                else:
                    json.dump(
                        results,
                        file,
                        indent=4,
                        sort_keys=True,
                    )


class BombeRLeWorld(GenericWorld):
    def __init__(self, args: WorldArgs, agents):
        super().__init__(args)

        self.rng = np.random.default_rng(args.seed)
        self.setup_agents(agents)

    def setup_agents(self, agents):
        # Add specified agents and start their subprocesses
        self.agents = []
        configurations = agent_configurations()
        counts = {}
        roster = []
        for agent_dir, train in agents:
            if list([d for d, t in agents]).count(agent_dir) > 1:
                name = agent_dir + '_' + str(counts.get(agent_dir, 0))
            else:
                name = agent_dir
            counts[agent_dir] = counts.get(agent_dir, 0) + 1
            roster.append((agent_dir, name, train))
        unknown = set(configurations) - {name for _, name, _ in roster}
        if unknown:
            raise ValueError(f"Agent configurations refer to unknown players: {sorted(unknown)}")
        for agent_dir, name, train in roster:
            self.add_agent(agent_dir, name, train=train)

    def build_arena(self):
        WALL = -1
        FREE = 0
        CRATE = 1
        arena = np.zeros((s.COLS, s.ROWS), int)

        scenario_info = s.SCENARIOS[self.args.scenario]

        # Crates in random locations
        arena[self.rng.random((s.COLS, s.ROWS)) < scenario_info["CRATE_DENSITY"]] = CRATE

        # Walls
        arena[:1, :] = WALL
        arena[-1:, :] = WALL
        arena[:, :1] = WALL
        arena[:, -1:] = WALL
        for x in range(s.COLS):
            for y in range(s.ROWS):
                if (x + 1) * (y + 1) % 2 == 1:
                    arena[x, y] = WALL

        # Clean the start positions
        start_positions = [(1, 1), (1, s.ROWS - 2), (s.COLS - 2, 1), (s.COLS - 2, s.ROWS - 2)]
        for (x, y) in start_positions:
            for (xx, yy) in [(x, y), (x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]:
                if arena[xx, yy] == 1:
                    arena[xx, yy] = FREE

        # Place coins at random, at preference under crates
        coins = []
        all_positions = np.stack(np.meshgrid(np.arange(s.COLS), np.arange(s.ROWS), indexing="ij"), -1)
        crate_positions = self.rng.permutation(all_positions[arena == CRATE])
        free_positions = self.rng.permutation(all_positions[arena == FREE])
        coin_positions = np.concatenate([
            crate_positions,
            free_positions
        ], 0)[:scenario_info["COIN_COUNT"]]
        for x, y in coin_positions:
            coins.append(Coin((x, y), collectable=arena[x, y] == FREE))

        # Reset agents and distribute starting positions
        active_agents = []
        for agent, start_position in zip(self.agents, self.rng.permutation(start_positions)):
            active_agents.append(agent)
            agent.x, agent.y = start_position

        return arena, coins, active_agents

    def get_state_for_agent(self, agent: Agent):
        if agent.dead:
            return None

        state = {
            'round': self.round,
            'step': self.step,
            'field': np.array(self.arena),
            'self': agent.get_state(),
            'others': [other.get_state() for other in self.active_agents if other is not agent],
            'bombs': [bomb.get_state() for bomb in self.bombs],
            'coins': [coin.get_state() for coin in self.coins if coin.collectable],
            'user_input': self.user_input,
        }

        explosion_map = np.zeros(self.arena.shape)
        for exp in self.explosions:
            if exp.is_dangerous():
                for (x, y) in exp.blast_coords:
                    explosion_map[x, y] = max(explosion_map[x, y], exp.timer - 1)
        state['explosion_map'] = explosion_map

        return state

    def poll_and_run_agents(self):
        # Tell agents to act
        observers = getattr(self, "evaluation_diagnostics", {})
        diagnostic_states = {}
        diagnostic_hazards = ()
        if observers:
            from evaluation.task3.diagnostics import snapshot_hazards
            # Snapshot BEFORE the action permutation. A bomb placed by an
            # earlier actor in this tick was not visible to the policy.
            diagnostic_hazards = snapshot_hazards(self.bombs, self.explosions, self.arena)
        for a in self.active_agents:
            state = self.get_state_for_agent(a)
            if a.name in observers:
                diagnostic_states[a.name] = state
            a.store_game_state(state)
            a.reset_game_events()
            if a.available_think_time > 0:
                a.act(state)

        # Give agents time to decide
        perm = self.rng.permutation(len(self.active_agents))
        self.replay['permutations'].append(perm)
        for i in perm:
            a = self.active_agents[i]
            if a.available_think_time > 0:
                try:
                    action, think_time = a.wait_for_act()
                except KeyboardInterrupt:
                    # Stop the game
                    raise
                except:
                    if not self.args.silence_errors:
                        raise
                    # Agents with errors cannot continue
                    action = "ERROR"
                    a.last_evaluation_diagnostic = None
                    think_time = float("inf")

                self.logger.info(f'Agent <{a.name}> chose action {action} in {think_time:.2f}s.')
                if think_time > a.available_think_time:
                    next_think_time = a.base_timeout - (think_time - a.available_think_time)
                    self.logger.warning(f'Agent <{a.name}> exceeded think time by {think_time - a.available_think_time:.2f}s. Setting action to "WAIT" and decreasing available time for next round to {next_think_time:.2f}s.')
                    action = "WAIT"
                    a.last_evaluation_diagnostic = None
                    a.trophies.append(Trophy.time_trophy)
                    a.available_think_time = next_think_time
                else:
                    self.logger.info(f'Agent <{a.name}> stayed within acceptable think time.')
                    a.available_think_time = a.base_timeout
            else:
                self.logger.info(f'Skipping agent <{a.name}> because of last slow think time.')
                a.available_think_time += a.base_timeout
                action = "WAIT"
                a.last_evaluation_diagnostic = None

            self.replay['actions'][a.name].append(action)
            observer = getattr(self, "evaluation_diagnostics", {}).get(a.name)
            if observer is not None:
                observer.observe(self.step, action, getattr(a, "last_evaluation_diagnostic", None),
                                 state=diagnostic_states[a.name], hazards=diagnostic_hazards)
            self.perform_agent_action(a, action)
            if observer is not None:
                observer.action_result(self.step, (a.x, a.y), a.events)

    def send_game_events(self):
        # Send events to all agents that expect them, then reset and wait for them
        for a in self.agents:
            if a.train:
                if not a.dead:
                    a.process_game_events(self.get_state_for_agent(a))
                for enemy in self.active_agents:
                    if enemy is not a:
                        pass
                        # a.process_enemy_game_events(self.get_state_for_agent(enemy), enemy)
        for a in self.agents:
            if a.train:
                if not a.dead:
                    a.wait_for_game_event_processing()
                for enemy in self.active_agents:
                    if enemy is not a:
                        pass
                        # a.wait_for_enemy_game_event_processing()

    def end_round(self):
        super().end_round()

        self.logger.info(f'WRAPPING UP ROUND #{self.round}')
        # Clean up survivors
        for a in self.active_agents:
            a.add_event(e.SURVIVED_ROUND)

        # Send final event to agents that expect them.
        for a in self.agents:
            if a.train:
                training_metrics = a.round_ended()

                if training_metrics is None:
                    continue

                if not isinstance(training_metrics, dict):
                    raise TypeError(
                        "end_of_round must return a dictionary "
                        "or None, but agent "
                        f"{a.name!r} returned "
                        f"{type(training_metrics).__name__}."
                    )

                self.training_statistics.setdefault(
                    a.name,
                    [],
                ).append(training_metrics)

        # Save course of the game for future replay
        if self.args.save_replay:
            self.replay['n_steps'] = self.step
            name = f'replays/{self.round_id}.pt' if self.args.save_replay is True else self.args.save_replay
            with open(name, 'wb') as f:
                pickle.dump(self.replay, f)

    def end(self):
        super().end()
        self.logger.info('SHUT DOWN')
        for a in self.agents:
            # Send exit message to shut down agent
            self.logger.debug(f'Sending exit message to agent <{a.name}>')
            # todo multiprocessing shutdown


class GUI:
    def __init__(self, world: GenericWorld):
        self.world = world
        self.screenshot_dir = Path(__file__).parent / "screenshots"

        # Initialize screen
        self.screen = pygame.display.set_mode((s.WIDTH, s.HEIGHT))
        pygame.display.set_caption('BombeRLe')
        icon = pygame.image.load(s.ASSET_DIR / f'bomb_yellow.png')
        pygame.display.set_icon(icon)

        # Background and tiles
        self.background = pygame.Surface((s.WIDTH, s.HEIGHT))
        self.background = self.background.convert()
        self.background.fill((0, 0, 0))
        self.t_wall = pygame.image.load(s.ASSET_DIR / 'brick.png')
        self.t_crate = pygame.image.load(s.ASSET_DIR / 'crate.png')

        # Font for scores and such
        font_name = s.ASSET_DIR / 'emulogic.ttf'
        self.fonts = {
            'huge': pygame.font.Font(font_name, 20),
            'big': pygame.font.Font(font_name, 16),
            'medium': pygame.font.Font(font_name, 10),
            'small': pygame.font.Font(font_name, 8),
        }

        self.frame = 0

    def render_text(self, text, x, y, color, halign='left', valign='top', size='medium', aa=False):
        text_surface = self.fonts[size].render(text, aa, color)
        text_rect = text_surface.get_rect()
        if halign == 'left':   text_rect.left = x
        if halign == 'center': text_rect.centerx = x
        if halign == 'right':  text_rect.right = x
        if valign == 'top':    text_rect.top = y
        if valign == 'center': text_rect.centery = y
        if valign == 'bottom': text_rect.bottom = y
        self.screen.blit(text_surface, text_rect)

    def render(self):
        self.screen.blit(self.background, (0, 0))

        if self.world.round == 0:
            return

        self.frame += 1
        pygame.display.set_caption(f'BombeRLe | Round #{self.world.round}')

        # World
        for x in range(self.world.arena.shape[1]):
            for y in range(self.world.arena.shape[0]):
                if self.world.arena[x, y] == -1:
                    self.screen.blit(self.t_wall,
                                     (s.GRID_OFFSET[0] + s.GRID_SIZE * x, s.GRID_OFFSET[1] + s.GRID_SIZE * y))
                if self.world.arena[x, y] == 1:
                    self.screen.blit(self.t_crate,
                                     (s.GRID_OFFSET[0] + s.GRID_SIZE * x, s.GRID_OFFSET[1] + s.GRID_SIZE * y))
        self.render_text(f'Step {self.world.step:d}', s.GRID_OFFSET[0], s.HEIGHT - s.GRID_OFFSET[1] / 2, (64, 64, 64),
                         valign='center', halign='left', size='medium')

        # Items
        for bomb in self.world.bombs:
            bomb.render(self.screen, s.GRID_OFFSET[0] + s.GRID_SIZE * bomb.x, s.GRID_OFFSET[1] + s.GRID_SIZE * bomb.y)
        for coin in self.world.coins:
            if coin.collectable:
                coin.render(self.screen, s.GRID_OFFSET[0] + s.GRID_SIZE * coin.x,
                            s.GRID_OFFSET[1] + s.GRID_SIZE * coin.y)

        # Agents
        for agent in self.world.active_agents:
            agent.render(self.screen, s.GRID_OFFSET[0] + s.GRID_SIZE * agent.x,
                         s.GRID_OFFSET[1] + s.GRID_SIZE * agent.y)

        # Explosions
        for explosion in self.world.explosions:
            explosion.render(self.screen)

        # Scores
        # agents = sorted(self.agents, key=lambda a: (a.score, -a.mean_time), reverse=True)
        agents = self.world.agents
        leading = max(agents, key=lambda a: (a.score, a.name))
        y_base = s.GRID_OFFSET[1] + 15
        for i, a in enumerate(agents):
            bounce = 0 if (a is not leading or self.world.running) else np.abs(10 * np.sin(5 * time()))
            a.render(self.screen, 600, y_base + 50 * i - 15 - bounce)
            self.render_text(a.display_name, 650, y_base + 50 * i,
                             (64, 64, 64) if a.dead else (255, 255, 255),
                             valign='center', size='small')
            for j, trophy in enumerate(a.trophies):
                self.screen.blit(trophy, (660 + 10 * j, y_base + 50 * i + 12))
            self.render_text(f'{a.score:d}', 830, y_base + 50 * i, (255, 255, 255),
                             valign='center', halign='right', size='big')
            self.render_text(f'{a.total_score:d}', 890, y_base + 50 * i, (64, 64, 64),
                             valign='center', halign='right', size='big')

        # End of round info
        if not self.world.running:
            x_center = (s.WIDTH - s.GRID_OFFSET[0] - s.COLS * s.GRID_SIZE) / 2 + s.GRID_OFFSET[0] + s.COLS * s.GRID_SIZE
            color = np.int_((255 * (np.sin(3 * time()) / 3 + .66),
                             255 * (np.sin(4 * time() + np.pi / 3) / 3 + .66),
                             255 * (np.sin(5 * time() - np.pi / 3) / 3 + .66)))
            self.render_text(leading.display_name, x_center, 320, color,
                             valign='top', halign='center', size='huge')
            self.render_text('has won the round!', x_center, 350, color,
                             valign='top', halign='center', size='big')
            leading_total = max(self.world.agents, key=lambda a: (a.total_score, a.display_name))
            if leading_total is leading:
                self.render_text(f'{leading_total.display_name} is also in the lead.', x_center, 390, (128, 128, 128),
                                 valign='top', halign='center', size='medium')
            else:
                self.render_text(f'But {leading_total.display_name} is in the lead.', x_center, 390, (128, 128, 128),
                                 valign='top', halign='center', size='medium')

        if self.world.running and self.world.args.make_video:
            self.world.logger.debug(f'Saving screenshot for frame {self.frame}')
            pygame.image.save(self.screen, str(self.screenshot_dir / f'{self.world.round_id}_{self.frame:05d}.png'))

    def make_video(self):
        # Turn screenshots into videos
        assert self.world.args.make_video is not False

        if self.world.args.make_video is True:
            files = [self.screenshot_dir / f'{self.world.round_id}_video.mp4',
                     self.screenshot_dir / f'{self.world.round_id}_video.webm']
        else:
            files = [Path(self.world.args.make_video)]

        self.world.logger.debug(f'Turning screenshots into video')

        PARAMS = {
            ".mp4": ['-preset', 'veryslow', '-tune', 'animation', '-crf', '5', '-c:v', 'libx264',
                     '-pix_fmt', 'yuv420p'],
            ".webm": ['-threads', '2', '-tile-columns', '2', '-frame-parallel', '0', '-g', '100', '-speed', '1', '-pix_fmt', 'yuv420p', '-qmin', '0', '-qmax', '10', '-crf', '5', '-b:v', '2M', '-c:v', 'libvpx-vp9', ]
        }

        for video_file in files:
            subprocess.call([
                'ffmpeg', '-y', '-framerate', f'{self.world.args.fps}',
                '-f', 'image2', '-pattern_type', 'glob',
                '-i', self.screenshot_dir / f'{self.world.round_id}_*.png',
                *PARAMS[video_file.suffix],
                video_file
            ])
        self.world.logger.info("Done writing videos.")
        for f in self.screenshot_dir.glob(f'{self.world.round_id}_*.png'):
            f.unlink()
