from collections import namedtuple, deque
import os
import json
from pathlib import Path
import shutil
import numpy as np
from typing import List

from .callbacks import (
    ACTIONS,
    clear_feature_context,
    extract_features,
    model_action_for_last_feature_state,
)
from .discount_factors import (
    DEFAULT_DISCOUNT_FACTOR_VERSION,
    get_discount_factor,
)
from .features import (
    CoinPriorityDiagnosticContext,
    TASK2_REPORT_V6_COIN_PRIORITY_FEATURE_VERSION,
)
from .exploration_schedules import (
    DEFAULT_EXPLORATION_VERSION,
    get_exploration_parameters,
)
from .learning_rates import (
    DEFAULT_LEARNING_RATE_VERSION,
    get_learning_rate_function,
    get_learning_rate_parameters,
)
from .model_io import save_checkpoint
from .rewards import (
    DEFAULT_REWARD_VERSION,
    RewardResult,
    get_reward_function,
    TASK3_TACTICAL_HOLD_REWARD_VERSION,
    TASK3_TACTICAL_REWARD_V4_VERSION,
    TASK3_TACTICAL_REWARD_V5_VERSION,
    KILLED_BY_OPPONENT,
    WAIT_AT_BLOCK_TARGET,
    SKIPPED_BLOCK_TARGET,
)
from .td_targets import (
    DEFAULT_TD_TARGET_VERSION,
    calculate_sampled_n_step_target,
    get_td_target_n_steps,
    get_td_target_parameters,
)

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

TRANSITION_HISTORY_SIZE = 3

# Version of the per-round training telemetry returned to the framework.
TRAINING_METRICS_SCHEMA_VERSION = 2


def _positive_environment_integer(
    variable_name: str,
    default: int | None,
) -> int | None:
    """Read an optional strictly positive integer setting."""
    configured_value = os.environ.get(variable_name)

    if configured_value is None:
        return default

    try:
        value = int(configured_value)
    except ValueError as error:
        raise ValueError(
            f"{variable_name} must be a positive integer, "
            f"got {configured_value!r}."
        ) from error

    if value <= 0:
        raise ValueError(
            f"{variable_name} must be a positive integer, "
            f"got {configured_value!r}."
        )

    return value


def reset_round_metrics(self) -> None:
    """Start the compact telemetry counters for one training round."""
    if getattr(self, "exploration_parameters", {}).get("schedule") == "multiplicative_temperature_decay":
        self.round_temperature_start = self.temperature
    else:
        self.round_epsilon_start = self.epsilon
    self.round_q_table_states_start = len(self.model)
    self.round_updated_pairs_start = len(
        self.updated_state_action_pairs
    )

    self.round_return = 0.0
    self.round_q_updates = 0
    self.round_td_error_abs_sum = 0.0
    self.round_td_error_squared_sum = 0.0
    self.round_td_error_abs_max = 0.0
    self.round_reward_components = {}
    if getattr(self, "reward_version", None) in (TASK3_TACTICAL_HOLD_REWARD_VERSION,
                                                TASK3_TACTICAL_REWARD_V4_VERSION,
                                                TASK3_TACTICAL_REWARD_V5_VERSION):
        # Registered zero counts keep both V9 events visible even in a round
        # without HOLD. Actual contributions use the ordinary recorder below.
        self.round_reward_components = {
            name: {"count": 0, "reward_sum": 0.0}
            for name in (WAIT_AT_BLOCK_TARGET, SKIPPED_BLOCK_TARGET)
        }
        if self.reward_version in (TASK3_TACTICAL_REWARD_V4_VERSION,
                                   TASK3_TACTICAL_REWARD_V5_VERSION):
            self.round_reward_components[KILLED_BY_OPPONENT] = {"count": 0, "reward_sum": 0.0}
    self.round_block_diagnostics = {}
    self.round_action_counts = {
        action: 0
        for action in ACTIONS
    }
    self.round_action_update_counts = {
        action: 0
        for action in ACTIONS
    }
    self.round_potential_hint_diagnostics = {
        "available": 0,
        "followed": 0,
        "exploitation_available": 0,
        "exploitation_followed": 0,
        "tie_break_available": 0,
        "tie_break_followed": 0,
        "tie_break_exploitation_available": 0,
        "tie_break_exploitation_followed": 0,
        "outside_bfs_available": 0,
        "outside_bfs_followed": 0,
        "outside_bfs_exploitation_available": 0,
        "outside_bfs_exploitation_followed": 0,
    }


def record_reward_result(
    self,
    reward_result: RewardResult,
) -> None:
    """Add one reward result to the current round's telemetry."""
    self.round_return += reward_result.total

    for event, reward in reward_result.contributions:
        component = self.round_reward_components.setdefault(
            event,
            {
                "count": 0,
                "reward_sum": 0.0,
            },
        )

        component["count"] += 1
        component["reward_sum"] += reward


def record_td_error(
    self,
    td_error: float | None,
) -> None:
    """Add one completed Q-update to the current round's telemetry."""
    if td_error is None:
        return

    absolute_td_error = abs(td_error)

    self.round_q_updates += 1
    self.round_td_error_abs_sum += absolute_td_error
    self.round_td_error_squared_sum += td_error ** 2
    self.round_td_error_abs_max = max(
        self.round_td_error_abs_max,
        absolute_td_error,
    )


def build_training_metrics(self) -> dict:
    """Build one JSON-compatible summary of the completed round."""
    reward_components = {
        event: {
            "count": int(component["count"]),
            "reward_sum": float(component["reward_sum"]),
        }
        for event, component in sorted(
            self.round_reward_components.items()
        )
    }

    if getattr(self, "exploration_parameters", {}).get("schedule", "multiplicative_epsilon_decay") == "multiplicative_epsilon_decay":
        exploration_metrics = {
            "epsilon_start": float(self.round_epsilon_start),
            "epsilon_end": float(self.epsilon),
        }
    elif self.exploration_parameters["schedule"] == "multiplicative_temperature_decay":
        exploration_metrics = {
            "epsilon_start": None,
            "epsilon_end": None,
            "temperature_start": float(self.round_temperature_start),
            "temperature_end": float(self.temperature),
        }
    else:
        raise ValueError(
            f"Unknown exploration schedule "
            f"{self.exploration_parameters['schedule']!r}."
        )

    return {
        "schema_version": TRAINING_METRICS_SCHEMA_VERSION,
        "round": int(self.rounds_completed),
        "undiscounted_return": float(self.round_return),
        **exploration_metrics,
        "q_updates": int(self.round_q_updates),
        "td_error_abs_sum": float(
            self.round_td_error_abs_sum
        ),
        "td_error_squared_sum": float(
            self.round_td_error_squared_sum
        ),
        "td_error_abs_max": float(
            self.round_td_error_abs_max
        ),
        "q_table_states": int(len(self.model)),
        "new_q_table_states": int(
            len(self.model) - self.round_q_table_states_start
        ),
        "updated_state_action_pairs": int(
            len(self.updated_state_action_pairs)
        ),
        "new_updated_state_action_pairs": int(
            len(self.updated_state_action_pairs)
            - self.round_updated_pairs_start
        ),
        "action_counts": dict(self.round_action_counts),
        "action_update_counts": dict(
            self.round_action_update_counts
        ),
        "potential_hint_diagnostics": dict(
            self.round_potential_hint_diagnostics
        ),
        "reward_components": reward_components,
        "block_diagnostics": dict(self.round_block_diagnostics),
    }


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    # Algorithmic n-step storage is independent of the bounded debug
    # history above. It is drained completely at every round boundary.
    self.td_target_buffer = deque()

    configured_exploration_version = os.environ.get(
        "BOMBERMAN_EXPLORATION_VERSION"
    )
    self.exploration_version = (
        configured_exploration_version
        or DEFAULT_EXPLORATION_VERSION
    )
    self.exploration_parameters = get_exploration_parameters(
        self.exploration_version
    )
    if self.exploration_parameters["schedule"] == "multiplicative_epsilon_decay":
        self.epsilon_start = float(self.exploration_parameters["epsilon_start"])
        self.epsilon_min = float(self.exploration_parameters["epsilon_min"])
        self.epsilon_decay = float(self.exploration_parameters["epsilon_decay"])
        self.epsilon = self.epsilon_start
    elif self.exploration_parameters["schedule"] == "multiplicative_temperature_decay":
        self.temperature_start = float(self.exploration_parameters["temperature_start"])
        self.temperature_min = float(self.exploration_parameters["temperature_min"])
        self.temperature_decay = float(self.exploration_parameters["temperature_decay"])
        self.temperature = self.temperature_start
    else:
        raise ValueError(
            f"Unknown exploration schedule "
            f"{self.exploration_parameters['schedule']!r}."
        )
    self.rounds_completed = 0
    # Direct/manual training retains the historical every-round save.
    # Controlled experiment runners may opt into less frequent saves.
    self.checkpoint_interval = _positive_environment_integer(
        "BOMBERMAN_CHECKPOINT_INTERVAL",
        1,
    )
    self.training_round_limit = _positive_environment_integer(
        "BOMBERMAN_TRAINING_ROUNDS",
        None,
    )
    raw_snapshots = json.loads(os.environ.get("BOMBERMAN_SNAPSHOT_ROUNDS", "[]"))
    if (not isinstance(raw_snapshots, list)
            or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
                   or (self.training_round_limit is not None and value >= self.training_round_limit)
                   for value in raw_snapshots)
            or len(set(raw_snapshots)) != len(raw_snapshots)):
        raise ValueError("BOMBERMAN_SNAPSHOT_ROUNDS requires unique positive intermediate rounds.")
    self.snapshot_rounds = frozenset(raw_snapshots)
    self.updated_state_action_pairs = set()
    self.state_action_update_counts = {}

    configured_discount_factor_version = os.environ.get(
        "BOMBERMAN_DISCOUNT_FACTOR_VERSION"
    )
    self.discount_factor_version = (
        configured_discount_factor_version
        or DEFAULT_DISCOUNT_FACTOR_VERSION
    )
    self.gamma = get_discount_factor(
        self.discount_factor_version
    )

    self.logger.info(
        "Training with discount-factor version "
        f"{self.discount_factor_version!r} "
        f"(gamma={self.gamma})."
    )

    configured_td_target_version = os.environ.get(
        "BOMBERMAN_TD_TARGET_VERSION"
    )
    self.td_target_version = (
        configured_td_target_version
        or DEFAULT_TD_TARGET_VERSION
    )
    self.td_target_parameters = get_td_target_parameters(
        self.td_target_version
    )
    self.td_target_n_steps = get_td_target_n_steps(
        self.td_target_version
    )

    self.logger.info(
        "Training with TD-target version "
        f"{self.td_target_version!r} "
        f"(n={self.td_target_n_steps})."
    )

    self.logger.info(
        f"Training with exploration version "
        f"{self.exploration_version!r}."
    )

    configured_learning_rate_version = os.environ.get(
        "BOMBERMAN_LEARNING_RATE_VERSION"
    )

    self.learning_rate_version = (
        configured_learning_rate_version
        or DEFAULT_LEARNING_RATE_VERSION
    )
    self.learning_rate_function = get_learning_rate_function(
        self.learning_rate_version
    )

    self.logger.info(
        f"Training with learning-rate version "
        f"{self.learning_rate_version!r}."
    )

    configured_reward_version = os.environ.get(
        "BOMBERMAN_REWARD_VERSION"
    )

    self.reward_version = (
        configured_reward_version
        or DEFAULT_REWARD_VERSION
    )
    self.reward_function = get_reward_function(
        self.reward_version
    )

    reset_round_metrics(self)

    self.logger.info(
        f"Training with reward version "
        f"{self.reward_version!r}."
    )

def maximum_learning_action_q_value(
    self,
    state,
) -> float:
    """Return max-a Q(state, a) over the configured learning actions."""
    if state not in self.model:
        self.model[state] = np.zeros(
            len(ACTIONS),
            dtype=float,
        )

    learning_q_values = self.model[state][
        list(self.learning_action_indices)
    ]

    return float(np.max(learning_q_values))


def update_q_table_from_target(
    self,
    transition: Transition,
    td_target: float,
) -> float | None:
    """Apply one tabular update from an already calculated TD target.

    :param self: The agent object shared by all callbacks.
    :param transition: Transition identifying the updated state-action pair.
    :param td_target: Finite scalar target for that state-action pair.
    """
    state, action, _, _ = transition

    if state is None:
        return None

    if not np.isfinite(td_target):
        raise ValueError("The TD target must be finite.")

    if state not in self.model:
        self.model[state] = np.zeros(
            len(ACTIONS),
            dtype=float,
        )

    action_index = ACTIONS.index(action)
    self.round_action_update_counts[action] += 1
    current_q_value = self.model[state][action_index]

    state_action_pair = (
        state,
        action_index,
    )
    self.updated_state_action_pairs.add(state_action_pair)

    visit_count = (
        self.state_action_update_counts.get(state_action_pair, 0)
        + 1
    )
    self.state_action_update_counts[state_action_pair] = visit_count
    learning_rate = self.learning_rate_function(visit_count)

    td_error = td_target - current_q_value

    self.model[state][action_index] += (
        learning_rate * td_error
    )

    return float(td_error)


def update_q_table(
    self,
    transition: Transition,
) -> float | None:
    """Perform the historical one-step Q-learning update."""
    state, _, next_state, reward = transition

    if state is None:
        return None

    if next_state is None:
        td_target = reward
    else:
        td_target = reward + self.gamma * (
            maximum_learning_action_q_value(
                self,
                next_state,
            )
        )

    return update_q_table_from_target(
        self,
        transition,
        float(td_target),
    )


def calculate_transition_sequence_target(
    self,
    transitions: tuple[Transition, ...],
) -> float:
    """Calculate one sampled return from a consecutive transition window."""
    if not transitions:
        raise ValueError(
            "A TD-target transition window cannot be empty."
        )

    if len(transitions) > self.td_target_n_steps:
        raise ValueError(
            "A TD-target transition window cannot exceed its "
            f"configured horizon of {self.td_target_n_steps}."
        )

    if any(
        transition.next_state is None
        for transition in transitions[:-1]
    ):
        raise ValueError(
            "A terminal transition may only appear at the end of a "
            "TD-target window."
        )

    final_next_state = transitions[-1].next_state
    bootstrap_value = (
        None
        if final_next_state is None
        else maximum_learning_action_q_value(
            self,
            final_next_state,
        )
    )

    return calculate_sampled_n_step_target(
        rewards=tuple(
            transition.reward
            for transition in transitions
        ),
        discount_factor=self.gamma,
        bootstrap_value=bootstrap_value,
    )


def update_oldest_buffered_transition(self) -> float | None:
    """Update and remove the oldest transition in the n-step buffer."""
    if not self.td_target_buffer:
        raise ValueError("Cannot update an empty TD-target buffer.")

    transitions = tuple(self.td_target_buffer)
    td_target = calculate_transition_sequence_target(
        self,
        transitions,
    )
    oldest_transition = self.td_target_buffer.popleft()

    return update_q_table_from_target(
        self,
        oldest_transition,
        td_target,
    )


def process_learning_transition(
    self,
    transition: Transition,
) -> tuple[float | None, ...]:
    """Process one transition under the configured TD-target horizon.

    One-step mode deliberately follows the historical immediate update
    path. Longer horizons update one oldest transition after ``n`` sampled
    rewards. A terminal transition drains the complete remaining buffer,
    producing successively shorter returns without bootstrapping.
    """
    if self.td_target_n_steps == 1:
        return (update_q_table(self, transition),)

    self.td_target_buffer.append(transition)

    if transition.next_state is None:
        td_errors = []

        while self.td_target_buffer:
            td_errors.append(
                update_oldest_buffered_transition(self)
            )

        return tuple(td_errors)

    if len(self.td_target_buffer) == self.td_target_n_steps:
        return (
            update_oldest_buffered_transition(self),
        )

    if len(self.td_target_buffer) > self.td_target_n_steps:
        raise RuntimeError(
            "The TD-target buffer exceeded its configured horizon."
        )

    return ()

def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Score the completed transition and update the oldest available TD target.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # Convert the framework data into one Q-learning transition.
    reward_result = calculate_reward(
        self,
        old_game_state,
        self_action,
        new_game_state,
        events,
    )

    record_reward_result(self, reward_result)

    transition = Transition(
        state=self.last_action_features,
        action=model_action_for_last_feature_state(
            self,
            self_action,
        ),
        next_state=extract_features(
            self,
            new_game_state,
            previous_position=tuple(
                old_game_state["self"][3]
            ),
            after_action=True,
        ),
        reward=reward_result.total,
    )
    # Keep the transition for debugging and possible later extensions.
    self.transitions.append(transition)

    for td_error in process_learning_transition(
        self,
        transition,
    ):
        record_td_error(self, td_error)


def end_of_round(
    self,
    last_game_state: dict,
    last_action: str,
    events: List[str],
) -> dict:
    """Learn from the terminal transition and return round telemetry."""
    self.logger.debug(
        f'Encountered event(s) '
        f'{", ".join(map(repr, events))} in final step'
    )

    reward_result = calculate_reward(
        self,
        last_game_state,
        last_action,
        None,
        events,
    )
    record_reward_result(self, reward_result)

    final_transition = Transition(
        state=self.last_action_features,
        action=model_action_for_last_feature_state(
            self,
            last_action,
        ),
        next_state=None,
        reward=reward_result.total,
    )

    self.transitions.append(final_transition)

    for td_error in process_learning_transition(
        self,
        final_transition,
    ):
        record_td_error(self, td_error)

    if self.td_target_buffer:
        raise RuntimeError(
            "The TD-target buffer was not empty after terminal flush."
        )

    clear_feature_context(self)

    if self.exploration_parameters["schedule"] == "multiplicative_epsilon_decay":
        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay,
        )
    elif self.exploration_parameters["schedule"] == "multiplicative_temperature_decay":
        self.temperature = max(
            self.temperature_min,
            self.temperature * self.temperature_decay,
        )
    else:
        raise ValueError(
            f"Unknown exploration schedule "
            f"{self.exploration_parameters['schedule']!r}."
        )

    self.rounds_completed += 1

    if self.exploration_parameters["schedule"] == "multiplicative_epsilon_decay":
        exploration_log_value = f"epsilon={self.epsilon:.4f}"
    else:
        exploration_log_value = f"temperature={self.temperature:.4f}"

    self.logger.info(
        f"End of round {self.rounds_completed}: "
        f"{exploration_log_value}, "
        f"known_states={len(self.model)}"
    )

    final_configured_round = (
        self.training_round_limit is not None
        and self.rounds_completed >= self.training_round_limit
    )
    checkpoint_due = (
        self.rounds_completed % self.checkpoint_interval == 0
        or final_configured_round
        or self.rounds_completed in getattr(self, "snapshot_rounds", ())
    )

    if checkpoint_due:
        if self.exploration_parameters["schedule"] == "multiplicative_epsilon_decay":
            exploration_checkpoint_metadata = {
                "epsilon_start": self.epsilon_start,
                "epsilon_min": self.epsilon_min,
                "epsilon_decay": self.epsilon_decay,
                "epsilon_current": self.epsilon,
            }
        elif self.exploration_parameters["schedule"] == "multiplicative_temperature_decay":
            exploration_checkpoint_metadata = {
                "epsilon_start": None,
                "epsilon_min": None,
                "epsilon_decay": None,
                "epsilon_current": None,
                "temperature_start": self.temperature_start,
                "temperature_min": self.temperature_min,
                "temperature_decay": self.temperature_decay,
                "temperature_current": self.temperature,
            }
        else:
            raise ValueError(
                f"Unknown exploration schedule "
                f"{self.exploration_parameters['schedule']!r}."
            )

        save_checkpoint(
            file_path=self.model_file,
            q_table=self.model,
            feature_version=self.feature_version,
            action_order=tuple(ACTIONS),
            # The terminal flush above updates every action exactly once,
            # including short rounds and the final n-1 buffered transitions.
            # These are the same canonical counters used by visit-decay alpha.
            state_action_update_counts=self.state_action_update_counts,
            training_metadata={
                "reward_version": self.reward_version,
                "action_space_version": self.action_space_version,
                "agent_seed": self.agent_seed,
                "rounds_completed": self.rounds_completed,
                # Retain the historical field as the initial learning rate.
                "alpha": self.learning_rate_function(1),
                "learning_rate_version": self.learning_rate_version,
                "learning_rate_parameters": (
                    get_learning_rate_parameters(
                        self.learning_rate_version
                    )
                ),
                "discount_factor_version": (
                    self.discount_factor_version
                ),
                "gamma": self.gamma,
                "td_target_version": self.td_target_version,
                "td_target_parameters": dict(
                    self.td_target_parameters
                ),
                "exploration_version": self.exploration_version,
                "exploration_parameters": dict(
                    self.exploration_parameters
                ),
                **exploration_checkpoint_metadata,
            },
        )

        if self.rounds_completed in getattr(self, "snapshot_rounds", ()):
            model_path = Path(self.model_file)
            snapshot = model_path.with_name(
                f"{model_path.stem}_round_{self.rounds_completed:05d}{model_path.suffix}"
            )
            # Exclusive creation prevents replacement of an earlier frozen model.
            # Copy the checkpoint just saved, preserving its exact Q/count metadata.
            with model_path.open("rb") as source, snapshot.open("xb") as destination:
                shutil.copyfileobj(source, destination)

    training_metrics = build_training_metrics(self)
    reset_round_metrics(self)

    return training_metrics


def calculate_reward(
    self,
    old_game_state: dict | None,
    action: str | None,
    new_game_state: dict | None,
    events: List[str],
) -> RewardResult:
    """
    Calculate one versioned reward and retain its contributions.
    """
    reward_features = getattr(self, "last_action_features", None)
    if getattr(self, "feature_version", None) == TASK2_REPORT_V6_COIN_PRIORITY_FEATURE_VERSION:
        context = getattr(self, "last_action_coin_route_bomb_diagnostic", None)
        if not isinstance(context, CoinPriorityDiagnosticContext):
            raise ValueError("Coin-priority rewards require full V6 context from the action.")
        # This tuple is used only by the unchanged reward function. Policy
        # selection and both ends of every TD transition retain reduced states.
        reward_features = context.full_feature_state
    reward_result = self.reward_function(
        old_game_state,
        model_action_for_last_feature_state(
            self,
            action,
        ),
        new_game_state,
        list(events),
        getattr(
            self,
            "last_action_potential_guidance_context",
            None,
        ),
        self.gamma,
        reward_features,
    )

    self.logger.debug(
        f"Awarded {reward_result.total} for events "
        f"{', '.join(event for event, _ in reward_result.contributions)}"
    )

    return reward_result
