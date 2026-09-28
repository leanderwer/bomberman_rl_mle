"""Five-step tabular Q-learning and V5 training telemetry for the final agent."""

from collections import namedtuple, deque
from collections.abc import Sequence
from math import nextafter
import math
from numbers import Real
import os

import numpy as np

from .callbacks import (
    ACTIONS,
    clear_feature_context,
    extract_features,
    model_action_for_last_feature_state,
)
from .model_io import save_checkpoint
from .rewards import (
    REWARD_VERSION,
    RewardResult,
    calculate_reward as reward_for_transition,
    KILLED_BY_OPPONENT,
    WAIT_AT_BLOCK_TARGET,
    SKIPPED_BLOCK_TARGET,
)


Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))
TRAINING_METRICS_SCHEMA_VERSION = 2
LEARNING_RATE_VERSION = "task2_visit_decay_power_060_v1"
DISCOUNT_FACTOR_VERSION = "gamma_095_v1"
TD_TARGET_VERSION = "q_learning_5_step_v1"
EXPLORATION_VERSION = "task3_softmax_22000_v1"
GAMMA = 0.95
TD_TARGET_N_STEPS = 5
TEMPERATURE_START = 1.0
TEMPERATURE_MIN = 0.05
TEMPERATURE_DECAY_ROUNDS = 22000
# nextafter rounds down decay factor to make sure temperature gets to TEMPERATURE_MIN after TEMPERATURE_DECAY_ROUNDS
TEMPERATURE_DECAY = nextafter(
    (TEMPERATURE_MIN / TEMPERATURE_START) ** (1.0 / TEMPERATURE_DECAY_ROUNDS), 0.0
)


def _require_version(value: str, expected: str, setting: str) -> str:
    """Reject historical settings instead of silently changing the algorithm."""
    if value != expected:
        raise ValueError(f"{setting} must be {expected!r}, got {value!r}.")
    return value


def _environment_version(variable_name: str, expected: str) -> str:
    return _require_version(os.environ.get(variable_name, expected), expected, variable_name)


def learning_rate(visit_count: int) -> float:
    """Decay alpha by updates of this canonical state-action pair only."""
    if isinstance(visit_count, bool) or not isinstance(visit_count, int) or visit_count < 1:
        raise ValueError("The state-action visit count must be a positive integer.")
    return 0.1 * (100.0 / (99.0 + visit_count)) ** 0.6


def get_learning_rate_parameters(version: str = LEARNING_RATE_VERSION) -> dict:
    _require_version(version, LEARNING_RATE_VERSION, "Learning-rate version")
    return {
        "schedule": "power_visit_count", "initial_alpha": 0.1,
        "reference": 100.0, "offset": 99.0, "exponent": 0.6,
    }


def get_exploration_parameters(version: str = EXPLORATION_VERSION) -> dict:
    _require_version(version, EXPLORATION_VERSION, "Exploration version")
    return {
        "schedule": "multiplicative_temperature_decay",
        "temperature_start": TEMPERATURE_START, "temperature_min": TEMPERATURE_MIN,
        "temperature_decay": TEMPERATURE_DECAY, "rounds_to_minimum": TEMPERATURE_DECAY_ROUNDS,
    }


def get_td_target_parameters(version: str = TD_TARGET_VERSION) -> dict:
    _require_version(version, TD_TARGET_VERSION, "TD-target version")
    return {
        "algorithm": "sampled_n_step_q_learning", "n_steps": TD_TARGET_N_STEPS,
        "reward_source": "sampled_behavior_trajectory",
        "bootstrap_operator": "max_learning_action_q", "terminal_bootstrap": "none",
        "importance_sampling": "none",
    }


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
    self.round_temperature_start = self.temperature
    self.round_q_table_states_start = len(self.model)
    self.round_updated_pairs_start = len(
        self.updated_state_action_pairs
    )

    self.round_return = 0.0
    self.round_q_updates = 0
    self.round_td_error_abs_sum = 0.0
    self.round_td_error_squared_sum = 0.0
    self.round_td_error_abs_max = 0.0
    # Register zero counts as well, so these event rows remain visible in plots.
    self.round_reward_components = {
        name: {"count": 0, "reward_sum": 0.0}
        for name in (WAIT_AT_BLOCK_TARGET, SKIPPED_BLOCK_TARGET, KILLED_BY_OPPONENT)
    }
    self.round_block_diagnostics = {}
    self.round_action_counts = {
        action: 0
        for action in ACTIONS
    }
    self.round_action_update_counts = {
        action: 0
        for action in ACTIONS
    }
    # Stable telemetry schema; this agent has no potential-hint mechanism.
    self.round_potential_hint_diagnostics = {}


def record_reward_result(self, reward_result: RewardResult) -> None:
    """Add one reward result to the current round's telemetry."""
    self.round_return += reward_result.total
    for event, reward in reward_result.contributions:
        component = self.round_reward_components.setdefault(
            event, {"count": 0, "reward_sum": 0.0}
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
        event: {"count": int(component["count"]),
                "reward_sum": float(component["reward_sum"])}
        for event, component in sorted(self.round_reward_components.items())
    }
    return {
        "schema_version": TRAINING_METRICS_SCHEMA_VERSION,
        "round": int(self.rounds_completed),
        "undiscounted_return": float(self.round_return),
        "epsilon_start": None,
        "epsilon_end": None,
        "temperature_start": float(self.round_temperature_start),
        "temperature_end": float(self.temperature),
        "q_updates": int(self.round_q_updates),
        "td_error_abs_sum": float(self.round_td_error_abs_sum),
        "td_error_squared_sum": float(self.round_td_error_squared_sum),
        "td_error_abs_max": float(self.round_td_error_abs_max),
        "q_table_states": int(len(self.model)),
        "new_q_table_states": int(len(self.model) - self.round_q_table_states_start),
        "updated_state_action_pairs": int(len(self.updated_state_action_pairs)),
        "new_updated_state_action_pairs": int(
            len(self.updated_state_action_pairs) - self.round_updated_pairs_start
        ),
        "action_counts": dict(self.round_action_counts),
        "action_update_counts": dict(self.round_action_update_counts),
        "potential_hint_diagnostics": dict(self.round_potential_hint_diagnostics),
        "reward_components": reward_components,
        "block_diagnostics": dict(self.round_block_diagnostics),
    }


def setup_training(self):
    """Start fresh training with the final configuration and cumulative counts."""
    self.td_target_buffer = deque()
    self.exploration_version = _environment_version(
        "BOMBERMAN_EXPLORATION_VERSION", EXPLORATION_VERSION
    )
    self.learning_rate_version = _environment_version(
        "BOMBERMAN_LEARNING_RATE_VERSION", LEARNING_RATE_VERSION
    )
    self.discount_factor_version = _environment_version(
        "BOMBERMAN_DISCOUNT_FACTOR_VERSION", DISCOUNT_FACTOR_VERSION
    )
    self.td_target_version = _environment_version(
        "BOMBERMAN_TD_TARGET_VERSION", TD_TARGET_VERSION
    )
    self.reward_version = _environment_version("BOMBERMAN_REWARD_VERSION", REWARD_VERSION)
    self.exploration_parameters = get_exploration_parameters()
    self.temperature_start = TEMPERATURE_START
    self.temperature_min = TEMPERATURE_MIN
    self.temperature_decay = TEMPERATURE_DECAY
    self.temperature = self.temperature_start
    self.rounds_completed = 0
    self.checkpoint_interval = _positive_environment_integer("BOMBERMAN_CHECKPOINT_INTERVAL", 1)
    self.training_round_limit = _positive_environment_integer("BOMBERMAN_TRAINING_ROUNDS", None)
    self.updated_state_action_pairs = set()
    self.state_action_update_counts = {}
    self.gamma = GAMMA
    self.td_target_parameters = get_td_target_parameters()
    self.td_target_n_steps = TD_TARGET_N_STEPS
    self.learning_rate_function = learning_rate
    self.reward_function = reward_for_transition
    reset_round_metrics(self)
    self.logger.info(
        f"Training with {self.reward_version}, gamma={self.gamma}, "
        f"{self.td_target_n_steps}-step updates and temperature minimum after "
        f"{TEMPERATURE_DECAY_ROUNDS} rounds."
    )


def calculate_sampled_n_step_target(
    rewards: Sequence[float],
    discount_factor: float,
    bootstrap_value: float | None,
) -> float:
    r"""Return a sampled n-step Q-learning target.

    For ``m = len(rewards)``, this evaluates

    ``sum(gamma**k * rewards[k], k=0..m-1)``

    and, for a non-terminal sequence, adds

    ``gamma**m * bootstrap_value``.

    A ``None`` bootstrap value marks a terminal sequence and therefore
    contributes no value beyond its final observed reward.
    """
    if not rewards:
        raise ValueError(
            "An n-step target requires at least one reward."
        )

    if (
        isinstance(discount_factor, bool)
        or not isinstance(discount_factor, Real)
        or not math.isfinite(float(discount_factor))
        or not 0.0 <= float(discount_factor) <= 1.0
    ):
        raise ValueError(
            "The discount factor must be a finite number in [0, 1]."
        )

    gamma = float(discount_factor)
    discounted_return = 0.0
    discount = 1.0

    for index, reward in enumerate(rewards):
        if (
            isinstance(reward, bool)
            or not isinstance(reward, Real)
            or not math.isfinite(float(reward))
        ):
            raise ValueError(
                f"Reward {index} must be a finite number."
            )

        discounted_return += discount * float(reward)
        discount *= gamma

    if bootstrap_value is not None:
        if (
            isinstance(bootstrap_value, bool)
            or not isinstance(bootstrap_value, Real)
            or not math.isfinite(float(bootstrap_value))
        ):
            raise ValueError(
                "The bootstrap value must be a finite number or None."
            )

        discounted_return += discount * float(bootstrap_value)

    return float(discounted_return)


def maximum_learning_action_q_value(self, state) -> float:
    """Return max-a Q(state, a); all six actions remain learnable."""
    if state not in self.model:
        self.model[state] = np.zeros(len(ACTIONS), dtype=float)
    return float(np.max(self.model[state][list(self.learning_action_indices)]))


def update_q_table_from_target(
    self, transition: Transition, td_target: float,
) -> float | None:
    """Apply one update and increment its cumulative canonical visit count."""
    state, action, _, _ = transition
    if state is None:
        return None
    if not np.isfinite(td_target):
        raise ValueError("The TD target must be finite.")
    if state not in self.model:
        self.model[state] = np.zeros(len(ACTIONS), dtype=float)

    action_index = ACTIONS.index(action)
    self.round_action_update_counts[action] += 1
    current_q_value = self.model[state][action_index]
    pair = state, action_index
    self.updated_state_action_pairs.add(pair)
    visit_count = self.state_action_update_counts.get(pair, 0) + 1
    self.state_action_update_counts[pair] = visit_count
    alpha = self.learning_rate_function(visit_count)
    td_error = td_target - current_q_value
    self.model[state][action_index] += alpha * td_error
    return float(td_error)


def calculate_transition_sequence_target(
    self, transitions: tuple[Transition, ...],
) -> float:
    """Calculate the sampled return for a consecutive transition window."""
    if not transitions:
        raise ValueError("A TD-target transition window cannot be empty.")
    if len(transitions) > self.td_target_n_steps:
        raise ValueError("A TD-target window cannot exceed its configured horizon.")
    if any(transition.next_state is None for transition in transitions[:-1]):
        raise ValueError("A terminal transition must be last in its target window.")
    final_next_state = transitions[-1].next_state
    bootstrap = (None if final_next_state is None
                 else maximum_learning_action_q_value(self, final_next_state))
    return calculate_sampled_n_step_target(
        rewards=tuple(transition.reward for transition in transitions),
        discount_factor=self.gamma,
        bootstrap_value=bootstrap,
    )


def update_oldest_buffered_transition(self) -> float | None:
    """Update and remove the oldest transition in the n-step buffer."""
    if not self.td_target_buffer:
        raise ValueError("Cannot update an empty TD-target buffer.")
    target = calculate_transition_sequence_target(self, tuple(self.td_target_buffer))
    oldest = self.td_target_buffer.popleft()
    return update_q_table_from_target(self, oldest, target)


def process_learning_transition(
    self, transition: Transition,
) -> tuple[float | None, ...]:
    """Learn from each full five-step window and flush all terminal suffixes."""
    self.td_target_buffer.append(transition)
    if transition.next_state is None:
        td_errors = []
        while self.td_target_buffer:
            td_errors.append(update_oldest_buffered_transition(self))
        return tuple(td_errors)
    if len(self.td_target_buffer) == self.td_target_n_steps:
        return (update_oldest_buffered_transition(self),)
    if len(self.td_target_buffer) > self.td_target_n_steps:
        raise RuntimeError("The TD-target buffer exceeded its configured horizon.")
    return ()

def game_events_occurred(
    self, old_game_state: dict, self_action: str, new_game_state: dict,
    events: list[str],
):
    """Build one canonical transition from the action and the next game state."""
    reward_result = calculate_reward(self, old_game_state, self_action, new_game_state, events)
    record_reward_result(self, reward_result)
    transition = Transition(
        state=self.last_action_features,
        action=model_action_for_last_feature_state(self, self_action),
        next_state=extract_features(
            self, new_game_state,
            previous_position=tuple(old_game_state["self"][3]), after_action=True,
        ),
        reward=reward_result.total,
    )
    for td_error in process_learning_transition(self, transition):
        record_td_error(self, td_error)


def end_of_round(
    self, last_game_state: dict, last_action: str, events: list[str],
) -> dict:
    """Flush terminal updates before recording metrics and saving the model."""
    reward_result = calculate_reward(self, last_game_state, last_action, None, events)
    record_reward_result(self, reward_result)
    final_transition = Transition(
        state=self.last_action_features,
        action=model_action_for_last_feature_state(self, last_action),
        next_state=None,
        reward=reward_result.total,
    )
    for td_error in process_learning_transition(self, final_transition):
        record_td_error(self, td_error)
    if self.td_target_buffer:
        raise RuntimeError("The TD-target buffer was not empty after terminal flush.")

    clear_feature_context(self)
    self.temperature = max(self.temperature_min, self.temperature * self.temperature_decay)
    self.rounds_completed += 1
    self.logger.info(
        f"End of round {self.rounds_completed}: temperature={self.temperature:.4f}, "
        f"known_states={len(self.model)}"
    )
    final_configured_round = (
        self.training_round_limit is not None
        and self.rounds_completed >= self.training_round_limit
    )
    if self.rounds_completed % self.checkpoint_interval == 0 or final_configured_round:
        save_checkpoint(
            file_path=self.model_file,
            q_table=self.model,
            feature_version=self.feature_version,
            action_order=tuple(ACTIONS),
            # The terminal flush counted all n-1 remaining buffered actions.
            state_action_update_counts=self.state_action_update_counts,
            training_metadata=build_training_metadata(self),
        )
    metrics = build_training_metrics(self)
    reset_round_metrics(self)
    return metrics


def build_training_metadata(self) -> dict:
    """Keep checkpoint identifiers and parameters compatible with the evaluator."""
    return {
        "reward_version": self.reward_version,
        "action_space_version": self.action_space_version,
        "agent_seed": self.agent_seed,
        "rounds_completed": self.rounds_completed,
        "alpha": self.learning_rate_function(1),
        "learning_rate_version": self.learning_rate_version,
        "learning_rate_parameters": get_learning_rate_parameters(self.learning_rate_version),
        "discount_factor_version": self.discount_factor_version,
        "gamma": self.gamma,
        "td_target_version": self.td_target_version,
        "td_target_parameters": dict(self.td_target_parameters),
        "exploration_version": self.exploration_version,
        "exploration_parameters": dict(self.exploration_parameters),
        "epsilon_start": None,
        "epsilon_min": None,
        "epsilon_decay": None,
        "epsilon_current": None,
        "temperature_start": self.temperature_start,
        "temperature_min": self.temperature_min,
        "temperature_decay": self.temperature_decay,
        "temperature_current": self.temperature,
    }


def calculate_reward(
    self, old_game_state: dict | None, action: str | None,
    new_game_state: dict | None, events: list[str],
) -> RewardResult:
    """Evaluate the chosen canonical action and retain V5 event contributions."""
    result = self.reward_function(
        old_game_state, model_action_for_last_feature_state(self, action),
        new_game_state, list(events), None, self.gamma,
        getattr(self, "last_action_features", None),
    )
    self.logger.debug(
        f"Awarded {result.total} for events "
        f"{', '.join(event for event, _ in result.contributions)}"
    )
    return result
