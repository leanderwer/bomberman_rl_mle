"""Framework callbacks for the frozen V12/V5 tabular agent.

Features describe safe and useful actions; they never restrict the action
space. The learned Q-table chooses among all six actions, including at HOLD.
"""

from collections import OrderedDict
from pathlib import Path
import os
import random

import numpy as np

from .features import (
    ACTIONS as ACTION_ORDER,
    FEATURE_VERSION,
    IDENTITY_FEATURE_ACTION_TRANSFORM,
    FEATURE_ACTION_TRANSFORM_CONTEXT_KEY,
    THREAT_URGENCY_DIAGNOSTIC_CONTEXT_KEY,
    TARGET_CLASS_DIAGNOSTIC_CONTEXT_KEY,
    COIN_ROUTE_BOMB_DIAGNOSTIC_CONTEXT_KEY,
    EVALUATION_DIAGNOSTIC_CONTEXT_KEY,
    state_to_features,
)
from .diagnostics import action_statistics, block_action_statistics
from .model_io import load_checkpoint


ACTIONS = list(ACTION_ORDER)
ACTION_SPACE_VERSION = "task2_all_actions_v1"
FEATURE_CACHE_SIZE = 3
_CONTEXT_KEYS = (
    FEATURE_ACTION_TRANSFORM_CONTEXT_KEY,
    THREAT_URGENCY_DIAGNOSTIC_CONTEXT_KEY,
    TARGET_CLASS_DIAGNOSTIC_CONTEXT_KEY,
    COIN_ROUTE_BOMB_DIAGNOSTIC_CONTEXT_KEY,
)


def clear_feature_context(self):
    """Reset caches and per-action observations at a round boundary."""
    self.feature_cache = OrderedDict()
    self.feature_cache_round = None
    self.last_action_features = None
    self.latest_feature_action_transform = IDENTITY_FEATURE_ACTION_TRANSFORM
    self.last_action_feature_action_transform = IDENTITY_FEATURE_ACTION_TRANSFORM
    self.latest_threat_urgency_diagnostic = None
    self.last_action_threat_urgency_diagnostic = None
    self.latest_target_class_diagnostic = None
    self.last_action_target_class_diagnostic = None
    self.latest_coin_route_bomb_diagnostic = None
    self.last_action_coin_route_bomb_diagnostic = None


def _freeze(value):
    """Preserve array shape, dtype and values in the visible-state cache key."""
    if isinstance(value, np.ndarray):
        return ("numpy", value.shape, value.dtype.str, value.tobytes())
    if isinstance(value, dict):
        return tuple((key, _freeze(item)) for key, item in sorted(value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def extract_features(self, game_state, previous_position=None, *, after_action=False):
    """Reuse identical observations, including their diagnostic context.

    V12 depends only on the current board. The two optional arguments retain
    the training callback interface; neither adds history to the Q-state.
    Round and step labels are excluded so a transition's next state can be
    reused by the following action callback.
    """
    if game_state is None:
        return None
    if "round" not in game_state:
        return self.feature_extractor(game_state)
    if not hasattr(self, "feature_cache"):
        clear_feature_context(self)
    if self.feature_cache_round != game_state["round"]:
        clear_feature_context(self)
        self.feature_cache_round = game_state["round"]

    excluded = {"round", "step", *_CONTEXT_KEYS, EVALUATION_DIAGNOSTIC_CONTEXT_KEY}
    cache_key = tuple(
        (key, _freeze(value))
        for key, value in sorted(game_state.items())
        if key not in excluded
    )
    cached = self.feature_cache.pop(cache_key, None)
    if cached is None:
        contextual_state = dict(game_state)
        contextual_state.update(dict.fromkeys(_CONTEXT_KEYS))
        if getattr(self, "evaluation_diagnostics_enabled", False):
            contextual_state[EVALUATION_DIAGNOSTIC_CONTEXT_KEY] = {}
        state = self.feature_extractor(contextual_state)
        transform = contextual_state[FEATURE_ACTION_TRANSFORM_CONTEXT_KEY]
        cached = (
            state,
            transform if transform is not None else IDENTITY_FEATURE_ACTION_TRANSFORM,
            contextual_state[THREAT_URGENCY_DIAGNOSTIC_CONTEXT_KEY],
            contextual_state[TARGET_CLASS_DIAGNOSTIC_CONTEXT_KEY],
            contextual_state[COIN_ROUTE_BOMB_DIAGNOSTIC_CONTEXT_KEY],
        )
    self.feature_cache[cache_key] = cached
    while len(self.feature_cache) > FEATURE_CACHE_SIZE:
        self.feature_cache.popitem(last=False)
    (
        state,
        self.latest_feature_action_transform,
        self.latest_threat_urgency_diagnostic,
        self.latest_target_class_diagnostic,
        self.latest_coin_route_bomb_diagnostic,
    ) = cached
    return state


def setup(self):
    """Load the bundled checkpoint, or initialise a new training run."""
    for key, expected in (
        ("BOMBERMAN_FEATURE_VERSION", FEATURE_VERSION),
        ("BOMBERMAN_ACTION_SPACE_VERSION", ACTION_SPACE_VERSION),
    ):
        configured = os.environ.get(key)
        if configured is not None and configured != expected:
            raise ValueError(f"{key} must be {expected!r} for the final agent.")

    # Manual training writes a different file so the bundled policy remains available. Controlled experiments supply their own output path.
    default_name = "trained-model.pt" if self.train else "my-saved-model.pt"
    self.model_file = os.environ.get(
        "BOMBERMAN_MODEL_FILE", str(Path(__file__).resolve().with_name(default_name))
    )
    self.agent_seed = int(os.environ.get("BOMBERMAN_AGENT_SEED", "0"))
    self.random_generator = random.Random(self.agent_seed)
    self.feature_version = FEATURE_VERSION
    self.feature_extractor = state_to_features
    self.action_space_version = ACTION_SPACE_VERSION
    self.learning_actions = tuple(ACTIONS)
    self.learning_action_indices = tuple(range(len(ACTIONS)))
    self.evaluation_diagnostics_enabled = (
        not self.train and os.environ.get("BOMBERMAN_TASK3_DIAGNOSTICS") == "1"
    )

    if self.train:
        self.model = {}
        self.checkpoint_metadata = {}
    else:
        self.model, self.checkpoint_metadata, _, _ = load_checkpoint(
            self.model_file, tuple(ACTIONS), expected_feature_version=FEATURE_VERSION
        )
        if self.checkpoint_metadata.get("action_space_version") != ACTION_SPACE_VERSION:
            raise ValueError("The final checkpoint must use all six learning actions.")
    clear_feature_context(self)
    self.logger.info(f"Using V12 policy from '{self.model_file}', agent seed {self.agent_seed}.")


def _softmax_probabilities(q_values, temperature):
    """Return finite probabilities for the softmax policy."""
    with np.errstate(over="ignore", under="ignore"):
        weights = np.exp((q_values - np.max(q_values)) / temperature)
    return weights / np.sum(weights)


def _choose_model_action(self, state):
    """Preserve the original softmax draw and approximate-Q tie breaking."""
    if state not in self.model:
        self.model[state] = np.zeros(len(ACTIONS), dtype=float)
    q_values = self.model[state][list(self.learning_action_indices)]
    if self.train:
        probabilities = _softmax_probabilities(q_values, self.temperature)
        index = self.random_generator.choices(
            range(len(self.learning_action_indices)), weights=probabilities.tolist(), k=1,
        )[0]
        return self.learning_actions[index]
    best = np.flatnonzero(np.isclose(q_values, np.max(q_values)))
    index = self.random_generator.choice(best.tolist())
    return self.learning_actions[index]


def act(self, game_state: dict) -> str:
    """Choose a learned action and publish optional evaluation observations."""
    self.last_action_statistics = None
    self.last_evaluation_diagnostic = None
    state = extract_features(self, game_state)
    self.last_action_features = state
    self.last_action_feature_action_transform = self.latest_feature_action_transform
    self.last_action_threat_urgency_diagnostic = self.latest_threat_urgency_diagnostic
    self.last_action_target_class_diagnostic = self.latest_target_class_diagnostic
    self.last_action_coin_route_bomb_diagnostic = self.latest_coin_route_bomb_diagnostic

    model_action = _choose_model_action(self, state)
    board_action = self.last_action_feature_action_transform.to_board_action(model_action)
    context = self.last_action_coin_route_bomb_diagnostic

    if self.train:
        if hasattr(self, "round_action_counts"):
            self.round_action_counts[model_action] += 1
        if hasattr(self, "round_block_diagnostics"):
            for name, count in block_action_statistics(state, model_action, context).items():
                self.round_block_diagnostics[name] = self.round_block_diagnostics.get(name, 0) + count
    else:
        self.last_action_statistics = action_statistics(
            state, model_action,
            threat_urgency_override=self.last_action_threat_urgency_diagnostic,
            target_class_override=self.last_action_target_class_diagnostic,
            coin_route_bomb_override=context,
        )
        evaluation = getattr(context, "evaluation", None)
        if evaluation is not None:
            self.last_evaluation_diagnostic = {
                **evaluation,
                "schema_version": 1,
                "feature_version": self.feature_version,
                "round": int(game_state["round"]),
                "step": int(game_state["step"]),
            }
    return board_action


def model_action_for_last_feature_state(self, board_action):
    """Express the executed action in the orientation used by the last Q-key."""
    transform = getattr(
        self, "last_action_feature_action_transform", IDENTITY_FEATURE_ACTION_TRANSFORM
    )
    return transform.to_canonical_action(board_action)
