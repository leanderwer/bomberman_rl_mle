"""Resolve an experiment's versioned components from its selected agent."""

from dataclasses import dataclass
import importlib
from pathlib import Path
import re
import sys
from typing import Callable


@dataclass(frozen=True)
class AgentRegistry:
    """Small common interface for historical and standalone agent layouts."""

    defaults: dict[str, str]
    get_feature_extractor: Callable
    get_reward_function: Callable
    get_learning_actions: Callable
    get_learning_rate_parameters: Callable
    get_exploration_parameters: Callable
    get_discount_factor: Callable
    get_td_target_parameters: Callable
    load_checkpoint: Callable

    @property
    def default_td_target_version(self) -> str:
        return self.defaults["td_target_version"]


def _fixed_component(kind, expected, value):
    def resolve(version):
        if version != expected:
            raise ValueError(
                f"The final agent requires {kind} {expected!r}, got {version!r}."
            )
        return value
    return resolve


def get_agent_registry(
    agent_name: str,
    repository_root: Path | None = None,
) -> AgentRegistry:
    """Load the selected agent's registry, never silently use Q-learning's."""
    if (not isinstance(agent_name, str)
            or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", agent_name) is None):
        raise ValueError(f"Invalid agent module name: {agent_name!r}.")
    root = Path(repository_root or Path(__file__).resolve().parents[1]).resolve()
    agent_directory = root / "agent_code" / agent_name
    for filename in ("callbacks.py", "train.py"):
        if not (agent_directory / filename).is_file():
            raise ValueError(f"Agent {agent_name!r} is missing {filename!r}: {agent_directory}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    def module(name):
        module_name = f"agent_code.{agent_name}.{name}"
        try:
            return importlib.import_module(module_name)
        except ModuleNotFoundError as error:
            if error.name != module_name:
                raise
            raise ValueError(
                f"Agent {agent_name!r} does not expose versioned {name!r}."
            ) from error

    features, rewards, model_io = module("features"), module("rewards"), module("model_io")
    if agent_name == "final_agent":
        callbacks, train = module("callbacks"), module("train")
        defaults = {
            "feature_version": features.FEATURE_VERSION,
            "reward_version": rewards.REWARD_VERSION,
            "action_space_version": callbacks.ACTION_SPACE_VERSION,
            "learning_rate_version": train.LEARNING_RATE_VERSION,
            "exploration_version": train.EXPLORATION_VERSION,
            "discount_factor_version": train.DISCOUNT_FACTOR_VERSION,
            "td_target_version": train.TD_TARGET_VERSION,
        }
        return AgentRegistry(
            defaults,
            _fixed_component("feature version", features.FEATURE_VERSION, features.state_to_features),
            _fixed_component("reward version", rewards.REWARD_VERSION, rewards.calculate_reward),
            _fixed_component("action-space version", callbacks.ACTION_SPACE_VERSION, tuple(callbacks.ACTIONS)),
            train.get_learning_rate_parameters,
            train.get_exploration_parameters,
            _fixed_component("discount-factor version", train.DISCOUNT_FACTOR_VERSION, train.GAMMA),
            train.get_td_target_parameters,
            model_io.load_checkpoint,
        )

    actions = module("action_spaces")
    learning_rates = module("learning_rates")
    exploration = module("exploration_schedules")
    discount = module("discount_factors")
    targets = module("td_targets")
    return AgentRegistry(
        {
            "feature_version": features.DEFAULT_FEATURE_VERSION,
            "reward_version": rewards.DEFAULT_REWARD_VERSION,
            "action_space_version": actions.DEFAULT_ACTION_SPACE_VERSION,
            "learning_rate_version": learning_rates.DEFAULT_LEARNING_RATE_VERSION,
            "exploration_version": exploration.DEFAULT_EXPLORATION_VERSION,
            "discount_factor_version": discount.DEFAULT_DISCOUNT_FACTOR_VERSION,
            "td_target_version": targets.DEFAULT_TD_TARGET_VERSION,
        },
        features.get_feature_extractor,
        rewards.get_reward_function,
        actions.get_learning_actions,
        learning_rates.get_learning_rate_parameters,
        exploration.get_exploration_parameters,
        discount.get_discount_factor,
        targets.get_td_target_parameters,
        model_io.load_checkpoint,
    )


def validate_agent_versions(
    configuration: dict,
    repository_root: Path | None = None,
    *,
    fill_defaults: bool = False,
) -> AgentRegistry:
    """Validate explicit settings and optionally record the agent's defaults."""
    registry = get_agent_registry(configuration["agent"], repository_root)
    resolvers = {
        "feature_version": registry.get_feature_extractor,
        "reward_version": registry.get_reward_function,
        "action_space_version": registry.get_learning_actions,
        "learning_rate_version": registry.get_learning_rate_parameters,
        "exploration_version": registry.get_exploration_parameters,
        "discount_factor_version": registry.get_discount_factor,
        "td_target_version": registry.get_td_target_parameters,
    }
    for key, resolve in resolvers.items():
        if fill_defaults:
            configuration.setdefault(key, registry.defaults[key])
        if key in configuration:
            version = configuration[key]
            if not isinstance(version, str) or not version.strip():
                raise ValueError(f"'{key}' must be a non-empty string.")
            resolve(version)
    return registry
