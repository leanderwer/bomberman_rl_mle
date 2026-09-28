"""Run controlled multi-seed experiments for Task 2."""

from argparse import ArgumentParser
import json
import math
from pathlib import Path
import re
import os
import subprocess
import sys

if not __package__:  # Support direct script invocation from the repository.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.agent_registry import get_agent_registry, validate_agent_versions
from evaluation.checkpoint_evaluations import (
    analyze_and_plot, checkpoint_file, clean_environment, experiment_mode,
    model_seed_runs, print_checkpoint_plan, run_checkpoint_evaluations, sha256,
    validate_checkpoint_plan,
)

ACTION_ORDER = (
    "UP",
    "RIGHT",
    "DOWN",
    "LEFT",
    "WAIT",
    "BOMB",
)

TASK_2_ACTION_SPACE_VERSION = "task2_all_actions_v1"
DEFAULT_TRAINING_CHECKPOINT_INTERVAL = 100


def parse_arguments():
    """Read the experiment configuration path from the command line."""
    parser = ArgumentParser(
        description="Run a controlled Task 2 multi-seed experiment."
    )

    parser.add_argument(
        "config",
        type=Path,
        help="Path to the experiment JSON configuration.",
    )

    parser.add_argument(
        "--validate-only", "--dry-run",
        action="store_true",
        help="Validate and display the experiment without running it.",
    )

    return parser.parse_args()


def load_configuration(file_path: Path) -> dict:
    """Load one experiment configuration."""
    try:
        with file_path.open("r", encoding="utf-8") as file:
            configuration = json.load(file)
    except FileNotFoundError as error:
        raise ValueError(
            f"Configuration file does not exist: {file_path}"
        ) from error
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Configuration file is not valid JSON: {error}"
        ) from error

    if not isinstance(configuration, dict):
        raise ValueError("The top-level JSON value must be an object.")

    return configuration


def require_section(configuration: dict, name: str) -> dict:
    """Return a required configuration section."""
    section = configuration.get(name)

    if not isinstance(section, dict):
        raise ValueError(f"'{name}' must be a JSON object.")

    return section


def require_non_empty_string(configuration: dict, name: str) -> str:
    """Return a required non-empty string."""
    value = configuration.get(name)

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"'{name}' must be a non-empty string.")

    return value


def require_positive_integer(configuration: dict, name: str) -> int:
    """Return a required positive integer."""
    value = configuration.get(name)

    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
    ):
        raise ValueError(f"'{name}' must be a positive integer.")

    return value


def require_seed_runs(
    configuration: dict,
    section_name: str,
) -> tuple[tuple[int, int], ...]:
    """Return validated pairs of world and agent seeds."""
    runs = configuration.get("runs")

    if not isinstance(runs, list) or not runs:
        raise ValueError(
            f"'{section_name}.runs' must be a non-empty list."
        )

    seed_pairs = []

    for index, seed_run in enumerate(runs):
        location = f"{section_name}.runs[{index}]"

        if not isinstance(seed_run, dict):
            raise ValueError(
                f"'{location}' must be a JSON object."
            )

        values = {}

        for seed_name in ("world_seed", "agent_seed"):
            seed = seed_run.get(seed_name)

            if (
                isinstance(seed, bool)
                or not isinstance(seed, int)
                or seed < 0
            ):
                raise ValueError(
                    f"'{location}.{seed_name}' must be a "
                    "non-negative integer."
                )

            values[seed_name] = seed

        seed_pair = (
            values["world_seed"],
            values["agent_seed"],
        )

        if seed_pair in seed_pairs:
            raise ValueError(
                f"Duplicate seed pair in '{section_name}.runs': "
                f"{seed_pair}"
            )

        seed_pairs.append(seed_pair)

    return tuple(seed_pairs)

def validate_registered_configuration(
    configuration: dict,
    repository_root: Path,
) -> None:
    """Resolve every named component before an experiment can start."""
    repository_root_string = str(repository_root)

    if repository_root_string not in sys.path:
        sys.path.insert(0, repository_root_string)

    import settings

    agent_name = configuration["agent"]
    agent_directory = (
        repository_root
        / "agent_code"
        / agent_name
    )

    if not agent_directory.is_dir():
        raise ValueError(
            f"Unknown agent {agent_name!r}: "
            f"{agent_directory} does not exist."
        )

    for required_file in ("callbacks.py", "train.py"):
        if not (agent_directory / required_file).is_file():
            raise ValueError(
                f"Agent {agent_name!r} is missing "
                f"{required_file!r}."
            )

    scenario = configuration["scenario"]

    if scenario not in settings.SCENARIOS:
        raise ValueError(
            f"Unknown scenario {scenario!r}. Available scenarios: "
            f"{', '.join(sorted(settings.SCENARIOS))}."
        )

    expected_coin_count = settings.SCENARIOS[scenario][
        "COIN_COUNT"
    ]
    configured_coin_count = configuration["coins_per_round"]

    if configured_coin_count != expected_coin_count:
        raise ValueError(
            "'coins_per_round' does not match the selected scenario: "
            f"configured {configured_coin_count}, but {scenario!r} "
            f"contains {expected_coin_count}."
        )

    registry = validate_agent_versions(configuration, repository_root)
    configuration.setdefault("td_target_version", registry.default_td_target_version)
    registry.get_td_target_parameters(configuration["td_target_version"])


def validate_configuration(
    configuration: dict,
    repository_root: Path | None = None,
) -> None:
    """Validate the parts required for a controlled experiment."""
    if repository_root is None:
        repository_root = Path(__file__).resolve().parents[2]

    experiment_name = require_non_empty_string(
        configuration, "experiment_name"
    )

    if re.fullmatch(r"[A-Za-z0-9._-]+", experiment_name) is None:
        raise ValueError(
            "'experiment_name' may only contain letters, numbers, "
            "dots, underscores, and hyphens."
        )

    require_non_empty_string(configuration, "agent")
    require_non_empty_string(configuration, "scenario")
    require_non_empty_string(configuration, "feature_version")
    require_non_empty_string(configuration, "reward_version")
    action_space_version = require_non_empty_string(
        configuration,
        "action_space_version",
    )

    if action_space_version != TASK_2_ACTION_SPACE_VERSION:
        raise ValueError(
            "Task 2 experiments must use action-space version "
            f"{TASK_2_ACTION_SPACE_VERSION!r}."
        )

    require_positive_integer(configuration, "coins_per_round")
    require_non_empty_string(
        configuration,
        "learning_rate_version",
    )
    require_non_empty_string(
        configuration,
        "exploration_version",
    )
    require_non_empty_string(
        configuration,
        "discount_factor_version",
    )
    if "td_target_version" in configuration:
        require_non_empty_string(
            configuration,
            "td_target_version",
        )

    validate_registered_configuration(
        configuration,
        repository_root,
    )

    description = configuration.get("description", "")
    if not isinstance(description, str):
        raise ValueError("'description' must be a string.")

    mode = experiment_mode(configuration)
    evaluation = require_section(configuration, "evaluation")
    if mode == "train_and_evaluate":
        if "checkpoint" in configuration:
            raise ValueError("checkpoint requires evaluate_checkpoint mode.")
        training = require_section(configuration, "training")
        require_positive_integer(training, "rounds")
        if "checkpoint_interval" in training:
            require_positive_integer(training, "checkpoint_interval")
        training_seed_pairs = require_seed_runs(training, "training")
    else:
        if "training" in configuration:
            raise ValueError("training must be omitted in evaluate_checkpoint mode.")
        checkpoint_file(configuration, repository_root)
        training_seed_pairs = require_seed_runs({"runs": model_seed_runs(configuration)}, "checkpoint")
    validate_checkpoint_plan(configuration)
    require_non_empty_string(evaluation, "seed_set")
    require_positive_integer(evaluation, "rounds_per_seed")
    evaluation_seed_pairs = require_seed_runs(
    evaluation,
    "evaluation",
)
    training_world_seeds = {
        world_seed
        for world_seed, _ in training_seed_pairs
    }
    evaluation_world_seeds = {
        world_seed
        for world_seed, _ in evaluation_seed_pairs
    }

    overlapping_world_seeds = (
        training_world_seeds & evaluation_world_seeds
    )

    if overlapping_world_seeds:
        raise ValueError(
            "Training and evaluation world seeds must be disjoint. "
            f"Overlap: {sorted(overlapping_world_seeds)}"
        )


def print_experiment_plan(configuration: dict) -> None:
    """Print the work that the runner will perform later."""
    if experiment_mode(configuration) == "evaluate_checkpoint":
        print(f"Experiment: {configuration['experiment_name']}")
        print(f"Frozen checkpoint: {configuration['checkpoint']['path']}")
        print(f"Training rounds: {configuration['checkpoint']['training_rounds']}")
        print(f"Evaluation: {len(configuration['evaluation']['runs'])} seeds x "
              f"{configuration['evaluation']['rounds_per_seed']} rounds")
        return
    training = configuration["training"]
    evaluation = configuration["evaluation"]

    number_of_models = len(training["runs"])
    evaluations_per_model = len(evaluation["runs"])

    total_training_rounds = number_of_models * training["rounds"]
    total_evaluation_runs = number_of_models * evaluations_per_model
    total_evaluation_rounds = (
        total_evaluation_runs * evaluation["rounds_per_seed"]
    )
    checkpoint_interval = training.get(
        "checkpoint_interval",
        DEFAULT_TRAINING_CHECKPOINT_INTERVAL,
    )

    print("Configuration valid.")
    print(f"Experiment: {configuration['experiment_name']}")
    print(
        f"Agent and scenario: "
        f"{configuration['agent']} / {configuration['scenario']}"
    )
    print(f"Feature version: {configuration['feature_version']}")
    print(f"Reward version: {configuration['reward_version']}")
    print(
        "Action-space version: "
        f"{configuration['action_space_version']}"
    )
    print(
        "Learning-rate version: "
        f"{configuration.get('learning_rate_version', 'agent default')}"
    )
    print(
        "Exploration version: "
        f"{configuration['exploration_version']}"
    )
    print(
        "Discount-factor version: "
        f"{configuration['discount_factor_version']}"
    )
    print(
        "TD-target version: "
        f"{configuration['td_target_version']}"
    )

    print("Training seed pairs:")
    for seed_run in training["runs"]:
        print(
            f"  world={seed_run['world_seed']}, "
            f"agent={seed_run['agent_seed']}"
        )

    print("Evaluation seed pairs:")
    for seed_run in evaluation["runs"]:
        print(
            f"  world={seed_run['world_seed']}, "
            f"agent={seed_run['agent_seed']}"
        )

    print(f"Training models: {number_of_models}")
    print(f"Total training rounds: {total_training_rounds}")
    print(f"Checkpoint interval: {checkpoint_interval}")
    print_checkpoint_plan(configuration)
    print(f"Evaluations per model: {evaluations_per_model}")
    print(f"Total evaluation runs: {total_evaluation_runs}")
    print(f"Total evaluation rounds: {total_evaluation_rounds}")

def training_run_id(seed_run: dict) -> str:
    """Return the stable identifier of one training seed pair."""
    return (
        f"train_w{seed_run['world_seed']}_"
        f"a{seed_run['agent_seed']}"
    )


def evaluation_run_id(seed_run: dict) -> str:
    """Return the stable identifier of one evaluation seed pair."""
    return (
        f"eval_w{seed_run['world_seed']}_"
        f"a{seed_run['agent_seed']}"
    )

def get_clean_git_commit(repository_root: Path) -> str:
    """Return the current commit and require a clean repository."""
    try:
        status_result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repository_root,
            capture_output=True,
            text=True,
            check=True,
        )

        commit_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            "Could not determine the Git state."
        ) from error

    if status_result.stdout.strip():
        raise RuntimeError(
            "The repository contains uncommitted changes. "
            "Commit them before running a controlled experiment."
        )

    return commit_result.stdout.strip()

def build_experiment_index(
    configuration: dict,
    git_commit: str,
    repository_root: Path | None = None,
) -> dict:
    """Build the machine-readable index of experiment artifacts."""
    repository_root = repository_root or Path(__file__).resolve().parents[2]
    runs = []

    for training_run in model_seed_runs(configuration):
        training_identifier = training_run_id(training_run)
        evaluations = []

        for evaluation_run in configuration["evaluation"]["runs"]:
            evaluation_identifier = evaluation_run_id(
                evaluation_run
            )

            evaluations.append(
                {
                    "evaluation_world_seed": (
                        evaluation_run["world_seed"]
                    ),
                    "evaluation_agent_seed": (
                        evaluation_run["agent_seed"]
                    ),
                    "results": (
                        f"evaluation/{training_identifier}/"
                        f"{evaluation_identifier}.json"
                    ),
                }
            )

        runs.append(
            {
                "training_world_seed": training_run["world_seed"],
                "training_agent_seed": training_run["agent_seed"],
                "model": f"models/{training_identifier}.pt",
                "training_results": (
                    f"training/{training_identifier}.json"
                ),
                "evaluations": evaluations,
            }
        )

    if experiment_mode(configuration) == "evaluate_checkpoint":
        source = checkpoint_file(configuration, repository_root)
        runs[0].update(model=configuration["checkpoint"]["path"],
                       model_source="external_checkpoint", model_sha256=sha256(source),
                       training_results=None)
    return {
        "status": "running",
        "git_commit": git_commit,
        "experiment_mode": experiment_mode(configuration),
        "configuration": configuration,
        "runs": runs,
    }


def write_experiment_index(
    experiment_directory: Path,
    experiment_index: dict,
) -> None:
    """Write the experiment index atomically."""
    index_file = experiment_directory / "experiment.json"
    temporary_file = experiment_directory / "experiment.json.tmp"

    with temporary_file.open("w", encoding="utf-8") as file:
        json.dump(
            experiment_index,
            file,
            indent=4,
            sort_keys=True,
        )

    temporary_file.replace(index_file)

def prepare_experiment_directory(
    repository_root: Path,
    experiment_name: str,
) -> Path:
    """Create a new directory for one complete experiment."""
    experiment_directory = (
        repository_root
        / "results"
        / "task2"
        / "experiments"
        / experiment_name
    )

    if experiment_directory.exists():
        raise ValueError(
            f"Experiment directory already exists: "
            f"{experiment_directory}\n"
            "Choose a new experiment name to avoid overwriting results."
        )

    (experiment_directory / "models").mkdir(parents=True)
    (experiment_directory / "training").mkdir()
    (experiment_directory / "evaluation").mkdir()
    (experiment_directory / "logs").mkdir()

    return experiment_directory

def verify_training_checkpoint(
    model_file: Path,
    configuration: dict,
    training_agent_seed: int,
    repository_root: Path,
) -> None:
    """Verify that a trained checkpoint matches its experiment config."""
    repository_root_string = str(repository_root)

    if repository_root_string not in sys.path:
        sys.path.insert(0, repository_root_string)

    registry = get_agent_registry(configuration["agent"], repository_root)
    load_checkpoint = registry.load_checkpoint

    q_table, metadata, feature_version, is_legacy = load_checkpoint(
        file_path=str(model_file),
        expected_action_order=ACTION_ORDER,
        expected_feature_version=configuration["feature_version"],
    )

    if is_legacy:
        raise RuntimeError(
            f"Training produced an unexpected legacy model: {model_file}"
        )

    reward_version = metadata.get("reward_version")
    if reward_version != configuration["reward_version"]:
        raise RuntimeError(
            "Reward-version mismatch:\n"
            f"  configured: {configuration['reward_version']}\n"
            f"  checkpoint: {reward_version}"
        )

    action_space_version = metadata.get("action_space_version")
    if action_space_version != configuration["action_space_version"]:
        raise RuntimeError(
            "Action-space-version mismatch:\n"
            f"  configured: {configuration['action_space_version']}\n"
            f"  checkpoint: {action_space_version}"
        )

    configured_learning_rate_version = configuration.get(
        "learning_rate_version"
    )
    saved_learning_rate_version = metadata.get(
        "learning_rate_version"
    )

    if (
        configured_learning_rate_version is not None
        and saved_learning_rate_version
        != configured_learning_rate_version
    ):
        raise RuntimeError(
            "Learning-rate-version mismatch:\n"
            f"  configured: {configured_learning_rate_version}\n"
            f"  checkpoint: {saved_learning_rate_version}"
        )

    configured_exploration_version = configuration.get(
        "exploration_version"
    )
    saved_exploration_version = metadata.get(
        "exploration_version"
    )

    if (
        configured_exploration_version is not None
        and saved_exploration_version
        != configured_exploration_version
    ):
        raise RuntimeError(
            "Exploration-version mismatch:\n"
            f"  configured: {configured_exploration_version}\n"
            f"  checkpoint: {saved_exploration_version}"
        )

    configured_discount_factor_version = configuration[
        "discount_factor_version"
    ]
    saved_discount_factor_version = metadata.get(
        "discount_factor_version"
    )

    if (
        saved_discount_factor_version
        != configured_discount_factor_version
    ):
        raise RuntimeError(
            "Discount-factor-version mismatch:\n"
            f"  configured: {configured_discount_factor_version}\n"
            f"  checkpoint: {saved_discount_factor_version}"
        )

    configured_td_target_version = configuration[
        "td_target_version"
    ]
    saved_td_target_version = metadata.get(
        "td_target_version"
    )

    if saved_td_target_version != configured_td_target_version:
        raise RuntimeError(
            "TD-target-version mismatch:\n"
            f"  configured: {configured_td_target_version}\n"
            f"  checkpoint: {saved_td_target_version}"
        )

    get_td_target_parameters = registry.get_td_target_parameters

    configured_td_target_parameters = get_td_target_parameters(
        configured_td_target_version
    )
    saved_td_target_parameters = metadata.get(
        "td_target_parameters"
    )

    if saved_td_target_parameters != configured_td_target_parameters:
        raise RuntimeError(
            "TD-target-parameter mismatch:\n"
            f"  configured: {configured_td_target_parameters}\n"
            f"  checkpoint: {saved_td_target_parameters}"
        )

    get_discount_factor = registry.get_discount_factor

    configured_gamma = get_discount_factor(
        configured_discount_factor_version
    )
    saved_gamma = metadata.get("gamma")

    if (
        isinstance(saved_gamma, bool)
        or not isinstance(saved_gamma, (int, float))
        or not math.isclose(
            float(saved_gamma),
            configured_gamma,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    ):
        raise RuntimeError(
            "Discount-factor value mismatch:\n"
            f"  configured: {configured_gamma}\n"
            f"  checkpoint: {saved_gamma}"
        )

    saved_agent_seed = metadata.get("agent_seed")
    if saved_agent_seed != training_agent_seed:
        raise RuntimeError(
            "Agent-seed mismatch:\n"
            f"  configured: {training_agent_seed}\n"
            f"  checkpoint: {saved_agent_seed}"
        )

    configured_rounds = (configuration["training"]["rounds"]
                         if experiment_mode(configuration) == "train_and_evaluate"
                         else configuration["checkpoint"]["training_rounds"])
    completed_rounds = metadata.get("rounds_completed")

    if completed_rounds != configured_rounds:
        raise RuntimeError(
            "Training-round mismatch:\n"
            f"  configured: {configured_rounds}\n"
            f"  checkpoint: {completed_rounds}"
        )

    print(
        "Checkpoint verified: "
        f"{len(q_table)} states, "
        f"feature={feature_version}, "
        f"reward={reward_version}, "
        f"seed={saved_agent_seed}, "
        f"rounds={completed_rounds}"
    )

def run_training_runs(
    configuration: dict,
    repository_root: Path,
    experiment_directory: Path,
) -> None:
    """Train one independent model for every training seed pair."""
    training = configuration["training"]
    training_runs = training["runs"]

    for run_number, training_run in enumerate(
        training_runs,
        start=1,
    ):
        world_seed = training_run["world_seed"]
        agent_seed = training_run["agent_seed"]
        run_identifier = training_run_id(training_run)

        model_file = (
            experiment_directory
            / "models"
            / f"{run_identifier}.pt"
        )
        statistics_file = (
            experiment_directory
            / "training"
            / f"{run_identifier}.json"
        )
        log_directory = (
            experiment_directory
            / "logs"
            / run_identifier
        )
        log_directory.mkdir()

        environment = (clean_environment() if configuration["experiment_name"].startswith("Report_")
                           or "checkpoints" in configuration["evaluation"] else os.environ.copy())
        environment["BOMBERMAN_AGENT_SEED"] = str(agent_seed)
        environment["BOMBERMAN_MODEL_FILE"] = str(model_file)
        environment["BOMBERMAN_FEATURE_VERSION"] = configuration[
            "feature_version"
        ]
        environment["BOMBERMAN_REWARD_VERSION"] = configuration[
            "reward_version"
        ]
        environment["BOMBERMAN_ACTION_SPACE_VERSION"] = configuration[
            "action_space_version"
        ]

        environment["BOMBERMAN_LEARNING_RATE_VERSION"] = (
            configuration["learning_rate_version"]
        )
        environment["BOMBERMAN_EXPLORATION_VERSION"] = (
            configuration["exploration_version"]
        )
        environment["BOMBERMAN_DISCOUNT_FACTOR_VERSION"] = (
            configuration["discount_factor_version"]
        )
        environment["BOMBERMAN_TD_TARGET_VERSION"] = configuration[
            "td_target_version"
        ]
        environment["BOMBERMAN_CHECKPOINT_INTERVAL"] = str(
            training.get(
                "checkpoint_interval",
                DEFAULT_TRAINING_CHECKPOINT_INTERVAL,
            )
        )
        environment["BOMBERMAN_TRAINING_ROUNDS"] = str(
            training["rounds"]
        )
        environment["BOMBERMAN_SNAPSHOT_ROUNDS"] = json.dumps(training.get("snapshot_rounds", []))
        environment["BOMBERMAN_COMPACT_TRAINING_STATS"] = "1"
        environment["BOMBERMAN_GAME_LOG_LEVEL"] = "WARNING"
        environment["BOMBERMAN_AGENT_WRAPPER_LOG_LEVEL"] = "WARNING"
        environment["BOMBERMAN_AGENT_CODE_LOG_LEVEL"] = "WARNING"

        command = [
            sys.executable,
            str(repository_root / "main.py"),
            "play",
            "--agents",
            configuration["agent"],
            "--train",
            "1",
            "--scenario",
            configuration["scenario"],
            "--n-rounds",
            str(training["rounds"]),
            "--seed",
            str(world_seed),
            "--no-gui",
            "--save-stats",
            str(statistics_file),
            "--log-dir",
            str(log_directory),
        ]

        print()
        print(
            f"Training run {run_number}/{len(training_runs)}: "
            f"world seed {world_seed}, agent seed {agent_seed}"
        )
        print(f"Model: {model_file}")

        try:
            subprocess.run(
                command,
                cwd=repository_root,
                env=environment,
                check=True,
            )
        except subprocess.CalledProcessError as error:
            raise RuntimeError(
                "Training failed for "
                f"world seed {world_seed} and "
                f"agent seed {agent_seed}."
            ) from error

        if not model_file.is_file():
            raise RuntimeError(
                f"Training finished without creating model: {model_file}"
            )

        if not statistics_file.is_file():
            raise RuntimeError(
                "Training finished without creating statistics: "
                f"{statistics_file}"
            )

        verify_training_checkpoint(
            model_file=model_file,
            configuration=configuration,
            training_agent_seed=agent_seed,
            repository_root=repository_root,
        )

        print(
            "Training completed successfully for "
            f"world seed {world_seed} and agent seed {agent_seed}."
        )

def run_evaluation_runs(
    configuration: dict,
    repository_root: Path,
    experiment_directory: Path,
) -> None:
    """Evaluate every model on every evaluation seed pair."""
    training_runs = model_seed_runs(configuration)
    evaluation = configuration["evaluation"]
    evaluation_runs = evaluation["runs"]

    total_runs = len(training_runs) * len(evaluation_runs)
    run_number = 0

    for training_run in training_runs:
        training_world_seed = training_run["world_seed"]
        training_agent_seed = training_run["agent_seed"]
        training_identifier = training_run_id(training_run)

        model_file = (
            experiment_directory
            / "models"
            / f"{training_identifier}.pt"
        )
        if experiment_mode(configuration) == "evaluate_checkpoint":
            model_file = checkpoint_file(configuration, repository_root)

        if not model_file.is_file():
            raise RuntimeError(
                f"Cannot evaluate missing model: {model_file}"
            )

        model_evaluation_directory = (
            experiment_directory
            / "evaluation"
            / training_identifier
        )
        model_evaluation_directory.mkdir()

        for evaluation_run in evaluation_runs:
            run_number += 1

            evaluation_world_seed = evaluation_run["world_seed"]
            evaluation_agent_seed = evaluation_run["agent_seed"]
            evaluation_identifier = evaluation_run_id(
                evaluation_run
            )

            statistics_file = (
                model_evaluation_directory
                / f"{evaluation_identifier}.json"
            )

            log_directory = (
                experiment_directory
                / "logs"
                / training_identifier
                / evaluation_identifier
            )
            log_directory.mkdir(parents=True)

            environment = (clean_environment() if configuration["experiment_name"].startswith("Report_")
                           or "checkpoints" in configuration["evaluation"] else os.environ.copy())
            environment["BOMBERMAN_AGENT_SEED"] = str(
                evaluation_agent_seed
            )
            environment["BOMBERMAN_MODEL_FILE"] = str(model_file)
            environment["BOMBERMAN_FEATURE_VERSION"] = configuration[
                "feature_version"
            ]
            environment["BOMBERMAN_ACTION_SPACE_VERSION"] = configuration[
                "action_space_version"
            ]
            environment["BOMBERMAN_GAME_LOG_LEVEL"] = "WARNING"
            environment[
                "BOMBERMAN_AGENT_WRAPPER_LOG_LEVEL"
            ] = "WARNING"
            environment["BOMBERMAN_AGENT_CODE_LOG_LEVEL"] = "WARNING"

            command = [
                sys.executable,
                str(repository_root / "main.py"),
                "play",
                "--agents",
                configuration["agent"],
                "--train",
                "0",
                "--scenario",
                configuration["scenario"],
                "--n-rounds",
                str(evaluation["rounds_per_seed"]),
                "--seed",
                str(evaluation_world_seed),
                "--no-gui",
                "--save-stats",
                str(statistics_file),
                "--log-dir",
                str(log_directory),
            ]

            print()
            print(
                f"Evaluation run {run_number}/{total_runs}: "
                "training seeds "
                f"w={training_world_seed}, a={training_agent_seed}; "
                "evaluation seeds "
                f"w={evaluation_world_seed}, a={evaluation_agent_seed}"
            )

            try:
                subprocess.run(
                    command,
                    cwd=repository_root,
                    env=environment,
                    check=True,
                )
            except subprocess.CalledProcessError as error:
                raise RuntimeError(
                    "Evaluation failed for "
                    "training seeds "
                    f"w={training_world_seed}, "
                    f"a={training_agent_seed} and "
                    "evaluation seeds "
                    f"w={evaluation_world_seed}, "
                    f"a={evaluation_agent_seed}."
                ) from error

            if not statistics_file.is_file():
                raise RuntimeError(
                    "Evaluation finished without creating statistics: "
                    f"{statistics_file}"
                )

            print("Evaluation completed successfully.")

def main():
    args = parse_arguments()

    repository_root = Path(__file__).resolve().parents[2]

    experiment_directory = None
    experiment_index = None

    try:
        configuration = load_configuration(args.config)
        validate_configuration(configuration, repository_root)
        print_experiment_plan(configuration)
        if args.validate_only:
            print()
            print(
                "Configuration validation completed; "
                "no experiment was started."
            )
            return

        if experiment_mode(configuration) == "evaluate_checkpoint":
            verify_training_checkpoint(checkpoint_file(configuration, repository_root), configuration,
                                       configuration["checkpoint"]["training_agent_seed"], repository_root)
        git_commit = get_clean_git_commit(repository_root)

        print(f"Git commit: {git_commit}")

        experiment_directory = prepare_experiment_directory(
            repository_root,
            configuration["experiment_name"],
        )

        experiment_index = build_experiment_index(
            configuration,
            git_commit,
        )
        write_experiment_index(
            experiment_directory,
            experiment_index,
        )

        if experiment_mode(configuration) == "train_and_evaluate":
            run_training_runs(configuration, repository_root, experiment_directory)

        run_evaluation_runs(
            configuration,
            repository_root,
            experiment_directory,
        )

        run_checkpoint_evaluations(sys.modules[__name__], configuration, repository_root,
                                   experiment_directory, experiment_index, "task2")
        experiment_index["status"] = "completed"
        write_experiment_index(
            experiment_directory,
            experiment_index,
        )
        if "checkpoints" in configuration["evaluation"]:
            analyze_and_plot(repository_root, "task2", experiment_directory / "experiment.json")

    except (ValueError, RuntimeError, OSError, subprocess.CalledProcessError, KeyboardInterrupt) as error:
        if experiment_directory is not None and experiment_index is not None:
            experiment_index["status"] = "failed"
            write_experiment_index(
                experiment_directory,
                experiment_index,
            )

        raise SystemExit(f"Experiment failed: {error}")

    print()
    print("All training and evaluation runs completed successfully.")
    print(f"Experiment directory: {experiment_directory}")

if __name__ == "__main__":
    main()
