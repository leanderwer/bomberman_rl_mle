"""Run controlled multi-seed experiments for Task 1."""

from argparse import ArgumentParser
import json
from pathlib import Path
import re
import os
import subprocess
import sys

if not __package__:  # Support direct script invocation from the repository.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.agent_registry import get_agent_registry, validate_agent_versions

TASK_1_ACTION_ORDER = (
    "UP",
    "RIGHT",
    "DOWN",
    "LEFT",
    "WAIT",
    "BOMB",
)


def parse_arguments():
    """Read the experiment configuration path from the command line."""
    parser = ArgumentParser(
        description="Run a controlled Task 1 multi-seed experiment."
    )

    parser.add_argument(
        "config",
        type=Path,
        help="Path to the experiment JSON configuration.",
    )

    parser.add_argument(
        "--validate-only",
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

def validate_configuration(
    configuration: dict,
    repository_root: Path | None = None,
) -> None:
    """Validate the parts required for a controlled experiment."""
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

    validate_agent_versions(configuration, repository_root, fill_defaults=True)

    if "coins_per_round" in configuration:
        require_positive_integer(
            configuration,
            "coins_per_round",
        )

    learning_rate_version = configuration.get(
        "learning_rate_version"
    )
    if learning_rate_version is not None:
        require_non_empty_string(
            configuration,
            "learning_rate_version",
        )

    exploration_version = configuration.get(
        "exploration_version"
    )
    if exploration_version is not None:
        require_non_empty_string(
            configuration,
            "exploration_version",
        )

    description = configuration.get("description", "")
    if not isinstance(description, str):
        raise ValueError("'description' must be a string.")

    training = require_section(configuration, "training")
    evaluation = require_section(configuration, "evaluation")

    require_positive_integer(training, "rounds")
    training_seed_pairs = require_seed_runs(
    training,
    "training",
)
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
    training = configuration["training"]
    evaluation = configuration["evaluation"]

    number_of_models = len(training["runs"])
    evaluations_per_model = len(evaluation["runs"])

    total_training_rounds = number_of_models * training["rounds"]
    total_evaluation_runs = number_of_models * evaluations_per_model
    total_evaluation_rounds = (
        total_evaluation_runs * evaluation["rounds_per_seed"]
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
        "Learning-rate version: "
        f"{configuration.get('learning_rate_version', 'agent default')}"
    )
    print(
        "Exploration version: "
        f"{configuration.get('exploration_version', 'agent default')}"
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
) -> dict:
    """Build the machine-readable index of experiment artifacts."""
    runs = []

    for training_run in configuration["training"]["runs"]:
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

    return {
        "status": "running",
        "git_commit": git_commit,
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
        / "task1"
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
        expected_action_order=TASK_1_ACTION_ORDER,
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

    for version_key in (
        "action_space_version", "discount_factor_version", "td_target_version",
    ):
        configured_version = configuration.get(version_key)
        if (configured_version is not None
                and metadata.get(version_key) != configured_version):
            raise RuntimeError(
                f"{version_key} mismatch: configured {configured_version!r}, "
                f"checkpoint {metadata.get(version_key)!r}."
            )
    if "td_target_version" in configuration:
        expected_parameters = registry.get_td_target_parameters(
            configuration["td_target_version"]
        )
        if metadata.get("td_target_parameters") != expected_parameters:
            raise RuntimeError("TD-target-parameter mismatch in training checkpoint.")
    if "discount_factor_version" in configuration:
        expected_gamma = registry.get_discount_factor(
            configuration["discount_factor_version"]
        )
        if metadata.get("gamma") != expected_gamma:
            raise RuntimeError("Discount-factor value mismatch in training checkpoint.")

    saved_agent_seed = metadata.get("agent_seed")
    if saved_agent_seed != training_agent_seed:
        raise RuntimeError(
            "Agent-seed mismatch:\n"
            f"  configured: {training_agent_seed}\n"
            f"  checkpoint: {saved_agent_seed}"
        )

    configured_rounds = configuration["training"]["rounds"]
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

        environment = os.environ.copy()
        environment["BOMBERMAN_AGENT_SEED"] = str(agent_seed)
        environment["BOMBERMAN_MODEL_FILE"] = str(model_file)
        environment["BOMBERMAN_FEATURE_VERSION"] = configuration[
            "feature_version"
        ]
        environment["BOMBERMAN_REWARD_VERSION"] = configuration[
            "reward_version"
        ]
        for setting in (
            "action_space_version", "discount_factor_version", "td_target_version",
        ):
            environment[f"BOMBERMAN_{setting.upper()}"] = configuration[setting]

        learning_rate_version = configuration.get(
            "learning_rate_version"
        )
        if learning_rate_version is not None:
            environment[
                "BOMBERMAN_LEARNING_RATE_VERSION"
            ] = learning_rate_version

        exploration_version = configuration.get(
            "exploration_version"
        )
        if exploration_version is not None:
            environment[
                "BOMBERMAN_EXPLORATION_VERSION"
            ] = exploration_version

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
    training_runs = configuration["training"]["runs"]
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

            environment = os.environ.copy()
            environment["BOMBERMAN_AGENT_SEED"] = str(
                evaluation_agent_seed
            )
            environment["BOMBERMAN_MODEL_FILE"] = str(model_file)
            environment["BOMBERMAN_FEATURE_VERSION"] = configuration[
                "feature_version"
            ]

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

    experiment_directory = None
    experiment_index = None

    try:
        configuration = load_configuration(args.config)
        validate_configuration(configuration)
        print_experiment_plan(configuration)
        if args.validate_only:
            print()
            print(
                "Configuration validation completed; "
                "no experiment was started."
            )
            return

        repository_root = Path(__file__).resolve().parents[2]
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

        run_training_runs(
            configuration,
            repository_root,
            experiment_directory,
        )

        run_evaluation_runs(
            configuration,
            repository_root,
            experiment_directory,
        )

        experiment_index["status"] = "completed"
        write_experiment_index(
            experiment_directory,
            experiment_index,
        )

    except (ValueError, RuntimeError) as error:
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
