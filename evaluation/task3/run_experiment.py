"""Run controlled multi-seed experiments for Task 3."""

from argparse import ArgumentParser
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import os
import subprocess
import sys
import shutil

if not __package__:  # Support direct script invocation from the repository.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.agent_registry import get_agent_registry, validate_agent_versions
from evaluation.checkpoint_evaluations import (
    analyze_and_plot, clean_environment, print_checkpoint_plan,
    run_checkpoint_evaluations, validate_checkpoint_plan,
)

if __package__:
    from .combat_metrics import opponent_instance_names
else:  # Direct script invocation.
    from combat_metrics import opponent_instance_names

ACTION_ORDER = (
    "UP",
    "RIGHT",
    "DOWN",
    "LEFT",
    "WAIT",
    "BOMB",
)

TASK_3_ACTION_SPACE_VERSION = "task2_all_actions_v1"
DEFAULT_TRAINING_CHECKPOINT_INTERVAL = 100
TRAIN_AND_EVALUATE_MODE = "train_and_evaluate"
EVALUATE_CHECKPOINT_MODE = "evaluate_checkpoint"
SUPPORTED_EXPERIMENT_MODES = {
    TRAIN_AND_EVALUATE_MODE,
    EVALUATE_CHECKPOINT_MODE,
}

PRESSURE_EVALUATION_PARAMETERS = {
    "pressure_threshold": "BOMBERMAN_EVAL_PRESSURE_THRESHOLD",
    "pressure_distance_discount": "BOMBERMAN_EVAL_PRESSURE_DISTANCE_DISCOUNT",
}
V12_FEATURE_VERSION = "task3_parallel_masks_tactical_bombs_v12"


def pressure_parameter(value, name):
    """Validate an explicitly recorded evaluation parameter, never a gamma."""
    lower_valid = value > 0 if isinstance(value, (int, float)) else False
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not 0 <= value <= 1 or not math.isfinite(value)
            or (name == "pressure_distance_discount" and not lower_valid)):
        interval = "(0, 1]" if name == "pressure_distance_discount" else "[0, 1]"
        raise ValueError(f"'{name}' must be a finite number in {interval}.")
    return float(value)


def validate_pressure_evaluation(configuration):
    """Keep parameter sweeps separate from training and historical features."""
    evaluation = configuration["evaluation"]
    supplied = set(PRESSURE_EVALUATION_PARAMETERS) & evaluation.keys()
    sweep = configuration.get("pressure_sweep")
    if "pressure_sweep" not in configuration and not supplied:
        return
    if (experiment_mode(configuration) != EVALUATE_CHECKPOINT_MODE
            or configuration["agent"] != "q_learning_agent"
            or configuration["feature_version"] != V12_FEATURE_VERSION):
        raise ValueError("Pressure overrides require evaluate_checkpoint with Q-learning V12.")
    if "pressure_sweep" in configuration:
        if supplied:
            raise ValueError("Do not mix pressure_sweep with individual pressure overrides.")
        if not isinstance(sweep, dict) or set(sweep) != {"thresholds", "distance_discounts"}:
            raise ValueError("'pressure_sweep' requires thresholds and distance_discounts lists.")
        for key, parameter in (("thresholds", "pressure_threshold"),
                               ("distance_discounts", "pressure_distance_discount")):
            values = sweep[key]
            if not isinstance(values, list) or not values:
                raise ValueError(f"'pressure_sweep.{key}' must be a non-empty list.")
            validated = [pressure_parameter(value, parameter) for value in values]
            if len(set(validated)) != len(validated):
                raise ValueError(f"'pressure_sweep.{key}' contains duplicate values.")
    else:
        if supplied != set(PRESSURE_EVALUATION_PARAMETERS):
            raise ValueError("Record both pressure_threshold and pressure_distance_discount.")
        for name in supplied:
            pressure_parameter(evaluation[name], name)


def pressure_variants(configuration):
    """Expand one grid without writing redundant input config files."""
    sweep = configuration["pressure_sweep"]
    variants = []
    for index, (discount, threshold) in enumerate(
            ((d, t) for d in sweep["distance_discounts"] for t in sweep["thresholds"]), 1):
        # The ordinal is unique even for values not representable in hundredths.
        identifier = f"variant_{index:02d}_pressure_{threshold:g}_discount_{discount:g}"
        child = deepcopy(configuration)
        del child["pressure_sweep"]
        child["experiment_name"] += "__" + identifier
        child["evaluation"].update(
            pressure_threshold=float(threshold), pressure_distance_discount=float(discount))
        variants.append((identifier, child))
    return variants


def archive_evaluation_replays(
    *,
    statistics_file: Path,
    source_directory: Path,
    destination_directory: Path,
    agent_name: str,
    expected_rounds: int,
) -> Path:
    """Move this run's replays only, retaining all explosion-hit attribution.

    Replay paths in the index are relative to its own directory. Recorded
    killers identify bomb owners, not the tactical reason an escape failed.
    No replay pickle is loaded while building the index.
    """
    statistics = json.loads(statistics_file.read_text(encoding="utf-8"))
    rounds = statistics.get("by_round", {})
    if len(rounds) != expected_rounds:
        raise RuntimeError(
            f"Expected {expected_rounds} recorded rounds, found {len(rounds)}."
        )
    if destination_directory.exists():
        raise RuntimeError(f"Replay destination already exists: {destination_directory}")

    records = []
    sources = []
    # Validate every source before moving any file. Do not glob or clear the
    # shared engine replay directory; it may contain unrelated matches.
    for round_id, result in rounds.items():
        number = re.search(r"(?:^| \| )Round (\d+) \(", round_id)
        if number is None or Path(round_id).name != round_id:
            raise RuntimeError(f"Unexpected engine round ID: {round_id!r}")
        source = source_directory / f"{round_id}.pt"
        if not source.is_file():
            raise RuntimeError(f"Missing evaluation replay: {source}")
        agent = result.get("by_agent", {}).get(agent_name)
        if agent is None or not isinstance(agent.get("survived"), bool):
            raise RuntimeError(f"Missing survival statistics for {agent_name}: {round_id}")
        hits = [
            event for event in result.get("kill_events", [])
            if event["victim"] == agent_name
        ]
        round_number = int(number.group(1))
        record = {
            "round": round_number,
            "round_id": round_id,
            "replay": f"round_{round_number:04d}.pt",
            "survived": agent["survived"],
            "death_step": min((event["step"] for event in hits), default=None),
            # Several explosions can hit during the same step. Keep all owners.
            "killers": sorted({event["killer"] for event in hits}),
            "kills": agent.get("kills", 0),
        }
        sources.append((source, record["replay"]))
        records.append(record)

    if len({record["round"] for record in records}) != len(records):
        raise RuntimeError("Duplicate round numbers in evaluation replay index.")
    records.sort(key=lambda record: record["round"])
    destination_directory.mkdir(parents=True)
    for source, name in sources:
        shutil.move(str(source), str(destination_directory / name))

    index = {
        "agent": agent_name,
        "replay_paths_relative_to": "index_directory",
        "round_count": len(records),
        "death_count": sum(not record["survived"] for record in records),
        "rounds": records,
    }
    index_file = destination_directory / "index.json"
    index_file.write_text(json.dumps(index, indent=4) + "\n", encoding="utf-8")
    return index_file


def parse_arguments():
    """Read the experiment configuration path from the command line."""
    parser = ArgumentParser(
        description="Run a controlled Task 3 multi-seed experiment."
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


def require_non_negative_integer(
    configuration: dict,
    name: str,
) -> int:
    """Return a required non-negative integer."""
    value = configuration.get(name)

    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise ValueError(
            f"'{name}' must be a non-negative integer."
        )

    return value


def experiment_mode(configuration: dict) -> str:
    """Return the explicit or backward-compatible experiment mode."""
    mode = configuration.get(
        "experiment_mode",
        TRAIN_AND_EVALUATE_MODE,
    )

    if mode not in SUPPORTED_EXPERIMENT_MODES:
        raise ValueError(
            "'experiment_mode' must be one of: "
            + ", ".join(sorted(SUPPORTED_EXPERIMENT_MODES))
            + "."
        )

    return mode


def resolve_repository_file(
    repository_root: Path,
    relative_path: str,
    location: str,
) -> Path:
    """Resolve one repository-relative file without allowing escape."""
    path = Path(relative_path)

    if path.is_absolute():
        raise ValueError(
            f"'{location}' must be relative to the repository root."
        )

    repository_root = repository_root.resolve()
    resolved_path = (repository_root / path).resolve()

    try:
        resolved_path.relative_to(repository_root)
    except ValueError as error:
        raise ValueError(
            f"'{location}' must stay inside the repository root."
        ) from error

    if not resolved_path.is_file():
        raise ValueError(
            f"'{location}' does not exist: {resolved_path}"
        )

    return resolved_path


def require_checkpoint(
    configuration: dict,
    repository_root: Path,
) -> dict:
    """Return and validate the external checkpoint declaration."""
    checkpoint = require_section(configuration, "checkpoint")
    checkpoint_path = require_non_empty_string(checkpoint, "path")
    require_non_negative_integer(
        checkpoint,
        "training_world_seed",
    )
    require_non_negative_integer(
        checkpoint,
        "training_agent_seed",
    )
    require_positive_integer(checkpoint, "training_rounds")

    source_tag = checkpoint.get("source_tag")
    if source_tag is not None and (
        not isinstance(source_tag, str) or not source_tag.strip()
    ):
        raise ValueError(
            "'checkpoint.source_tag' must be a non-empty string "
            "when provided."
        )

    resolve_repository_file(
        repository_root,
        checkpoint_path,
        "checkpoint.path",
    )

    return checkpoint


def checkpoint_file(
    configuration: dict,
    repository_root: Path,
) -> Path:
    """Return the validated external checkpoint path."""
    checkpoint = require_checkpoint(
        configuration,
        repository_root,
    )
    return resolve_repository_file(
        repository_root,
        checkpoint["path"],
        "checkpoint.path",
    )


def file_sha256(file_path: Path) -> str:
    """Return the SHA-256 digest of one model artifact."""
    digest = hashlib.sha256()

    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def require_opponents(configuration: dict) -> tuple[str, ...]:
    """Return the non-empty, ordered opponent roster."""
    opponents = configuration.get("opponents")

    if not isinstance(opponents, list) or not opponents:
        raise ValueError(
            "'opponents' must be a non-empty JSON list."
        )

    validated_opponents = []

    for index, opponent in enumerate(opponents):
        if not isinstance(opponent, str) or not opponent.strip():
            raise ValueError(
                f"'opponents[{index}]' must be a non-empty string."
            )

        validated_opponents.append(opponent)

    agent_name = configuration.get("agent")

    if (agent_name in validated_opponents
            or agent_name in opponent_instance_names(validated_opponents)):
        raise ValueError(
            "The learning agent must not also appear in 'opponents'."
        )

    return tuple(validated_opponents)


def configured_agent_roster(configuration: dict) -> tuple[str, ...]:
    """Return the learning agent followed by its fixed opponents."""
    return (
        configuration["agent"],
        *require_opponents(configuration),
    )


def validate_opponent_seeds(configuration: dict, section_name: str) -> None:
    """Validate optional explicit opponent streams in either experiment phase."""
    for index, seed_run in enumerate(configuration[section_name]["runs"]):
        if "opponent_agent_seeds" not in seed_run:
            continue
        seeds = seed_run["opponent_agent_seeds"]
        if (not isinstance(seeds, list)
                or len(seeds) != len(configuration["opponents"])
                or any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
                       for seed in seeds)):
            raise ValueError(
                f"'{section_name}.runs[{index}].opponent_agent_seeds' requires "
                "one non-negative integer per opponent."
            )


def agent_seed_configurations(configuration: dict, seed_run: dict) -> dict:
    """Use engine instance names to give every participant its own RNG stream."""
    players = {
        configuration["agent"]: {"BOMBERMAN_AGENT_SEED": str(seed_run["agent_seed"])},
    }
    players.update({
        name: {"BOMBERMAN_AGENT_SEED": str(seed)}
        for name, seed in zip(opponent_instance_names(configuration["opponents"]),
                              seed_run["opponent_agent_seeds"])
    })
    return players


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
    agent_directory = (
        repository_root
        / "agent_code"
        / configuration["agent"]
    )

    if not agent_directory.is_dir():
        raise ValueError(
            f"Unknown agent {configuration['agent']!r}: "
            f"{agent_directory} does not exist."
        )

    for required_file in ("callbacks.py", "train.py"):
        if not (agent_directory / required_file).is_file():
            raise ValueError(
                f"Agent {configuration['agent']!r} is missing "
                f"{required_file!r}."
            )

    opponents = require_opponents(configuration)

    if len(opponents) + 1 > settings.MAX_AGENTS:
        raise ValueError(
            "The configured roster exceeds the engine limit of "
            f"{settings.MAX_AGENTS} agents."
        )

    for opponent in opponents:
        opponent_directory = (
            repository_root / "agent_code" / opponent
        )

        if not opponent_directory.is_dir():
            raise ValueError(
                f"Unknown opponent {opponent!r}: "
                f"{opponent_directory} does not exist."
            )

        if not (opponent_directory / "callbacks.py").is_file():
            raise ValueError(
                f"Opponent {opponent!r} is missing 'callbacks.py'."
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

    mode = experiment_mode(configuration)

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

    if action_space_version != TASK_3_ACTION_SPACE_VERSION:
        raise ValueError(
            "Task 3 experiments must use action-space version "
            f"{TASK_3_ACTION_SPACE_VERSION!r}."
        )

    require_opponents(configuration)

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

    evaluation = require_section(configuration, "evaluation")
    require_non_empty_string(evaluation, "seed_set")
    require_positive_integer(evaluation, "rounds_per_seed")
    if not isinstance(evaluation.get("save_replays", False), bool):
        raise ValueError("'evaluation.save_replays' must be a boolean.")
    evaluation_seed_pairs = require_seed_runs(
        evaluation,
        "evaluation",
    )
    validate_opponent_seeds(configuration, "evaluation")
    validate_pressure_evaluation(configuration)

    if mode == TRAIN_AND_EVALUATE_MODE:
        if "checkpoint" in configuration:
            raise ValueError(
                "'checkpoint' is only valid in evaluate_checkpoint mode."
            )

        training = require_section(configuration, "training")
        require_positive_integer(training, "rounds")
        if "checkpoint_interval" in training:
            require_positive_integer(
                training,
                "checkpoint_interval",
            )
        training_seed_pairs = require_seed_runs(
            training,
            "training",
        )
        validate_opponent_seeds(configuration, "training")
        training_world_seeds = {
            world_seed
            for world_seed, _ in training_seed_pairs
        }
    else:
        if "training" in configuration:
            raise ValueError(
                "'training' must be omitted in evaluate_checkpoint mode."
            )

        checkpoint = require_checkpoint(
            configuration,
            repository_root,
        )
        training_world_seeds = {
            checkpoint["training_world_seed"]
        }

    validate_checkpoint_plan(configuration)
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
    mode = experiment_mode(configuration)
    evaluation = configuration["evaluation"]
    print_checkpoint_plan(configuration)

    if mode == TRAIN_AND_EVALUATE_MODE:
        training = configuration["training"]
        number_of_models = len(training["runs"])
        total_training_rounds = (
            number_of_models * training["rounds"]
        )
    else:
        training = None
        number_of_models = 1
        total_training_rounds = 0

    evaluations_per_model = len(evaluation["runs"])
    total_evaluation_runs = number_of_models * evaluations_per_model
    total_evaluation_rounds = (
        total_evaluation_runs * evaluation["rounds_per_seed"]
    )
    variants = pressure_variants(configuration) if "pressure_sweep" in configuration else []
    if variants:
        total_evaluation_runs *= len(variants)
        total_evaluation_rounds *= len(variants)

    print("Configuration valid.")
    print(f"Experiment: {configuration['experiment_name']}")
    print(f"Experiment mode: {mode}")
    print(
        f"Agent and scenario: "
        f"{configuration['agent']} / {configuration['scenario']}"
    )
    print(
        "Fixed opponents: "
        + ", ".join(configuration["opponents"])
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

    if mode == TRAIN_AND_EVALUATE_MODE:
        print("Training seed pairs:")
        for seed_run in training["runs"]:
            print(
                f"  world={seed_run['world_seed']}, "
                f"agent={seed_run['agent_seed']}"
            )
    else:
        checkpoint = configuration["checkpoint"]
        print("Zero-shot checkpoint:")
        print(
            f"  path={checkpoint['path']}"
        )
        print(
            "  original training seeds: "
            f"world={checkpoint['training_world_seed']}, "
            f"agent={checkpoint['training_agent_seed']}"
        )
        print(
            "  original training rounds: "
            f"{checkpoint['training_rounds']}"
        )

    print("Evaluation seed pairs:")
    for seed_run in evaluation["runs"]:
        print(
            f"  world={seed_run['world_seed']}, "
            f"agent={seed_run['agent_seed']}"
        )

    print(f"Evaluated models: {number_of_models}")
    print(f"Total training rounds: {total_training_rounds}")
    if training is not None:
        checkpoint_interval = training.get(
            "checkpoint_interval",
            DEFAULT_TRAINING_CHECKPOINT_INTERVAL,
        )
        print(f"Checkpoint interval: {checkpoint_interval}")
    print(f"Evaluations per model: {evaluations_per_model}")
    print(f"Total evaluation runs: {total_evaluation_runs}")
    print(f"Total evaluation rounds: {total_evaluation_rounds}")
    if variants:
        print(f"Pressure combinations: {len(variants)}; "
              f"{len(evaluation['runs']) * evaluation['rounds_per_seed']} rounds each")
        for _, child in variants:
            parameters = child["evaluation"]
            print(f"  threshold={parameters['pressure_threshold']:g}, "
                  f"pressure distance discount={parameters['pressure_distance_discount']:g}")
    elif "pressure_threshold" in evaluation:
        print(f"Evaluation pressure threshold: {evaluation['pressure_threshold']:g}")
        print(f"Evaluation pressure distance discount: {evaluation['pressure_distance_discount']:g}")
    if evaluation.get("save_replays", False):
        print("Replay recording: all evaluation rounds, with a death index")

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
    repository_root: Path,
) -> dict:
    """Build the machine-readable index of experiment artifacts."""
    runs = []

    mode = experiment_mode(configuration)

    if mode == TRAIN_AND_EVALUATE_MODE:
        model_runs = configuration["training"]["runs"]
    else:
        checkpoint = configuration["checkpoint"]
        model_runs = [
            {
                "world_seed": checkpoint["training_world_seed"],
                "agent_seed": checkpoint["training_agent_seed"],
            }
        ]

    for training_run in model_runs:
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
            if configuration["evaluation"].get("save_replays", False):
                evaluations[-1]["replay_index"] = (
                    f"replays/{training_identifier}/"
                    f"{evaluation_identifier}/index.json"
                )
            if "opponent_agent_seeds" in evaluation_run:
                evaluations[-1]["opponent_agent_seeds"] = evaluation_run["opponent_agent_seeds"]

        run = {
            "training_world_seed": training_run["world_seed"],
            "training_agent_seed": training_run["agent_seed"],
            "evaluations": evaluations,
        }

        if mode == TRAIN_AND_EVALUATE_MODE:
            run.update(
                {
                    "model": f"models/{training_identifier}.pt",
                    "training_results": (
                        f"training/{training_identifier}.json"
                    ),
                }
            )
            if "opponent_agent_seeds" in training_run:
                run["opponent_agent_seeds"] = training_run["opponent_agent_seeds"]
        else:
            source_model = checkpoint_file(
                configuration,
                repository_root,
            )
            run.update(
                {
                    "model": checkpoint["path"],
                    "model_source": "external_checkpoint",
                    "model_sha256": file_sha256(source_model),
                    "training_results": None,
                }
            )

        runs.append(run)

    index = {
        "status": "running",
        "git_commit": git_commit,
        "experiment_mode": mode,
        "configuration": configuration,
        "runs": runs,
    }
    if "pressure_sweep" in configuration:
        index.update(
            experiment_kind="pressure_sweep", runs=[],
            checkpoint_sha256=file_sha256(checkpoint_file(configuration, repository_root)),
            variants=[{
                "pressure_threshold": child["evaluation"]["pressure_threshold"],
                "pressure_distance_discount": child["evaluation"]["pressure_distance_discount"],
                "experiment": f"variants/{identifier}/experiment.json",
            } for identifier, child in pressure_variants(configuration)],
        )
    return index


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
        / "task3"
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

def verify_checkpoint(
    model_file: Path,
    configuration: dict,
    expected_agent_seed: int,
    expected_rounds: int,
    repository_root: Path,
) -> None:
    """Verify that a checkpoint matches the configured model contract."""
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
            f"Expected a versioned checkpoint, got legacy model: "
            f"{model_file}"
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
    if saved_agent_seed != expected_agent_seed:
        raise RuntimeError(
            "Agent-seed mismatch:\n"
            f"  configured: {expected_agent_seed}\n"
            f"  checkpoint: {saved_agent_seed}"
        )

    completed_rounds = metadata.get("rounds_completed")

    if completed_rounds != expected_rounds:
        raise RuntimeError(
            "Training-round mismatch:\n"
            f"  configured: {expected_rounds}\n"
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

        seeded_opponents = "opponent_agent_seeds" in training_run
        environment = (clean_environment() if seeded_opponents
                       or configuration["experiment_name"].startswith("Report_")
                       or "checkpoints" in configuration["evaluation"] else os.environ.copy())
        if seeded_opponents:
            environment["BOMBERMAN_AGENT_CONFIGS"] = json.dumps(
                agent_seed_configurations(configuration, training_run), sort_keys=True,
            )
        environment["BOMBERMAN_TASK3_DIAGNOSTICS"] = "0"
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
            *configured_agent_roster(configuration),
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

        verify_checkpoint(
            model_file=model_file,
            configuration=configuration,
            expected_agent_seed=agent_seed,
            expected_rounds=training["rounds"],
            repository_root=repository_root,
        )

        print(
            "Training completed successfully for "
            f"world seed {world_seed} and agent seed {agent_seed}."
        )

def configured_evaluation_models(
    configuration: dict,
    repository_root: Path,
    experiment_directory: Path,
) -> list[tuple[dict, Path]]:
    """Return model seed descriptors and paths for evaluation."""
    if experiment_mode(configuration) == EVALUATE_CHECKPOINT_MODE:
        checkpoint = configuration["checkpoint"]
        model_run = {
            "world_seed": checkpoint["training_world_seed"],
            "agent_seed": checkpoint["training_agent_seed"],
        }
        return [
            (
                model_run,
                checkpoint_file(configuration, repository_root),
            )
        ]

    models = []

    for training_run in configuration["training"]["runs"]:
        training_identifier = training_run_id(training_run)
        models.append(
            (
                training_run,
                experiment_directory
                / "models"
                / f"{training_identifier}.pt",
            )
        )

    return models


def evaluation_environment(configuration, seed_run, model_file):
    """Pass frozen evaluation settings and optional per-opponent RNG seeds."""
    evaluation = configuration["evaluation"]
    seeded_opponents = "opponent_agent_seeds" in seed_run
    environment = (clean_environment() if seeded_opponents
                   or "pressure_threshold" in evaluation
                   or configuration["experiment_name"].startswith("Report_")
                   or "checkpoints" in evaluation else os.environ.copy())
    for variable in PRESSURE_EVALUATION_PARAMETERS.values():
        environment.pop(variable, None)
    environment.update({
        "BOMBERMAN_TASK3_DIAGNOSTICS": "1",
        "BOMBERMAN_DIAGNOSTIC_AGENT": configuration["agent"],
        "BOMBERMAN_AGENT_SEED": str(seed_run["agent_seed"]),
        "BOMBERMAN_MODEL_FILE": str(model_file),
        "BOMBERMAN_FEATURE_VERSION": configuration["feature_version"],
        "BOMBERMAN_ACTION_SPACE_VERSION": configuration["action_space_version"],
        "BOMBERMAN_GAME_LOG_LEVEL": "WARNING",
        "BOMBERMAN_AGENT_WRAPPER_LOG_LEVEL": "WARNING",
        "BOMBERMAN_AGENT_CODE_LOG_LEVEL": "WARNING",
    })
    for parameter, variable in PRESSURE_EVALUATION_PARAMETERS.items():
        if parameter in evaluation:
            environment[variable] = str(evaluation[parameter])
    if seeded_opponents:
        environment["BOMBERMAN_AGENT_CONFIGS"] = json.dumps(
            agent_seed_configurations(configuration, seed_run), sort_keys=True,
        )
    return environment


def run_evaluation_runs(
    configuration: dict,
    repository_root: Path,
    experiment_directory: Path,
) -> None:
    """Evaluate every model on every evaluation seed pair."""
    model_runs = configured_evaluation_models(
        configuration,
        repository_root,
        experiment_directory,
    )
    evaluation = configuration["evaluation"]
    evaluation_runs = evaluation["runs"]

    total_runs = len(model_runs) * len(evaluation_runs)
    run_number = 0

    for training_run, model_file in model_runs:
        training_world_seed = training_run["world_seed"]
        training_agent_seed = training_run["agent_seed"]
        training_identifier = training_run_id(training_run)

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

            environment = evaluation_environment(configuration, evaluation_run, model_file)

            command = [
                sys.executable,
                str(repository_root / "main.py"),
                "play",
                "--agents",
                *configured_agent_roster(configuration),
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
            if evaluation.get("save_replays", False):
                # The engine uses round IDs as filenames. A run-specific match
                # name prevents unrelated evaluations from sharing filenames.
                (repository_root / "replays").mkdir(exist_ok=True)
                command.extend([
                    "--save-replay",
                    "--match-name",
                    f"{configuration['experiment_name']}--"
                    f"{training_identifier}--{evaluation_identifier}",
                ])

            print()
            print(
                f"Evaluation run {run_number}/{total_runs}: "
                "model training seeds "
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
                    "model training seeds "
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

            if evaluation.get("save_replays", False):
                replay_index = archive_evaluation_replays(
                    statistics_file=statistics_file,
                    source_directory=repository_root / "replays",
                    destination_directory=(
                        experiment_directory / "replays"
                        / training_identifier / evaluation_identifier
                    ),
                    agent_name=configuration["agent"],
                    expected_rounds=evaluation["rounds_per_seed"],
                )
                print(f"Replays and death index: {replay_index}")

            print("Evaluation completed successfully.")

def run_pressure_sweep(configuration, repository_root, experiment_directory, experiment_index):
    """Run each combination through the ordinary checkpoint evaluator."""
    source = checkpoint_file(configuration, repository_root)
    expected_hash = experiment_index["checkpoint_sha256"]
    variants = pressure_variants(configuration)
    for number, (identifier, child) in enumerate(variants, 1):
        if file_sha256(source) != expected_hash:
            raise RuntimeError("The source checkpoint changed during the pressure sweep.")
        directory = experiment_directory / "variants" / identifier
        (directory / "evaluation").mkdir(parents=True)
        (directory / "logs").mkdir()
        index = build_experiment_index(child, experiment_index["git_commit"], repository_root)
        write_experiment_index(directory, index)
        print(f"\nPressure combination {number}/{len(variants)}: "
              f"threshold={child['evaluation']['pressure_threshold']:g}, "
              f"distance discount={child['evaluation']['pressure_distance_discount']:g}")
        try:
            run_evaluation_runs(child, repository_root, directory)
            if file_sha256(source) != expected_hash:
                raise RuntimeError("The source checkpoint changed during the pressure sweep.")
            index["status"] = "completed"
        except (Exception, KeyboardInterrupt):
            index["status"] = "failed"
            raise
        finally:
            write_experiment_index(directory, index)


def main():
    args = parse_arguments()

    repository_root = Path(__file__).resolve().parents[2]

    experiment_directory = None
    experiment_index = None

    try:
        configuration = load_configuration(args.config)
        validate_configuration(configuration, repository_root)
        print_experiment_plan(configuration)
        mode = experiment_mode(configuration)
        if args.validate_only:
            print()
            print(
                "Configuration validation completed; "
                "no experiment was started."
            )
            return

        if mode == EVALUATE_CHECKPOINT_MODE:
            checkpoint = configuration["checkpoint"]
            verify_checkpoint(
                model_file=checkpoint_file(
                    configuration,
                    repository_root,
                ),
                configuration=configuration,
                expected_agent_seed=checkpoint[
                    "training_agent_seed"
                ],
                expected_rounds=checkpoint["training_rounds"],
                repository_root=repository_root,
            )

        git_commit = get_clean_git_commit(repository_root)

        print(f"Git commit: {git_commit}")

        experiment_directory = prepare_experiment_directory(
            repository_root,
            configuration["experiment_name"],
        )

        experiment_index = build_experiment_index(
            configuration,
            git_commit,
            repository_root,
        )
        write_experiment_index(
            experiment_directory,
            experiment_index,
        )

        if mode == TRAIN_AND_EVALUATE_MODE:
            run_training_runs(
                configuration,
                repository_root,
                experiment_directory,
            )

        if "pressure_sweep" in configuration:
            run_pressure_sweep(configuration, repository_root, experiment_directory, experiment_index)
        else:
            run_evaluation_runs(configuration, repository_root, experiment_directory)

        run_checkpoint_evaluations(sys.modules[__name__], configuration, repository_root,
                                   experiment_directory, experiment_index, "task3")
        experiment_index["status"] = "completed"
        write_experiment_index(
            experiment_directory,
            experiment_index,
        )
        if "checkpoints" in configuration["evaluation"]:
            analyze_and_plot(repository_root, "task3", experiment_directory / "experiment.json")

    except (ValueError, RuntimeError, OSError, subprocess.CalledProcessError, KeyboardInterrupt) as error:
        if experiment_directory is not None and experiment_index is not None:
            experiment_index["status"] = "failed"
            write_experiment_index(
                experiment_directory,
                experiment_index,
            )

        message = "interrupted by user" if isinstance(error, KeyboardInterrupt) else str(error)
        raise SystemExit(f"Experiment failed: {message}")

    print()
    if mode == EVALUATE_CHECKPOINT_MODE:
        print("All zero-shot evaluation runs completed successfully.")
    else:
        print("All training and evaluation runs completed successfully.")
    print(f"Experiment directory: {experiment_directory}")

if __name__ == "__main__":
    main()
