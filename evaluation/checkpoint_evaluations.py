"""Optional frozen intermediate evaluations for the existing task runners."""

from copy import deepcopy
import hashlib
import os
from pathlib import Path
import subprocess
import sys


def clean_environment():
    """Prevent shell settings from silently changing a controlled experiment."""
    return {key: value for key, value in os.environ.items()
            if not key.startswith("BOMBERMAN_")}


def snapshot_path(model_file, rounds):
    path = Path(model_file)
    return path.with_name(f"{path.stem}_round_{rounds:05d}{path.suffix}")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def experiment_mode(configuration):
    mode = configuration.get("experiment_mode", "train_and_evaluate")
    if mode not in {"train_and_evaluate", "evaluate_checkpoint"}:
        raise ValueError("Unsupported experiment_mode.")
    return mode


def model_seed_runs(configuration):
    if experiment_mode(configuration) == "train_and_evaluate":
        return configuration["training"]["runs"]
    checkpoint = configuration["checkpoint"]
    return [{"world_seed": checkpoint["training_world_seed"],
             "agent_seed": checkpoint["training_agent_seed"]}]


def checkpoint_file(configuration, repository_root):
    checkpoint = configuration.get("checkpoint")
    if not isinstance(checkpoint, dict):
        raise ValueError("'checkpoint' must be an object.")
    for key, minimum in (("training_world_seed", 0), ("training_agent_seed", 0),
                         ("training_rounds", 1)):
        value = checkpoint.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"Invalid checkpoint.{key}.")
    name = checkpoint.get("path")
    if not isinstance(name, str) or not name or Path(name).is_absolute():
        raise ValueError("checkpoint.path must be a nonempty repository-relative path.")
    root = Path(repository_root).resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"checkpoint.path must identify an existing repository file: {name}")
    return path


def validate_checkpoint_plan(configuration):
    """Require every evaluation milestone to name one retained intermediate model."""
    training = configuration.get("training")
    checkpoints = configuration["evaluation"].get("checkpoints", [])
    if not isinstance(checkpoints, list):
        raise ValueError("'evaluation.checkpoints' must be a list.")
    if training is None:
        if checkpoints:
            raise ValueError("Checkpoint milestones require a training experiment.")
        return
    snapshots = training.get("snapshot_rounds", [])
    if not isinstance(snapshots, list):
        raise ValueError("'training.snapshot_rounds' must be a list.")
    if snapshots and configuration["agent"] != "q_learning_agent":
        raise ValueError("Intermediate snapshots currently require q_learning_agent.")
    seen = set()
    for value in snapshots:
        if (isinstance(value, bool) or not isinstance(value, int)
                or not 0 < value < training["rounds"] or value in seen):
            raise ValueError("snapshot_rounds must contain unique positive intermediate rounds.")
        seen.add(value)
    evaluated = set()
    for entry in checkpoints:
        if not isinstance(entry, dict) or set(entry) != {"rounds", "rounds_per_seed"}:
            raise ValueError("Each evaluation checkpoint requires rounds and rounds_per_seed.")
        rounds, count = entry["rounds"], entry["rounds_per_seed"]
        if (isinstance(rounds, bool) or not isinstance(rounds, int)
                or rounds not in seen or rounds in evaluated):
            raise ValueError("Evaluation checkpoints must name distinct retained snapshot_rounds.")
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("Checkpoint rounds_per_seed must be a positive integer.")
        evaluated.add(rounds)


def print_checkpoint_plan(configuration):
    snapshots = configuration.get("training", {}).get("snapshot_rounds", [])
    if snapshots:
        print("Retained intermediate snapshots: " + ", ".join(map(str, snapshots)))
    for point in configuration["evaluation"].get("checkpoints", []):
        print(f"Frozen checkpoint evaluation after {point['rounds']} rounds: "
              f"{len(configuration['evaluation']['runs'])} seeds x "
              f"{point['rounds_per_seed']} rounds")


def analyze_and_plot(repository_root, task, experiment_file):
    """Use the existing analysis and PDF commands, retaining their output contract."""
    for script in ("analyze_experiment.py", "plot_experiment.py"):
        subprocess.run([sys.executable, str(repository_root / "evaluation" / task / script),
                        str(experiment_file)], cwd=repository_root,
                       env=clean_environment(), check=True)


def run_checkpoint_evaluations(runner, configuration, repository_root,
                               experiment_directory, experiment_index, task):
    """Evaluate saved models after training without consuming the training RNG streams."""
    for point in configuration["evaluation"].get("checkpoints", []):
        for training_run in configuration["training"]["runs"]:
            identifier = runner.training_run_id(training_run)
            rounds = point["rounds"]
            source = snapshot_path(experiment_directory / "models" / f"{identifier}.pt", rounds)
            child = deepcopy(configuration)
            child.pop("training")
            child["experiment_mode"] = "evaluate_checkpoint"
            child["experiment_name"] += f"__round_{rounds:05d}__{identifier}"
            child["checkpoint"] = {
                "path": str(source.relative_to(repository_root)),
                "training_world_seed": training_run["world_seed"],
                "training_agent_seed": training_run["agent_seed"],
                "training_rounds": rounds,
            }
            if "opponent_agent_seeds" in training_run:
                child["checkpoint"]["opponent_agent_seeds"] = training_run["opponent_agent_seeds"]
            child["evaluation"].pop("checkpoints", None)
            child["evaluation"]["rounds_per_seed"] = point["rounds_per_seed"]
            child["evaluation"]["seed_set"] += f"__{point['rounds_per_seed']}_rounds_per_seed"
            runner.validate_configuration(child, repository_root)
            if task == "task2":
                runner.verify_training_checkpoint(source, child, training_run["agent_seed"], repository_root)
            else:
                runner.verify_checkpoint(source, child, training_run["agent_seed"], rounds, repository_root)
            directory = experiment_directory / "checkpoints" / f"round_{rounds:05d}" / identifier
            (directory / "evaluation").mkdir(parents=True)
            (directory / "logs").mkdir()
            index = runner.build_experiment_index(child, experiment_index["git_commit"], repository_root)
            index["parent_experiment"] = str(experiment_directory.relative_to(repository_root) / "experiment.json")
            runner.write_experiment_index(directory, index)
            expected_hash = sha256(source)
            entry = {"rounds": rounds, "training_run": identifier,
                     "rounds_per_seed": point["rounds_per_seed"],
                     "experiment": str((directory / "experiment.json").relative_to(experiment_directory))}
            experiment_index.setdefault("checkpoint_evaluations", []).append(entry)
            runner.write_experiment_index(experiment_directory, experiment_index)
            try:
                runner.run_evaluation_runs(child, repository_root, directory)
                if sha256(source) != expected_hash:
                    raise RuntimeError("A frozen intermediate checkpoint changed during evaluation.")
                index["status"] = "completed"
            except (Exception, KeyboardInterrupt):
                index["status"] = "failed"
                raise
            finally:
                runner.write_experiment_index(directory, index)
            analyze_and_plot(repository_root, task, directory / "experiment.json")
