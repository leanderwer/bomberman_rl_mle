"""Analyze each learner separately without duplicating shared training data."""

import argparse
from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.task3 import analyze_experiment as task3
from evaluation.task3.analyze_results import analyze_evaluation_file
from evaluation.task3.analyze_training import calculate_training_metrics
from evaluation.task3.combat_metrics import opponent_instance_names
from evaluation.task3.diagnostics import analyze_evaluation_diagnostics
from evaluation.task4.run_experiment import (
    SCHEMA_VERSION, VERSION_KEYS, configuration_sha256, file_sha256,
    learner_configuration, read_json, result_path, write_json,
)


def load_index(experiment_file):
    index = read_json(experiment_file)
    if index.get("schema_version") != SCHEMA_VERSION or index.get("experiment_mode") != "joint_train_and_evaluate":
        raise ValueError("Not a supported Task 4 experiment manifest.")
    if index.get("status") != "completed":
        raise ValueError("Only fully completed experiments can be analyzed.")
    config = index["configuration"]
    if index.get("configuration_sha256") != configuration_sha256(config):
        raise ValueError("Experiment configuration differs from its recorded SHA-256.")
    if not index.get("runs"):
        raise ValueError("Experiment has no joint runs.")
    return index


def checked_result(directory, record, path_key, hash_key):
    path = result_path(directory, record[path_key])
    if file_sha256(path) != record.get(hash_key):
        raise ValueError(f"Result changed since the experiment completed: {path}")
    return path


def project_learner(index, learner):
    """Present one learner to the existing Task 3 aggregators and diagnostics."""
    original = index["configuration"]
    name = learner["agent"]
    training_opponents = [entry["agent"] for entry in original["learners"] if entry["agent"] != name] + original["training"]["opponents"]
    config = {
        **learner_configuration(original, learner),
        "experiment_name": f"{original['experiment_name']} / {name}",
        "experiment_mode": "train_and_evaluate",
        "description": original.get("description", ""),
        "opponents": original["evaluation"]["opponents"],
        "training_roster": index["training_roster"],
        "training_opponents": training_opponents,
        "evaluation_opponents": original["evaluation"]["opponents"],
        "training": {
            **original["training"], "opponents": training_opponents,
            "runs": [{"world_seed": run["world_seed"], "agent_seed": run["agent_seeds"][name]} for run in original["training"]["runs"]],
        },
        "evaluation": original["evaluation"],
    }
    runs = [{
        **run["learners"][name], "training_world_seed": run["training_world_seed"],
        "training_results": run["training_results"],
        "training_results_sha256": run["training_results_sha256"],
        "model_source": "joint_training",
    } for run in index["runs"]]
    return {**index, "experiment_mode": "train_and_evaluate", "configuration": config, "runs": runs}


def analyze_learner(experiment_file, index, learner, shared_results):
    projected = project_learner(index, learner)
    config = projected["configuration"]
    name = learner["agent"]
    directory = Path(experiment_file).resolve().parent
    analyzed = []
    expected_evaluations = {(run["world_seed"], run["agent_seed"]) for run in config["evaluation"]["runs"]}
    for run in projected["runs"]:
        if run.get("status") != "completed":
            raise ValueError(f"Incomplete learner: {name}.")
        configured_run = next(item for item in index["configuration"]["training"]["runs"]
                              if item["world_seed"] == run["training_world_seed"])
        if run["training_agent_seed"] != configured_run["agent_seeds"][name]:
            raise ValueError(f"Training seed differs from the configuration for {name}.")
        raw = shared_results[run["training_results"]]
        metadata = raw.get("training_metadata_by_agent", {}).get(name)
        if not isinstance(metadata, dict):
            raise ValueError(f"Missing training metadata for {name}.")
        # Shallow projection only; other learners' raw data is never copied to disk.
        metrics = calculate_training_metrics({**raw, "training_metadata": metadata}, name)
        expected_metadata = {key: config[key] for key in VERSION_KEYS}
        expected_metadata.update(world_seed=run["training_world_seed"], agent_seed=run["training_agent_seed"], scenario=config["scenario"])
        if any(metadata.get(key) != value for key, value in expected_metadata.items()):
            raise ValueError(f"Training metadata inconsistent with {name}'s configuration.")
        if metrics["summary"]["rounds"] != config["training"]["rounds"]:
            raise ValueError(f"Incorrect training round count for {name}.")
        if set(metrics["opponents"]) != set(opponent_instance_names(config["training_opponents"])):
            raise ValueError(f"Incorrect training opponents for {name}.")
        if sum(row["q_updates"] for row in raw["training_by_agent"][name]) != run["learning_updates"]:
            raise ValueError("Recorded checkpoint updates differ from the learner's history.")
        evaluations = []
        if len(run["evaluations"]) != len(expected_evaluations) or {
            (entry["evaluation_world_seed"], entry["evaluation_agent_seed"]) for entry in run["evaluations"]
        } != expected_evaluations:
            raise ValueError(f"Incorrect evaluation seeds for {name}.")
        for evaluation in run["evaluations"]:
            if evaluation.get("status") != "completed":
                raise ValueError("An evaluation did not complete.")
            path = checked_result(directory, evaluation, "results", "results_sha256")
            evaluation_metrics = analyze_evaluation_file(path, name, config["coins_per_round"])
            if (evaluation_metrics["rounds"] != config["evaluation"]["rounds_per_seed"]
                    or set(evaluation_metrics["opponents"]) != set(opponent_instance_names(config["opponents"]))):
                raise ValueError(f"Incorrect evaluation rounds/opponents: {path}")
            evaluations.append({
                "evaluation_world_seed": evaluation["evaluation_world_seed"],
                "evaluation_agent_seed": evaluation["evaluation_agent_seed"],
                "results_file": evaluation["results"], "metrics": evaluation_metrics,
            })
        analyzed.append({
            "training_world_seed": run["training_world_seed"],
            "training_agent_seed": run["training_agent_seed"],
            "training_results_file": run["training_results"],
            "training_metrics": metrics, "model": run["model"],
            "model_source": "joint_training", "model_sha256": run["model_sha256"],
            "evaluations": evaluations,
        })
    summary = task3.build_analysis_summary(
        projected, config, analyzed,
        task3.aggregate_evaluation_metrics(analyzed),
        task3.aggregate_evaluations_by_seed(analyzed),
        task3.aggregate_training_metrics(analyzed),
        task3.aggregate_training_curves(analyzed),
        task3.aggregate_reward_components(analyzed),
        task3.aggregate_action_counts(analyzed),
    )
    summary["source"].update(
        task=4, configuration_sha256=index["configuration_sha256"],
        experiment_index="../../experiment.json", learner=name,
        joint_experiment_name=index["configuration"]["experiment_name"],
    )
    summary["experiment"].update(
        training_mode="joint_training", joint_training_runs=len(index["runs"]),
        training_roster=index["training_roster"], evaluation_opponents=config["opponents"],
        training_wall_time_scope="whole_shared_match_not_individual_agent_cpu_time",
    )
    for row in summary["runs"]:
        row.update(training_opponents=config["training_opponents"], evaluation_opponents=config["opponents"])
    summary["evaluation_diagnostics"] = analyze_evaluation_diagnostics(experiment_file, projected)
    return summary


def analyze_experiment(experiment_file):
    index = load_index(experiment_file)
    directory = Path(experiment_file).resolve().parent
    # The large shared history is read once and reused across the two summaries.
    shared_results = {}
    expected_worlds = {entry["world_seed"] for entry in index["configuration"]["training"]["runs"]}
    if len(index["runs"]) != len(expected_worlds) or {run["training_world_seed"] for run in index["runs"]} != expected_worlds:
        raise ValueError("Wrong joint training seed runs.")
    for run in index["runs"]:
        path = checked_result(directory, run, "training_results", "training_results_sha256")
        shared_results[run["training_results"]] = read_json(path)
    paths = []
    for learner in index["configuration"]["learners"]:
        summary = analyze_learner(experiment_file, index, learner, shared_results)
        path = directory / "analysis" / learner["agent"] / "summary.json"
        write_json(path, summary)
        paths.append(path)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_file", type=Path)
    args = parser.parse_args()
    try:
        for path in analyze_experiment(args.experiment_file):
            print(f"Summary: {path}")
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise SystemExit(f"Task 4 analysis failed: {error}") from error


if __name__ == "__main__":
    main()
