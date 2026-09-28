"""Train two independent learners in one game, then evaluate each separately."""

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.agent_registry import get_agent_registry, validate_agent_versions
from evaluation.task3.combat_metrics import opponent_instance_names
from evaluation.task3.run_experiment import get_clean_git_commit


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
VERSION_KEYS = (
    "feature_version", "reward_version", "action_space_version",
    "learning_rate_version", "exploration_version", "discount_factor_version",
    "td_target_version",
)
ACTION_ORDER = ("UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB")
SCHEMA_VERSION = 1


def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def configuration_sha256(configuration):
    return hashlib.sha256(json.dumps(configuration, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def require_integer(value, location, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{location} must be an integer >= {minimum}.")
    return value


def learner_configuration(configuration, learner):
    return {**{key: configuration[key] for key in (
        "feature_version", "reward_version", "scenario", "coins_per_round",
    )}, **learner}


def validate_configuration(configuration, repository_root=REPOSITORY_ROOT):
    """Validate every agent's actual registry before creating output files."""
    config = deepcopy(configuration)
    name = config.get("experiment_name")
    if not isinstance(name, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name) is None:
        raise ValueError("experiment_name must be a simple directory name.")
    if config.get("experiment_mode", "joint_train_and_evaluate") != "joint_train_and_evaluate":
        raise ValueError("Task 4 supports joint_train_and_evaluate; use Task 3 for existing checkpoints.")
    config["experiment_mode"] = "joint_train_and_evaluate"
    for key in ("feature_version", "reward_version"):
        if not isinstance(config.get(key), str) or not config[key].strip():
            raise ValueError(f"Missing explicit shared default: {key}.")
    if "checkpoint" in config or config.get("resume", False):
        raise ValueError("Task 4 starts fresh models; resuming checkpoints is not supported.")
    from settings import SCENARIOS
    if config.get("scenario") not in SCENARIOS:
        raise ValueError("Unknown scenario.")
    actual_coins = SCENARIOS[config["scenario"]]["COIN_COUNT"]
    config.setdefault("coins_per_round", actual_coins)
    if require_integer(config["coins_per_round"], "coins_per_round", 1) != actual_coins:
        raise ValueError("coins_per_round differs from the scenario.")
    learners = config.get("learners")
    if not isinstance(learners, list) or len(learners) != 2:
        raise ValueError("Exactly two learners are required.")
    names = []
    for learner in learners:
        if not isinstance(learner, dict):
            raise ValueError("Each learner must be an object.")
        unknown = set(learner) - set(VERSION_KEYS) - {"agent", "checkpoint_interval", "description"}
        if unknown:
            raise ValueError(f"Unknown learner settings: {sorted(unknown)}. Model paths are generated; the scenario is shared.")
        if not isinstance(learner.get("agent"), str) or not learner["agent"]:
            raise ValueError("Each learner needs an agent folder name.")
        if "checkpoint_interval" in learner:
            require_integer(learner["checkpoint_interval"], "learner.checkpoint_interval", 1)
        effective = learner_configuration(config, learner)
        for key in VERSION_KEYS:
            if not isinstance(effective.get(key), str) or not effective[key]:
                raise ValueError(f"Missing explicit learner setting: {key}.")
        registry = validate_agent_versions(effective, repository_root)
        if tuple(registry.get_learning_actions(effective["action_space_version"])) != ACTION_ORDER:
            raise ValueError("Task 4 requires all six learning actions.")
        names.append(learner["agent"])
    if len(set(names)) != 2:
        raise ValueError("Learners must use distinct agent folders so engine identities remain unambiguous.")
    for phase in ("training", "evaluation"):
        section = config.get(phase)
        if not isinstance(section, dict):
            raise ValueError(f"Missing {phase} object.")
        opponents = section.get("opponents")
        expected_count = 2 if phase == "training" else 3
        if not isinstance(opponents, list) or len(opponents) != expected_count:
            raise ValueError(f"{phase}.opponents must contain {expected_count} fixed agents.")
        for opponent in opponents:
            if (not isinstance(opponent, str)
                    or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", opponent) is None
                    or not (Path(repository_root) / "agent_code" / opponent / "callbacks.py").is_file()
                    or opponent in names):
                raise ValueError(f"Invalid fixed opponent: {opponent!r}.")
        instance_names = opponent_instance_names(opponents)
        if set(instance_names).intersection(names):
            raise ValueError("A fixed opponent collides with a learner's instance name.")
        rounds_key = "rounds" if phase == "training" else "rounds_per_seed"
        require_integer(section.get(rounds_key), f"{phase}.{rounds_key}", 1)
        runs = section.get("runs")
        if not isinstance(runs, list) or not runs:
            raise ValueError(f"{phase}.runs must be a non-empty list.")
        world_seeds = []
        for run in runs:
            if not isinstance(run, dict):
                raise ValueError(f"{phase} seed runs must be objects.")
            world_seeds.append(require_integer(run.get("world_seed"), f"{phase}.world_seed"))
            if phase == "training":
                seeds = run.get("agent_seeds")
                if not isinstance(seeds, dict) or set(seeds) != set(names):
                    raise ValueError("Every training run needs one agent_seeds entry for each learner.")
                for seed in seeds.values():
                    require_integer(seed, "training.agent_seeds")
                if len(set(seeds.values())) != len(seeds):
                    raise ValueError("Use distinct seeds for the two training learners.")
            else:
                require_integer(run.get("agent_seed"), "evaluation.agent_seed")
        if len(set(world_seeds)) != len(world_seeds):
            raise ValueError(f"{phase}.runs has duplicate world seeds.")
    if {run["world_seed"] for run in config["training"]["runs"]}.intersection(
        run["world_seed"] for run in config["evaluation"]["runs"]
    ):
        raise ValueError("Training and evaluation world seeds must be disjoint.")
    config["training"].setdefault("checkpoint_interval", 100)
    require_integer(config["training"]["checkpoint_interval"], "training.checkpoint_interval", 1)
    return config


def opponent_seed_map(opponents, world_seed, seed=0):
    """Fixed, explicit independent RNG streams, identical across compared models."""
    return {name: {"BOMBERMAN_AGENT_SEED": str(int.from_bytes(
        hashlib.sha256(f"task4:{world_seed}:{seed}:{name}".encode()).digest()[:4], "big"
    ))} for name in opponent_instance_names(opponents)}


def learner_environment(configuration, learner, seed, model_file):
    effective = learner_configuration(configuration, learner)
    return {
        **{f"BOMBERMAN_{key.upper()}": effective[key] for key in VERSION_KEYS},
        "BOMBERMAN_AGENT_SEED": str(seed),
        "BOMBERMAN_MODEL_FILE": str(Path(model_file).resolve()),
        "BOMBERMAN_CHECKPOINT_INTERVAL": str(learner.get("checkpoint_interval", configuration["training"]["checkpoint_interval"])),
        "BOMBERMAN_TRAINING_ROUNDS": str(configuration["training"]["rounds"]),
    }


def build_environment(configuration, agent_configurations, diagnostic_agent=None):
    # Do not inherit a stale model path, learner version or observer from a shell.
    environment = {key: value for key, value in os.environ.items() if not key.startswith("BOMBERMAN_")}
    environment.update({
        "BOMBERMAN_AGENT_CONFIGS": json.dumps(agent_configurations, sort_keys=True),
        "BOMBERMAN_TASK3_DIAGNOSTICS": "1" if diagnostic_agent else "0",
        "BOMBERMAN_COMPACT_TRAINING_STATS": "1",
        "BOMBERMAN_FEATURE_VERSION": configuration["feature_version"],
        "BOMBERMAN_GAME_LOG_LEVEL": "WARNING",
        "BOMBERMAN_AGENT_WRAPPER_LOG_LEVEL": "WARNING",
        "BOMBERMAN_AGENT_CODE_LOG_LEVEL": "WARNING",
    })
    if diagnostic_agent:
        environment["BOMBERMAN_DIAGNOSTIC_AGENT"] = diagnostic_agent
        environment["BOMBERMAN_FEATURE_VERSION"] = agent_configurations[diagnostic_agent]["BOMBERMAN_FEATURE_VERSION"]
    return environment


def build_index(configuration, commit):
    runs = []
    for seed_run in configuration["training"]["runs"]:
        run_id = f"train_w{seed_run['world_seed']}"
        learners = {}
        for learner in configuration["learners"]:
            name = learner["agent"]
            seed = seed_run["agent_seeds"][name]
            learners[name] = {
                "training_agent_seed": seed,
                "model": f"models/{name}/{run_id}_a{seed}.pt",
                "status": "pending",
                "evaluations": [{
                    "evaluation_world_seed": evaluation["world_seed"],
                    "evaluation_agent_seed": evaluation["agent_seed"],
                    "results": f"evaluation/{name}/{run_id}/eval_w{evaluation['world_seed']}_a{evaluation['agent_seed']}.json",
                    "opponent_seeds": opponent_seed_map(configuration["evaluation"]["opponents"], evaluation["world_seed"], evaluation["agent_seed"]),
                    "status": "pending",
                } for evaluation in configuration["evaluation"]["runs"]],
            }
        runs.append({
            "training_world_seed": seed_run["world_seed"],
            "training_results": f"training/{run_id}.json",
            "opponent_seeds": opponent_seed_map(configuration["training"]["opponents"], seed_run["world_seed"]),
            "status": "pending", "learners": learners,
        })
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_mode": "joint_train_and_evaluate",
        "status": "running", "git_commit": commit,
        "configuration_sha256": configuration_sha256(configuration),
        "configuration": configuration,
        "training_initialization": "fresh_q_tables_and_update_counts",
        "continue_without_training": True,
        "training_roster": [entry["agent"] for entry in configuration["learners"]] + configuration["training"]["opponents"],
        "evaluation_opponents": configuration["evaluation"]["opponents"],
        "runs": runs,
    }


def result_path(directory, relative):
    directory = Path(directory).resolve()
    path = (directory / relative).resolve()
    if not path.is_relative_to(directory):
        raise ValueError("Artifact path escapes experiment directory.")
    return path


def run_game(repository_root, configuration, roster, train, rounds, world_seed, result, logs, environment):
    result.parent.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(repository_root / "main.py"), "play", "--agents", *roster,
               "--train", str(train), "--scenario", configuration["scenario"], "--n-rounds", str(rounds),
               "--seed", str(world_seed), "--no-gui", "--save-stats", str(result), "--log-dir", str(logs)]
    if train:
        # Let the fixed opponents finish after both learners die; training wins
        # then use the actual final scoreboard rather than an interrupted game.
        command.append("--continue-without-training")
    subprocess.run(command, cwd=repository_root, env=environment, check=True)
    raw = read_json(result)
    if len(raw.get("by_round", {})) != rounds or set(raw.get("by_agent", {})) != set(opponent_instance_names(roster)):
        raise ValueError(f"Game output has wrong round count or roster: {result}")
    return raw


def verify_checkpoint(model_file, configuration, learner, expected_seed, rounds, training_rows, repository_root=REPOSITORY_ROOT):
    effective = learner_configuration(configuration, learner)
    registry = get_agent_registry(learner["agent"], repository_root)
    q, metadata, version, legacy = registry.load_checkpoint(
        str(model_file), ACTION_ORDER, expected_feature_version=effective["feature_version"], include_update_counts=True,
    )
    if legacy or version != effective["feature_version"] or not q:
        raise ValueError("Training produced an empty, legacy or incompatible checkpoint.")
    for key in VERSION_KEYS[1:]:
        if metadata.get(key) != effective[key]:
            raise ValueError(f"Checkpoint mismatch for {learner['agent']}: {key}.")
    expected_parameters = {
        "learning_rate_parameters": registry.get_learning_rate_parameters(effective["learning_rate_version"]),
        "exploration_parameters": registry.get_exploration_parameters(effective["exploration_version"]),
        "td_target_parameters": registry.get_td_target_parameters(effective["td_target_version"]),
        "gamma": registry.get_discount_factor(effective["discount_factor_version"]),
        "agent_seed": expected_seed, "rounds_completed": rounds,
    }
    for key, expected in expected_parameters.items():
        if metadata.get(key) != expected:
            raise ValueError(f"Checkpoint mismatch for {learner['agent']}: {key}.")
    counts = metadata.get("state_action_update_counts")
    if counts is None or sum(counts.values()) != sum(row["q_updates"] for row in training_rows):
        raise ValueError("Checkpoint update counts do not match this learner's complete training history.")
    return {"model_sha256": file_sha256(model_file), "q_states": len(q), "learning_updates": sum(counts.values())}


def run_experiment(configuration, repository_root=REPOSITORY_ROOT):
    configuration = validate_configuration(configuration, repository_root)
    commit = get_clean_git_commit(repository_root)
    directory = repository_root / "results" / "task4" / "experiments" / configuration["experiment_name"]
    # A failed run also remains intact. A new experiment name is required to retry.
    directory.mkdir(parents=True, exist_ok=False)
    manifest = directory / "experiment.json"
    index = build_index(configuration, commit)
    write_json(manifest, index)
    try:
        for run in index["runs"]:
            agent_configs = dict(run["opponent_seeds"])
            for learner in configuration["learners"]:
                record = run["learners"][learner["agent"]]
                model_file = result_path(directory, record["model"])
                model_file.parent.mkdir(parents=True, exist_ok=True)
                agent_configs[learner["agent"]] = learner_environment(configuration, learner, record["training_agent_seed"], model_file)
            run["status"] = "training"
            write_json(manifest, index)
            print(f"Joint training: world={run['training_world_seed']}, rounds={configuration['training']['rounds']}", flush=True)
            raw = run_game(repository_root, configuration, index["training_roster"], 2,
                           configuration["training"]["rounds"], run["training_world_seed"],
                           result_path(directory, run["training_results"]), directory / "logs" / f"train_w{run['training_world_seed']}",
                           build_environment(configuration, agent_configs))
            metadata = raw.get("training_metadata_by_agent", {})
            histories = raw.get("training_by_agent", {})
            for learner in configuration["learners"]:
                name = learner["agent"]
                record = run["learners"][name]
                if name not in metadata or len(histories.get(name, [])) != configuration["training"]["rounds"]:
                    raise ValueError(f"Missing independent training metadata/history for {name}.")
                effective = learner_configuration(configuration, learner)
                expected = {**{key: effective[key] for key in VERSION_KEYS}, "agent_seed": record["training_agent_seed"], "world_seed": run["training_world_seed"]}
                if any(metadata[name].get(key) != value for key, value in expected.items()):
                    raise ValueError(f"Wrong independent training settings for {name}.")
                record.update(verify_checkpoint(result_path(directory, record["model"]), configuration, learner,
                                                record["training_agent_seed"], configuration["training"]["rounds"], histories[name], repository_root))
                record["status"] = "trained"
            run["training_results_sha256"] = file_sha256(result_path(directory, run["training_results"]))
            run["status"] = "trained"
            write_json(manifest, index)
        for run in index["runs"]:
            for learner in configuration["learners"]:
                name = learner["agent"]
                record = run["learners"][name]
                model = result_path(directory, record["model"])
                for evaluation in record["evaluations"]:
                    if file_sha256(model) != record["model_sha256"]:
                        raise ValueError("A trained checkpoint changed before evaluation.")
                    agent_configs = dict(evaluation["opponent_seeds"])
                    agent_configs[name] = learner_environment(configuration, learner, evaluation["evaluation_agent_seed"], model)
                    evaluation["status"] = "running"
                    write_json(manifest, index)
                    print(f"Evaluation: {name}, train world={run['training_world_seed']}, eval world={evaluation['evaluation_world_seed']}", flush=True)
                    result = result_path(directory, evaluation["results"])
                    run_game(repository_root, configuration, [name] + configuration["evaluation"]["opponents"], 0,
                             configuration["evaluation"]["rounds_per_seed"], evaluation["evaluation_world_seed"], result,
                             directory / "logs" / name / f"train_w{run['training_world_seed']}" / result.stem,
                             build_environment(configuration, agent_configs, name))
                    if file_sha256(model) != record["model_sha256"]:
                        raise ValueError("Evaluation modified the trained checkpoint.")
                    evaluation.update(status="completed", results_sha256=file_sha256(result))
                    write_json(manifest, index)
                record["status"] = "completed"
            run["status"] = "completed"
        index["status"] = "completed"
        write_json(manifest, index)
    except Exception as error:
        index.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(manifest, index)
        raise
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    try:
        configuration = validate_configuration(read_json(args.config))
        print(f"Configuration valid: {configuration['experiment_name']}")
        for learner in configuration["learners"]:
            print(f"  {learner['agent']}: {learner['td_target_version']} / {learner['exploration_version']}")
        print(f"Fresh joint training: {configuration['training']['rounds']} rounds per world seed; separate evaluation against {configuration['evaluation']['opponents']}.")
        if not args.validate_only:
            print(f"Experiment completed: {run_experiment(configuration)}")
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"Task 4 experiment failed: {error}") from error


if __name__ == "__main__":
    main()
