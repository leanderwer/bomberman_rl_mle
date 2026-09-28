"""Render the existing tactical plots separately for each Task 4 learner."""

import argparse
from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.task3.plot_experiment import write_plot_document
from evaluation.task4.analyze_experiment import load_index
from evaluation.task4.run_experiment import read_json


def plot_experiment(experiment_file):
    index = load_index(experiment_file)
    directory = Path(experiment_file).resolve().parent
    paths = []
    for learner in index["configuration"]["learners"]:
        name = learner["agent"]
        summary_file = directory / "analysis" / name / "summary.json"
        summary = read_json(summary_file)
        if (summary.get("source", {}).get("configuration_sha256") != index["configuration_sha256"]
                or summary["source"].get("git_commit") != index["git_commit"]
                or summary["source"].get("learner") != name):
            raise ValueError(f"Summary does not belong to this learner/experiment: {summary_file}")
        paths.append(write_plot_document(summary, summary_file))
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_file", type=Path)
    args = parser.parse_args()
    try:
        for path in plot_experiment(args.experiment_file):
            print(f"PDF: {path}")
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise SystemExit(f"Task 4 plotting failed: {error}") from error


if __name__ == "__main__":
    main()
