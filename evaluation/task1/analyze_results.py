import argparse
import json
from pathlib import Path
from statistics import mean, pstdev

AGENT_NAME = "q_learning_agent"
DEFAULT_COINS_PER_ROUND = 50
MAX_STEPS_PER_ROUND = 400
DECISION_TIME_LIMIT_SECONDS = 0.5

def load_results(file_path: Path) -> dict:
    """
    Load one JSON result file created by the Bomberman framework.

    :param file_path: Path to the result file.
    :return: The parsed result dictionary.
    """
    with file_path.open("r", encoding="utf-8") as file:
        return json.load(file)

def calculate_metrics(
    agent_statistics: dict,
    round_statistics: dict,
    agent_name: str = AGENT_NAME,
    coins_per_round: int = DEFAULT_COINS_PER_ROUND,
) -> dict:
    """
    Calculate the central Task 1 evaluation metrics.

    :param agent_statistics: Aggregated statistics of the agent.
    :param round_statistics: Individual statistics of all rounds.
    :param coins_per_round: Coins available in each evaluated round.
    :return: A dictionary containing the calculated metrics.
    """
    if not isinstance(coins_per_round, int) or coins_per_round <= 0:
        raise ValueError("'coins_per_round' must be a positive integer.")

    coins_collected_per_round = [
        round_data.get("coins", 0)
        for round_data in round_statistics.values()
    ]

    steps_per_round = [
    round_data.get("steps", 0)
    for round_data in round_statistics.values()
    ]

    completed_round_steps = [
        round_data.get("steps", 0)
        for round_data in round_statistics.values()
        if round_data.get("coins", 0) == coins_per_round
    ]

    coin_auc_per_round = []
    auc_data_complete = True

    decision_times = []
    decision_time_data_complete = True

    for round_data in round_statistics.values():
        agent_round_data = round_data.get(
            "by_agent",
            {},
        ).get(agent_name)

        if agent_round_data is None:
            auc_data_complete = False
            decision_time_data_complete = False
            continue

        if "coin_collection_steps" in agent_round_data:
            coin_auc_per_round.append(
                calculate_coin_auc(
                    agent_round_data[
                        "coin_collection_steps"
                    ],
                    coins_per_round=coins_per_round,
                )
            )
        else:
            auc_data_complete = False

        if "decision_times" in agent_round_data:
            decision_times.extend(
                agent_round_data["decision_times"]
            )
        else:
            decision_time_data_complete = False

    number_of_rounds = len(coins_collected_per_round)
    total_coins = agent_statistics.get("coins", 0)
    total_steps = agent_statistics.get("steps", 0)
    invalid_actions = agent_statistics.get("invalid", 0)
    total_decision_time = agent_statistics.get("time", 0.0)

    if number_of_rounds > 0:
        mean_coins = mean(coins_collected_per_round)
        coin_standard_deviation = pstdev(
            coins_collected_per_round
        )

        completed_rounds = sum(
            coins == coins_per_round
            for coins in coins_collected_per_round
        )

        completion_rate = completed_rounds / number_of_rounds

        coin_collection_rate = (
            total_coins
            / (coins_per_round * number_of_rounds)
        )

        mean_steps_per_round = mean(steps_per_round)
    else:
        mean_coins = 0.0
        coin_standard_deviation = 0.0
        completion_rate = 0.0
        coin_collection_rate = 0.0
        mean_steps_per_round = 0.0

    if completed_round_steps:
        mean_steps_to_completion = mean(
            completed_round_steps
        )
    else:
        mean_steps_to_completion = None

    if (
        number_of_rounds > 0
        and auc_data_complete
        and len(coin_auc_per_round) == number_of_rounds
    ):
        mean_coin_auc = mean(coin_auc_per_round)

        coin_auc_standard_deviation = pstdev(
            coin_auc_per_round
        )
    else:
        mean_coin_auc = None
        coin_auc_standard_deviation = None

    if (
        number_of_rounds > 0
        and decision_time_data_complete
        and decision_times
    ):
        decision_time_p95_ms = (
            1000
            * calculate_percentile(
                decision_times,
                0.95,
            )
        )

        maximum_decision_time_ms = (
            1000 * max(decision_times)
        )

        timeout_rate = (
            sum(
                decision_time
                > DECISION_TIME_LIMIT_SECONDS
                for decision_time in decision_times
            )
            / len(decision_times)
        )
    else:
        decision_time_p95_ms = None
        maximum_decision_time_ms = None
        timeout_rate = None

    if total_steps > 0:
        invalid_action_rate = invalid_actions / total_steps
        coins_per_100_steps = 100 * total_coins / total_steps

        mean_decision_time_ms = (
            1000 * total_decision_time / total_steps
        )
    else:
        invalid_action_rate = 0.0
        coins_per_100_steps = 0.0
        mean_decision_time_ms = 0.0

    return {
        "rounds": number_of_rounds,
        "coins_per_round_target": coins_per_round,
        "mean_coins_per_round": mean_coins,
        "coin_standard_deviation": coin_standard_deviation,
        "coin_collection_rate": coin_collection_rate,
        "completion_rate": completion_rate,
        "mean_steps_per_round": mean_steps_per_round,
        "mean_steps_to_completion": mean_steps_to_completion,
        "invalid_action_rate": invalid_action_rate,
        "coins_per_100_steps": coins_per_100_steps,
        "mean_decision_time_ms": mean_decision_time_ms,
        "decision_time_p95_ms": decision_time_p95_ms,
        "maximum_decision_time_ms": maximum_decision_time_ms,
        "timeout_rate": timeout_rate,
        "mean_coin_auc": mean_coin_auc,
        "coin_auc_standard_deviation": (
            coin_auc_standard_deviation
        ),
    }

def analyze_evaluation_file(
    file_path: Path,
    agent_name: str = AGENT_NAME,
    coins_per_round: int = DEFAULT_COINS_PER_ROUND,
) -> dict:
    """Load and analyze one evaluation statistics file."""
    results = load_results(file_path)

    if not isinstance(results, dict):
        raise ValueError(
            "The top-level JSON value must be an object."
        )

    by_agent = results.get("by_agent")
    round_statistics = results.get("by_round")

    if not isinstance(by_agent, dict):
        raise ValueError(
            "The file does not contain 'by_agent'."
        )

    if not isinstance(round_statistics, dict):
        raise ValueError(
            "The file does not contain 'by_round'."
        )

    agent_statistics = by_agent.get(agent_name)

    if not isinstance(agent_statistics, dict):
        raise ValueError(
            f"No statistics found for agent {agent_name!r}."
        )

    return calculate_metrics(
        agent_statistics,
        round_statistics,
        agent_name,
        coins_per_round,
    )

def calculate_coin_auc(
    coin_collection_steps: list[int],
    coins_per_round: int = DEFAULT_COINS_PER_ROUND,
) -> float:
    """
    Calculate the normalized area under the cumulative
    coin-collection curve for one round.

    A coin collected early contributes more than a coin
    collected late.

    :param coin_collection_steps:
        Steps at which coins were collected.
    :param coins_per_round:
        Coins available in the round.
    :return:
        Normalized coin AUC between 0 and 1.
    """
    if not isinstance(coins_per_round, int) or coins_per_round <= 0:
        raise ValueError("'coins_per_round' must be a positive integer.")

    if len(coin_collection_steps) > coins_per_round:
        raise ValueError(
            "More coin collection steps than available coins."
        )

    if coin_collection_steps != sorted(
        coin_collection_steps
    ):
        raise ValueError(
            "Coin collection steps are not chronological."
        )

    for step in coin_collection_steps:
        if not 1 <= step <= MAX_STEPS_PER_ROUND:
            raise ValueError(
                f"Invalid coin collection step: {step}"
            )

    accumulated_coin_time = sum(
        MAX_STEPS_PER_ROUND - step + 1
        for step in coin_collection_steps
    )

    maximum_coin_time = (
        coins_per_round * MAX_STEPS_PER_ROUND
    )

    return accumulated_coin_time / maximum_coin_time

def calculate_percentile(
    values: list[float],
    quantile: float,
) -> float:
    """
    Calculate a linearly interpolated quantile.

    :param values:
        Numerical observations.
    :param quantile:
        Requested quantile between 0 and 1.
    :return:
        The interpolated quantile value.
    """
    if not values:
        raise ValueError(
            "Cannot calculate a percentile of empty data."
        )

    if not 0 <= quantile <= 1:
        raise ValueError(
            "Quantile must be between 0 and 1."
        )

    sorted_values = sorted(values)

    position = (
        len(sorted_values) - 1
    ) * quantile

    lower_index = int(position)

    upper_index = min(
        lower_index + 1,
        len(sorted_values) - 1,
    )

    interpolation_weight = position - lower_index

    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]

    return (
        lower_value
        + interpolation_weight
        * (upper_value - lower_value)
    )

def parse_arguments():
    """
    Parse the path of the result file from the command line.
    """
    parser = argparse.ArgumentParser(
        description="Analyze one Task 1 evaluation result."
    )

    parser.add_argument(
        "result_file",
        type=Path,
        help="JSON result file created with --save-stats.",
    )

    parser.add_argument(
        "--coins-per-round",
        type=int,
        default=DEFAULT_COINS_PER_ROUND,
        help=(
            "Number of coins available in each round "
            f"(default: {DEFAULT_COINS_PER_ROUND})."
        ),
    )

    return parser.parse_args()

def main():
    """
    Load one result file and inspect its basic structure.
    """
    args = parse_arguments()
    metrics = analyze_evaluation_file(
        args.result_file,
        coins_per_round=args.coins_per_round,
    )

    print(f"Rounds: {metrics['rounds']}")

    print(
        "Mean coins per round: "
        f"{metrics['mean_coins_per_round']:.2f}"
    )

    print(
        "Coin standard deviation: "
        f"{metrics['coin_standard_deviation']:.2f}"
    )

    print(
        "Coin collection rate: "
        f"{100 * metrics['coin_collection_rate']:.2f}%"
    )

    print(
        "Completion rate: "
        f"{100 * metrics['completion_rate']:.2f}%"
    )

    print(
        "Mean steps per round: "
        f"{metrics['mean_steps_per_round']:.2f}"
    )

    if metrics["mean_steps_to_completion"] is None:
        print("Mean steps to completion: N/A")
    else:
        print(
            "Mean steps to completion: "
            f"{metrics['mean_steps_to_completion']:.2f}"
        )

    if metrics["mean_coin_auc"] is None:
        print("Mean normalized coin AUC: N/A")
        print("Coin AUC standard deviation: N/A")
    else:
        print(
            "Mean normalized coin AUC: "
            f"{100 * metrics['mean_coin_auc']:.2f}%"
        )

        print(
            "Coin AUC standard deviation: "
            f"{100 * metrics['coin_auc_standard_deviation']:.2f} "
            "percentage points"
        )

    print(
        "Invalid action rate: "
        f"{100 * metrics['invalid_action_rate']:.2f}%"
    )

    print(
        "Coins per 100 steps: "
        f"{metrics['coins_per_100_steps']:.2f}"
    )

    print(
        "Mean decision time: "
        f"{metrics['mean_decision_time_ms']:.3f} ms"
    )

    if metrics["decision_time_p95_ms"] is None:
        print("Decision time p95: N/A")
        print("Maximum decision time: N/A")
        print("Timeout rate: N/A")
    else:
        print(
            "Decision time p95: "
            f"{metrics['decision_time_p95_ms']:.3f} ms"
        )

        print(
            "Maximum decision time: "
            f"{metrics['maximum_decision_time_ms']:.3f} ms"
        )

        print(
            "Timeout rate: "
            f"{100 * metrics['timeout_rate']:.2f}%"
        )

if __name__ == "__main__":
    main()
