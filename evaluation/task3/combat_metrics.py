"""Validate and aggregate Task 3 combat outcomes."""

from math import isfinite
from statistics import mean, pstdev


MAX_STEPS_PER_ROUND = 400


EVALUATION_COMBAT_METRICS = (
    "mean_kills_per_round",
    "rounds_with_kill_rate",
    "kills_per_bomb",
    "mean_opponents_eliminated_per_round",
    "any_opponent_eliminated_rate",
    "all_opponents_eliminated_rate",
    "win_rate",
    "score_win_rate",
    "score_draw_rate",
    "score_loss_rate",
    "death_by_opponent_rate",
    "mean_opponent_suicides_per_round",
    "rounds_with_opponent_suicide_rate",
    "mean_first_kill_step",
    "mean_kill_step",
    "mean_kill_auc",
    "mean_score_difference_per_round",
)

TRAINING_COMBAT_METRICS = (
    "mean_kills_per_round",
    "early_mean_kills_per_round",
    "late_mean_kills_per_round",
    "kills_per_round_improvement",
    "all_opponents_eliminated_rate",
    "late_all_opponents_eliminated_rate",
    "death_by_opponent_rate",
    "late_death_by_opponent_rate",
    "mean_kill_auc",
    "late_mean_kill_auc",
)

TRAINING_COMBAT_CURVE_METRICS = (
    "kills",
    "kills_moving_average",
    "all_opponents_eliminated",
    "all_opponents_eliminated_moving_average",
    "death_by_opponent",
    "death_by_opponent_moving_average",
    "kill_auc",
    "kill_auc_moving_average",
)


def opponent_instance_names(opponents: list[str] | tuple[str, ...]) -> list[str]:
    """Map configured agent folders to the engine's distinct instance names."""
    seen = {}
    names = []
    for code_name in opponents:
        index = seen.get(code_name, 0)
        seen[code_name] = index + 1
        names.append(f"{code_name}_{index}" if opponents.count(code_name) > 1
                     else code_name)
    if len(set(names)) != len(names):
        raise ValueError("The configured opponents produce colliding engine names.")
    return names


def require_non_negative_integer(value, location: str) -> int:
    """Return one validated non-negative integer."""
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise ValueError(
            f"'{location}' must be a non-negative integer."
        )

    return value


def require_boolean(value, location: str) -> bool:
    """Return one validated boolean."""
    if not isinstance(value, bool):
        raise ValueError(f"'{location}' must be a boolean.")

    return value


def _validated_agents(
    round_identifier: str,
    round_data: dict,
    agent_name: str,
) -> tuple[dict, tuple[str, ...], list[dict]]:
    """Return the learning agent and its fixed opponent roster."""
    by_agent = round_data.get("by_agent")

    if not isinstance(by_agent, dict):
        raise ValueError(
            f"Missing 'by_agent' object in {round_identifier!r}."
        )

    agent_data = by_agent.get(agent_name)

    if not isinstance(agent_data, dict):
        raise ValueError(
            f"Missing statistics for agent {agent_name!r} "
            f"in {round_identifier!r}."
        )

    opponent_names = tuple(
        name
        for name in by_agent
        if name != agent_name
    )

    if not opponent_names:
        raise ValueError(
            f"Task 3 round {round_identifier!r} contains no opponents."
        )

    opponent_data = []

    for opponent_name in opponent_names:
        statistics = by_agent[opponent_name]

        if not isinstance(statistics, dict):
            raise ValueError(
                f"Statistics for opponent {opponent_name!r} in "
                f"{round_identifier!r} must be an object."
            )

        opponent_data.append(statistics)

    return agent_data, opponent_names, opponent_data


def _validated_kill_events(
    round_identifier: str,
    round_data: dict,
    participant_names: set[str],
) -> list[dict] | None:
    """Return validated kill telemetry, if the result contains it."""
    kill_events = round_data.get("kill_events")

    if kill_events is None:
        return None

    if not isinstance(kill_events, list):
        raise ValueError(
            f"'kill_events' in {round_identifier!r} must be a list."
        )

    validated_events = []

    for event_index, event in enumerate(kill_events):
        location = (
            f"{round_identifier}.kill_events[{event_index}]"
        )

        if not isinstance(event, dict):
            raise ValueError(f"'{location}' must be an object.")

        step = require_non_negative_integer(
            event.get("step"),
            f"{location}.step",
        )
        killer = event.get("killer")
        victim = event.get("victim")

        if step < 1 or step > MAX_STEPS_PER_ROUND:
            raise ValueError(
                f"'{location}.step' must be between 1 and "
                f"{MAX_STEPS_PER_ROUND}."
            )
        if killer not in participant_names:
            raise ValueError(
                f"'{location}.killer' names an unknown participant."
            )
        if victim not in participant_names:
            raise ValueError(
                f"'{location}.victim' names an unknown participant."
            )

        validated_events.append(
            {
                "step": step,
                "killer": killer,
                "victim": victim,
            }
        )

    return validated_events


def combat_round_row(
    round_identifier: str,
    round_data: dict,
    agent_name: str,
) -> dict:
    """Validate and normalize one round's combat outcomes."""
    if not isinstance(round_data, dict):
        raise ValueError(
            f"Round {round_identifier!r} must be an object."
        )

    agent_data, opponent_names, opponent_data = _validated_agents(
        round_identifier,
        round_data,
        agent_name,
    )
    participant_names = {agent_name, *opponent_names}

    kills = require_non_negative_integer(
        agent_data.get("kills", 0),
        f"{round_identifier}.by_agent.{agent_name}.kills",
    )
    bombs = require_non_negative_integer(
        agent_data.get("bombs", 0),
        f"{round_identifier}.by_agent.{agent_name}.bombs",
    )
    suicides = require_non_negative_integer(
        agent_data.get("suicides", 0),
        f"{round_identifier}.by_agent.{agent_name}.suicides",
    )
    survived = require_boolean(
        agent_data.get("survived"),
        f"{round_identifier}.by_agent.{agent_name}.survived",
    )
    score = agent_data.get("score")

    if (
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not isfinite(score)
    ):
        raise ValueError(
            f"'{round_identifier}.by_agent.{agent_name}.score' "
            "must be a finite number."
        )

    opponents_eliminated = 0
    opponent_suicides = 0
    opponent_scores = []

    for opponent_name, statistics in zip(
        opponent_names,
        opponent_data,
    ):
        opponent_survived = require_boolean(
            statistics.get("survived"),
            (
                f"{round_identifier}.by_agent."
                f"{opponent_name}.survived"
            ),
        )
        opponents_eliminated += int(not opponent_survived)
        opponent_suicides += require_non_negative_integer(
            statistics.get("suicides", 0),
            (
                f"{round_identifier}.by_agent."
                f"{opponent_name}.suicides"
            ),
        )
        current_score = statistics.get("score")

        if (
            isinstance(current_score, bool)
            or not isinstance(current_score, (int, float))
            or not isfinite(current_score)
        ):
            raise ValueError(
                f"'{round_identifier}.by_agent."
                f"{opponent_name}.score' must be a finite number."
            )

        opponent_scores.append(float(current_score))

    kill_events = _validated_kill_events(
        round_identifier,
        round_data,
        participant_names,
    )
    if kill_events is None:
        kill_steps = None
        death_by_opponent = (not survived) and suicides == 0
    else:
        first_step_by_victim = {}

        for event in kill_events:
            if (
                event["killer"] == agent_name
                and event["victim"] != agent_name
            ):
                first_step_by_victim.setdefault(
                    event["victim"],
                    event["step"],
                )

        kill_steps = sorted(first_step_by_victim.values())
        death_by_opponent = any(
            event["victim"] == agent_name
            and event["killer"] != agent_name
            for event in kill_events
        )

    opponent_count = len(opponent_names)
    all_opponents_eliminated = (
        opponents_eliminated == opponent_count
    )
    kill_auc = (
        None
        if kill_steps is None
        else sum(
            MAX_STEPS_PER_ROUND - step + 1
            for step in kill_steps
        )
        / (opponent_count * MAX_STEPS_PER_ROUND)
    )

    # Rank the final engine scores, including eliminated participants. A tie
    # for the highest score is a draw; being the last survivor is independent.
    best_opponent_score = max(opponent_scores)
    return {
        "opponents": opponent_names,
        "opponent_count": opponent_count,
        "kills": kills,
        "bombs": bombs,
        "opponents_eliminated": opponents_eliminated,
        "all_opponents_eliminated": all_opponents_eliminated,
        "win": survived and all_opponents_eliminated,
        "score_win": score > best_opponent_score,
        "score_draw": score == best_opponent_score,
        "score_loss": score < best_opponent_score,
        "death_by_opponent": death_by_opponent,
        "opponent_suicides": opponent_suicides,
        "kill_steps": kill_steps,
        "kill_auc": kill_auc,
        "score_difference": float(score) - sum(opponent_scores),
    }


def calculate_combat_metrics(
    round_statistics: dict,
    agent_name: str,
) -> dict:
    """Aggregate combat quality while keeping one fixed opponent roster."""
    if not isinstance(round_statistics, dict) or not round_statistics:
        raise ValueError("'by_round' must be a non-empty object.")

    rows = [
        combat_round_row(identifier, round_data, agent_name)
        for identifier, round_data in round_statistics.items()
    ]
    opponent_names = rows[0]["opponents"]

    for row in rows[1:]:
        if row["opponents"] != opponent_names:
            raise ValueError(
                "Every round in one Task 3 result must use the same "
                "ordered opponent roster."
            )

    number_of_rounds = len(rows)
    total_kills = sum(row["kills"] for row in rows)
    total_bombs = sum(row["bombs"] for row in rows)
    telemetry_rows = [
        row for row in rows if row["kill_steps"] is not None
    ]
    complete_kill_telemetry = (
        len(telemetry_rows) == number_of_rounds
    )

    if complete_kill_telemetry:
        kill_steps = [
            step
            for row in rows
            for step in row["kill_steps"]
        ]
        first_kill_steps = [
            row["kill_steps"][0]
            for row in rows
            if row["kill_steps"]
        ]
        kill_auc_values = [row["kill_auc"] for row in rows]
    else:
        kill_steps = []
        first_kill_steps = []
        kill_auc_values = []

    return {
        "opponents": list(opponent_names),
        "opponent_count": rows[0]["opponent_count"],
        "mean_kills_per_round": total_kills / number_of_rounds,
        "kill_standard_deviation": pstdev(
            row["kills"] for row in rows
        ),
        "rounds_with_kill_rate": (
            sum(row["kills"] > 0 for row in rows)
            / number_of_rounds
        ),
        "kills_per_bomb": (
            total_kills / total_bombs
            if total_bombs
            else 0.0
        ),
        "mean_opponents_eliminated_per_round": mean(
            row["opponents_eliminated"] for row in rows
        ),
        "any_opponent_eliminated_rate": (
            sum(row["opponents_eliminated"] > 0 for row in rows)
            / number_of_rounds
        ),
        "all_opponents_eliminated_rate": (
            sum(row["all_opponents_eliminated"] for row in rows)
            / number_of_rounds
        ),
        "win_rate": sum(row["win"] for row in rows) / number_of_rounds,
        **{
            f"score_{outcome}_rate": mean(row[f"score_{outcome}"] for row in rows)
            for outcome in ("win", "draw", "loss")
        },
        "death_by_opponent_rate": (
            sum(row["death_by_opponent"] for row in rows)
            / number_of_rounds
        ),
        "mean_opponent_suicides_per_round": mean(
            row["opponent_suicides"] for row in rows
        ),
        "rounds_with_opponent_suicide_rate": (
            sum(row["opponent_suicides"] > 0 for row in rows)
            / number_of_rounds
        ),
        "kill_telemetry_rounds": len(telemetry_rows),
        "mean_first_kill_step": (
            mean(first_kill_steps) if first_kill_steps else None
        ),
        "mean_kill_step": mean(kill_steps) if kill_steps else None,
        "mean_kill_auc": (
            mean(kill_auc_values) if kill_auc_values else None
        ),
        "mean_score_difference_per_round": mean(
            row["score_difference"] for row in rows
        ),
    }
