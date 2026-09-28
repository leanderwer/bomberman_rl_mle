from pathlib import Path
from numbers import Integral
import os
import pickle
from typing import Any

import numpy as np


CHECKPOINT_KIND = "tabular_q_learning"
CHECKPOINT_SCHEMA_VERSION = 1
STATE_ACTION_UPDATE_COUNTS = "state_action_update_counts"


def _validate_update_counts(counts: Any, q_table: dict, action_count: int) -> dict:
    """Validate sparse counts keyed by (canonical Q-state, action index)."""
    if not isinstance(counts, dict):
        raise ValueError("State-action update counts must be a dictionary.")
    validated = {}
    for pair, count in counts.items():
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise ValueError("Update-count keys must be (state, action_index) pairs.")
        state, action_index = pair
        if not isinstance(state, tuple) or state not in q_table:
            raise ValueError("Update-count states must exist in the Q-table.")
        if (isinstance(action_index, (bool, np.bool_))
                or not isinstance(action_index, Integral)
                or not 0 <= action_index < action_count):
            raise ValueError("Update-count action indices must match action_order.")
        if (isinstance(count, (bool, np.bool_))
                or not isinstance(count, Integral) or count < 0):
            raise ValueError("Update counts must be non-negative integers.")
        validated[(state, int(action_index))] = int(count)
    return validated


def _validate_q_table(
    q_table: Any,
    action_order: tuple[str, ...],
) -> dict:
    """
    Validate and normalize a tabular Q-learning model.
    """
    if not isinstance(q_table, dict):
        raise ValueError("The stored Q-table must be a dictionary.")

    expected_shape = (len(action_order),)
    validated_q_table = {}

    for state, q_values in q_table.items():
        if not isinstance(state, tuple):
            raise ValueError(
                f"Invalid Q-table state {state!r}: expected a tuple."
            )

        values = np.asarray(q_values)

        if values.shape != expected_shape:
            raise ValueError(
                f"Invalid Q-values for state {state!r}: "
                f"expected shape {expected_shape}, got {values.shape}."
            )

        if not np.issubdtype(values.dtype, np.number):
            raise ValueError(
                f"Invalid Q-values for state {state!r}: "
                "values must be numeric."
            )

        values = values.astype(float, copy=False)

        if not np.all(np.isfinite(values)):
            raise ValueError(
                f"Invalid Q-values for state {state!r}: "
                "values must be finite."
            )

        validated_q_table[state] = values

    return validated_q_table


def load_checkpoint(
    file_path: str,
    expected_action_order: tuple[str, ...],
    expected_feature_version: str | None = None,
    legacy_feature_version: str | None = None,
    *,
    include_update_counts: bool = False,
) -> tuple[dict, dict, str, bool]:
    """
    Load and validate a Q-learning checkpoint.

    If expected_feature_version is provided, the stored version must
    match it. Otherwise, the stored feature version is returned and
    can be used to select the correct feature extractor.

    Legacy bare Q-tables do not contain a feature version. Their
    version must therefore be supplied explicitly.

    With include_update_counts=True, metadata["state_action_update_counts"]
    contains the validated sparse mapping (state, action_index) -> updates.
    A missing pair means zero ONLY when that mapping is present. Old files
    return None for the mapping: their historical counts are unknown.
    Counts are stored separately from training_metadata, so ordinary callers
    keep the existing JSON-compatible metadata and four-item return value.

    :return:
        Q-table, training metadata, feature version, legacy flag.
    """
    path = Path(file_path)

    with path.open("rb") as file:
        payload = pickle.load(file)

    expected_actions = tuple(expected_action_order)

    # Backward compatibility with bare Q-table dictionaries.
    if isinstance(payload, dict) and "schema_version" not in payload:
        if legacy_feature_version is None:
            raise ValueError(
                "The model is a legacy Q-table without feature "
                "metadata. A legacy feature version must be supplied."
            )

        if (
            expected_feature_version is not None
            and expected_feature_version != legacy_feature_version
        ):
            raise ValueError(
                "Feature version mismatch for legacy model: "
                f"expected {expected_feature_version!r}, "
                f"legacy model was assigned "
                f"{legacy_feature_version!r}."
            )

        q_table = _validate_q_table(payload, expected_actions)

        return (
            q_table,
            {STATE_ACTION_UPDATE_COUNTS: None} if include_update_counts else {},
            legacy_feature_version,
            True,
        )

    if not isinstance(payload, dict):
        raise ValueError(
            "The model file does not contain a valid checkpoint."
        )

    if payload.get("kind") != CHECKPOINT_KIND:
        raise ValueError(
            f"Unsupported checkpoint kind: "
            f"{payload.get('kind')!r}."
        )

    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            "Checkpoint schema mismatch: "
            f"expected {CHECKPOINT_SCHEMA_VERSION}, "
            f"got {payload.get('schema_version')!r}."
        )

    saved_feature_version = payload.get("feature_version")

    if (
        not isinstance(saved_feature_version, str)
        or not saved_feature_version
    ):
        raise ValueError(
            "Checkpoint feature version must be a non-empty string."
        )

    if (
        expected_feature_version is not None
        and saved_feature_version != expected_feature_version
    ):
        raise ValueError(
            "Feature version mismatch: "
            f"expected {expected_feature_version!r}, "
            f"got {saved_feature_version!r}."
        )

    saved_action_order = tuple(
        payload.get("action_order", ())
    )

    if saved_action_order != expected_actions:
        raise ValueError(
            "Action order mismatch: "
            f"expected {expected_actions!r}, "
            f"got {saved_action_order!r}."
        )

    metadata = payload.get("training_metadata", {})

    if not isinstance(metadata, dict):
        raise ValueError(
            "Checkpoint training metadata must be a dictionary."
        )

    q_table = _validate_q_table(
        payload.get("q_table"),
        expected_actions,
    )

    if include_update_counts:
        metadata = dict(metadata)
        metadata[STATE_ACTION_UPDATE_COUNTS] = (
            _validate_update_counts(
                payload[STATE_ACTION_UPDATE_COUNTS], q_table, len(expected_actions)
            )
            if STATE_ACTION_UPDATE_COUNTS in payload else None
        )

    return (
        q_table,
        metadata,
        saved_feature_version,
        False,
    )


def save_checkpoint(
    file_path: str,
    q_table: dict,
    feature_version: str,
    action_order: tuple[str, ...],
    training_metadata: dict,
    *,
    state_action_update_counts: dict | None = None,
) -> None:
    """
    Validate and atomically save a Q-learning checkpoint and optional counts.

    The optional top-level field extends schema 1 without changing old files
    or metadata consumers. Training supplies its cumulative update counters
    after the terminal n-step flush, once per executed training transition.
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    actions = tuple(action_order)
    validated_q_table = _validate_q_table(q_table, actions)

    checkpoint = {
        "kind": CHECKPOINT_KIND,
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "feature_version": feature_version,
        "action_order": actions,
        "q_table": validated_q_table,
        "training_metadata": dict(training_metadata),
    }
    if state_action_update_counts is not None:
        checkpoint[STATE_ACTION_UPDATE_COUNTS] = _validate_update_counts(
            state_action_update_counts, validated_q_table, len(actions)
        )

    temporary_path = path.with_name(f".{path.name}.tmp")

    try:
        with temporary_path.open("wb") as file:
            pickle.dump(
                checkpoint,
                file,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
            file.flush()
            os.fsync(file.fileno())

        os.replace(temporary_path, path)

    finally:
        if temporary_path.exists():
            temporary_path.unlink()
