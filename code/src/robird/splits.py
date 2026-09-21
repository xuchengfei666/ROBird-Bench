from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix


def _stable_jitter(seed: int, observer_id: int, split: str) -> float:
    payload = f"{seed}|{observer_id}|{split}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") / 2**64


def observer_species_counts(
    frame: pd.DataFrame,
) -> tuple[list[int], list[int], np.ndarray]:
    groups = frame.drop_duplicates("observation_id")
    observers = sorted(int(value) for value in groups["observer_id"].unique())
    taxa = sorted(int(value) for value in groups["taxon_id"].unique())
    observer_index = {value: index for index, value in enumerate(observers)}
    taxon_index = {value: index for index, value in enumerate(taxa)}
    counts = np.zeros((len(taxa), len(observers)), dtype=np.float64)
    for row in groups.itertuples(index=False):
        counts[taxon_index[int(row.taxon_id)], observer_index[int(row.observer_id)]] += 1.0
    return observers, taxa, counts


def solve_observer_split(
    frame: pd.DataFrame,
    names: Sequence[str],
    fractions: Sequence[float],
    seed: int,
    time_limit_seconds: int,
) -> tuple[dict[int, str], dict[str, Any]]:
    if frame.empty:
        raise ValueError("Cannot split an empty manifest")
    if len(names) != len(fractions) or not names:
        raise ValueError("Split names and fractions must be non-empty and equally sized")
    fractions_array = np.asarray(fractions, dtype=np.float64)
    if np.any(fractions_array <= 0) or not np.isclose(fractions_array.sum(), 1.0):
        raise ValueError("Split fractions must be positive and sum to one")

    observers, taxa, counts = observer_species_counts(frame)
    num_observers, num_splits, num_taxa = len(observers), len(names), len(taxa)
    num_x = num_observers * num_splits
    num_slack = num_taxa * num_splits * 2
    num_vars = num_x + num_slack
    objective = np.zeros(num_vars, dtype=np.float64)
    for observer_pos, observer_id in enumerate(observers):
        for split_pos, split in enumerate(names):
            objective[observer_pos * num_splits + split_pos] = (
                _stable_jitter(seed, observer_id, split) * 1e-9
            )
    totals = counts.sum(axis=1)
    for taxon_pos in range(num_taxa):
        weight = 1.0 / max(float(totals[taxon_pos]), 1.0)
        start = num_x + taxon_pos * num_splits * 2
        objective[start : start + num_splits * 2] = weight

    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    lower: list[float] = []
    upper: list[float] = []

    def add_constraint(coefficients: Mapping[int, float], lo: float, hi: float) -> None:
        row = len(lower)
        for column, value in coefficients.items():
            rows.append(row)
            columns.append(column)
            values.append(float(value))
        lower.append(float(lo))
        upper.append(float(hi))

    for observer_pos in range(num_observers):
        add_constraint(
            {observer_pos * num_splits + split_pos: 1.0 for split_pos in range(num_splits)},
            1.0,
            1.0,
        )

    for split_pos in range(num_splits):
        add_constraint(
            {
                observer_pos * num_splits + split_pos: 1.0
                for observer_pos in range(num_observers)
            },
            1.0,
            np.inf,
        )

    for taxon_pos in range(num_taxa):
        for split_pos in range(num_splits):
            coefficients = {
                observer_pos * num_splits + split_pos: counts[taxon_pos, observer_pos]
                for observer_pos in range(num_observers)
                if counts[taxon_pos, observer_pos] > 0
            }
            plus = num_x + (taxon_pos * num_splits + split_pos) * 2
            minus = plus + 1
            coefficients[plus] = -1.0
            coefficients[minus] = 1.0
            target = totals[taxon_pos] * fractions_array[split_pos]
            add_constraint(coefficients, target, target)
            add_constraint(
                {
                    observer_pos * num_splits + split_pos: counts[taxon_pos, observer_pos]
                    for observer_pos in range(num_observers)
                    if counts[taxon_pos, observer_pos] > 0
                },
                1.0,
                np.inf,
            )

    matrix = coo_matrix((values, (rows, columns)), shape=(len(lower), num_vars)).tocsr()
    variable_lower = np.zeros(num_vars, dtype=np.float64)
    variable_upper = np.full(num_vars, np.inf, dtype=np.float64)
    variable_upper[:num_x] = 1.0
    integrality = np.zeros(num_vars, dtype=np.int8)
    integrality[:num_x] = 1
    result = milp(
        c=objective,
        integrality=integrality,
        bounds=Bounds(variable_lower, variable_upper),
        constraints=LinearConstraint(matrix, np.asarray(lower), np.asarray(upper)),
        options={"time_limit": float(time_limit_seconds), "mip_rel_gap": 0.0},
    )
    if result.x is None:
        raise RuntimeError(f"Observer split MILP failed: {result.status} {result.message}")
    assignments: dict[int, str] = {}
    matrix_x = result.x[:num_x].reshape(num_observers, num_splits)
    if not np.allclose(matrix_x, np.rint(matrix_x), atol=1e-6):
        raise RuntimeError("MILP returned a fractional incumbent")
    for observer_pos, observer_id in enumerate(observers):
        split_pos = int(np.argmax(matrix_x[observer_pos]))
        if matrix_x[observer_pos, split_pos] < 0.5:
            raise RuntimeError(f"Non-integral assignment for observer {observer_id}")
        assignments[observer_id] = str(names[split_pos])
    return assignments, {
        "status": int(result.status),
        "message": str(result.message),
        "objective": float(result.fun),
        "optimal": bool(result.success),
        "observers": num_observers,
        "taxa": num_taxa,
    }


def apply_split(frame: pd.DataFrame, assignments: Mapping[int, str]) -> pd.DataFrame:
    output = frame.copy()
    output["split"] = output["observer_id"].map({int(k): str(v) for k, v in assignments.items()})
    if output["split"].isna().any():
        missing = output.loc[output["split"].isna(), "observer_id"].unique().tolist()
        raise ValueError(f"Missing split assignments for observers: {missing[:10]}")
    return output


def validate_splits(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        raise ValueError("Cannot validate empty splits")
    groups = frame.drop_duplicates("observation_id")
    observer_counts = groups.groupby("observer_id")["split"].nunique()
    observation_counts = frame.groupby("observation_id")["split"].nunique()
    violations: list[str] = []
    if int(observer_counts.max()) != 1:
        violations.append("observer overlap")
    if int(observation_counts.max()) != 1:
        violations.append("observation overlap")
    exact = frame.loc[frame["sha256"].fillna("").astype(str).str.len() > 0]
    if len(exact):
        hash_splits = exact.groupby("sha256")["split"].nunique()
        if int(hash_splits.max()) > 1:
            violations.append("exact file hash overlap")
    table = (
        groups.groupby(["taxon_id", "split"])["observation_id"]
        .nunique()
        .unstack(fill_value=0)
    )
    if bool((table == 0).any(axis=None)):
        violations.append("one or more species are absent from a split")
    return {
        "passed": not violations,
        "violations": violations,
        "split_groups": {str(k): int(v) for k, v in groups["split"].value_counts().items()},
        "per_species_min": {str(k): int(v) for k, v in table.min(axis=0).items()},
        "per_species_max": {str(k): int(v) for k, v in table.max(axis=0).items()},
    }
