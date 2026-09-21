from __future__ import annotations

import pandas as pd

from robird.splits import apply_split, solve_observer_split, validate_splits


def _split_frame() -> pd.DataFrame:
    rows = []
    for observer_id in range(1, 9):
        taxon_id = 10 if observer_id <= 4 else 20
        for photo_offset in range(2):
            rows.append(
                {
                    "observation_id": observer_id * 10,
                    "observer_id": observer_id,
                    "taxon_id": taxon_id,
                    "photo_id": observer_id * 100 + photo_offset,
                    "sha256": f"hash-{observer_id}-{photo_offset}",
                }
            )
    return pd.DataFrame.from_records(rows)


def test_solver_is_deterministic_and_observer_disjoint() -> None:
    frame = _split_frame()
    first, _ = solve_observer_split(frame, ["train", "test"], [0.5, 0.5], 17, 10)
    second, _ = solve_observer_split(frame, ["train", "test"], [0.5, 0.5], 17, 10)

    assert first == second
    assigned = apply_split(frame, first)
    assert validate_splits(assigned)["passed"] is True


def test_validator_detects_observer_leakage() -> None:
    frame = _split_frame()
    frame["split"] = "train"
    extra = frame.iloc[[0]].copy()
    extra["observation_id"] = 999
    extra["photo_id"] = 999
    extra["sha256"] = "hash-999"
    extra["split"] = "test"
    report = validate_splits(pd.concat([frame, extra], ignore_index=True))

    assert report["passed"] is False
    assert "observer overlap" in report["violations"]
