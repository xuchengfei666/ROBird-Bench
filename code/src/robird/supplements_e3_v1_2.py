from __future__ import annotations

from typing import Any

import pandas as pd


def build_strict_cross_observation_shuffle_manifest(manifest: pd.DataFrame, split: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create mixed sets with no destination photo from its original observation.

    A stratum is eligible only when it has at least three observations and at
    least two photos per observation. For each within-group photo position j,
    source observation indices are shifted by a nonzero deterministic offset;
    the mapping is a permutation for every j, so every source photo is used
    exactly once and no destination keeps a photo from its original group.
    """
    selected = manifest.loc[manifest["split"] == split].copy()
    selected["photo_count"] = selected.groupby("observation_id")["photo_id"].transform("count")
    rows: list[dict[str, Any]] = []
    strata: dict[str, Any] = {}
    assignment: dict[int, list[int]] = {}
    eligible_observations: list[int] = []
    excluded_single_view = 0
    excluded_two_group = 0
    for (taxon_id, photo_count), group in selected.groupby(["taxon_id", "photo_count"], sort=True):
        observation_ids = sorted(int(value) for value in group["observation_id"].unique())
        key = f"taxon={int(taxon_id)}|photos={int(photo_count)}"
        strata[key] = {"observations": len(observation_ids), "photos": int(len(group))}
        if int(photo_count) < 2:
            excluded_single_view += len(observation_ids)
            continue
        if len(observation_ids) < 3:
            excluded_two_group += len(observation_ids)
            continue
        eligible_observations.extend(observation_ids)
        source_groups = {
            obs: selected.loc[selected["observation_id"] == obs].sort_values("photo_id").to_dict(orient="records")
            for obs in observation_ids
        }
        base_rows = {obs: dict(source_groups[obs][0]) for obs in observation_ids}
        for destination_index, destination in enumerate(observation_ids):
            assigned: list[dict[str, Any]] = []
            for position in range(int(photo_count)):
                offset = 1 + (position % (len(observation_ids) - 1))
                source_index = (destination_index + offset) % len(observation_ids)
                source = observation_ids[source_index]
                assigned.append(source_groups[source][position])
            destination_original = {int(item["photo_id"]) for item in source_groups[destination]}
            if destination_original.intersection(int(item["photo_id"]) for item in assigned):
                raise RuntimeError(f"Strict shuffle retained original membership for {destination}")
            assignment[destination] = [int(item["photo_id"]) for item in assigned]
            base = base_rows[destination]
            for source_row in assigned:
                row = dict(base)
                for field in ("photo_id", "local_path", "sha256", "width", "height", "url", "attribution", "license_code"):
                    row[field] = source_row[field]
                rows.append(row)
    shuffled = pd.DataFrame.from_records(rows).drop(columns=["photo_count"], errors="ignore")
    if shuffled.empty:
        raise RuntimeError("No eligible strict cross-observation strata")
    if shuffled["photo_id"].duplicated().any():
        raise RuntimeError("Strict shuffle duplicated photo IDs")
    metadata = {
        "split": split,
        "eligible_observations": len(eligible_observations),
        "eligible_strata": sum(1 for value in strata.values() if int(value["observations"]) >= 3),
        "excluded_single_view_observations": excluded_single_view,
        "excluded_two_group_stratum_observations": excluded_two_group,
        "strata": strata,
        "assigned_photo_ids": {str(key): value for key, value in sorted(assignment.items())},
    }
    return shuffled, metadata
