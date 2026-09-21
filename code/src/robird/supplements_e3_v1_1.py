from __future__ import annotations

from typing import Any

import pandas as pd


def build_photo_level_shuffle_manifest(manifest: pd.DataFrame, split: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Break observation membership while preserving taxon and group cardinality.

    Within each taxon/cardinality stratum, photos are flattened in deterministic
    observation/photo order and circularly shifted by one photo. Groups with a
    single photo are excluded because a photo-level permutation cannot change
    their membership semantics.
    """
    selected = manifest.loc[manifest["split"] == split].copy()
    selected["photo_count"] = selected.groupby("observation_id")["photo_id"].transform("count")
    rows: list[dict[str, Any]] = []
    strata: dict[str, Any] = {}
    eligible_observations: list[int] = []
    excluded_single_view = 0
    excluded_single_group_strata = 0
    assignment: dict[int, list[int]] = {}
    for (taxon_id, photo_count), group in selected.groupby(["taxon_id", "photo_count"], sort=True):
        ordered = group.sort_values(["observation_id", "photo_id"])
        observation_ids = sorted(int(value) for value in ordered["observation_id"].unique())
        key = f"taxon={int(taxon_id)}|photos={int(photo_count)}"
        strata[key] = {"observations": len(observation_ids), "photos": int(len(ordered))}
        if int(photo_count) < 2:
            excluded_single_view += len(observation_ids)
            continue
        if len(observation_ids) < 2:
            excluded_single_group_strata += len(observation_ids)
            continue
        eligible_observations.extend(observation_ids)
        photos = [row for row in ordered.to_dict(orient="records")]
        shifted = photos[-1:] + photos[:-1]
        by_destination: dict[int, list[dict[str, Any]]] = {obs: [] for obs in observation_ids}
        destination_position = 0
        for observation_id in observation_ids:
            count = int(photo_count)
            assigned = shifted[destination_position : destination_position + count]
            destination_position += count
            by_destination[observation_id] = assigned
            assignment[observation_id] = [int(item["photo_id"]) for item in assigned]
        base_rows = {obs: selected.loc[selected["observation_id"] == obs].iloc[0].to_dict() for obs in observation_ids}
        for destination in observation_ids:
            base = dict(base_rows[destination])
            for source_row in by_destination[destination]:
                row = dict(base)
                for field in ("photo_id", "local_path", "sha256", "width", "height", "url", "attribution", "license_code"):
                    row[field] = source_row[field]
                rows.append(row)
    shuffled = pd.DataFrame.from_records(rows).drop(columns=["photo_count"], errors="ignore")
    if shuffled.empty:
        raise RuntimeError("No eligible photo-level shuffle strata")
    for observation_id, group in shuffled.groupby("observation_id"):
        if len(group) != len(selected.loc[selected["observation_id"] == observation_id]):
            raise RuntimeError("Photo-level shuffle changed group cardinality")
        if group["photo_id"].duplicated().any():
            raise RuntimeError(f"Photo-level shuffle duplicated photo IDs in {observation_id}")
    metadata = {
        "split": split,
        "eligible_observations": len(eligible_observations),
        "excluded_single_view_observations": excluded_single_view,
        "excluded_single_group_stratum_observations": excluded_single_group_strata,
        "strata": strata,
        "assigned_photo_ids": {str(key): value for key, value in sorted(assignment.items())},
    }
    return shuffled, metadata
