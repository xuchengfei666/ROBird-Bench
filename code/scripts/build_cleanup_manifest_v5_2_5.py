from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from robird.io import (
    atomic_write_csv,
    atomic_write_json,
    load_yaml,
    require_frozen_protocol,
    resolve_config_path,
    sha256_file,
)
from robird.manifest import validate_manifest_schema
from robird.scaleup import canonical_hash, groups_to_canonical, select_bounded_total_groups, stable_hash_int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the frozen v5.2.5 cleanup metadata manifest.")
    parser.add_argument("--config", type=Path, default=Path("configs/scaleup_p0_v5_2_5.yaml"))
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _vertex_cover(
    manifest: pd.DataFrame, candidates: pd.DataFrame, fixed_removed: set[int], seed: int
) -> set[int]:
    edges: list[tuple[int, int]] = []
    manifest_photo_ids = set(manifest["photo_id"].astype(int))
    for row in candidates.itertuples(index=False):
        left, right = int(row.left_photo_id), int(row.right_photo_id)
        if left not in manifest_photo_ids or right not in manifest_photo_ids:
            raise ValueError("Near-duplicate candidate references a photo outside the input manifest")
        if left in fixed_removed or right in fixed_removed:
            continue
        edges.append(tuple(sorted((left, right))))
    edges = sorted(set(edges))
    if not edges:
        return set()
    cohort_by_photo = dict(zip(manifest.photo_id.astype(int), manifest.cohort.astype(str)))
    variables = sorted({photo_id for edge in edges for photo_id in edge})
    index = {photo_id: position for position, photo_id in enumerate(variables)}
    matrix = lil_matrix((len(edges), len(variables)), dtype=np.float64)
    for row_index, (left, right) in enumerate(edges):
        matrix[row_index, index[left]] = 1.0
        matrix[row_index, index[right]] = 1.0
    costs = np.asarray(
        [
            (100.0 if cohort_by_photo[photo_id].startswith("roq_") else 1.0)
            + 1e-6 * stable_hash_int(seed, "v5.2.5-vertex-cover", photo_id) / 2**64
            for photo_id in variables
        ],
        dtype=np.float64,
    )
    result = milp(
        c=costs,
        integrality=np.ones(len(variables), dtype=np.int8),
        bounds=Bounds(np.zeros(len(variables)), np.ones(len(variables))),
        constraints=LinearConstraint(
            matrix.tocsr(), np.ones(len(edges)), np.full(len(edges), np.inf)
        ),
        options={"time_limit": 300.0},
    )
    if result.x is None or not result.success:
        raise RuntimeError(f"Near-duplicate vertex-cover MILP failed: {result.message}")
    rounded = np.rint(result.x)
    if not np.allclose(result.x, rounded, atol=1e-6):
        raise RuntimeError("Near-duplicate vertex-cover MILP returned a fractional solution")
    removed = {variables[index] for index, value in enumerate(rounded) if int(value) == 1}
    if any(left not in removed and right not in removed for left, right in edges):
        raise RuntimeError("Computed vertex cover does not cover every candidate edge")
    return removed


def _group_dicts(frame: pd.DataFrame) -> dict[int, pd.DataFrame]:
    return {
        int(observation_id): group.copy()
        for observation_id, group in frame.groupby("observation_id", sort=False)
    }


def main() -> int:
    args = parse_args()
    config = load_yaml(args.config)
    require_frozen_protocol(config)
    paths = config["paths"]
    input_path = resolve_config_path(config, paths["input_manifest_csv"])
    input_audit_path = resolve_config_path(config, paths["input_audit_json"])
    candidate_path = resolve_config_path(config, paths["near_duplicate_csv"])
    candidate_meta_path = resolve_config_path(config, paths["near_duplicate_meta_json"])
    class_table_path = resolve_config_path(config, paths["class_table_csv"])
    census_path = resolve_config_path(config, paths["census_snapshot_json"])
    metadata_path = resolve_config_path(config, paths["metadata_manifest_csv"])
    audit_path = resolve_config_path(config, paths["selection_audit_json"])
    if metadata_path.exists() or audit_path.exists():
        raise FileExistsError("v5.2.5 metadata outputs already exist and cannot be overwritten")

    expected = config["expected_inputs"]
    paths_and_hashes = {
        "input_manifest_sha256": (input_path, expected["input_manifest_sha256"]),
        "input_audit_sha256": (input_audit_path, expected["input_audit_sha256"]),
        "near_duplicate_csv_sha256": (candidate_path, expected["near_duplicate_csv_sha256"]),
        "near_duplicate_meta_sha256": (candidate_meta_path, expected["near_duplicate_meta_sha256"]),
        "census_snapshot_sha256": (census_path, expected["census_snapshot_sha256"]),
        "class_table_sha256": (class_table_path, expected["class_table_sha256"]),
    }
    for label, (path, expected_hash) in paths_and_hashes.items():
        actual = sha256_file(path)
        if actual != str(expected_hash):
            raise RuntimeError(f"Frozen input hash mismatch for {label}: {actual}")

    input_audit = _load_json(input_audit_path)
    if input_audit.get("decision") != "STOP_DATASET_P0":
        raise RuntimeError("v5.2.4 input audit is not the immutable STOP decision")
    manifest = pd.read_csv(input_path, keep_default_na=False).fillna("")
    candidates = pd.read_csv(candidate_path, keep_default_na=False)
    candidate_meta = _load_json(candidate_meta_path)
    if int(candidate_meta.get("candidate_count", -1)) != len(candidates):
        raise RuntimeError("Near-duplicate candidate count does not match its sidecar")
    if candidate_meta.get("candidate_sha256") != sha256_file(candidate_path):
        raise RuntimeError("Near-duplicate candidate sidecar hash mismatch")
    if len(candidates) != int(expected["expected_near_duplicate_candidates"]):
        raise RuntimeError("Unexpected frozen near-duplicate candidate count")

    duplicate_hashes = manifest.loc[manifest["sha256"].duplicated(False), "sha256"]
    duplicate_hashes = {str(value) for value in duplicate_hashes if len(str(value)) == 64}
    fixed_removed: set[int] = set()
    for _, group in manifest[manifest["sha256"].isin(duplicate_hashes)].groupby("sha256", sort=True):
        fixed_removed.update(int(value) for value in sorted(group["photo_id"].astype(int))[1:])
    if len(duplicate_hashes) != int(expected["expected_exact_duplicate_hashes"]):
        raise RuntimeError("Unexpected frozen exact-duplicate hash count")
    near_removed = _vertex_cover(manifest, candidates, fixed_removed, int(config["seed"]))
    removed_photo_ids = fixed_removed | near_removed
    cleaned_rows = manifest[~manifest["photo_id"].astype(int).isin(removed_photo_ids)].copy()
    photo_counts = cleaned_rows.groupby("observation_id").size()
    quarantined_observations = {int(value) for value in photo_counts[photo_counts < 2].index}
    cleaned_rows = cleaned_rows[
        ~cleaned_rows["observation_id"].astype(int).isin(quarantined_observations)
    ].copy()

    seed_rows = cleaned_rows[cleaned_rows["cohort"].astype(str).str.startswith("roq_")].copy()
    existing_expansion = cleaned_rows[
        ~cleaned_rows["cohort"].astype(str).str.startswith("roq_")
    ].copy()
    class_table = pd.read_csv(class_table_path, keep_default_na=False)
    class_map = dict(zip(class_table["taxon_id"].astype(int), class_table["class_index"].astype(int), strict=True))
    taxon_ids = set(class_map)
    seed_observers = set(seed_rows["observer_id"].astype(int))
    old_observation_ids = set(manifest["observation_id"].astype(int))
    quarantined_observations |= set(
        int(value)
        for value in manifest.loc[manifest["observation_id"].astype(int).isin(quarantined_observations), "observation_id"]
    )

    census = _load_json(census_path)
    census_groups = census.get("groups")
    if not isinstance(census_groups, list) or census.get("status") != "COMPLETE":
        raise RuntimeError("The complete v5.1 census snapshot is required")
    existing_groups = _group_dicts(existing_expansion)
    candidate_rows: list[dict[str, Any]] = []
    for observation_id, group in existing_groups.items():
        row = group.iloc[0]
        candidate_rows.append(
            {
                "taxon_id": int(row["taxon_id"]),
                "observation_id": observation_id,
                "observer_id": int(row["observer_id"]),
                "_existing": True,
            }
        )
    added_groups: dict[int, dict[str, Any]] = {}
    for group in census_groups:
        taxon_id = int(group["taxon_id"])
        observation_id = int(group["observation_id"])
        observer_id = int(group["observer_id"])
        if taxon_id not in taxon_ids or observation_id in old_observation_ids:
            continue
        if observation_id in quarantined_observations or observer_id in seed_observers:
            continue
        if not (int(config["source"]["min_photos_per_group"]) <= len(group.get("photos", [])) <= int(config["source"]["max_photos_per_group"])):
            continue
        if set(str(photo.get("license_code", "")).casefold() for photo in group.get("photos", [])) - {
            str(value).casefold() for value in config["source"]["photo_licenses"]
        }:
            continue
        added_groups[observation_id] = dict(group)
    candidate_rows.extend(
        {
            "taxon_id": int(group["taxon_id"]),
            "observation_id": int(group["observation_id"]),
            "observer_id": int(group["observer_id"]),
            "_existing": False,
        }
        for group in added_groups.values()
    )

    seed_counts = seed_rows.groupby("taxon_id")["observation_id"].nunique().astype(int).to_dict()
    lower_targets = {
        taxon_id: max(0, int(config["selection"]["final_groups_per_species_min"]) - int(seed_counts.get(taxon_id, 0)))
        for taxon_id in sorted(taxon_ids)
    }
    upper_targets = {
        taxon_id: int(config["selection"]["final_groups_per_species_max"]) - int(seed_counts.get(taxon_id, 0))
        for taxon_id in sorted(taxon_ids)
    }
    replacement_target = int(config["selection"]["expected_groups"]) - int(seed_rows["observation_id"].nunique())
    selected, solver = select_bounded_total_groups(
        candidate_rows,
        lower_targets,
        upper_targets,
        replacement_target,
        int(config["seed"]),
        observer_capacity=int(config["selection"]["observer_capacity"]),
        time_limit_seconds=float(config["selection"]["milp_time_limit_seconds"]),
    )
    if not solver.get("feasible"):
        raise RuntimeError(f"v5.2.5 bounded-total MILP failed: {solver}")

    selected_existing = [existing_groups[int(row["observation_id"])] for row in selected if row.get("_existing")]
    selected_existing_frame = pd.concat(selected_existing, ignore_index=True) if selected_existing else pd.DataFrame(columns=manifest.columns)
    selected_new = [added_groups[int(row["observation_id"])] for row in selected if not row.get("_existing")]
    selected_new_frame = groups_to_canonical(selected_new, class_map, "robird_scaleup_v5_2_5_candidate")
    combined = pd.concat([seed_rows, selected_existing_frame, selected_new_frame], ignore_index=True, sort=False).fillna("")
    combined["class_index"] = combined["taxon_id"].astype(int).map(class_map).astype(np.int64)
    combined = combined.sort_values(["class_index", "observation_id", "photo_id"]).reset_index(drop=True)
    combined["row_id"] = np.arange(len(combined), dtype=np.int64)
    violations = validate_manifest_schema(combined)
    group_counts = combined.groupby("taxon_id")["observation_id"].nunique().astype(int).to_dict()
    group_photo_counts = combined.groupby("observation_id").size().astype(int)
    known_hashes = combined.loc[combined["sha256"].astype(str).str.len() == 64, "sha256"]
    gates = {
        "schema": not violations,
        "species_exact": combined["taxon_id"].nunique() == int(config["selection"]["expected_species"]),
        "groups_exact": combined["observation_id"].nunique() == int(config["selection"]["expected_groups"]),
        "photos_minimum": len(combined) >= int(config["selection"]["expected_photos_minimum"]),
        "final_species_balance": set(group_counts) == taxon_ids and all(
            int(config["selection"]["final_groups_per_species_min"]) <= group_counts[taxon_id] <= int(config["selection"]["final_groups_per_species_max"])
            for taxon_id in taxon_ids
        ),
        "group_photo_bounds": bool(
            len(group_photo_counts) == combined["observation_id"].nunique()
            and group_photo_counts.between(
                int(config["source"]["min_photos_per_group"]), int(config["source"]["max_photos_per_group"])
            ).all()
        ),
        "licenses_allowed": set(combined["license_code"].astype(str).str.casefold()) <= {
            str(value).casefold() for value in config["source"]["photo_licenses"]
        },
        "photo_ids_unique": not combined["photo_id"].duplicated().any(),
        "known_exact_duplicates_removed": not known_hashes.duplicated(False).any(),
        "seed_observer_disjoint": not (seed_observers & set(combined.loc[~combined["cohort"].astype(str).str.startswith("roq_"), "observer_id"].astype(int))),
        "selected_observer_capacity": solver.get("maximum_groups_per_selected_observer") == int(config["selection"]["observer_capacity"]),
    }
    if violations or not all(gates.values()):
        raise RuntimeError(f"v5.2.5 metadata gates failed: violations={violations}, gates={gates}")

    atomic_write_csv(metadata_path, combined, refuse_if_exists=True)
    audit = {
        "status": "PASS_METADATA_TO_DOWNLOAD",
        "gate_pass": True,
        "dataset_p0_decision_authorized": False,
        "download_authorized": True,
        "byte_audit_complete": False,
        "near_duplicate_audit_complete": False,
        "input_hashes": {label: sha256_file(path) for label, (path, _) in paths_and_hashes.items()},
        "metadata_manifest_sha256": sha256_file(metadata_path),
        "counts": {
            "species": int(combined["taxon_id"].nunique()),
            "groups": int(combined["observation_id"].nunique()),
            "photos": int(len(combined)),
            "retained_seed_groups": int(seed_rows["observation_id"].nunique()),
            "selected_existing_expansion_groups": int(len(selected_existing)),
            "selected_new_groups": int(len(selected_new)),
            "quarantined_observations": int(len(quarantined_observations)),
            "fixed_exact_removed_photos": int(len(fixed_removed)),
            "near_vertex_cover_removed_photos": int(len(near_removed)),
        },
        "cleanup": {
            "duplicate_hash_count": int(len(duplicate_hashes)),
            "candidate_pair_count": int(len(candidates)),
            "fixed_removed_photo_ids": sorted(fixed_removed),
            "near_vertex_cover_photo_ids": sorted(near_removed),
            "quarantined_observation_ids": sorted(quarantined_observations),
        },
        "solver": solver,
        "gates": gates,
        "final_groups_per_taxon": {str(key): int(value) for key, value in group_counts.items()},
    }
    atomic_write_json(audit_path, audit, refuse_if_exists=True)
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
