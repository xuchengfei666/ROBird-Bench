from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from robird.io import atomic_write_csv, atomic_write_json, load_yaml, require_frozen_protocol, resolve_config_path, sha256_file
from robird.manifest import combine_seed_manifests, normalize_legacy_manifest, validate_manifest_schema
from robird.scaleup import groups_to_canonical


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the v5.2.4 metadata manifest from the frozen replacement PASS.")
    parser.add_argument("--config", type=Path, default=Path("configs/scaleup_p0_v5_2_4.yaml"))
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def main() -> int:
    args = parse_args()
    config = load_yaml(args.config)
    require_frozen_protocol(config)
    paths = config["paths"]
    replacement_path = resolve_config_path(config, paths["replacement_audit_json"])
    class_table_path = resolve_config_path(config, paths["class_table_csv"])
    class_audit_path = resolve_config_path(config, paths["class_table_audit_json"])
    manifest_path = resolve_config_path(config, paths["metadata_manifest_csv"])
    audit_path = resolve_config_path(config, paths["selection_audit_json"])
    if manifest_path.exists() or audit_path.exists():
        raise FileExistsError("v5.2.4 metadata outputs already exist and cannot be overwritten")
    replacement = _load_json(replacement_path)
    if replacement.get("status") != "PASS_REPLACEMENT_METADATA_CANDIDATE" or replacement.get("gate_pass") is not True:
        raise RuntimeError("Replacement metadata PASS is missing")
    class_audit = _load_json(class_audit_path)
    if class_audit.get("status") != "PASS_FROZEN_REPLACEMENT_CLASS_TABLE" or class_audit.get("gate_pass") is not True:
        raise RuntimeError("Frozen class-table PASS is missing")
    if class_audit.get("class_table_sha256") != sha256_file(class_table_path):
        raise RuntimeError("Class-table hash does not match its audit")
    table = pd.read_csv(class_table_path, keep_default_na=False)
    if len(table) != 100 or table["taxon_id"].nunique() != 100:
        raise ValueError("Frozen class table must contain 100 unique taxa")
    class_map = dict(zip(table["taxon_id"].astype(int), table["class_index"].astype(int), strict=True))
    selected = list(replacement.get("selected_groups") or [])
    if len(selected) != int(config["selection"]["expected_new_groups"]):
        raise ValueError("Replacement PASS selected-group count does not match the frozen endpoint")
    original = normalize_legacy_manifest(
        resolve_config_path(config, paths["original_manifest_csv"]),
        resolve_config_path(config, paths["original_source_json"]),
        resolve_config_path(config, paths["original_download_ledger"]),
        "roq_original_inspected",
    )
    confirmation = normalize_legacy_manifest(
        resolve_config_path(config, paths["confirmation_manifest_csv"]),
        resolve_config_path(config, paths["confirmation_source_json"]),
        resolve_config_path(config, paths["confirmation_download_ledger"]),
        "roq_confirmation_inspected",
    )
    seed_all = combine_seed_manifests(original, confirmation)
    excluded_observations = set(seed_all["observation_id"].astype(np.int64))
    excluded_observers = set(seed_all["observer_id"].astype(np.int64))
    removed_ids = {
        int(config["taxonomy"]["historical_removed_seed_taxon_id"]),
        int(config["taxonomy"]["removed_seed_taxon_id"]),
    }
    seed = seed_all[~seed_all["taxon_id"].astype(int).isin(removed_ids)].copy()
    seed["class_index"] = seed["taxon_id"].astype(int).map(class_map)
    if seed["class_index"].isna().any():
        raise ValueError("Retained seed taxon is absent from the frozen class table")
    seed["class_index"] = seed["class_index"].astype(np.int64)
    if seed["taxon_id"].nunique() != int(config["taxonomy"]["expected_seed_taxa"]):
        raise ValueError("Unexpected retained seed-taxa count")
    if seed["observation_id"].nunique() != int(config["selection"]["expected_seed_groups"]):
        raise ValueError("Unexpected retained seed-group count")
    expansion = groups_to_canonical(selected, class_map, "robird_scaleup_v5_2_4_candidate")
    combined = pd.concat([seed, expansion], ignore_index=True, sort=False)
    combined = combined.sort_values(["class_index", "observation_id", "photo_id"]).reset_index(drop=True)
    combined["row_id"] = np.arange(len(combined), dtype=np.int64)
    violations = validate_manifest_schema(combined)
    groups = int(combined["observation_id"].nunique())
    species = int(combined["taxon_id"].nunique())
    photos = int(len(combined))
    group_counts = combined[["taxon_id", "observation_id"]].drop_duplicates().groupby("taxon_id").size().astype(int).to_dict()
    photo_counts = combined[["observation_id", "photo_id"]].drop_duplicates().groupby("observation_id").size().astype(int).to_dict()
    selected_observers = [int(row["observer_id"]) for row in selected]
    selected_observations = {int(row["observation_id"]) for row in selected}
    selected_photo_ids = [int(photo["photo_id"]) for row in selected for photo in row.get("photos", [])]
    all_photo_ids = combined["photo_id"].astype(int).tolist()
    gates = {
        "schema": not violations,
        "species_exact": species == int(config["selection"]["expected_species"]),
        "groups_exact": groups == int(config["selection"]["expected_groups"]),
        "photos_minimum": photos >= int(config["selection"]["expected_photos_minimum"]),
        "final_species_balance": set(group_counts) == set(class_map)
        and all(int(config["selection"]["final_groups_per_species_min"]) <= group_counts[taxon_id] <= int(config["selection"]["final_groups_per_species_max"]) for taxon_id in class_map),
        "final_group_photo_bounds": all(int(config["source"]["min_photos_per_group"]) <= count <= int(config["source"]["max_photos_per_group"]) for count in photo_counts.values()),
        "selected_group_photo_bounds": all(int(config["source"]["min_photos_per_group"]) <= len(row.get("photos", [])) <= int(config["source"]["max_photos_per_group"]) for row in selected),
        "licenses_allowed": set(combined["license_code"].astype(str).str.lower()) <= {str(v).lower() for v in config["source"]["photo_licenses"]},
        "photo_ids_unique": len(all_photo_ids) == len(set(all_photo_ids)) and len(selected_photo_ids) == len(set(selected_photo_ids)),
        "new_observers_unique": len(selected_observers) == len(set(selected_observers)),
        "new_observers_exclude_seed": not (set(selected_observers) & excluded_observers),
        "new_observations_exclude_seed": not (selected_observations & excluded_observations),
    }
    if not all(gates.values()):
        raise RuntimeError(f"Metadata manifest gate failed: {gates}")
    atomic_write_csv(manifest_path, combined, refuse_if_exists=True)
    audit = {
        "status": "PASS_METADATA_TO_DOWNLOAD",
        "gate_pass": True,
        "dataset_p0_decision_authorized": False,
        "download_authorized": True,
        "byte_audit_complete": False,
        "near_duplicate_audit_complete": False,
        "replacement_audit_sha256": sha256_file(replacement_path),
        "class_table_sha256": sha256_file(class_table_path),
        "metadata_manifest_sha256": sha256_file(manifest_path),
        "counts": {"species": species, "groups": groups, "photos": photos, "seed_groups": int(seed["observation_id"].nunique()), "selected_new_groups": len(selected)},
        "gates": gates,
        "final_groups_per_species": {str(key): value for key, value in group_counts.items()},
    }
    atomic_write_json(audit_path, audit, refuse_if_exists=True)
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
