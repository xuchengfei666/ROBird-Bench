from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robird.audit import difference_hash, hamming_distance
from robird.download import download_image
from robird.io import (
    atomic_write_csv,
    atomic_write_json,
    load_yaml,
    require_frozen_protocol,
    resolve_config_path,
    sha256_file,
)
from robird.manifest import validate_manifest_schema
from robird.scaleup import groups_to_canonical, stable_hash_int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage and finalize ROBird Dataset P0-v5.3.")
    parser.add_argument("--config", type=Path, default=Path("configs/finalize_p0_v5_3.yaml"))
    parser.add_argument("--reviewed-staging-candidates", type=Path)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def verify_hash(path: Path, expected: str, label: str) -> None:
    actual = sha256_file(path)
    if actual != str(expected):
        raise RuntimeError(f"Frozen input hash mismatch for {label}: {actual}")


def canonical_pairs(frame: pd.DataFrame) -> dict[tuple[int, int], dict[str, Any]]:
    pairs: dict[tuple[int, int], dict[str, Any]] = {}
    for row in frame.to_dict(orient="records"):
        key = (int(row["candidate_photo_id"]), int(row["base_photo_id"]))
        if key in pairs:
            raise ValueError(f"Duplicate staging review pair: {key}")
        pairs[key] = row
    return pairs


def main() -> int:
    args = parse_args()
    config = load_yaml(args.config)
    require_frozen_protocol(config)
    paths = config["paths"]
    expected = config["expected_inputs"]
    base_path = resolve_config_path(config, paths["base_manifest_csv"])
    base_audit_path = resolve_config_path(config, paths["base_audit_json"])
    base_candidates_path = resolve_config_path(config, paths["base_near_candidates_csv"])
    base_review_path = resolve_config_path(config, paths["base_near_review_csv"])
    census_path = resolve_config_path(config, paths["census_snapshot_json"])
    class_table_path = resolve_config_path(config, paths["class_table_csv"])
    pair_path = resolve_config_path(config, paths["staging_pairs_csv"])
    stage_audit_path = resolve_config_path(config, paths["staging_audit_json"])
    ledger_path = resolve_config_path(config, paths["staging_ledger_json"])
    staging_root = resolve_config_path(config, paths["staging_root"])
    final_path = resolve_config_path(config, paths["canonical_manifest_csv"])
    final_audit_path = resolve_config_path(config, paths["finalization_audit_json"])

    verify_hash(base_path, expected["base_manifest_sha256"], "base_manifest")
    verify_hash(base_audit_path, expected["base_audit_sha256"], "base_audit")
    verify_hash(base_candidates_path, expected["base_near_candidates_sha256"], "base_candidates")
    verify_hash(census_path, expected["census_snapshot_sha256"], "census_snapshot")
    verify_hash(class_table_path, expected["class_table_sha256"], "class_table")
    base_candidates = pd.read_csv(base_candidates_path)
    base_reviews = pd.read_csv(base_review_path)
    if set(tuple(sorted(value)) for value in base_candidates[["left_photo_id", "right_photo_id"]].astype(int).itertuples(index=False, name=None)) != set(
        tuple(sorted(value)) for value in base_reviews[["left_photo_id", "right_photo_id"]].astype(int).itertuples(index=False, name=None)
    ):
        raise RuntimeError("The visual review does not match the frozen v5.2.5 candidate set")
    if set(base_reviews["review_decision"].astype(str).str.casefold()) != {"distinct"}:
        raise RuntimeError("The frozen v5.2.5 visual review is incomplete")

    base = pd.read_csv(base_path, keep_default_na=False).fillna("")
    removed_observation = int(config["replacement"]["removed_observation_id"])
    removed = base[base["observation_id"].astype(int) == removed_observation]
    if len(removed) != 2 or set(removed["sha256"].astype(str)) != {str(config["replacement"]["expected_duplicate_sha256"])}:
        raise RuntimeError("The declared exact-duplicate observation does not match v5.2.5")
    retained = base[base["observation_id"].astype(int) != removed_observation].copy()
    if retained["sha256"].duplicated(False).any():
        raise RuntimeError("Retained v5.2.5 base still contains exact duplicate bytes")

    historical_exact_observations: set[int] = set()
    for value in paths["historical_manifest_csvs"]:
        historical = pd.read_csv(resolve_config_path(config, value), keep_default_na=False)
        historical_exact_observations |= set(
            historical.loc[historical["sha256"].duplicated(False), "observation_id"].astype(int)
        )
    census = load_json(census_path)
    groups = [dict(row) for row in census.get("groups", [])]
    target_taxon = int(config["replacement"]["taxon_id"])
    base_observations = set(base["observation_id"].astype(int))
    base_observers = set(base["observer_id"].astype(int))
    allowed = {str(value).casefold() for value in config["selection"]["allowed_licenses"]}
    candidates = []
    for group in groups:
        if int(group["taxon_id"]) != target_taxon:
            continue
        if int(group["observation_id"]) in base_observations or int(group["observer_id"]) in base_observers:
            continue
        if int(group["observation_id"]) in historical_exact_observations:
            continue
        photos = list(group.get("photos") or [])
        if not int(config["selection"]["min_photos_per_group"]) <= len(photos) <= int(config["selection"]["max_photos_per_group"]):
            continue
        if {str(photo.get("license_code", "")).casefold() for photo in photos} - allowed:
            continue
        candidates.append(group)
    candidates.sort(key=lambda row: stable_hash_int(int(config["seed"]), "v5.3-staging", row["observation_id"]))
    candidate_order = [int(row["observation_id"]) for row in candidates]
    if candidate_order != [int(value) for value in config["replacement"]["candidate_order"]]:
        raise RuntimeError(f"Frozen v5.3 candidate order mismatch: {candidate_order}")

    ledger: dict[int, dict[str, Any]] = {}
    if ledger_path.exists():
        previous = load_json(ledger_path)
        if previous.get("base_manifest_sha256") != sha256_file(base_path) or previous.get("candidate_order") != candidate_order:
            raise RuntimeError("Existing v5.3 staging ledger has a different contract")
        ledger = {int(row["photo_id"]): dict(row) for row in previous.get("photos", [])}
    for group in candidates:
        for photo in group["photos"]:
            photo_id = int(photo["photo_id"])
            if ledger.get(photo_id, {}).get("ok"):
                continue
            suffix = Path(str(photo["url"]).split("?", 1)[0]).suffix.lower() or ".jpg"
            destination = staging_root / str(int(group["observation_id"])) / f"{photo_id}{suffix}"
            try:
                result = download_image(
                    str(photo["url"]),
                    destination,
                    int(config["download"]["request_attempts"]),
                    float(config["download"]["timeout_seconds"]),
                )
                ledger[photo_id] = {**photo, **result, "observation_id": int(group["observation_id"]), "photo_id": photo_id, "ok": True, "error": ""}
            except Exception as error:
                ledger[photo_id] = {**photo, "observation_id": int(group["observation_id"]), "photo_id": photo_id, "ok": False, "error": f"{type(error).__name__}: {error}"}
            atomic_write_json(
                ledger_path,
                {
                    "status": "RUNNING",
                    "base_manifest_sha256": sha256_file(base_path),
                    "candidate_order": candidate_order,
                    "photos": sorted(ledger.values(), key=lambda row: int(row["photo_id"])),
                },
            )
    required_ids = {int(photo["photo_id"]) for group in candidates for photo in group["photos"]}
    failed = sorted(photo_id for photo_id in required_ids if not ledger.get(photo_id, {}).get("ok"))
    if failed:
        raise RuntimeError(f"v5.3 staging download incomplete: {failed}")
    atomic_write_json(
        ledger_path,
        {
            "status": "COMPLETE",
            "base_manifest_sha256": sha256_file(base_path),
            "candidate_order": candidate_order,
            "photos": sorted(ledger.values(), key=lambda row: int(row["photo_id"])),
        },
    )

    hash_size = int(config["selection"]["dhash_size"])
    threshold = int(config["selection"]["hamming_threshold"])
    base_hashes = set(retained["sha256"].astype(str))
    base_dhashes = [
        (int(row.photo_id), int(row.observation_id), difference_hash(Path(str(row.local_path)), hash_size))
        for row in retained.itertuples(index=False)
    ]
    pair_rows: list[dict[str, Any]] = []
    candidate_diagnostics: list[dict[str, Any]] = []
    for group in candidates:
        observation_id = int(group["observation_id"])
        records = [ledger[int(photo["photo_id"])] for photo in group["photos"]]
        byte_hashes = [str(row["sha256"]) for row in records]
        exact_clean = len(byte_hashes) == len(set(byte_hashes)) and not (set(byte_hashes) & base_hashes)
        pair_count = 0
        for record in records:
            candidate_hash = difference_hash(Path(str(record["path"])), hash_size)
            for base_photo_id, base_observation_id, base_hash in base_dhashes:
                distance = hamming_distance(candidate_hash, base_hash)
                if distance <= threshold:
                    pair_rows.append(
                        {
                            "candidate_observation_id": observation_id,
                            "candidate_photo_id": int(record["photo_id"]),
                            "base_observation_id": base_observation_id,
                            "base_photo_id": base_photo_id,
                            "hamming_distance": distance,
                        }
                    )
                    pair_count += 1
        candidate_diagnostics.append(
            {
                "observation_id": observation_id,
                "observer_id": int(group["observer_id"]),
                "photo_count": len(records),
                "exact_clean": exact_clean,
                "near_candidate_count": pair_count,
            }
        )
    pairs = pd.DataFrame.from_records(
        pair_rows,
        columns=["candidate_observation_id", "candidate_photo_id", "base_observation_id", "base_photo_id", "hamming_distance"],
    ).sort_values(["candidate_observation_id", "candidate_photo_id", "base_photo_id"]).reset_index(drop=True)
    if pair_path.exists():
        existing_pairs = pd.read_csv(pair_path)
        if not existing_pairs.equals(pairs):
            raise RuntimeError("Existing v5.3 staging pair artifact differs from recomputation")
    else:
        atomic_write_csv(pair_path, pairs, refuse_if_exists=True)
    stage_audit = {
        "status": "COMPLETE_STAGING_REQUIRES_REVIEW" if len(pairs) else "COMPLETE_STAGING_NO_CANDIDATES",
        "base_manifest_sha256": sha256_file(base_path),
        "candidate_order": candidate_order,
        "candidate_diagnostics": candidate_diagnostics,
        "staging_pair_count": int(len(pairs)),
        "staging_pairs_sha256": sha256_file(pair_path),
        "staging_ledger_sha256": sha256_file(ledger_path),
        "final_manifest_written": False,
    }
    if stage_audit_path.exists():
        if load_json(stage_audit_path) != stage_audit:
            raise RuntimeError("Existing v5.3 staging audit differs from recomputation")
    else:
        atomic_write_json(stage_audit_path, stage_audit, refuse_if_exists=True)

    reviewed: dict[tuple[int, int], dict[str, Any]] = {}
    if len(pairs):
        if args.reviewed_staging_candidates is None:
            print(json.dumps({"status": "PENDING_STAGING_VISUAL_REVIEW", **stage_audit}, indent=2, sort_keys=True))
            return 3
        reviews = pd.read_csv(args.reviewed_staging_candidates)
        required = {"candidate_photo_id", "base_photo_id", "review_decision"}
        if required - set(reviews.columns):
            raise ValueError("Staging review CSV is missing required columns")
        expected_pairs = canonical_pairs(pairs)
        reviewed = canonical_pairs(reviews)
        if set(expected_pairs) != set(reviewed):
            raise RuntimeError("Staging visual reviews do not match the frozen pair set")
        decisions = {str(row["review_decision"]).casefold() for row in reviewed.values()}
        if decisions - {"distinct", "quarantine"}:
            raise ValueError("Unknown staging review decision")

    selected: dict[str, Any] | None = None
    for group, diagnostic in zip(candidates, candidate_diagnostics, strict=True):
        if not diagnostic["exact_clean"]:
            continue
        observation_id = int(group["observation_id"])
        group_pairs = [row for row in pairs.to_dict(orient="records") if int(row["candidate_observation_id"]) == observation_id]
        if any(str(reviewed[(int(row["candidate_photo_id"]), int(row["base_photo_id"]))]["review_decision"]).casefold() == "quarantine" for row in group_pairs):
            continue
        selected = group
        break
    if selected is None:
        raise RuntimeError("No v5.3 replacement candidate passed exact and visual review gates")
    if final_path.exists() or final_audit_path.exists():
        raise FileExistsError("v5.3 finalization outputs already exist")

    class_table = pd.read_csv(class_table_path, keep_default_na=False)
    class_map = dict(zip(class_table["taxon_id"].astype(int), class_table["class_index"].astype(int), strict=True))
    replacement_frame = groups_to_canonical([selected], class_map, "robird_scaleup_v5_3_candidate")
    for index in replacement_frame.index:
        photo_id = int(replacement_frame.at[index, "photo_id"])
        record = ledger[photo_id]
        replacement_frame.at[index, "local_path"] = record["path"]
        replacement_frame.at[index, "sha256"] = record["sha256"]
        replacement_frame.at[index, "width"] = int(record["width"])
        replacement_frame.at[index, "height"] = int(record["height"])
    combined = pd.concat([retained, replacement_frame], ignore_index=True, sort=False).fillna("")
    combined = combined.sort_values(["class_index", "observation_id", "photo_id"]).reset_index(drop=True)
    combined["row_id"] = np.arange(len(combined), dtype=np.int64)
    violations = validate_manifest_schema(combined)
    group_counts = combined.groupby("taxon_id")["observation_id"].nunique().astype(int)
    group_sizes = combined.groupby("observation_id").size().astype(int)
    gates = {
        "schema": not violations,
        "species_exact": combined["taxon_id"].nunique() == int(config["selection"]["expected_species"]),
        "groups_exact": combined["observation_id"].nunique() == int(config["selection"]["expected_groups"]),
        "photos_minimum": len(combined) >= int(config["selection"]["expected_photos_minimum"]),
        "species_balance": bool(group_counts.between(int(config["selection"]["final_groups_per_species_min"]), int(config["selection"]["final_groups_per_species_max"])).all()),
        "group_photo_bounds": bool(group_sizes.between(int(config["selection"]["min_photos_per_group"]), int(config["selection"]["max_photos_per_group"])).all()),
        "photo_ids_unique": not combined["photo_id"].duplicated().any(),
        "exact_hashes_unique": not combined["sha256"].duplicated().any(),
        "replacement_observer_excludes_base": int(selected["observer_id"]) not in set(retained["observer_id"].astype(int)),
    }
    if violations or not all(gates.values()):
        raise RuntimeError(f"v5.3 final manifest gates failed: {gates}, violations={violations}")
    atomic_write_csv(final_path, combined, refuse_if_exists=True)
    final_audit = {
        "status": "PASS_V5_3_FINALIZATION_TO_BYTE_AUDIT",
        "gate_pass": True,
        "dataset_p0_decision_authorized": False,
        "removed_observation_id": removed_observation,
        "selected_replacement_observation_id": int(selected["observation_id"]),
        "selected_replacement_observer_id": int(selected["observer_id"]),
        "base_visual_review_sha256": sha256_file(base_review_path),
        "staging_review_sha256": sha256_file(args.reviewed_staging_candidates) if args.reviewed_staging_candidates else None,
        "staging_audit_sha256": sha256_file(stage_audit_path),
        "canonical_manifest_sha256": sha256_file(final_path),
        "counts": {"species": int(combined["taxon_id"].nunique()), "groups": int(combined["observation_id"].nunique()), "photos": int(len(combined))},
        "gates": gates,
    }
    atomic_write_json(final_audit_path, final_audit, refuse_if_exists=True)
    print(json.dumps(final_audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
