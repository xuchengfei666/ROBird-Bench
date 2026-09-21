from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from robird.audit import audit_manifest, duplicate_candidates
from robird.io import (
    atomic_write_csv,
    atomic_write_json,
    load_yaml,
    require_frozen_protocol,
    resolve_config_path,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the frozen ROBird Dataset P0 manifest.")
    parser.add_argument("--config", type=Path, default=Path("configs/dataset_p0.yaml"))
    parser.add_argument("--inspect-files", action="store_true")
    parser.add_argument("--near-duplicates", action="store_true")
    parser.add_argument("--reviewed-candidates", type=Path)
    return parser.parse_args()


def _canonical_pairs(frame: pd.DataFrame) -> dict[tuple[int, int], dict[str, Any]]:
    pairs: dict[tuple[int, int], dict[str, Any]] = {}
    for row in frame.to_dict(orient="records"):
        left, right = sorted((int(row["left_photo_id"]), int(row["right_photo_id"])))
        key = (left, right)
        if key in pairs:
            raise ValueError(f"Duplicate near-duplicate pair: {key}")
        pairs[key] = row
    return pairs


def _review_status(candidates: pd.DataFrame, review_path: Path | None) -> tuple[bool, str]:
    if candidates.empty:
        return True, "completed_no_candidates"
    if review_path is None:
        return False, f"pending_manual_review:{len(candidates)}"
    reviews = pd.read_csv(review_path)
    required = {"left_photo_id", "right_photo_id", "review_decision"}
    missing = sorted(required - set(reviews.columns))
    if missing:
        raise ValueError(f"Reviewed candidate file missing columns: {missing}")
    candidate_pairs = _canonical_pairs(candidates)
    review_pairs = _canonical_pairs(reviews)
    if set(candidate_pairs) != set(review_pairs):
        missing_reviews = sorted(set(candidate_pairs) - set(review_pairs))
        stale_reviews = sorted(set(review_pairs) - set(candidate_pairs))
        raise ValueError(
            "Reviewed candidates do not match current candidates: "
            f"missing={missing_reviews[:5]}, stale={stale_reviews[:5]}"
        )
    decisions = {str(row["review_decision"]).strip().lower() for row in review_pairs.values()}
    invalid = sorted(decisions - {"distinct", "quarantine"})
    if invalid:
        raise ValueError(f"Unknown review decisions: {invalid}")
    quarantine_count = sum(
        str(row["review_decision"]).strip().lower() == "quarantine"
        for row in review_pairs.values()
    )
    if quarantine_count:
        return False, f"quarantine_required:{quarantine_count}"
    return True, f"completed_all_distinct:{len(review_pairs)}"


def _holdout_status(
    config: dict[str, Any], development: pd.DataFrame
) -> tuple[bool, str, dict[str, Any]]:
    holdout_config = config.get("holdout", {})
    manifest_value = holdout_config.get("registered_manifest")
    expected_hash = str(holdout_config.get("registered_manifest_sha256") or "").lower()
    if not manifest_value:
        return False, "not_registered", {}
    holdout_path = resolve_config_path(config, str(manifest_value))
    if not holdout_path.is_file():
        return False, "registered_manifest_missing", {"path": str(holdout_path)}
    actual_hash = sha256_file(holdout_path)
    if len(expected_hash) != 64 or actual_hash != expected_hash:
        return False, "registered_manifest_hash_mismatch", {
            "path": str(holdout_path),
            "expected_sha256": expected_hash,
            "actual_sha256": actual_hash,
        }
    holdout = pd.read_csv(holdout_path)
    required = {"observation_id", "observer_id", "taxon_id"}
    missing = sorted(required - set(holdout.columns))
    if missing:
        return False, f"registered_manifest_missing_columns:{missing}", {
            "path": str(holdout_path),
            "sha256": actual_hash,
        }
    observation_overlap = {int(value) for value in development["observation_id"]} & {
        int(value) for value in holdout["observation_id"]
    }
    observer_overlap = {int(value) for value in development["observer_id"]} & {
        int(value) for value in holdout["observer_id"]
    }
    development_taxa = {int(value) for value in development["taxon_id"]}
    holdout_taxa = {int(value) for value in holdout["taxon_id"]}
    taxon_match = development_taxa == holdout_taxa
    metadata = {
        "path": str(holdout_path),
        "sha256": actual_hash,
        "observations": int(holdout["observation_id"].nunique()),
        "observers": int(holdout["observer_id"].nunique()),
        "taxa": int(holdout["taxon_id"].nunique()),
        "observation_overlap": len(observation_overlap),
        "observer_overlap": len(observer_overlap),
        "taxon_set_matches": taxon_match,
    }
    passed = not observation_overlap and not observer_overlap and taxon_match and not holdout.empty
    return passed, "registered_and_disjoint" if passed else "registered_but_ineligible", metadata


def _candidate_contract(manifest_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    duplicate_config = config["duplicates"]
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "config_sha256": config["_config_hash"],
        "dhash_size": int(duplicate_config["dhash_size"]),
        "hamming_threshold": int(duplicate_config["hamming_threshold"]),
    }


def _load_or_generate_candidates(
    manifest: pd.DataFrame, manifest_path: Path, config: dict[str, Any]
) -> tuple[pd.DataFrame, Path, Path]:
    duplicate_config = config["duplicates"]
    candidate_path = resolve_config_path(config, duplicate_config["candidate_csv"])
    meta_path = resolve_config_path(config, duplicate_config["candidate_meta_json"])
    contract = _candidate_contract(manifest_path, config)
    if candidate_path.exists() or meta_path.exists():
        if not candidate_path.exists() or not meta_path.exists():
            raise RuntimeError("Near-duplicate candidate CSV and metadata sidecar must exist together")
        with meta_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        for key, expected in contract.items():
            if metadata.get(key) != expected:
                raise RuntimeError(f"Near-duplicate candidate contract mismatch for {key}")
        if metadata.get("candidate_sha256") != sha256_file(candidate_path):
            raise RuntimeError("Near-duplicate candidate CSV hash mismatch")
        candidates = pd.read_csv(candidate_path)
        if int(metadata.get("candidate_count", -1)) != len(candidates):
            raise RuntimeError("Near-duplicate candidate count does not match its sidecar")
        return candidates, candidate_path, meta_path

    candidates = duplicate_candidates(
        manifest,
        hash_size=int(duplicate_config["dhash_size"]),
        threshold=int(duplicate_config["hamming_threshold"]),
    )
    atomic_write_csv(candidate_path, candidates, refuse_if_exists=True)
    atomic_write_json(
        meta_path,
        {
            **contract,
            "candidate_path": str(candidate_path),
            "candidate_sha256": sha256_file(candidate_path),
            "candidate_count": int(len(candidates)),
        },
        refuse_if_exists=True,
    )
    return candidates, candidate_path, meta_path


def main() -> int:
    args = parse_args()
    config = load_yaml(args.config)
    require_frozen_protocol(config)
    if args.reviewed_candidates is not None and not args.near_duplicates:
        raise ValueError("--reviewed-candidates requires --near-duplicates")

    paths = config["paths"]
    manifest_path = resolve_config_path(config, paths["canonical_manifest_csv"])
    output = resolve_config_path(config, paths["audit_json"])
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite immutable Dataset P0 audit: {output}")
    manifest = pd.read_csv(manifest_path)
    holdout_passed, holdout_status, holdout_metadata = _holdout_status(config, manifest)
    candidates = pd.DataFrame()
    near_duplicate_passed = False
    near_duplicate_status = "not_audited"
    if args.near_duplicates:
        candidates, candidate_path, candidate_meta_path = _load_or_generate_candidates(
            manifest, manifest_path, config
        )
        review_path = args.reviewed_candidates.resolve() if args.reviewed_candidates else None
        if review_path is not None and review_path == candidate_path:
            raise ValueError("Reviewed input must not be the candidate output file")
        near_duplicate_passed, near_duplicate_status = _review_status(candidates, review_path)
        if not candidates.empty and review_path is None:
            print(
                json.dumps(
                    {
                        "status": "PENDING_NEAR_DUPLICATE_REVIEW",
                        "candidate_csv": str(candidate_path),
                        "candidate_meta_json": str(candidate_meta_path),
                        "candidate_count": int(len(candidates)),
                        "dataset_audit_written": False,
                    },
                    indent=2,
                )
            )
            return 3

    report = audit_manifest(
        manifest,
        config,
        inspect_files=args.inspect_files,
        near_duplicate_audit_passed=near_duplicate_passed,
        near_duplicate_status=near_duplicate_status,
        holdout_registered=holdout_passed,
        holdout_status=holdout_status,
    )
    report["provenance"] = {
        "config_sha256": config["_config_hash"],
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "file_existence_inspected": bool(args.inspect_files),
        "near_duplicate_candidates_generated": bool(args.near_duplicates),
        "near_duplicate_candidate_count": int(len(candidates)),
        "holdout_registration": holdout_metadata,
    }
    atomic_write_json(output, report, refuse_if_exists=True)
    print(json.dumps({"output": str(output), "decision": report["decision"]}, indent=2))
    return 0 if report["decision"] != "STOP_DATASET_P0" else 2


if __name__ == "__main__":
    raise SystemExit(main())
