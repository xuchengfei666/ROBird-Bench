from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from robird.io import (
    atomic_write_json,
    load_yaml,
    require_frozen_protocol,
    resolve_config_path,
    sha256_file,
)
from robird.manifest import combine_seed_manifests, normalize_legacy_manifest
from robird.scaleup import (
    _api_url,
    api_get_json,
    canonical_hash,
    stable_hash_int,
)


SCRIPT_DIR = Path(__file__).resolve().parent
FROZEN_V5_SCRIPT = SCRIPT_DIR / "audit_scaleup_census_v5.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Continue the frozen metadata-only P0-v5 census with the v5.1 URL parser."
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/scaleup_census_v5_1.yaml")
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _load_frozen_v5_script() -> Any:
    spec = importlib.util.spec_from_file_location("robird_frozen_census_v5", FROZEN_V5_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load frozen v5 census script: {FROZEN_V5_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalize_large_url_v5_1(url: str) -> str:
    """Normalize known iNaturalist size objects, including an empty extension."""
    path, separator, query = url.partition("?")
    filename = path.rsplit("/", 1)[-1]
    stem, dot, extension = filename.rpartition(".")
    if stem not in {"square", "small", "medium", "large"} or not dot:
        raise ValueError(f"Unexpected iNaturalist photo URL: {url}")
    path = path[: -len(filename)] + f"large.{extension}"
    return path + (separator + query if separator else "")


def fetch_observation_candidates_v5_1(
    taxon: Mapping[str, Any],
    source: Mapping[str, Any],
    excluded_observations: set[int],
    excluded_observers: set[int],
    *,
    seed: int,
    early_stop_unique_observers: int | None = None,
) -> list[dict[str, Any]]:
    """Copy the frozen v5 query contract with only the versioned URL parser changed."""
    allowed = {str(value).strip().casefold() for value in source["photo_licenses"]}
    by_observation: dict[int, dict[str, Any]] = {}
    for page in range(1, int(source["pages"]) + 1):
        params = {
            "taxon_id": int(taxon["taxon_id"]),
            "photos": "true",
            "photo_license": ",".join(source["photo_licenses"]),
            "quality_grade": source["quality_grade"],
            "created_d2": source["created_d2"],
            "per_page": int(source["per_page"]),
            "page": page,
            "order": "desc",
            "order_by": "created_at",
        }
        payload = api_get_json(
            _api_url(str(source["api_base"]), "observations", params),
            int(source["request_attempts"]),
        )
        for observation in payload.get("results", []):
            if not observation.get("user"):
                continue
            observation_id = int(observation["id"])
            observer_id = int(observation["user"]["id"])
            if observation_id in excluded_observations or observer_id in excluded_observers:
                continue
            observed_taxon = observation.get("taxon") or {}
            lineage = {int(value) for value in observed_taxon.get("ancestor_ids", [])}
            if observed_taxon.get("id") is not None:
                lineage.add(int(observed_taxon["id"]))
            if int(taxon["taxon_id"]) not in lineage:
                continue
            photos = []
            for photo in observation.get("photos", []):
                license_code = str(photo.get("license_code", "")).strip().casefold()
                url = str(photo.get("url", "")).strip()
                if license_code not in allowed or photo.get("hidden", False) or not url:
                    continue
                dimensions = photo.get("original_dimensions") or {}
                photos.append(
                    {
                        "photo_id": int(photo["id"]),
                        "license_code": license_code,
                        "attribution": str(photo.get("attribution", "")),
                        "width": dimensions.get("width"),
                        "height": dimensions.get("height"),
                        "url": normalize_large_url_v5_1(url),
                    }
                )
            if len(photos) < int(source["min_photos_per_group"]):
                continue
            photos = sorted(
                photos,
                key=lambda photo: stable_hash_int(
                    seed,
                    "scaleup-photo",
                    taxon["taxon_id"],
                    observation_id,
                    photo["photo_id"],
                ),
            )[: int(source["max_photos_per_group"])]
            by_observation[observation_id] = {
                "taxon_id": int(taxon["taxon_id"]),
                "scientific_name": str(taxon["scientific_name"]),
                "common_name": str(taxon.get("common_name", "")),
                "observation_id": observation_id,
                "observer_id": observer_id,
                "observed_on": observation.get("observed_on"),
                "created_at": observation.get("created_at"),
                "place_ids": observation.get("place_ids", []),
                "photos": photos,
            }
        if early_stop_unique_observers is not None:
            unique_observers = {row["observer_id"] for row in by_observation.values()}
            if len(unique_observers) >= early_stop_unique_observers:
                break
        time.sleep(float(source["request_delay_seconds"]))
    return list(by_observation.values())


def _build_contract(config: Mapping[str, Any], module: Any, pool: list[dict[str, Any]]) -> tuple[dict[str, Any], str]:
    paths = config["paths"]
    feasibility_path = resolve_config_path(config, paths["feasibility_audit_json"])
    taxa_path = resolve_config_path(config, paths["frozen_taxa_v4_csv"])
    v4_snapshot_path = resolve_config_path(config, paths["v4_candidate_snapshot_json"])
    v4_audit_path = resolve_config_path(config, paths["v4_selection_audit_json"])
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
    class_map = {
        int(row["taxon_id"]): int(row["class_index"])
        for row in pool
        if row["pool"] == "current_v4"
    }
    _, _, _ = module.prepare_seed_manifest_for_taxonomy(
        seed_all,
        class_map,
        {"removed_seed_taxon_id": 145224, "expected_seed_taxa": 79},
    )
    input_paths = {
        key: resolve_config_path(config, paths[key])
        for key in (
            "original_manifest_csv",
            "original_source_json",
            "original_download_ledger",
            "confirmation_manifest_csv",
            "confirmation_source_json",
            "confirmation_download_ledger",
        )
    }
    contract = {
        "config_sha256": config["_config_hash"],
        "freeze_record_sha256": sha256_file(
            resolve_config_path(config, config["protocol"]["freeze_record"])
        ),
        "v4_candidate_sha256": sha256_file(v4_snapshot_path),
        "v4_audit_sha256": sha256_file(v4_audit_path),
        "frozen_taxa_v4_sha256": sha256_file(taxa_path),
        "feasibility_sha256": sha256_file(feasibility_path),
        "source_sha256": canonical_hash(config["source"]),
        "pool_sha256": canonical_hash(pool),
        "seed_inputs": {key: sha256_file(path) for key, path in input_paths.items()},
    }
    return contract, canonical_hash(contract)


def bridge_v5_checkpoint(config: Mapping[str, Any], module: Any) -> None:
    paths = config["paths"]
    legacy_path = resolve_config_path(config, paths["legacy_v5_snapshot_json"])
    target_path = resolve_config_path(config, paths["census_snapshot_json"])
    bootstrap = config["bootstrap"]
    replace_preflight_bridge = False
    if target_path.exists():
        existing = _load_json(target_path)
        existing_completed = existing.get("completed_taxa", [])
        existing_groups = existing.get("groups", [])
        if (
            existing.get("status") != "RUNNING"
            or len(existing_completed) != int(bootstrap["expected_completed_taxa"])
            or len(existing_groups) != int(bootstrap["expected_candidate_groups"])
        ):
            raise FileExistsError(
                "Existing v5.1 snapshot is not the exact preflight bridge and cannot be overwritten"
            )
        replace_preflight_bridge = True
    actual_hash = sha256_file(legacy_path)
    if actual_hash != str(bootstrap["expected_legacy_v5_candidate_sha256"]):
        raise RuntimeError(
            f"Legacy v5 snapshot hash mismatch: expected {bootstrap['expected_legacy_v5_candidate_sha256']}, got {actual_hash}"
        )
    legacy = _load_json(legacy_path)
    if legacy.get("status") != bootstrap["expected_legacy_v5_status"]:
        raise RuntimeError("Legacy v5 snapshot is not the expected RUNNING checkpoint")
    if legacy.get("contract_hash") != bootstrap["expected_legacy_v5_contract_hash"]:
        raise RuntimeError("Legacy v5 snapshot contract hash does not match the frozen checkpoint")
    completed = [int(value) for value in legacy.get("completed_taxa", [])]
    groups = [dict(row) for row in legacy.get("groups", [])]
    per_taxon = [dict(row) for row in legacy.get("per_taxon", [])]
    if len(completed) != int(bootstrap["expected_completed_taxa"]):
        raise RuntimeError("Legacy v5 completed-taxa count does not match the frozen checkpoint")
    if len(groups) != int(bootstrap["expected_candidate_groups"]):
        raise RuntimeError("Legacy v5 group count does not match the frozen checkpoint")
    if completed[-1] != int(bootstrap["expected_last_taxon_id"]):
        raise RuntimeError("Legacy v5 last completed taxon does not match the frozen checkpoint")
    feasibility = _load_json(resolve_config_path(config, paths["feasibility_audit_json"]))
    frozen_taxa = pd.read_csv(resolve_config_path(config, paths["frozen_taxa_v4_csv"]))
    pool = module.build_taxon_pool(frozen_taxa, feasibility, config["census"])
    expected_prefix = [int(row["taxon_id"]) for row in pool[: len(completed)]]
    if completed != expected_prefix or len(per_taxon) != len(completed):
        raise RuntimeError("Legacy v5 checkpoint is not the expected fixed-pool prefix")
    if legacy.get("pool_sha256") != canonical_hash(pool):
        raise RuntimeError("Legacy v5 checkpoint pool hash does not match the fixed pool")
    contract, contract_hash = _build_contract(config, module, pool)
    atomic_write_json(
        target_path,
        {
            "status": "RUNNING",
            "contract_hash": contract_hash,
            "contract": contract,
            "pool_sha256": canonical_hash(pool),
            "completed_taxa": completed,
            "total_taxa": len(pool),
            "per_taxon": per_taxon,
            "groups": groups,
        },
        refuse_if_exists=not replace_preflight_bridge,
    )


def main() -> int:
    args = parse_args()
    config = load_yaml(args.config)
    require_frozen_protocol(config)
    frozen_v5 = _load_frozen_v5_script()
    bridge_v5_checkpoint(config, frozen_v5)
    frozen_v5.fetch_observation_candidates = fetch_observation_candidates_v5_1
    sys.argv = [sys.argv[0], "--config", str(args.config)]
    return int(frozen_v5.main())


if __name__ == "__main__":
    raise SystemExit(main())
