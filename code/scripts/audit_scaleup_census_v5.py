from __future__ import annotations

import argparse
import json
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
    canonical_hash,
    compact_candidates_by_observer,
    fetch_observation_candidates,
    prepare_seed_manifest_for_taxonomy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the frozen metadata-only P0-v5 taxon preselection census."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/scaleup_census_v5.yaml"))
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _clean_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value)


def build_taxon_pool(
    frozen_taxa: pd.DataFrame, feasibility: Mapping[str, Any], census: Mapping[str, Any]
) -> list[dict[str, Any]]:
    expected_current = int(census["expected_current_taxa"])
    expected_reserve = int(census["expected_reserve_taxa"])
    expected_total = int(census["expected_total_taxa"])
    final_min = int(census["final_groups_per_species_min"])
    final_max = int(census["final_groups_per_species_max"])
    if len(frozen_taxa) != expected_current or frozen_taxa["taxon_id"].nunique() != expected_current:
        raise ValueError("The v4 table does not contain the exact current-taxon count")
    ordered = frozen_taxa.sort_values("class_index")
    if ordered["class_index"].astype(int).tolist() != list(range(expected_current)):
        raise ValueError("The v4 class order is not contiguous")

    pool: list[dict[str, Any]] = []
    current_ids: set[int] = set()
    for row in ordered.to_dict("records"):
        taxon_id = int(row["taxon_id"])
        existing = int(row["existing_groups"])
        lower = final_min - existing
        upper = final_max - existing
        if lower < 1 or upper < lower:
            raise ValueError(f"Invalid current-taxon bounds for taxon_id={taxon_id}")
        current_ids.add(taxon_id)
        pool.append(
            {
                "census_index": len(pool),
                "pool": "current_v4",
                "class_index": int(row["class_index"]),
                "feasibility_audit_order": None,
                "taxon_id": taxon_id,
                "scientific_name": _clean_text(row["scientific_name"]),
                "common_name": _clean_text(row.get("common_name")),
                "origin": _clean_text(row["origin"]),
                "existing_groups": existing,
                "lower_target": lower,
                "upper_target": upper,
            }
        )

    reserve_min_groups = int(census["reserve_original_min_groups"])
    reserve_min_observers = int(census["reserve_original_min_observers"])
    for audit_order, row in enumerate(feasibility.get("candidates", [])):
        if not row.get("resolved"):
            continue
        taxon_id = int(row["taxon_id"])
        if taxon_id in current_ids:
            continue
        if int(row["eligible_groups"]) < reserve_min_groups:
            continue
        if int(row["eligible_observers"]) < reserve_min_observers:
            continue
        pool.append(
            {
                "census_index": len(pool),
                "pool": "reserve",
                "class_index": None,
                "feasibility_audit_order": audit_order,
                "taxon_id": taxon_id,
                "scientific_name": _clean_text(row["scientific_name"]),
                "common_name": _clean_text(row.get("resolved_common_name") or row.get("common_name")),
                "origin": "frozen_feasibility_reserve",
                "existing_groups": 0,
                "lower_target": final_min,
                "upper_target": final_max,
                "original_eligible_groups": int(row["eligible_groups"]),
                "original_eligible_observers": int(row["eligible_observers"]),
            }
        )

    reserves = [row for row in pool if row["pool"] == "reserve"]
    ids = [int(row["taxon_id"]) for row in pool]
    if len(reserves) != expected_reserve:
        raise ValueError(f"Expected {expected_reserve} reserve taxa, found {len(reserves)}")
    if len(pool) != expected_total or len(set(ids)) != expected_total:
        raise ValueError("Census pool count or uniqueness does not match the frozen contract")
    return pool


def _per_taxon_record(
    taxon: Mapping[str, Any], candidates: list[dict[str, Any]], retained: list[dict[str, Any]], source: str
) -> dict[str, Any]:
    lower = int(taxon["lower_target"])
    upper = int(taxon["upper_target"])
    return {
        "census_index": int(taxon["census_index"]),
        "pool": str(taxon["pool"]),
        "class_index": taxon.get("class_index"),
        "feasibility_audit_order": taxon.get("feasibility_audit_order"),
        "taxon_id": int(taxon["taxon_id"]),
        "scientific_name": str(taxon["scientific_name"]),
        "existing_groups": int(taxon["existing_groups"]),
        "lower_target": lower,
        "upper_target": upper,
        "eligible_groups": len(candidates),
        "eligible_observers": len({int(row["observer_id"]) for row in candidates}),
        "retained_candidates": len(retained),
        "shortfall": max(0, lower - len(retained)),
        "record_source": source,
    }


def bootstrap_v4_prefix(
    pool: list[dict[str, Any]], v4_snapshot: Mapping[str, Any], v4_audit: Mapping[str, Any], config: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[int], list[dict[str, Any]]]:
    bootstrap = config["bootstrap"]
    expected_count = int(bootstrap["expected_completed_current_taxa"])
    expected_ids = [int(row["taxon_id"]) for row in pool[:expected_count]]
    completed = [int(value) for value in v4_snapshot.get("completed_taxa", [])]
    if v4_snapshot.get("status") != bootstrap["expected_v4_status"]:
        raise RuntimeError("The v4 candidate snapshot status does not match the census bootstrap contract")
    if v4_audit.get("status") != bootstrap["expected_v4_audit_status"]:
        raise RuntimeError("The v4 audit status does not match the census bootstrap contract")
    if v4_snapshot.get("contract_hash") != bootstrap["expected_v4_contract_hash"]:
        raise RuntimeError("The v4 snapshot contract hash does not match the census bootstrap contract")
    if v4_audit.get("candidate_snapshot_sha256") != bootstrap["expected_v4_candidate_sha256"]:
        raise RuntimeError("The v4 audit does not reference the expected candidate snapshot")
    if completed != expected_ids or len(completed) != expected_count:
        raise RuntimeError("The v4 bootstrap is not the exact expected current-taxon prefix")
    groups = [dict(row) for row in v4_snapshot.get("groups", [])]
    if len(groups) != int(bootstrap["expected_candidate_groups"]):
        raise RuntimeError("The v4 bootstrap candidate-group count is unexpected")
    old_rows = list(v4_snapshot.get("per_species", []))
    if len(old_rows) != expected_count:
        raise RuntimeError("The v4 bootstrap per-species table is incomplete")

    records: list[dict[str, Any]] = []
    for taxon, old in zip(pool[:expected_count], old_rows, strict=True):
        if int(old["taxon_id"]) != int(taxon["taxon_id"]):
            raise RuntimeError("The v4 bootstrap per-species order does not match the fixed pool")
        if int(old["lower_target"]) != int(taxon["lower_target"]):
            raise RuntimeError("The v4 bootstrap lower target differs from the census contract")
        if int(old["upper_target"]) != int(taxon["upper_target"]):
            raise RuntimeError("The v4 bootstrap upper target differs from the census contract")
        copied = dict(old)
        copied.update(
            {
                "census_index": int(taxon["census_index"]),
                "pool": "current_v4",
                "feasibility_audit_order": None,
                "existing_groups": int(taxon["existing_groups"]),
                "shortfall": max(0, int(taxon["lower_target"]) - int(old["retained_candidates"])),
                "record_source": "v4_hash_verified_bootstrap",
            }
        )
        records.append(copied)
    return groups, completed, records


def _write_snapshot(
    path: Path,
    status: str,
    contract: Mapping[str, Any],
    contract_hash: str,
    pool: list[dict[str, Any]],
    completed: list[int],
    per_taxon: list[dict[str, Any]],
    groups: list[dict[str, Any]],
) -> None:
    atomic_write_json(
        path,
        {
            "status": status,
            "contract_hash": contract_hash,
            "contract": dict(contract),
            "pool_sha256": canonical_hash(pool),
            "completed_taxa": completed,
            "total_taxa": len(pool),
            "per_taxon": per_taxon,
            "groups": groups,
        },
    )


def main() -> int:
    args = parse_args()
    config = load_yaml(args.config)
    require_frozen_protocol(config)
    paths = config["paths"]
    census = config["census"]

    snapshot_path = resolve_config_path(config, paths["census_snapshot_json"])
    audit_path = resolve_config_path(config, paths["census_audit_json"])
    if audit_path.exists():
        raise FileExistsError("The final frozen census audit already exists and cannot be overwritten")

    feasibility_path = resolve_config_path(config, paths["feasibility_audit_json"])
    taxa_path = resolve_config_path(config, paths["frozen_taxa_v4_csv"])
    v4_snapshot_path = resolve_config_path(config, paths["v4_candidate_snapshot_json"])
    v4_audit_path = resolve_config_path(config, paths["v4_selection_audit_json"])
    for path, expected_key in (
        (v4_snapshot_path, "expected_v4_candidate_sha256"),
        (v4_audit_path, "expected_v4_audit_sha256"),
    ):
        actual = sha256_file(path)
        expected = str(config["bootstrap"][expected_key])
        if actual != expected:
            raise RuntimeError(f"Bootstrap hash mismatch for {path}: expected {expected}, got {actual}")

    feasibility = _load_json(feasibility_path)
    if feasibility.get("status") != "PASS_FREEZE_100_TAXA" or not feasibility.get("gate_pass"):
        raise RuntimeError("The frozen taxonomy feasibility audit is not a PASS")
    frozen_taxa = pd.read_csv(taxa_path)
    pool = build_taxon_pool(frozen_taxa, feasibility, census)

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
    taxonomy_contract = {
        "removed_seed_taxon_id": 145224,
        "expected_seed_taxa": 79,
    }
    _, excluded_observations, excluded_observers = prepare_seed_manifest_for_taxonomy(
        seed_all, class_map, taxonomy_contract
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
    contract_hash = canonical_hash(contract)

    all_groups: list[dict[str, Any]]
    completed: list[int]
    per_taxon: list[dict[str, Any]]
    if snapshot_path.exists():
        previous = _load_json(snapshot_path)
        if previous.get("contract_hash") != contract_hash:
            raise RuntimeError("Existing census snapshot does not match the frozen census contract")
        if previous.get("status") not in {"RUNNING", "COMPLETE"}:
            raise RuntimeError(f"Unrecognized census snapshot status: {previous.get('status')}")
        all_groups = [dict(row) for row in previous.get("groups", [])]
        completed = [int(value) for value in previous.get("completed_taxa", [])]
        per_taxon = [dict(row) for row in previous.get("per_taxon", [])]
    else:
        all_groups, completed, per_taxon = bootstrap_v4_prefix(
            pool,
            _load_json(v4_snapshot_path),
            _load_json(v4_audit_path),
            config,
        )
        _write_snapshot(
            snapshot_path, "RUNNING", contract, contract_hash, pool, completed, per_taxon, all_groups
        )

    expected_order = [int(row["taxon_id"]) for row in pool]
    if completed != expected_order[: len(completed)]:
        raise RuntimeError("Census resume artifact is not a prefix of the frozen pool order")
    if len(per_taxon) != len(completed):
        raise RuntimeError("Census resume per-taxon records do not match completed taxa")

    source = config["source"]
    multiplier = int(census["candidate_multiplier"])
    for position, taxon in enumerate(pool[len(completed) :], start=len(completed) + 1):
        cap = int(taxon["upper_target"]) * multiplier
        candidates = fetch_observation_candidates(
            taxon,
            source,
            excluded_observations,
            excluded_observers,
            seed=int(config["seed"]),
            early_stop_unique_observers=cap,
        )
        retained = compact_candidates_by_observer(
            candidates, cap, int(config["seed"]), int(taxon["taxon_id"])
        )
        all_groups.extend(retained)
        completed.append(int(taxon["taxon_id"]))
        record = _per_taxon_record(taxon, candidates, retained, "census_query")
        per_taxon.append(record)
        _write_snapshot(
            snapshot_path, "RUNNING", contract, contract_hash, pool, completed, per_taxon, all_groups
        )
        print(
            f"[{position:03d}/{len(pool)}] {str(taxon['pool']).upper()} "
            f"{taxon['scientific_name']}: target={taxon['lower_target']}-{taxon['upper_target']} "
            f"candidates={len(retained)} shortfall={record['shortfall']}",
            flush=True,
        )

    _write_snapshot(
        snapshot_path, "COMPLETE", contract, contract_hash, pool, completed, per_taxon, all_groups
    )
    current = [row for row in per_taxon if row["pool"] == "current_v4"]
    reserve = [row for row in per_taxon if row["pool"] == "reserve"]
    current_shortfalls = [row for row in current if int(row["shortfall"]) > 0]
    reserve_feasible = [row for row in reserve if int(row["shortfall"]) == 0]
    reserve_shortfalls = [row for row in reserve if int(row["shortfall"]) > 0]
    audit = {
        "status": "COMPLETE_P0_V5_PRESELECTION_CENSUS",
        "dataset_p0_decision_authorized": False,
        "contract_hash": contract_hash,
        "candidate_snapshot_sha256": sha256_file(snapshot_path),
        "pool_sha256": canonical_hash(pool),
        "counts": {
            "total_taxa": len(per_taxon),
            "current_taxa": len(current),
            "reserve_taxa": len(reserve),
            "current_shortfalls": len(current_shortfalls),
            "reserve_individually_feasible": len(reserve_feasible),
            "reserve_shortfalls": len(reserve_shortfalls),
            "retained_candidate_groups": len(all_groups),
        },
        "current_shortfalls": current_shortfalls,
        "reserve_individually_feasible": reserve_feasible,
        "reserve_shortfalls": reserve_shortfalls,
        "replacement_capacity_sufficient_individually": len(reserve_feasible)
        >= len(current_shortfalls),
        "global_milp_evaluated": False,
        "class_table_written": False,
        "metadata_manifest_written": False,
        "download_authorized": False,
        "per_taxon": per_taxon,
    }
    atomic_write_json(audit_path, audit, refuse_if_exists=True)
    print(
        json.dumps(
            {
                "status": audit["status"],
                "counts": audit["counts"],
                "dataset_p0_decision_authorized": False,
                "output": str(audit_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
