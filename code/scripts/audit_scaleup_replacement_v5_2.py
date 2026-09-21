from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from robird.io import (
    atomic_write_json,
    load_yaml,
    require_frozen_protocol,
    resolve_config_path,
    sha256_file,
)
from robird.manifest import (
    combine_seed_manifests,
    normalize_legacy_manifest,
    validate_manifest_schema,
)
from robird.scaleup import (
    canonical_hash,
    groups_to_canonical,
    prepare_seed_manifest_for_taxonomy,
    select_bounded_total_groups,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit the frozen P0-v5.2 joint replacement rule over all reserve taxa."
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/scaleup_replacement_v5_2.yaml")
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _replace_table(
    v4: pd.DataFrame,
    replacement: Mapping[str, Any],
    feasibility_row: Mapping[str, Any],
    removed_id: int,
) -> pd.DataFrame:
    table = v4.copy()
    matches = table.index[table["taxon_id"].astype(int) == removed_id].tolist()
    if len(matches) != 1:
        raise ValueError("The v4 table must contain exactly one removed seed class")
    index = matches[0]
    table.loc[index, "taxon_id"] = int(replacement["taxon_id"])
    table.loc[index, "scientific_name"] = str(replacement["scientific_name"])
    table.loc[index, "common_name"] = str(
        feasibility_row.get("resolved_common_name") or feasibility_row.get("common_name", "")
    )
    table.loc[index, "family"] = str(feasibility_row.get("family", ""))
    table.loc[index, "genus"] = str(feasibility_row.get("genus", ""))
    table.loc[index, "origin"] = "inat2021_taxonomic_neighbor"
    table.loc[index, "taxonomy_category_id"] = int(feasibility_row["category_id"])
    table.loc[index, "existing_groups"] = 0
    table.loc[index, "new_group_target"] = 50
    table["class_index"] = np.arange(len(table), dtype=np.int64)
    if len(table) != 100 or table["taxon_id"].nunique() != 100:
        raise ValueError("Temporary replacement table must contain 100 unique taxa")
    if sorted(table["class_index"].astype(int)) != list(range(100)):
        raise ValueError("Temporary replacement class indices are not contiguous")
    if int((table["origin"] == "inspected_seed").sum()) != 78:
        raise ValueError("Temporary replacement table must contain 78 seed taxa")
    if int((table["origin"] == "inat2021_taxonomic_neighbor").sum()) != 22:
        raise ValueError("Temporary replacement table must contain 22 expansion taxa")
    return table


def _targets(table: pd.DataFrame, selection: Mapping[str, Any]) -> tuple[dict[int, int], dict[int, int], int]:
    final_min = int(selection["final_groups_per_species_min"])
    final_max = int(selection["final_groups_per_species_max"])
    existing = dict(
        zip(table["taxon_id"].astype(int), table["existing_groups"].astype(int), strict=True)
    )
    lower = {taxon_id: final_min - count for taxon_id, count in existing.items()}
    upper = {taxon_id: final_max - count for taxon_id, count in existing.items()}
    total_new = int(selection["expected_new_groups"])
    if sum(existing.values()) + total_new != int(selection["expected_groups"]):
        raise ValueError("Replacement new-group accounting does not reach exactly 5,000 groups")
    if not sum(lower.values()) <= total_new <= sum(upper.values()):
        raise ValueError("Replacement total is outside the summed per-taxon bounds")
    return lower, upper, total_new


def _metadata_audit(
    selected: list[dict[str, Any]],
    table: pd.DataFrame,
    seed_manifest: pd.DataFrame,
    excluded_observations: set[int],
    excluded_observers: set[int],
    lower: Mapping[int, int],
    upper: Mapping[int, int],
    expected_species: int,
    expected_groups: int,
    allowed_licenses: set[str],
    min_photos: int,
    max_photos: int,
) -> tuple[dict[str, bool], dict[str, Any], pd.DataFrame]:
    class_map = dict(zip(table["taxon_id"].astype(int), table["class_index"].astype(int), strict=True))
    expansion = groups_to_canonical(selected, class_map, "robird_scaleup_v5_2_candidate")
    combined = pd.concat([seed_manifest, expansion], ignore_index=True, sort=False)
    combined = combined.sort_values(["class_index", "observation_id", "photo_id"]).reset_index(drop=True)
    combined["row_id"] = np.arange(len(combined), dtype=np.int64)
    violations = validate_manifest_schema(combined)
    group_counts = (
        combined[["taxon_id", "observation_id"]]
        .drop_duplicates()
        .groupby("taxon_id")
        .size()
        .astype(int)
        .to_dict()
    )
    photo_counts = (
        combined[["taxon_id", "observation_id", "photo_id"]]
        .drop_duplicates()
        .groupby("observation_id")
        .size()
        .astype(int)
        .to_dict()
    )
    selected_observers = [int(row["observer_id"]) for row in selected]
    selected_observations = {int(row["observation_id"]) for row in selected}
    all_group_photos = [len(row.get("photos", [])) for row in selected]
    final_min = int(min(group_counts.values(), default=0))
    final_max = int(max(group_counts.values(), default=0))
    gates = {
        "schema": not violations,
        "species_exact": int(combined["taxon_id"].nunique()) == expected_species,
        "groups_exact": int(combined["observation_id"].nunique()) == expected_groups,
        "species_balance": set(group_counts) == set(lower)
        and all(lower[taxon_id] <= group_counts.get(taxon_id, 0) <= upper[taxon_id] for taxon_id in lower),
        "photos_minimum": len(combined) >= 12000,
        "new_observers_unique": len(selected_observers) == len(set(selected_observers)),
        "new_observers_exclude_seed": not (set(selected_observers) & excluded_observers),
        "new_observations_exclude_seed": not (selected_observations & excluded_observations),
        "licenses_allowed": set(combined["license_code"].astype(str).str.lower()) <= allowed_licenses,
        "selected_group_photo_bounds": all(min_photos <= count <= max_photos for count in all_group_photos),
        "selected_photo_ids_unique": len(
            [photo["photo_id"] for row in selected for photo in row.get("photos", [])]
        )
        == len(
            {
                int(photo["photo_id"])
                for row in selected
                for photo in row.get("photos", [])
            }
        ),
        "final_group_photo_bounds": all(min_photos <= count <= max_photos for count in photo_counts.values()),
    }
    details = {
        "counts": {
            "species": int(combined["taxon_id"].nunique()),
            "groups": int(combined["observation_id"].nunique()),
            "photos": int(len(combined)),
            "seed_groups": int(seed_manifest["observation_id"].nunique()),
            "selected_new_groups": len(selected),
            "selected_new_observers": len(set(selected_observers)),
            "final_min_groups_per_species": final_min,
            "final_max_groups_per_species": final_max,
        },
        "violations": violations,
        "final_groups_per_species": {str(key): value for key, value in group_counts.items()},
    }
    return gates, details, combined


def _load_seed_inputs(config: Mapping[str, Any], class_map: Mapping[int, int]) -> tuple[pd.DataFrame, set[int], set[int], dict[str, str]]:
    paths = config["paths"]
    original_path = resolve_config_path(config, paths["original_manifest_csv"])
    original_source = resolve_config_path(config, paths["original_source_json"])
    original_ledger = resolve_config_path(config, paths["original_download_ledger"])
    confirmation_path = resolve_config_path(config, paths["confirmation_manifest_csv"])
    confirmation_source = resolve_config_path(config, paths["confirmation_source_json"])
    confirmation_ledger = resolve_config_path(config, paths["confirmation_download_ledger"])
    original = normalize_legacy_manifest(original_path, original_source, original_ledger, "roq_original_inspected")
    confirmation = normalize_legacy_manifest(
        confirmation_path, confirmation_source, confirmation_ledger, "roq_confirmation_inspected"
    )
    seed_all = combine_seed_manifests(original, confirmation)
    seed, excluded_observations, excluded_observers = prepare_seed_manifest_for_taxonomy(
        seed_all,
        class_map,
        {
            "removed_seed_taxon_id": int(config["removed_seed_taxon_id"]),
            "expected_seed_taxa": int(config["selection"]["expected_current_seed_taxa"]),
        },
    )
    hashes = {
        "original_manifest_csv": sha256_file(original_path),
        "confirmation_manifest_csv": sha256_file(confirmation_path),
        "original_source_json": sha256_file(original_source),
        "confirmation_source_json": sha256_file(confirmation_source),
        "original_download_ledger": sha256_file(original_ledger),
        "confirmation_download_ledger": sha256_file(confirmation_ledger),
    }
    return seed, excluded_observations, excluded_observers, hashes


def _seed_input_hashes(config: Mapping[str, Any]) -> dict[str, str]:
    paths = config["paths"]
    values = {
        "original_manifest_csv": resolve_config_path(config, paths["original_manifest_csv"]),
        "confirmation_manifest_csv": resolve_config_path(config, paths["confirmation_manifest_csv"]),
        "original_source_json": resolve_config_path(config, paths["original_source_json"]),
        "confirmation_source_json": resolve_config_path(config, paths["confirmation_source_json"]),
        "original_download_ledger": resolve_config_path(config, paths["original_download_ledger"]),
        "confirmation_download_ledger": resolve_config_path(config, paths["confirmation_download_ledger"]),
    }
    return {key: sha256_file(path) for key, path in values.items()}


def main() -> int:
    args = parse_args()
    config = load_yaml(args.config)
    require_frozen_protocol(config)
    paths = config["paths"]
    selection = config["selection"]
    audit_config = config["audit"]
    attempts_path = resolve_config_path(config, paths["attempts_json"])
    audit_path = resolve_config_path(config, paths["audit_json"])
    if audit_path.exists():
        raise FileExistsError(f"Replacement audit already exists: {audit_path}")

    snapshot_path = resolve_config_path(config, paths["v5_1_snapshot_json"])
    snapshot_hash = sha256_file(snapshot_path)
    if snapshot_hash != str(audit_config["expected_v5_1_snapshot_sha256"]):
        raise RuntimeError("v5.1 census snapshot hash does not match the frozen replacement contract")
    v5_snapshot = _load_json(snapshot_path)
    if v5_snapshot.get("status") != audit_config["expected_v5_1_snapshot_status"]:
        raise RuntimeError("v5.1 census snapshot status is not COMPLETE")
    if len(v5_snapshot.get("completed_taxa", [])) != int(audit_config["expected_v5_1_taxa"]):
        raise RuntimeError("v5.1 census snapshot taxon count is unexpected")
    if len(v5_snapshot.get("groups", [])) != int(audit_config["expected_v5_1_groups"]):
        raise RuntimeError("v5.1 census snapshot group count is unexpected")

    census_audit_path = resolve_config_path(config, paths["v5_1_audit_json"])
    census_audit_hash = sha256_file(census_audit_path)
    if census_audit_hash != str(audit_config["expected_v5_1_audit_sha256"]):
        raise RuntimeError("v5.1 census audit hash does not match the frozen replacement contract")
    census_audit = _load_json(census_audit_path)
    if census_audit.get("status") != audit_config["expected_v5_1_audit_status"]:
        raise RuntimeError("v5.1 census audit status is not COMPLETE")
    counts = census_audit.get("counts", {})
    for key in ("current_shortfalls", "reserve_individually_feasible", "reserve_shortfalls"):
        if int(counts.get(key, -1)) != int(audit_config[f"expected_v5_1_{key}"]):
            raise RuntimeError(f"v5.1 audit count mismatch: {key}")
    if census_audit.get("dataset_p0_decision_authorized") is not False:
        raise RuntimeError("v5.1 census is unexpectedly authorizing Dataset P0")

    v4_table_path = resolve_config_path(config, paths["frozen_taxa_v4_csv"])
    v4_table = pd.read_csv(v4_table_path)
    feasibility = _load_json(resolve_config_path(config, paths["feasibility_audit_json"]))
    feasibility_by_id = {int(row["taxon_id"]): row for row in feasibility["candidates"]}
    current_shortfalls = [row for row in census_audit.get("current_shortfalls", [])]
    if len(current_shortfalls) != 1 or int(current_shortfalls[0]["taxon_id"]) != int(config["removed_seed_taxon_id"]):
        raise RuntimeError("v5.1 current shortfall does not match the frozen v5.2 removal rule")

    reserves = sorted(
        [dict(row) for row in census_audit.get("reserve_individually_feasible", [])],
        key=lambda row: (int(row["census_index"]), int(row["taxon_id"])),
    )
    if len(reserves) != int(audit_config["expected_v5_1_reserve_feasible"]):
        raise RuntimeError("v5.1 reserve pool size does not match the frozen replacement contract")
    reserve_ids = [int(row["taxon_id"]) for row in reserves]
    if len(set(reserve_ids)) != len(reserve_ids):
        raise RuntimeError("v5.1 reserve pool contains duplicate taxa")

    base_ids = set(v4_table["taxon_id"].astype(int))
    if int(config["removed_seed_taxon_id"]) not in base_ids:
        raise RuntimeError("Removed seed taxon is absent from the v4 table")
    if set(reserve_ids) & base_ids:
        raise RuntimeError("Reserve pool overlaps the v4 class table")

    source_hash = canonical_hash(config["source"])
    contract = {
        "config_sha256": config["_config_hash"],
        "freeze_record_sha256": sha256_file(resolve_config_path(config, config["protocol"]["freeze_record"])),
        "v4_table_sha256": sha256_file(v4_table_path),
        "v5_1_snapshot_sha256": snapshot_hash,
        "v5_1_audit_sha256": census_audit_hash,
        "source_contract_hash": source_hash,
        "reserve_order": reserve_ids,
        "seed_inputs": _seed_input_hashes(config),
    }

    attempts: list[dict[str, Any]] = []
    if attempts_path.exists():
        previous = _load_json(attempts_path)
        if previous.get("contract_hash") != canonical_hash(contract):
            raise RuntimeError("Existing replacement attempt ledger has a different contract")
        attempts = [dict(row) for row in previous.get("attempts", [])]
        if [int(row["taxon_id"]) for row in attempts] != reserve_ids[: len(attempts)]:
            raise RuntimeError("Replacement attempt ledger is not a reserve-order prefix")

    selected_candidate: dict[str, Any] | None = None
    selected_groups: list[dict[str, Any]] = []
    selected_details: dict[str, Any] = {}
    selected_gates: dict[str, bool] = {}
    selected_table: pd.DataFrame | None = None

    for reserve in reserves[len(attempts) :]:
        taxon_id = int(reserve["taxon_id"])
        feasibility_row = feasibility_by_id.get(taxon_id)
        if feasibility_row is None:
            raise RuntimeError(f"Reserve taxon {taxon_id} is missing from feasibility audit")
        table = _replace_table(v4_table, reserve, feasibility_row, int(config["removed_seed_taxon_id"]))
        lower, upper, total_new = _targets(table, selection)
        active_ids = set(table["taxon_id"].astype(int))
        candidate_rows = [
            dict(row)
            for row in v5_snapshot["groups"]
            if int(row["taxon_id"]) in active_ids
        ]
        cap_ok = all(
            sum(int(row["taxon_id"]) == taxon_id for row in candidate_rows)
            <= upper[taxon_id] * int(selection["candidate_multiplier"])
            for taxon_id in active_ids
        )
        class_map = dict(zip(table["taxon_id"].astype(int), table["class_index"].astype(int), strict=True))
        seed_manifest, excluded_observations, excluded_observers, seed_hashes = _load_seed_inputs(
            config, class_map
        )
        contract_hash = canonical_hash(contract)
        selected, solver_audit = select_bounded_total_groups(
            candidate_rows,
            lower,
            upper,
            total_new,
            int(config["seed"]),
            observer_capacity=int(selection["max_groups_per_new_observer"]),
        )
        if selected:
            gates, details, combined = _metadata_audit(
                selected,
                table,
                seed_manifest,
                excluded_observations,
                excluded_observers,
                lower,
                upper,
                int(selection["expected_species"]),
                int(selection["expected_groups"]),
                {str(value).lower() for value in config["source"]["photo_licenses"]},
                int(config["source"]["min_photos_per_group"]),
                int(config["source"]["max_photos_per_group"]),
            )
        else:
            gates = {"candidate_cap": cap_ok}
            details = {"counts": {}, "violations": []}
            combined = pd.DataFrame()
        gates["candidate_cap"] = cap_ok
        gate_pass = bool(solver_audit.get("feasible")) and all(gates.values())
        attempt = {
            "census_index": int(reserve["census_index"]),
            "feasibility_audit_order": int(reserve["feasibility_audit_order"]),
            "taxon_id": taxon_id,
            "scientific_name": str(reserve["scientific_name"]),
            "lower_target": int(reserve["lower_target"]),
            "upper_target": int(reserve["upper_target"]),
            "solver": solver_audit,
            "gates": gates,
            "gate_pass": gate_pass,
            "details": details,
        }
        attempts.append(attempt)
        atomic_write_json(
            attempts_path,
            {
                "status": "RUNNING",
                "contract_hash": contract_hash,
                "contract": contract,
                "attempts": attempts,
            },
        )
        print(
            f"[{len(attempts):02d}/{len(reserves)}] {reserve['scientific_name']}: "
            f"milp={bool(solver_audit.get('feasible'))} metadata={gate_pass}",
            flush=True,
        )
        if gate_pass:
            selected_candidate = attempt
            selected_groups = selected
            selected_details = details
            selected_gates = gates
            selected_table = table
            break

    if selected_candidate is None:
        status = "STOP_NO_REPLACEMENT_PASSES_METADATA_GATES"
        gate_pass = False
    else:
        status = "PASS_REPLACEMENT_METADATA_CANDIDATE"
        gate_pass = True

    final_contract_hash = canonical_hash(contract)
    final_attempt_status = "COMPLETE" if gate_pass else "STOP"
    atomic_write_json(
        attempts_path,
        {
            "status": final_attempt_status,
            "contract_hash": final_contract_hash,
            "contract": contract,
            "attempts": attempts,
        },
    )
    audit = {
        "status": status,
        "gate_pass": gate_pass,
        "dataset_p0_decision_authorized": False,
        "class_table_written": False,
        "metadata_manifest_written": False,
        "download_authorized": False,
        "global_milp_evaluated": bool(attempts),
        "contract_hash": final_contract_hash,
        "attempts_sha256": sha256_file(attempts_path),
        "v5_1_snapshot_sha256": snapshot_hash,
        "v5_1_audit_sha256": census_audit_hash,
        "attempt_count": len(attempts),
        "reserve_count": len(reserves),
        "replacement_rule": selection["replacement_rule"],
        "attempts": attempts,
        "selected_replacement": selected_candidate,
        "selected_gates": selected_gates,
        "selected_details": selected_details,
        "selected_groups": selected_groups,
        "selected_table_preview": (
            selected_table.to_dict(orient="records") if selected_table is not None else []
        ),
    }
    atomic_write_json(audit_path, audit, refuse_if_exists=True)
    print(
        json.dumps(
            {
                "status": status,
                "gate_pass": gate_pass,
                "attempt_count": len(attempts),
                "selected_replacement": selected_candidate,
                "output": str(audit_path),
            },
            indent=2,
        )
    )
    return 0 if gate_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
