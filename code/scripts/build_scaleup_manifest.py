from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from robird.io import (
    atomic_write_csv,
    atomic_write_json,
    load_yaml,
    require_frozen_protocol,
    resolve_config_path,
    sha256_file,
)
from robird.manifest import combine_seed_manifests, normalize_legacy_manifest, validate_manifest_schema
from robird.scaleup import (
    canonical_hash,
    compact_candidates_by_observer,
    fetch_observation_candidates,
    groups_to_canonical,
    prepare_seed_manifest_for_taxonomy,
    select_balanced_groups,
    select_bounded_total_groups,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the frozen 5,000-group ROBird metadata manifest.")
    parser.add_argument("--config", type=Path, default=Path("configs/scaleup_p0.yaml"))
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _validate_frozen_taxa(
    frame: pd.DataFrame,
    feasibility: dict,
    selection: dict,
    taxonomy_contract: dict | None = None,
) -> None:
    required = {
        "class_index",
        "taxon_id",
        "scientific_name",
        "common_name",
        "origin",
        "existing_groups",
        "new_group_target",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Frozen taxon table is missing columns: {missing}")
    if len(frame) != int(selection["expected_species"]) or frame["taxon_id"].nunique() != len(frame):
        raise ValueError("Frozen taxon table must contain exactly 100 unique taxa")
    if sorted(frame["class_index"].astype(int)) != list(range(len(frame))):
        raise ValueError("Frozen class_index values must be contiguous")
    seed = frame[frame["origin"] == "inspected_seed"]
    added = frame[frame["origin"] == "inat2021_taxonomic_neighbor"]
    expected_seed_taxa = int((taxonomy_contract or {}).get("expected_seed_taxa", 80))
    expected_expansion_taxa = int((taxonomy_contract or {}).get("expected_expansion_taxa", 20))
    if len(seed) != expected_seed_taxa or len(added) != expected_expansion_taxa:
        raise ValueError(
            f"Frozen taxonomy must contain {expected_seed_taxa} seed and "
            f"{expected_expansion_taxa} expansion taxa"
        )
    expected_new_ids = {int(row["taxon_id"]) for row in feasibility["selected_new_taxa"]}
    if taxonomy_contract:
        removed_id = int(taxonomy_contract["removed_seed_taxon_id"])
        replacement_id = int(taxonomy_contract["replacement_taxon_id"])
        frame_ids = set(frame["taxon_id"].astype(int))
        if removed_id in frame_ids or replacement_id not in frame_ids:
            raise ValueError("Frozen taxonomy does not implement the declared replacement")
        if str(taxonomy_contract["removed_seed_scientific_name"]) == str(
            taxonomy_contract["replacement_scientific_name"]
        ):
            raise ValueError("Removed and replacement scientific names must differ")
        v3_ids = (frame_ids - {replacement_id}) | {removed_id}
        min_groups = int(feasibility["gates"]["min_eligible_groups"])
        min_observers = int(feasibility["gates"]["min_eligible_observers"])
        fallback = next(
            (
                row
                for row in feasibility["candidates"]
                if row.get("resolved")
                and int(row["taxon_id"]) not in v3_ids
                and int(row["eligible_groups"]) >= min_groups
                and int(row["eligible_observers"]) >= min_observers
            ),
            None,
        )
        if fallback is None or int(fallback["taxon_id"]) != replacement_id:
            raise ValueError("Replacement is not the first unused feasible audit candidate")
        if str(fallback["scientific_name"]) != str(
            taxonomy_contract["replacement_scientific_name"]
        ):
            raise ValueError("Replacement scientific name does not match the feasibility audit")
        expected_new_ids.add(replacement_id)
    if set(added["taxon_id"].astype(int)) != expected_new_ids:
        raise ValueError("Frozen expansion taxa do not match the passed feasibility audit")
    if set(seed["existing_groups"].astype(int)) != {26}:
        raise ValueError("Seed taxa must retain exactly 26 existing groups")
    if set(added["existing_groups"].astype(int)) != {0}:
        raise ValueError("Expansion taxa must have zero existing groups")
    mode = str(selection.get("mode", "fixed"))
    if mode == "fixed":
        if set(seed["new_group_target"].astype(int)) != {24}:
            raise ValueError("Seed taxon targets do not match the frozen 26+24 design")
        if set(added["new_group_target"].astype(int)) != {50}:
            raise ValueError("Expansion taxon targets do not match the frozen 0+50 design")
    elif mode == "bounded_total":
        final_min = int(selection["final_groups_per_species_min"])
        final_max = int(selection["final_groups_per_species_max"])
        total_new = int(selection["new_groups_total"])
        existing_total = int(frame["existing_groups"].astype(int).sum())
        if final_min < 1 or final_max < final_min:
            raise ValueError("Invalid final per-species balance interval")
        if existing_total + total_new != int(selection["expected_groups"]):
            raise ValueError("Bounded new-group total does not reach expected_groups exactly")
        lower_sum = int((final_min - frame["existing_groups"].astype(int)).sum())
        upper_sum = int((final_max - frame["existing_groups"].astype(int)).sum())
        if (frame["existing_groups"].astype(int) > final_max).any() or not lower_sum <= total_new <= upper_sum:
            raise ValueError("Bounded selection total is unreachable from the frozen taxon table")
    else:
        raise ValueError(f"Unsupported selection mode: {mode}")


def _selection_targets(
    frame: pd.DataFrame, selection: dict
) -> tuple[str, dict[int, int], dict[int, int], int]:
    mode = str(selection.get("mode", "fixed"))
    if mode == "fixed":
        targets = dict(
            zip(frame["taxon_id"].astype(int), frame["new_group_target"].astype(int))
        )
        return mode, targets, dict(targets), int(sum(targets.values()))
    if mode == "bounded_total":
        final_min = int(selection["final_groups_per_species_min"])
        final_max = int(selection["final_groups_per_species_max"])
        existing = dict(
            zip(frame["taxon_id"].astype(int), frame["existing_groups"].astype(int))
        )
        lower = {taxon_id: final_min - count for taxon_id, count in existing.items()}
        upper = {taxon_id: final_max - count for taxon_id, count in existing.items()}
        return mode, lower, upper, int(selection["new_groups_total"])
    raise ValueError(f"Unsupported selection mode: {mode}")


def _per_species_shortfalls(per_species: list[dict]) -> list[dict]:
    shortfalls = []
    for row in per_species:
        lower_target = int(row.get("lower_target", row["target"]))
        retained = int(row["retained_candidates"])
        if retained >= lower_target:
            continue
        shortfalls.append(
            {
                "class_index": int(row["class_index"]),
                "taxon_id": int(row["taxon_id"]),
                "scientific_name": str(row["scientific_name"]),
                "lower_target": lower_target,
                "upper_target": int(row.get("upper_target", lower_target)),
                "retained_candidates": retained,
                "deficit": lower_target - retained,
            }
        )
    return shortfalls


def main() -> int:
    args = parse_args()
    config = load_yaml(args.config)
    require_frozen_protocol(config)
    paths = config["paths"]
    selection = config["selection"]

    feasibility_path = resolve_config_path(config, paths["feasibility_audit_json"])
    frozen_taxa_path = resolve_config_path(config, paths["frozen_taxa_csv"])
    candidate_path = resolve_config_path(config, paths["candidate_snapshot_json"])
    audit_path = resolve_config_path(config, paths["selection_audit_json"])
    manifest_path = resolve_config_path(config, paths["metadata_manifest_csv"])
    if audit_path.exists() or manifest_path.exists():
        raise FileExistsError("Frozen scale-up decision artifacts already exist and cannot be overwritten")

    feasibility = _load_json(feasibility_path)
    if feasibility.get("status") != "PASS_FREEZE_100_TAXA" or not feasibility.get("gate_pass"):
        raise RuntimeError("Scale-up feasibility did not pass")
    frozen_taxa = pd.read_csv(frozen_taxa_path)
    taxonomy_contract = config.get("taxonomy_contract")
    _validate_frozen_taxa(frozen_taxa, feasibility, selection, taxonomy_contract)
    selection_mode, lower_targets, upper_targets, total_new_target = _selection_targets(
        frozen_taxa, selection
    )
    class_map = dict(
        zip(frozen_taxa["taxon_id"].astype(int), frozen_taxa["class_index"].astype(int))
    )

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
    seed_manifest_all = combine_seed_manifests(original, confirmation)
    seed_manifest, excluded_observations, excluded_observers = prepare_seed_manifest_for_taxonomy(
        seed_manifest_all, class_map, taxonomy_contract
    )

    contract = {
        "config_sha256": config["_config_hash"],
        "freeze_record_sha256": sha256_file(resolve_config_path(config, config["protocol"]["freeze_record"])),
        "frozen_taxa_sha256": sha256_file(frozen_taxa_path),
        "feasibility_sha256": sha256_file(feasibility_path),
        "seed_manifest_inputs": {
            "original": sha256_file(resolve_config_path(config, paths["original_manifest_csv"])),
            "confirmation": sha256_file(resolve_config_path(config, paths["confirmation_manifest_csv"])),
        },
    }
    contract_hash = canonical_hash(contract)
    all_candidates: list[dict] = []
    completed_taxa: list[int] = []
    per_species: list[dict] = []
    if candidate_path.exists():
        previous = _load_json(candidate_path)
        if previous.get("contract_hash") != contract_hash:
            raise RuntimeError("Existing candidate snapshot does not match the frozen collection contract")
        if previous.get("status") not in {"RUNNING", "COMPLETE"}:
            raise RuntimeError(f"Unrecognized candidate snapshot status: {previous.get('status')}")
        all_candidates = list(previous.get("groups", []))
        completed_taxa = [int(value) for value in previous.get("completed_taxa", [])]
        per_species = list(previous.get("per_species", []))
        if previous.get("status") == "COMPLETE" and len(completed_taxa) != len(frozen_taxa):
            raise RuntimeError("Complete candidate snapshot has an incomplete taxon list")

    expected_order = frozen_taxa.sort_values("class_index")["taxon_id"].astype(int).tolist()
    if completed_taxa != expected_order[: len(completed_taxa)]:
        raise RuntimeError("Candidate resume artifact is not a prefix of the frozen taxon order")
    shortfalls = _per_species_shortfalls(per_species)
    if shortfalls:
        atomic_write_json(
            candidate_path,
            {
                "status": "STOPPED_PRECHECK",
                "contract_hash": contract_hash,
                "contract": contract,
                "completed_taxa": completed_taxa,
                "total_taxa": len(frozen_taxa),
                "per_species": per_species,
                "groups": all_candidates,
            },
        )
        atomic_write_json(
            audit_path,
            {
                "status": "STOP_SCALEUP_PER_SPECIES_INFEASIBLE",
                "gate_pass": False,
                "contract_hash": contract_hash,
                "candidate_snapshot_sha256": sha256_file(candidate_path),
                "completed_taxa_before_stop": len(completed_taxa),
                "shortfalls": shortfalls,
                "per_species": per_species,
            },
            refuse_if_exists=True,
        )
        print(json.dumps({
            "status": "STOP_SCALEUP_PER_SPECIES_INFEASIBLE",
            "shortfalls": shortfalls,
            "output": str(audit_path),
        }, indent=2))
        return 2
    source = config["source"]
    multiplier = int(selection["candidate_multiplier"])
    for position, taxon_row in enumerate(
        frozen_taxa.sort_values("class_index").iloc[len(completed_taxa) :].to_dict("records"),
        start=len(completed_taxa) + 1,
    ):
        taxon_id = int(taxon_row["taxon_id"])
        lower_target = lower_targets[taxon_id]
        upper_target = upper_targets[taxon_id]
        cap = upper_target * multiplier
        candidates = fetch_observation_candidates(
            taxon_row,
            source,
            excluded_observations,
            excluded_observers,
            seed=int(config["seed"]),
            early_stop_unique_observers=cap,
        )
        compacted = compact_candidates_by_observer(candidates, cap, int(config["seed"]), taxon_id)
        all_candidates.extend(compacted)
        completed_taxa.append(taxon_id)
        per_species.append(
            {
                "class_index": int(taxon_row["class_index"]),
                "taxon_id": taxon_id,
                "scientific_name": str(taxon_row["scientific_name"]),
                "target": lower_target,
                "lower_target": lower_target,
                "upper_target": upper_target,
                "eligible_groups": len(candidates),
                "eligible_observers": len({row["observer_id"] for row in candidates}),
                "retained_candidates": len(compacted),
            }
        )
        atomic_write_json(
            candidate_path,
            {
                "status": "RUNNING",
                "contract_hash": contract_hash,
                "contract": contract,
                "completed_taxa": completed_taxa,
                "total_taxa": len(frozen_taxa),
                "per_species": per_species,
                "groups": all_candidates,
            },
        )
        print(
            f"[{position:03d}/{len(frozen_taxa)}] {taxon_row['scientific_name']}: "
            f"target={lower_target}-{upper_target} candidates={len(compacted)}",
            flush=True,
        )
        shortfalls = _per_species_shortfalls(per_species[-1:])
        if shortfalls:
            atomic_write_json(
                candidate_path,
                {
                    "status": "STOPPED_PRECHECK",
                    "contract_hash": contract_hash,
                    "contract": contract,
                    "completed_taxa": completed_taxa,
                    "total_taxa": len(frozen_taxa),
                    "per_species": per_species,
                    "groups": all_candidates,
                },
            )
            atomic_write_json(
                audit_path,
                {
                    "status": "STOP_SCALEUP_PER_SPECIES_INFEASIBLE",
                    "gate_pass": False,
                    "contract_hash": contract_hash,
                    "candidate_snapshot_sha256": sha256_file(candidate_path),
                    "completed_taxa_before_stop": len(completed_taxa),
                    "shortfalls": shortfalls,
                    "per_species": per_species,
                },
                refuse_if_exists=True,
            )
            print(json.dumps({
                "status": "STOP_SCALEUP_PER_SPECIES_INFEASIBLE",
                "shortfalls": shortfalls,
                "output": str(audit_path),
            }, indent=2))
            return 2
    atomic_write_json(
        candidate_path,
        {
            "status": "COMPLETE",
            "contract_hash": contract_hash,
            "contract": contract,
            "completed_taxa": completed_taxa,
            "total_taxa": len(frozen_taxa),
            "per_species": per_species,
            "groups": all_candidates,
        },
    )

    if selection_mode == "fixed":
        selected, solver_audit = select_balanced_groups(
            all_candidates,
            lower_targets,
            int(config["seed"]),
            observer_capacity=int(selection["max_groups_per_new_observer"]),
        )
    else:
        selected, solver_audit = select_bounded_total_groups(
            all_candidates,
            lower_targets,
            upper_targets,
            total_new_target,
            int(config["seed"]),
            observer_capacity=int(selection["max_groups_per_new_observer"]),
        )
    if not solver_audit["feasible"]:
        atomic_write_json(
            audit_path,
            {
                "status": "STOP_SCALEUP_MILP_INFEASIBLE",
                "gate_pass": False,
                "contract_hash": contract_hash,
                "candidate_snapshot_sha256": sha256_file(candidate_path),
                "solver": solver_audit,
                "per_species": per_species,
            },
            refuse_if_exists=True,
        )
        print(json.dumps({"status": "STOP_SCALEUP_MILP_INFEASIBLE", "output": str(audit_path)}, indent=2))
        return 2

    expansion_manifest = groups_to_canonical(selected, class_map, "robird_scaleup_frozen")
    combined = pd.concat([seed_manifest, expansion_manifest], ignore_index=True, sort=False)
    combined = combined.sort_values(["class_index", "observation_id", "photo_id"]).reset_index(drop=True)
    combined["row_id"] = np.arange(len(combined), dtype=np.int64)
    violations = validate_manifest_schema(combined)
    groups = int(combined["observation_id"].nunique())
    species = int(combined["taxon_id"].nunique())
    photos = int(len(combined))
    new_observers = [int(row["observer_id"]) for row in selected]
    final_species_counts = (
        combined[["taxon_id", "observation_id"]]
        .drop_duplicates()
        .groupby("taxon_id")
        .size()
        .astype(int)
        .to_dict()
    )
    if selection_mode == "bounded_total":
        final_min = int(selection["final_groups_per_species_min"])
        final_max = int(selection["final_groups_per_species_max"])
        species_balance_ok = all(
            final_min <= final_species_counts.get(int(taxon_id), 0) <= final_max
            for taxon_id in frozen_taxa["taxon_id"]
        )
    else:
        existing_counts = dict(
            zip(frozen_taxa["taxon_id"].astype(int), frozen_taxa["existing_groups"].astype(int))
        )
        species_balance_ok = all(
            final_species_counts.get(taxon_id, 0) == existing_counts[taxon_id] + lower_targets[taxon_id]
            for taxon_id in lower_targets
        )
    gate_checks = {
        "schema": not violations,
        "species_exact": species == int(selection["expected_species"]),
        "groups_exact": groups == int(selection["expected_groups"]),
        "species_balance": species_balance_ok,
        "photos_minimum": photos >= 12000,
        "new_observers_unique": len(new_observers) == len(set(new_observers)),
        "new_observers_exclude_seed": not (set(new_observers) & excluded_observers),
        "new_observations_exclude_seed": not (
            {int(row["observation_id"]) for row in selected} & excluded_observations
        ),
    }
    if not all(gate_checks.values()):
        atomic_write_json(
            audit_path,
            {
                "status": "STOP_SCALEUP_METADATA_GATE_FAILED",
                "gate_pass": False,
                "contract_hash": contract_hash,
                "candidate_snapshot_sha256": sha256_file(candidate_path),
                "counts": {"species": species, "groups": groups, "photos": photos},
                "gate_checks": gate_checks,
                "violations": violations,
                "solver": solver_audit,
                "per_species": per_species,
            },
            refuse_if_exists=True,
        )
        print(json.dumps({"status": "STOP_SCALEUP_METADATA_GATE_FAILED", "gate_checks": gate_checks}, indent=2))
        return 2

    atomic_write_csv(manifest_path, combined, refuse_if_exists=True)
    audit = {
        "status": "PASS_METADATA_TO_DOWNLOAD",
        "gate_pass": True,
        "contract_hash": contract_hash,
        "candidate_snapshot_sha256": sha256_file(candidate_path),
        "metadata_manifest_sha256": sha256_file(manifest_path),
        "counts": {
            "species": species,
            "groups": groups,
            "photos": photos,
            "seed_groups": int(seed_manifest["observation_id"].nunique()),
            "selected_new_groups": len(selected),
            "selected_new_observers": len(set(new_observers)),
        },
        "gate_checks": gate_checks,
        "solver": solver_audit,
        "selection_mode": selection_mode,
        "final_groups_per_species": {str(key): value for key, value in final_species_counts.items()},
        "per_species": per_species,
    }
    atomic_write_json(audit_path, audit, refuse_if_exists=True)
    print(json.dumps({"status": audit["status"], "counts": audit["counts"], "output": str(manifest_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
