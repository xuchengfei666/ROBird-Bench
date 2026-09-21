from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from robird.io import atomic_write_json, load_yaml, protocol_status, resolve_config_path, sha256_file
from robird.scaleup import (
    canonical_hash,
    fetch_observation_candidates,
    load_inat2021_birds,
    load_seed_identity,
    rank_expansion_taxa,
    resolve_exact_species,
)


FINAL_STATUSES = {"PASS_FREEZE_100_TAXA", "STOP_SCALEUP_INFEASIBLE"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit metadata-only feasibility for the 100-taxon scale-up.")
    parser.add_argument("--config", type=Path, default=Path("configs/scaleup_feasibility.yaml"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_yaml(args.config)
    if protocol_status(config) != "metadata_feasibility_preregistered":
        raise RuntimeError("Feasibility CLI requires status metadata_feasibility_preregistered")

    paths = config["paths"]
    original_csv = resolve_config_path(config, paths["original_manifest_csv"])
    confirmation_csv = resolve_config_path(config, paths["confirmation_manifest_csv"])
    taxonomy_path = resolve_config_path(config, paths["taxonomy_json"])
    output_path = resolve_config_path(config, paths["output_json"])
    current_taxa, excluded_observations, excluded_observers, seed_audit = load_seed_identity(
        original_csv, confirmation_csv
    )
    birds = load_inat2021_birds(taxonomy_path, config["taxonomy"]["expected_sha256"])
    ranked = rank_expansion_taxa(birds, current_taxa)
    selection = config["candidate_selection"]
    candidate_prefix = ranked[: int(selection["scan_limit"])]
    if len(candidate_prefix) != int(selection["scan_limit"]):
        raise RuntimeError("Frozen candidate scan prefix is shorter than scan_limit")

    contract = {
        "config_sha256": config["_config_hash"],
        "taxonomy_sha256": sha256_file(taxonomy_path),
        "seed_audit": seed_audit,
        "candidate_prefix": [
            {"category_id": row["category_id"], "scientific_name": row["scientific_name"]}
            for row in candidate_prefix
        ],
    }
    contract_hash = canonical_hash(contract)
    rows: list[dict] = []
    if output_path.exists():
        previous = json.loads(output_path.read_text(encoding="utf-8"))
        if previous.get("contract_hash") != contract_hash:
            raise RuntimeError("Existing feasibility artifact does not match the current preregistered contract")
        if previous.get("status") in FINAL_STATUSES:
            print(json.dumps({
                "status": previous["status"],
                "gate_pass": previous["gate_pass"],
                "selected_new_species": len(previous.get("selected_new_taxa", [])),
                "output": str(output_path),
                "resumed_complete_artifact": True,
            }, indent=2, sort_keys=True))
            return 0 if previous.get("gate_pass") else 2
        if previous.get("status") != "RUNNING":
            raise RuntimeError(f"Unrecognized feasibility artifact status: {previous.get('status')}")
        rows = list(previous.get("candidates", []))
        expected_prefix = [row["scientific_name"] for row in candidate_prefix[: len(rows)]]
        observed_prefix = [row.get("scientific_name") for row in rows]
        if observed_prefix != expected_prefix:
            raise RuntimeError("Resume artifact is not a prefix of the frozen candidate order")

    source = config["source"]
    for index, candidate in enumerate(candidate_prefix[len(rows) :], start=len(rows) + 1):
        row = dict(candidate)
        resolved = resolve_exact_species(
            str(source["api_base"]),
            str(candidate["scientific_name"]),
            attempts=int(source["request_attempts"]),
        )
        time.sleep(float(source["request_delay_seconds"]))
        row.update(
            {
                "resolved": resolved is not None,
                "taxon_id": resolved["taxon_id"] if resolved else None,
                "resolved_common_name": resolved["common_name"] if resolved else "",
                "eligible_groups": 0,
                "eligible_observers": 0,
            }
        )
        if resolved is not None:
            groups = fetch_observation_candidates(
                {**candidate, **resolved},
                source,
                excluded_observations,
                excluded_observers,
                seed=int(config["seed"]),
                early_stop_unique_observers=int(source["early_stop_unique_observers"]),
            )
            row["eligible_groups"] = len(groups)
            row["eligible_observers"] = len({group["observer_id"] for group in groups})
        rows.append(row)
        atomic_write_json(
            output_path,
            {
                "status": "RUNNING",
                "protocol": config["protocol"]["version"],
                "contract_hash": contract_hash,
                "contract": contract,
                "completed": index,
                "total": len(candidate_prefix),
                "seed_audit": seed_audit,
                "candidates": rows,
            },
        )
        print(
            f"[{index:03d}/{len(candidate_prefix)}] {candidate['scientific_name']}: "
            f"resolved={row['resolved']} groups={row['eligible_groups']} "
            f"observers={row['eligible_observers']}",
            flush=True,
        )

    eligible = [
        row
        for row in rows
        if row["resolved"]
        and int(row["eligible_groups"]) >= int(selection["min_eligible_groups"])
        and int(row["eligible_observers"]) >= int(selection["min_eligible_observers"])
    ]
    selected = eligible[: int(selection["new_species_count"])]
    gate_pass = len(selected) == int(selection["new_species_count"])
    status = "PASS_FREEZE_100_TAXA" if gate_pass else "STOP_SCALEUP_INFEASIBLE"
    final = {
        "status": status,
        "protocol": config["protocol"]["version"],
        "gate_pass": gate_pass,
        "contract_hash": contract_hash,
        "contract": contract,
        "seed_audit": seed_audit,
        "gates": {
            "scan_limit": int(selection["scan_limit"]),
            "required_new_species": int(selection["new_species_count"]),
            "min_eligible_groups": int(selection["min_eligible_groups"]),
            "min_eligible_observers": int(selection["min_eligible_observers"]),
        },
        "resolved_candidates": sum(bool(row["resolved"]) for row in rows),
        "eligible_candidates": len(eligible),
        "selected_new_taxa": selected,
        "candidates": rows,
    }
    atomic_write_json(output_path, final)
    print(json.dumps({
        "status": status,
        "gate_pass": gate_pass,
        "resolved_candidates": final["resolved_candidates"],
        "eligible_candidates": len(eligible),
        "selected_new_species": len(selected),
        "output": str(output_path),
    }, indent=2, sort_keys=True))
    return 0 if gate_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
