from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from robird.io import atomic_write_csv, atomic_write_json, load_yaml, require_frozen_protocol, resolve_config_path, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write the frozen v5.2.4 replacement class table from its PASS audit.")
    parser.add_argument("--config", type=Path, default=Path("configs/scaleup_class_table_v5_2_4.yaml"))
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
    audit_path = resolve_config_path(config, paths["replacement_audit_json"])
    table_path = resolve_config_path(config, paths["class_table_csv"])
    output_audit_path = resolve_config_path(config, paths["class_table_audit_json"])
    if table_path.exists() or output_audit_path.exists():
        raise FileExistsError("v5.2.4 class-table outputs already exist and cannot be overwritten")
    audit = _load_json(audit_path)
    if audit.get("status") != "PASS_REPLACEMENT_METADATA_CANDIDATE" or audit.get("gate_pass") is not True:
        raise RuntimeError("v5.2.4 replacement metadata gate has not passed")
    if audit.get("dataset_p0_decision_authorized") is not False:
        raise RuntimeError("Replacement audit unexpectedly authorizes Dataset P0")
    selected = audit.get("selected_replacement") or {}
    if int(selected.get("taxon_id", -1)) != int(config["selection"]["replacement_taxon_id"]):
        raise RuntimeError("Selected replacement taxon does not match the frozen class-table rule")
    if str(selected.get("scientific_name")) != str(config["selection"]["replacement_scientific_name"]):
        raise RuntimeError("Selected replacement name does not match the frozen class-table rule")
    table = pd.DataFrame.from_records(audit.get("selected_table_preview", []))
    required = {"class_index", "taxon_id", "scientific_name", "common_name", "family", "genus", "origin", "taxonomy_category_id", "existing_groups", "new_group_target"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"Selected class table is missing columns: {missing}")
    if len(table) != int(config["selection"]["expected_species"]) or table["taxon_id"].nunique() != len(table):
        raise ValueError("Frozen replacement class table must contain 100 unique taxa")
    if sorted(table["class_index"].astype(int)) != list(range(len(table))):
        raise ValueError("Class indices must be contiguous")
    if int((table["origin"] == "inspected_seed").sum()) != int(config["selection"]["expected_seed_taxa"]):
        raise ValueError("Unexpected retained seed-taxa count")
    if int((table["origin"] == "inat2021_taxonomic_neighbor").sum()) != int(config["selection"]["expected_expansion_taxa"]):
        raise ValueError("Unexpected expansion-taxa count")
    if int(table.loc[table["origin"] == "inspected_seed", "existing_groups"].sum()) != int(config["selection"]["expected_existing_seed_groups"]):
        raise ValueError("Unexpected retained seed-group count")
    if set(table.loc[table["origin"] == "inspected_seed", "new_group_target"].astype(int)) != {24}:
        raise ValueError("Retained seed targets must be 24")
    if set(table.loc[table["origin"] == "inat2021_taxonomic_neighbor", "existing_groups"].astype(int)) != {0}:
        raise ValueError("Expansion taxa must have zero existing groups")
    if set(table.loc[table["origin"] == "inat2021_taxonomic_neighbor", "new_group_target"].astype(int)) != {50}:
        raise ValueError("Expansion targets must be 50")
    if int(table["existing_groups"].astype(int).sum()) + int(config["selection"]["expected_new_groups"]) != int(config["selection"]["expected_groups"]):
        raise ValueError("Class-table group accounting does not reach 5,000")
    atomic_write_csv(table_path, table, refuse_if_exists=True)
    class_audit = {
        "status": "PASS_FROZEN_REPLACEMENT_CLASS_TABLE",
        "gate_pass": True,
        "dataset_p0_decision_authorized": False,
        "replacement_audit_sha256": sha256_file(audit_path),
        "class_table_sha256": sha256_file(table_path),
        "counts": {
            "species": int(len(table)),
            "seed_taxa": int((table["origin"] == "inspected_seed").sum()),
            "expansion_taxa": int((table["origin"] == "inat2021_taxonomic_neighbor").sum()),
            "existing_seed_groups": int(table.loc[table["origin"] == "inspected_seed", "existing_groups"].sum()),
            "new_groups": int(config["selection"]["expected_new_groups"]),
            "total_groups": int(config["selection"]["expected_groups"]),
        },
        "replacement": selected,
    }
    atomic_write_json(output_audit_path, class_audit, refuse_if_exists=True)
    print(json.dumps(class_audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
