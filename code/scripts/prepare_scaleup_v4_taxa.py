from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from robird.io import atomic_write_csv, load_yaml, resolve_config_path, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write the deterministic P0-v4 repaired class table.")
    parser.add_argument("--config", type=Path, default=Path("configs/scaleup_p0_v4.yaml"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_yaml(args.config)
    paths = config["paths"]
    contract = config["taxonomy_contract"]
    feasibility_path = resolve_config_path(config, paths["feasibility_audit_json"])
    v3_table_path = resolve_config_path(config, "../data/manifests/frozen_100_taxa.csv")
    output_path = resolve_config_path(config, paths["frozen_taxa_csv"])
    feasibility = json.loads(feasibility_path.read_text(encoding="utf-8"))
    if feasibility.get("status") != "PASS_FREEZE_100_TAXA" or not feasibility.get("gate_pass"):
        raise RuntimeError("The frozen taxonomy feasibility audit is not a PASS")
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite v4 class table: {output_path}")

    v3 = pd.read_csv(v3_table_path)
    removed_id = int(contract["removed_seed_taxon_id"])
    replacement_id = int(contract["replacement_taxon_id"])
    v3_ids = set(v3["taxon_id"].astype(int))
    if removed_id not in v3_ids:
        raise ValueError("The v3 class table does not contain the declared failed seed taxon")
    if replacement_id in v3_ids:
        raise ValueError("The replacement taxon is already present in the v3 class table")

    eligible = [
        row
        for row in feasibility["candidates"]
        if row.get("resolved")
        and int(row["taxon_id"]) not in v3_ids
        and int(row["eligible_groups"]) >= 50
        and int(row["eligible_observers"]) >= 50
    ]
    if not eligible or int(eligible[0]["taxon_id"]) != replacement_id:
        raise ValueError("Declared v4 replacement is not the first unused feasible audit candidate")
    replacement = eligible[0]

    rows = v3[v3["taxon_id"].astype(int) != removed_id].copy()
    rows = pd.concat(
        [
            rows,
            pd.DataFrame(
                [
                    {
                        "class_index": -1,
                        "taxon_id": replacement_id,
                        "scientific_name": str(replacement["scientific_name"]),
                        "common_name": str(
                            replacement.get("resolved_common_name")
                            or replacement.get("common_name", "")
                        ),
                        "family": str(replacement.get("family", "")),
                        "genus": str(replacement.get("genus", "")),
                        "origin": "inat2021_taxonomic_neighbor",
                        "taxonomy_category_id": int(replacement["category_id"]),
                        "existing_groups": 0,
                        "new_group_target": 50,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    rows["_origin_order"] = rows["origin"].map(
        {"inspected_seed": 0, "inat2021_taxonomic_neighbor": 1}
    )
    if rows["_origin_order"].isna().any():
        raise ValueError("v4 class table contains an unrecognized taxon origin")
    rows = rows.sort_values(["_origin_order", "class_index", "taxon_id"]).reset_index(drop=True)
    rows["class_index"] = range(len(rows))
    rows = rows[
        [
            "class_index",
            "taxon_id",
            "scientific_name",
            "common_name",
            "family",
            "genus",
            "origin",
            "taxonomy_category_id",
            "existing_groups",
            "new_group_target",
        ]
    ]
    if len(rows) != 100 or rows["taxon_id"].nunique() != 100:
        raise ValueError("v4 class table must contain exactly 100 unique taxa")
    if int((rows["origin"] == "inspected_seed").sum()) != int(contract["expected_seed_taxa"]):
        raise ValueError("v4 seed count does not match taxonomy contract")
    if int((rows["origin"] == "inat2021_taxonomic_neighbor").sum()) != int(
        contract["expected_expansion_taxa"]
    ):
        raise ValueError("v4 expansion count does not match taxonomy contract")

    atomic_write_csv(output_path, rows, refuse_if_exists=True)
    print(
        json.dumps(
            {
                "status": "PROPOSED_V4_TABLE_WRITTEN",
                "removed": {
                    "taxon_id": removed_id,
                    "scientific_name": contract["removed_seed_scientific_name"],
                },
                "replacement": {
                    "taxon_id": replacement_id,
                    "scientific_name": replacement["scientific_name"],
                    "eligible_groups": replacement["eligible_groups"],
                    "eligible_observers": replacement["eligible_observers"],
                },
                "feasibility_sha256": sha256_file(feasibility_path),
                "output": str(output_path),
                "output_sha256": sha256_file(output_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
