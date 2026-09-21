from __future__ import annotations

import argparse
import json
from pathlib import Path

from robird.io import atomic_write_csv, load_yaml, resolve_config_path, sha256_file
from robird.scaleup import build_frozen_taxa_frame, load_inat2021_birds, load_seed_identity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write the exact proposed 100-taxon freeze table.")
    parser.add_argument("--feasibility-config", type=Path, default=Path("configs/scaleup_feasibility.yaml"))
    parser.add_argument("--scaleup-config", type=Path, default=Path("configs/scaleup_p0.yaml"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    feasibility_config = load_yaml(args.feasibility_config)
    scaleup_config = load_yaml(args.scaleup_config)
    feasibility_path = resolve_config_path(
        feasibility_config, feasibility_config["paths"]["output_json"]
    )
    with feasibility_path.open("r", encoding="utf-8") as handle:
        feasibility = json.load(handle)
    if feasibility.get("status") != "PASS_FREEZE_100_TAXA" or not feasibility.get("gate_pass"):
        raise RuntimeError("The metadata feasibility gate has not passed")

    shared_source_keys = {
        "api_base",
        "created_d2",
        "quality_grade",
        "photo_licenses",
        "min_photos_per_group",
        "max_photos_per_group",
        "pages",
        "per_page",
    }
    differing = sorted(
        key
        for key in shared_source_keys
        if feasibility_config["source"].get(key) != scaleup_config["source"].get(key)
    )
    if differing:
        raise RuntimeError(f"Feasibility and frozen collection API contracts differ: {differing}")
    original_csv = resolve_config_path(
        feasibility_config, feasibility_config["paths"]["original_manifest_csv"]
    )
    confirmation_csv = resolve_config_path(
        feasibility_config, feasibility_config["paths"]["confirmation_manifest_csv"]
    )
    current_taxa, _, _, seed_audit = load_seed_identity(original_csv, confirmation_csv)
    taxonomy_path = resolve_config_path(
        feasibility_config, feasibility_config["paths"]["taxonomy_json"]
    )
    birds = load_inat2021_birds(
        taxonomy_path, feasibility_config["taxonomy"]["expected_sha256"]
    )
    by_name = {str(row["name"]).strip().casefold(): row for row in birds}
    for row in current_taxa:
        taxonomy_row = by_name.get(str(row["scientific_name"]).strip().casefold())
        if taxonomy_row is not None:
            row["family"] = str(taxonomy_row.get("family", ""))
            row["genus"] = str(taxonomy_row.get("genus", row["genus"]))

    frame = build_frozen_taxa_frame(current_taxa, feasibility["selected_new_taxa"])
    output_path = resolve_config_path(scaleup_config, scaleup_config["paths"]["frozen_taxa_csv"])
    atomic_write_csv(output_path, frame, refuse_if_exists=True)
    print(
        json.dumps(
            {
                "status": "PROPOSED_100_TAXA_WRITTEN_NOT_FROZEN",
                "species": len(frame),
                "seed_audit": seed_audit,
                "feasibility_audit_sha256": sha256_file(feasibility_path),
                "output": str(output_path),
                "output_sha256": sha256_file(output_path),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
