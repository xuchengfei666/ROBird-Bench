from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from robird.io import (
    atomic_write_csv,
    atomic_write_json,
    load_yaml,
    require_frozen_protocol,
    resolve_config_path,
    sha256_file,
)
from robird.splits import apply_split, solve_observer_split, validate_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build deterministic observer-disjoint splits.")
    parser.add_argument("--config", type=Path, default=Path("configs/dataset_p0.yaml"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    require_frozen_protocol(config)
    paths = config["paths"]
    manifest_path = resolve_config_path(config, paths["canonical_manifest_csv"])
    split_path = resolve_config_path(config, paths["split_csv"])
    audit_path = resolve_config_path(config, paths["split_audit_json"])
    existing = [str(path) for path in (split_path, audit_path) if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite immutable split artifacts: {existing}")

    manifest = pd.read_csv(manifest_path)
    split_config = config["split"]
    assignments, solver = solve_observer_split(
        manifest,
        names=[str(value) for value in split_config["names"]],
        fractions=[float(value) for value in split_config["fractions"]],
        seed=int(config["seed"]),
        time_limit_seconds=int(split_config["milp_time_limit_seconds"]),
    )
    photo_splits = apply_split(manifest, assignments)
    validation = validate_splits(photo_splits)
    group_columns = [
        "observation_id",
        "observer_id",
        "taxon_id",
        "class_index",
        "cohort",
        "split",
    ]
    group_splits = photo_splits.drop_duplicates("observation_id")[group_columns]
    diagnostics = {
        "protocol": dict(config["protocol"]),
        "solver": solver,
        "validation": validation,
        "provenance": {
            "config_sha256": config["_config_hash"],
            "manifest_sha256": sha256_file(manifest_path),
            "manifest_path": str(manifest_path),
        },
    }
    atomic_write_csv(split_path, group_splits, refuse_if_exists=True)
    atomic_write_json(audit_path, diagnostics, refuse_if_exists=True)
    print(json.dumps({"split_csv": str(split_path), "passed": validation["passed"]}, indent=2))
    if not validation["passed"]:
        raise RuntimeError(f"Split validation failed: {validation['violations']}")


if __name__ == "__main__":
    main()
