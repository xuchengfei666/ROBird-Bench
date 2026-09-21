from __future__ import annotations

import argparse
import json
from pathlib import Path

from robird.io import atomic_write_csv, load_yaml, require_frozen_protocol, resolve_config_path
from robird.manifest import combine_seed_manifests, normalize_legacy_manifest, validate_manifest_schema


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the inspected ROBird development manifest.")
    parser.add_argument("--config", type=Path, default=Path("configs/dataset_p0.yaml"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    require_frozen_protocol(config)
    paths = config["paths"]

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
    manifest = combine_seed_manifests(original, confirmation)
    violations = validate_manifest_schema(manifest)
    if violations:
        raise RuntimeError(f"Canonical manifest validation failed: {violations}")

    output = resolve_config_path(config, paths["canonical_manifest_csv"])
    atomic_write_csv(output, manifest, refuse_if_exists=True)
    summary = {
        "output": str(output),
        "photos": int(len(manifest)),
        "observations": int(manifest["observation_id"].nunique()),
        "species": int(manifest["taxon_id"].nunique()),
        "status": "development_seed_only",
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
