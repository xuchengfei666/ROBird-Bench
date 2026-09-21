from __future__ import annotations

import argparse
import json
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from robird.download import download_image
from robird.io import atomic_write_csv, atomic_write_json, load_yaml, require_frozen_protocol, resolve_config_path, sha256_file
from robird.manifest import validate_manifest_schema
from robird.scaleup import canonical_hash


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download byte-verified images for the frozen scale-up manifest.")
    parser.add_argument("--config", type=Path, default=Path("configs/scaleup_p0.yaml"))
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
    download_config = config["download"]
    metadata_path = resolve_config_path(config, paths["metadata_manifest_csv"])
    selection_path = resolve_config_path(config, paths["selection_audit_json"])
    output_path = resolve_config_path(config, paths["canonical_manifest_csv"])
    download_root = resolve_config_path(config, paths["download_root"])
    ledger_path = resolve_config_path(config, paths["download_ledger_json"])
    if output_path.exists():
        raise FileExistsError(f"Canonical frozen manifest already exists: {output_path}")

    selection = _load_json(selection_path)
    if selection.get("status") != "PASS_METADATA_TO_DOWNLOAD" or not selection.get("gate_pass"):
        raise RuntimeError("Metadata scale-up gate has not passed")
    metadata_sha256 = sha256_file(metadata_path)
    if selection.get("metadata_manifest_sha256") != metadata_sha256:
        raise RuntimeError("Metadata manifest hash does not match the PASS audit")
    frame = pd.read_csv(metadata_path, keep_default_na=False)
    violations = validate_manifest_schema(frame)
    if violations:
        raise ValueError(f"Metadata manifest schema violations: {violations}")
    pending = frame[frame["local_path"].astype(str).str.len() == 0].copy()
    if pending.empty:
        raise ValueError("Metadata manifest contains no new photos to download")

    contract = {
        "config_sha256": config["_config_hash"],
        "metadata_manifest_sha256": metadata_sha256,
        "selection_audit_sha256": sha256_file(selection_path),
        "download_root": str(download_root),
    }
    contract_hash = canonical_hash(contract)
    ledger_rows: dict[int, dict] = {}
    if ledger_path.exists():
        previous = _load_json(ledger_path)
        if previous.get("contract_hash") != contract_hash:
            raise RuntimeError("Existing download ledger does not match the frozen contract")
        for row in previous.get("photos", []):
            ledger_rows[int(row["photo_id"])] = row

    tasks = pending[
        ["photo_id", "taxon_id", "observation_id", "url"]
    ].drop_duplicates("photo_id").to_dict("records")
    tasks = [row for row in tasks if not ledger_rows.get(int(row["photo_id"]), {}).get("ok", False)]
    failures = 0
    completed_since_checkpoint = 0

    def run_task(row: dict) -> dict:
        photo_id = int(row["photo_id"])
        suffix = Path(str(row["url"]).split("?", 1)[0]).suffix.lower() or ".jpg"
        destination = (
            download_root
            / str(int(row["taxon_id"]))
            / str(int(row["observation_id"]))
            / f"{photo_id}{suffix}"
        )
        try:
            result = download_image(
                str(row["url"]),
                destination,
                int(download_config["request_attempts"]),
                float(download_config["timeout_seconds"]),
            )
            return {**row, **result, "photo_id": photo_id, "ok": True, "error": ""}
        except Exception as error:
            return {**row, "photo_id": photo_id, "ok": False, "error": f"{type(error).__name__}: {error}"}

    with ThreadPoolExecutor(max_workers=int(download_config["max_workers"])) as executor:
        futures: dict[Future, dict] = {executor.submit(run_task, row): row for row in tasks}
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            ledger_rows[int(result["photo_id"])] = result
            failures += int(not result["ok"])
            completed_since_checkpoint += 1
            if completed_since_checkpoint >= int(download_config["checkpoint_every"]):
                atomic_write_json(
                    ledger_path,
                    {
                        "status": "RUNNING",
                        "contract_hash": contract_hash,
                        "contract": contract,
                        "completed": sum(bool(row.get("ok")) for row in ledger_rows.values()),
                        "total": len(pending),
                        "failures": sum(not bool(row.get("ok")) for row in ledger_rows.values()),
                        "photos": sorted(ledger_rows.values(), key=lambda row: int(row["photo_id"])),
                    },
                )
                completed_since_checkpoint = 0
                print(f"downloaded={index}/{len(tasks)} failures={failures}", flush=True)
            if failures >= int(download_config["max_failures_before_stop"]):
                for queued in futures:
                    queued.cancel()
                break

    atomic_write_json(
        ledger_path,
        {
            "status": "RUNNING",
            "contract_hash": contract_hash,
            "contract": contract,
            "completed": sum(bool(row.get("ok")) for row in ledger_rows.values()),
            "total": len(pending),
            "failures": sum(not bool(row.get("ok")) for row in ledger_rows.values()),
            "photos": sorted(ledger_rows.values(), key=lambda row: int(row["photo_id"])),
        },
    )
    required_photo_ids = set(pending["photo_id"].astype(int))
    failed_ids = sorted(
        photo_id
        for photo_id in required_photo_ids
        if not ledger_rows.get(photo_id, {}).get("ok", False)
    )
    if failed_ids:
        raise RuntimeError(f"Download incomplete: {len(failed_ids)} photos failed; rerun resumes from the ledger")

    for row_index in pending.index:
        photo_id = int(frame.at[row_index, "photo_id"])
        record = ledger_rows[photo_id]
        frame.at[row_index, "local_path"] = record["path"]
        frame.at[row_index, "sha256"] = record["sha256"]
        frame.at[row_index, "width"] = int(record["width"])
        frame.at[row_index, "height"] = int(record["height"])
    if (frame["local_path"].astype(str).str.len() == 0).any() or (frame["sha256"].astype(str).str.len() != 64).any():
        raise RuntimeError("Canonical manifest still contains missing paths or invalid hashes")
    atomic_write_csv(output_path, frame, refuse_if_exists=True)
    final_ledger = {
        "status": "COMPLETE",
        "contract_hash": contract_hash,
        "contract": contract,
        "completed": len(required_photo_ids),
        "total": len(required_photo_ids),
        "failures": 0,
        "canonical_manifest": str(output_path),
        "canonical_manifest_sha256": sha256_file(output_path),
        "photos": sorted(ledger_rows.values(), key=lambda row: int(row["photo_id"])),
    }
    atomic_write_json(ledger_path, final_ledger)
    print(json.dumps({
        "status": "COMPLETE",
        "downloaded_photos": len(required_photo_ids),
        "canonical_manifest": str(output_path),
        "canonical_manifest_sha256": final_ledger["canonical_manifest_sha256"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
