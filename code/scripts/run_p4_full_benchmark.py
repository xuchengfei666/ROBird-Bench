from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from robird.io import atomic_write_json, load_yaml, require_frozen_protocol, resolve_config_path, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the frozen v5.3 P4 comparative suite.")
    parser.add_argument("--config", type=Path, default=Path("configs/p4_full_benchmark_v5_3.yaml"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_yaml(args.config)
    require_frozen_protocol(config)
    project_root = Path(config["_config_path"]).resolve().parents[1]
    result_root = resolve_config_path(config, config["paths"]["result_root"])
    manifest_path = resolve_config_path(config, config["paths"]["run_manifest_json"])
    if manifest_path.exists():
        raise FileExistsError(f"P4 run manifest already exists: {manifest_path}")
    train_script = project_root / "code" / "scripts" / "train.py"
    evaluate_script = project_root / "code" / "scripts" / "evaluate.py"
    records: list[dict[str, Any]] = []
    for model_name in [str(value) for value in config["runner"]["models"]]:
        run_name = f"{model_name.replace('_', '-')}-all"
        run_dir = result_root / run_name
        if any((run_dir / name).exists() for name in ("best.pth", "history.json", "metrics.json", "predictions.csv")):
            raise FileExistsError(f"P4 output already exists for {run_name}: {run_dir}")
        train_command = [
            sys.executable, str(train_script), "--config", str(Path(config["_config_path"]).resolve()),
            "--model", model_name, "--run-name", run_name, "--device", args.device,
        ]
        subprocess.run(train_command, cwd=project_root, check=True)
        checkpoint = run_dir / "best.pth"
        evaluate_command = [
            sys.executable, str(evaluate_script), "--config", str(Path(config["_config_path"]).resolve()),
            "--checkpoint", str(checkpoint), "--split", "development_test", "--output-dir", str(run_dir),
            "--device", args.device,
        ]
        subprocess.run(evaluate_command, cwd=project_root, check=True)
        records.append({
            "model": model_name,
            "run_name": run_name,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "history": str(run_dir / "history.json"),
            "history_sha256": sha256_file(run_dir / "history.json"),
            "metrics": str(run_dir / "metrics.json"),
            "metrics_sha256": sha256_file(run_dir / "metrics.json"),
            "predictions": str(run_dir / "predictions.csv"),
            "predictions_sha256": sha256_file(run_dir / "predictions.csv"),
        })
        print(json.dumps({"model": model_name, "status": "COMPLETE"}), flush=True)
    atomic_write_json(
        manifest_path,
        {"status": "COMPLETE_P4_FULL_BENCHMARK", "config_sha256": config["_config_hash"], "runs": records},
        refuse_if_exists=True,
    )
    print(json.dumps({"run_manifest": str(manifest_path), "runs": len(records)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
