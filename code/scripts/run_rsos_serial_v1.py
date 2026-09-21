from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from robird.io import atomic_write_csv, atomic_write_json, sha256_file
from robird.rsos_suite_v1 import (e3_run, evaluate_run, features_resnet, finish,
                                 group_similarity, marker, train_run)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--freeze", type=Path, required=True)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[2]
    config = json.loads(args.config.read_text(encoding="utf-8"))
    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    contract = sha256_file(args.freeze)
    for relative, digest in freeze["files"].items():
        if sha256_file(root/relative) != digest:
            raise RuntimeError(f"Queue input/code changed: {relative}")
    report_root = root/"code/results/rsos_serial_v1"
    data_root = Path(config["data_root"])
    report_root.mkdir(parents=True, exist_ok=True)
    # Windows OS lock is released even on a crashed worker, unlike stale PID files.
    import msvcrt
    lock = (report_root/"worker.lock").open("a+b")
    if lock.tell() == 0:
        lock.write(b"0")
        lock.flush()
    lock.seek(0)
    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    status_path = report_root/"queue_status.json"
    stage = "preflight"

    def status(state="RUNNING", **extra):
        atomic_write_json(status_path, dict(
            status=state, stage=stage, pid=os.getpid(), contract=contract,
            updated_unix=time.time(), **extra))
        print(json.dumps(dict(status=state, stage=stage, **extra)), flush=True)

    def child(script, parameters):
        log = report_root/(Path(script).stem+".log")
        with log.open("a", encoding="utf-8") as handle:
            subprocess.run([sys.executable, str(root/"code/scripts"/script), *parameters],
                           cwd=root, stdout=handle, stderr=subprocess.STDOUT,
                           check=True, creationflags=subprocess.CREATE_NO_WINDOW)

    try:
        status()
        stage = "metrics_v2"
        status()
        old_summary = root/"code/results/rsos_metrics_v2/summary.json"
        inv_path = root/"code/configs/rsos_metrics_v2_inventory.json"
        if not old_summary.exists():
            child("reanalyze_rsos_metrics_v2.py", ["--inventory",str(inv_path)])
        existing = json.loads(old_summary.read_text(encoding="utf-8"))
        if (existing["contract_sha256"] != sha256_file(inv_path)
                or existing["decision"] != "INTEGRITY_PASS_PREPARE_FAIR_BASELINES"):
            raise RuntimeError("Reanalysis integrity gate not passed")
        for name,digest in existing["output_hashes"].items():
            if sha256_file(old_summary.parent/name) != digest:
                raise RuntimeError("Reanalysis completed output changed")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; CPU fallback intentionally disabled")
        manifest_path = root/"code/data/manifests/development_photos_v5_3.csv"
        splits_path = root/"code/data/manifests/development_splits_v5_3.csv"
        manifest = pd.read_csv(manifest_path)
        splits = pd.read_csv(splits_path)
        manifest = manifest.merge(splits[["observation_id","observer_id","taxon_id","split"]],
                                  on=["observation_id","observer_id","taxon_id"],
                                  how="left", validate="many_to_one")
        if manifest.split.isna().any() or manifest.photo_id.duplicated().any():
            raise RuntimeError("Invalid suite manifest/split alignment")
        feature_path = Path(config["dino_features"])
        index_path = Path(config["dino_index"])
        index = pd.read_csv(index_path).set_index("photo_id")
        dino = np.load(feature_path,mmap_mode="r")
        feature_arrays = {"dinov2": np.asarray(dino[index.loc[manifest.photo_id].feature_row])}
        stage = "resnet50_features"
        status()
        feature_arrays["resnet50"] = features_resnet(manifest, data_root/"resnet50_features", contract)
        specs = []
        for backbone in ("dinov2","resnet50"):
            for model in ("mean_feature","probability_mlp","deepsets","set_transformer"):
                for budget in (1,None):
                    for seed in (20260819,20260820,20260821):
                        run_id = f"{backbone}-{model}-{'k1' if budget else 'all'}-seed{seed}"
                        specs.append(dict(backbone=backbone,model=model,budget=budget,seed=seed,run_id=run_id))
        if len(specs) != 48:
            raise AssertionError("Suite coverage changed")
        # Reuse is allowed only when inherited training settings agree exactly.
        import yaml
        original = yaml.safe_load((root/"code/configs/e4_seed_sensitivity_v1.yaml").read_text(encoding="utf-8"))
        expected_training = dict(batch_size=64,epochs=60,learning_rate=.0003,
                                 weight_decay=.0001,patience=10,label_smoothing=0.)
        if original["training"] != expected_training:
            raise RuntimeError("E4 training contract not reusable")
        expected_hashes = dict(
            config_sha256=sha256_file(root/"code/configs/e4_seed_sensitivity_v1.yaml"),
            manifest_sha256=sha256_file(manifest_path), splits_sha256=sha256_file(splits_path),
            dataset_audit_sha256=sha256_file(root/"code/results/dataset_p0/audit_v5_3.json"),
            feature_index_sha256=sha256_file(index_path))
        checkpoints = {}
        # Training barrier: no new development-test evaluation before all checkpoints are fixed.
        for i,spec in enumerate(specs):
            stage = "train/"+spec["run_id"]
            status(run=i+1,total=48)
            reuse = None
            if spec["backbone"]=="dinov2" and spec["budget"] is None and spec["model"] in ("deepsets","set_transformer"):
                reuse = root/"code/results/benchmark_v5_3_e4_seed_v1"/f"{spec['model'].replace('_','-')}-seed{spec['seed']}"/"best.pth"
            checkpoints[spec["run_id"]] = train_run(
                manifest,feature_arrays[spec["backbone"]],spec,
                data_root/"runs"/spec["run_id"],contract,reuse,expected_hashes)
        barrier = report_root/"training_barrier.json"
        if not marker(barrier,contract):
            finish(barrier,contract,list(checkpoints.values()),checkpoint_count=48)
        for i,spec in enumerate(specs):
            stage = "evaluate/"+spec["run_id"]
            status(run=i+1,total=48)
            evaluate_run(manifest,feature_arrays[spec["backbone"]],spec,
                         checkpoints[spec["run_id"]],data_root/"runs"/spec["run_id"],
                         report_root/"runs"/spec["run_id"],contract)
        for i,spec in enumerate([s for s in specs if s["budget"] is None]):
            stage = "E3/"+spec["run_id"]
            status(run=i+1,total=24,randomizations=20)
            e3_run(manifest,feature_arrays[spec["backbone"]],spec,
                   checkpoints[spec["run_id"]],data_root/"runs"/spec["run_id"],
                   report_root/"runs"/spec["run_id"],contract)
        stage = "E5_and_local_summary"
        status()
        local_done = report_root/"local_done.json"
        if not marker(local_done,contract):
            similarities = {b:group_similarity(manifest,x) for b,x in feature_arrays.items()}
            all_rows, correlations = [], []
            for spec in specs:
                directory = report_root/"runs"/spec["run_id"]
                metrics = json.loads((directory/"metrics.json").read_text(encoding="utf-8"))
                for k,values in metrics["eligible_per_budget"].items():
                    all_rows.append(dict(**spec,evaluation_budget=int(k),
                                         macro_expected_accuracy=values["macro_expected_accuracy"],
                                         micro_expected_accuracy=values["micro_expected_accuracy"]))
                edges = pd.read_csv(directory/"nested.csv")
                joined = edges.merge(similarities[spec["backbone"]].drop(columns="label"),
                                     on="observation_id",validate="many_to_one")
                atomic_write_csv(directory/"e5_covariates.csv",joined)
                for k,part in joined.groupby("budget_from"):
                    for covariate in ("mean_pair_cosine","mean_area"):
                        x = part[covariate]-part.groupby("label")[covariate].transform("mean")
                        y = part.regression-part.groupby("label").regression.transform("mean")
                        value = x.corr(y)
                        correlations.append(dict(**spec,budget_from=int(k),covariate=covariate,
                                                 within_taxon_centered_correlation=float(value) if np.isfinite(value) else None,
                                                 groups=len(part),causal_claim=False))
            atomic_write_csv(report_root/"all_runs_true_budget.csv",pd.DataFrame(all_rows))
            atomic_write_csv(report_root/"e5_descriptive_correlations.csv",pd.DataFrame(correlations))
            atomic_write_json(report_root/"scope.json",dict(
                original_g6_pass=False,g6x_duplicate_review="pending",
                new_holdout_images_downloaded=0,nabirds_stress="pending",
                quality_proxy_analysis="not yet implemented; size and cosine only",
                interpretation="development-only; 20 E3 randomizations are not independent observer trials"))
            finish(local_done,contract,[report_root/"all_runs_true_budget.csv",
                                       report_root/"e5_descriptive_correlations.csv",report_root/"scope.json"])
        stage = "holdout_metadata_only"
        status()
        census_done = data_root/"holdout_metadata/done.json"
        if not marker(census_done,contract):
            child("census_rsos_holdout_v1.py",["--contract",contract,"--config",str(args.config.resolve())])
        census = marker(census_done,contract)
        stage = "finished_local_suite_holdout_design_boundary"
        status("COMPLETE_LOCAL_SUITE_HOLDOUT_DESIGN_REQUIRED",
               trained_grid=48,new_trainings=42,reused_runs=6,
               holdout_individually_feasible=census["individually_feasible"],
               original_g6_pass=False,shortfalls=census["shortfalls"])
    except BaseException as exc:
        status("STOP_EXCEPTION",error_type=type(exc).__name__,error=str(exc),
               traceback=traceback.format_exc())
        raise
    finally:
        lock.close()


if __name__ == "__main__":
    main()
