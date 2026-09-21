from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from robird.budget_metrics_v2 import summarize_run, validate_records
from robird.io import atomic_write_csv, atomic_write_json, sha256_file


def verify_inventory(inventory_path: Path, root: Path) -> dict:
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    for relative, expected in inventory["files"].items():
        path = root / relative
        if sha256_file(path) != expected:
            raise RuntimeError(f"Immutable input hash mismatch: {relative}")
    return inventory


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory", type=Path, required=True)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[2]
    inventory = verify_inventory(args.inventory, root)
    contract = sha256_file(args.inventory)
    output = root / "code/results/rsos_metrics_v2"
    if (output / "summary.json").exists():
        raise FileExistsError("Completed v2 result already exists; no overwrite")
    output.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(root / inventory["manifest"])
    splits = pd.read_csv(root / inventory["splits"])
    if splits.observation_id.duplicated().any():
        raise ValueError("Duplicate split observation")
    joined = manifest.merge(
        splits[["observation_id", "observer_id", "taxon_id", "split"]],
        on=["observation_id", "observer_id", "taxon_id"], how="left", validate="many_to_one")
    if joined.split.isna().any():
        raise ValueError("Manifest/split identity mismatch")
    for field in ("observation_id", "observer_id"):
        if joined.groupby(field).split.nunique().max() != 1:
            raise ValueError(f"Cross-split identity leakage: {field}")
    test = joined[joined.split == "development_test"].copy()
    if test.observation_id.nunique() != 750 or test.taxon_id.nunique() != 100:
        raise ValueError("Unexpected development-test cohort")
    summaries, all_budgets, all_edges = [], [], []
    for source_index, source in enumerate(inventory["prediction_sources"]):
        shard = output / f"source_{source_index:02d}.json"
        budget_path = output / f"source_{source_index:02d}_groups.csv"
        edge_path = output / f"source_{source_index:02d}_nested.csv"
        if shard.exists():
            result = json.loads(shard.read_text(encoding="utf-8"))
            if result["contract_sha256"] != contract or result["source"] != source:
                raise RuntimeError("Completed shard belongs to another input contract")
            for path in (budget_path, edge_path):
                if sha256_file(path) != result["output_hashes"][path.name]:
                    raise RuntimeError("Completed shard output tampered")
            budgets = pd.read_csv(budget_path)
            edges = pd.read_csv(edge_path)
        else:
            if budget_path.exists() or edge_path.exists():
                raise RuntimeError("Incomplete shard preserved; inspect before versioned recovery")
            frame = pd.read_csv(root / source)
            partitions = list(frame.groupby("method", sort=True)) if "method" in frame else [("model", frame)]
            pieces, budget_pieces, edge_pieces = [], [], []
            for method, part in partitions:
                groups = validate_records(part, test)
                summary, budgets, edges = summarize_run(
                    groups, inventory["bootstrap_repeats"], inventory["seed"])
                for table in (budgets, edges):
                    table.insert(0, "method", str(method))
                    table.insert(0, "source", source)
                pieces.append(dict(source=source, method=str(method), **summary))
                budget_pieces.append(budgets)
                edge_pieces.append(edges)
            budgets = pd.concat(budget_pieces, ignore_index=True)
            edges = pd.concat(edge_pieces, ignore_index=True)
            atomic_write_csv(budget_path, budgets, refuse_if_exists=True)
            atomic_write_csv(edge_path, edges, refuse_if_exists=True)
            result = dict(
                source=source, contract_sha256=contract, methods=pieces,
                output_hashes={q.name: sha256_file(q) for q in (budget_path, edge_path)})
            atomic_write_json(shard, result, refuse_if_exists=True)
        summaries.extend(result["methods"])
        all_budgets.append(budgets)
        all_edges.append(edges)
        print(json.dumps(dict(stage="SOURCE_COMPLETE", index=source_index,
                              methods=len(result["methods"]), source=source)), flush=True)
    budgets = pd.concat(all_budgets, ignore_index=True)
    edges = pd.concat(all_edges, ignore_index=True)
    atomic_write_csv(output / "all_group_budgets.csv", budgets, refuse_if_exists=True)
    atomic_write_csv(output / "all_group_nested.csv", edges, refuse_if_exists=True)
    # Descriptive E5: same tested observations, source IDs and numeric taxa retained.
    photo_meta = test.assign(area=test.width * test.height).groupby("observation_id").agg(
        mean_area=("area", "mean"), scientific_name=("scientific_name", "first"))
    frequency = test[["observation_id","observer_id"]].drop_duplicates().groupby("observer_id").size()
    annotated = edges.merge(photo_meta, on="observation_id", validate="many_to_one")
    annotated["area_stratum"] = pd.cut(annotated.mean_area, [0, 500000, 1500000, np.inf],
                                       right=False, labels=["<0.5M","0.5-1.5M",">=1.5M"])
    annotated["observer_frequency"] = annotated.observer_id.map(frequency)
    stratum_rows = []
    for field in ("taxon_id", "n_photos", "area_stratum", "observer_frequency"):
        for key, part in annotated.groupby(["source","method","budget_from",field], observed=True):
            stratum_rows.append(dict(
                source=key[0], method=key[1], budget_from=int(key[2]), stratum_type=field,
                stratum=str(key[3]), groups=len(part), taxa=part.taxon_id.nunique(),
                observers=part.observer_id.nunique(),
                micro_before=float(part.before_accuracy.mean()), micro_after=float(part.after_accuracy.mean()),
                micro_correction=float(part.correction.mean()), micro_regression=float(part.regression.mean()),
                micro_any_harmful=float(part.any_harmful.mean())))
    atomic_write_csv(output / "e5_descriptive_nested_strata.csv", pd.DataFrame(stratum_rows), refuse_if_exists=True)
    # E4 training seed dispersion; never pool seeds as independent test observations.
    seed_rows = []
    for item in summaries:
        src = item["source"]
        if "benchmark_v5_3_e4_seed_v1/" not in src:
            continue
        run = Path(src).parent.name
        model, seed = run.split("-seed")
        for k, metrics in item["eligible_per_budget"].items():
            seed_rows.append(dict(model=model, seed=int(seed), budget=int(k),
                                  macro_expected_accuracy=metrics["macro_expected_accuracy"]))
    seeds = pd.DataFrame(seed_rows)
    atomic_write_csv(output / "e4_true_budget_seeds.csv", seeds, refuse_if_exists=True)
    aggregate = seeds.groupby(["model","budget"]).macro_expected_accuracy.agg(["mean","std","min","max"]).reset_index()
    atomic_write_csv(output / "e4_true_budget_aggregate.csv", aggregate, refuse_if_exists=True)
    verify_inventory(args.inventory, root)
    result = dict(
        status="COMPLETE_RSOS_TRUE_BUDGET_REANALYSIS_V2",
        decision="INTEGRITY_PASS_PREPARE_FAIR_BASELINES",
        scientific_winner_gate=None, claim_scope="RETROSPECTIVE_DEVELOPMENT_ONLY",
        original_p0_g6=False, contract_sha256=contract,
        source_count=len(inventory["prediction_sources"]), method_run_count=len(summaries),
        methods=summaries,
        limitations=[
            "Intervals conditional on existing taxa; no multiplicity correction",
            "E3 lacks saved raw subset probabilities; true-budget E3 still pending",
            "Old STOP files preserved; old budget interpretation superseded",
            "E5 raw strata are associations, not causal quality effects",
            "No features, new training or prospective holdout run in this stage"],
        output_hashes={str(q.relative_to(output)): sha256_file(q)
                       for q in output.iterdir() if q.is_file()})
    atomic_write_json(output / "summary.json", result, refuse_if_exists=True)
    print(json.dumps({k: result[k] for k in ("status","decision","source_count","method_run_count")}), flush=True)


if __name__ == "__main__":
    main()
