"""Retrospective true-budget metrics. Never imports the legacy ensemble scorer."""
from __future__ import annotations

import itertools
import json
import math
from typing import Any

import numpy as np
import pandas as pd


def validate_records(frame: pd.DataFrame, manifest: pd.DataFrame,
                     num_classes: int | None = None) -> dict[int, dict]:
    """Fail closed unless every nonempty subset is present once for every group."""
    required = {"observation_id", "budget", "photo_ids", "label",
                "prediction", "probabilities_json"}
    if not required.issubset(frame):
        raise ValueError(f"Missing prediction columns: {required - set(frame)}")
    groups = {}
    classes = sorted(manifest.class_index.unique())
    class_count = len(classes) if num_classes is None else num_classes
    if num_classes is None and classes != list(range(class_count)):
        raise ValueError("Noncontiguous class map")
    if any(int(c) < 0 or int(c) >= class_count for c in classes):
        raise ValueError("Labels outside fixed class map")
    for obs, part in manifest.groupby("observation_id", sort=True):
        if any(part[c].nunique() != 1 for c in ("class_index", "observer_id", "taxon_id")):
            raise ValueError("Inconsistent observation identity")
        ids = tuple(sorted(int(x) for x in part.photo_id))
        if not 2 <= len(ids) <= 5 or len(set(ids)) != len(ids):
            raise ValueError("Invalid observation photo IDs/cardinality")
        groups[int(obs)] = dict(
            photos=ids, label=int(part.class_index.iloc[0]),
            observer_id=int(part.observer_id.iloc[0]),
            taxon_id=int(part.taxon_id.iloc[0]), subsets={})
    for row in frame.itertuples(index=False):
        obs = int(row.observation_id)
        if obs not in groups:
            raise ValueError(f"Unknown/wrong-split observation: {obs}")
        group = groups[obs]
        photos = tuple(sorted(int(x) for x in str(row.photo_ids).split(";")))
        if len(photos) != int(row.budget) or len(set(photos)) != len(photos):
            raise ValueError("Subset cardinality/duplicate error")
        if not set(photos).issubset(group["photos"]):
            raise ValueError("Foreign photo ownership")
        if int(row.label) != group["label"]:
            raise ValueError("Label mismatch")
        if photos in group["subsets"]:
            raise ValueError("Duplicate subset")
        prob = np.asarray(json.loads(row.probabilities_json), dtype=np.float64)
        if (prob.shape != (class_count,) or not np.isfinite(prob).all()
                or np.any(prob < 0) or np.any(prob > 1)
                or abs(prob.sum() - 1) > 1e-5):
            raise ValueError("Invalid probability vector")
        if int(row.prediction) != int(prob.argmax()):
            raise ValueError("Prediction is not argmax")
        if hasattr(row, "selected_photo_ids"):
            selected = {int(x) for x in str(row.selected_photo_ids).split(";")}
            if not selected or not selected.issubset(photos):
                raise ValueError("Selector reads outside acquired subset")
        group["subsets"][photos] = prob
    for obs, group in groups.items():
        expected = {s for k in range(1, len(group["photos"]) + 1)
                    for s in itertools.combinations(group["photos"], k)}
        if set(group["subsets"]) != expected:
            raise ValueError(f"Incomplete exhaustive coverage: {obs}")
    return groups


def cluster_interval(frame: pd.DataFrame, column: str,
                     repeats: int = 2000, seed: int = 20260908) -> dict:
    """Linearized cluster bootstrap for fixed-species macro means.

    Each observer's influence is sum((x_i - class_mean)/(L*n_class)).
    Resampling these clusters shares bootstrap multiplicities across species.
    This is an asymptotic, design-conditional interval, not a new taxon interval.
    """
    class_mean = frame.groupby("label")[column].transform("mean")
    counts = frame.groupby("label")[column].transform("size")
    point = float(frame.groupby("label")[column].mean().mean())
    influence = (frame[column] - class_mean) / (frame.label.nunique() * counts)
    cluster = influence.groupby(frame.observer_id).sum().to_numpy(dtype=float)
    assert abs(cluster.sum()) < 1e-10
    rng = np.random.default_rng(seed)
    samples = np.empty(repeats)
    for start in range(0, repeats, 100):
        count = min(100, repeats - start)
        draws = rng.multinomial(len(cluster), np.full(len(cluster), 1 / len(cluster)),
                                size=count)
        samples[start:start + count] = point + draws @ cluster
    return dict(point=point, interval95=np.quantile(samples, [.025, .975]).tolist(),
                clusters=len(cluster), repeats=repeats,
                method="linearized_observer_cluster_percentile_bootstrap_fixed_taxa")


def group_statistics(groups: dict[int, dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    budget_rows, edge_rows = [], []
    for obs, group in groups.items():
        y = group["label"]
        subsets = group["subsets"]
        base = dict(observation_id=obs, label=y, observer_id=group["observer_id"],
                    taxon_id=group["taxon_id"], n_photos=len(group["photos"]))
        means = {}
        for k in range(1, len(group["photos"]) + 1):
            probs = np.stack([p for s, p in subsets.items() if len(s) == k])
            correct = probs.argmax(1) == y
            means[k] = float(correct.mean())
            target = np.zeros(probs.shape[1])
            target[y] = 1
            budget_rows.append(dict(
                **base, budget=k, subset_count=len(probs),
                expected_accuracy=means[k],
                expected_top5=float(np.mean(np.any(np.argsort(probs, axis=1)[:, -5:] == y, axis=1))),
                expected_nll=float(-np.log(np.clip(probs[:, y], 1e-12, 1)).mean()),
                expected_brier=float(np.square(probs - target).sum(1).mean()),
                expected_confidence=float(probs.max(1).mean()),
                legacy_all_subsets_ensemble_accuracy=float(probs.mean(0).argmax() == y)))
        for k in range(1, len(group["photos"])):
            edges = []
            for subset, prob in subsets.items():
                if len(subset) != k:
                    continue
                for new_photo in group["photos"]:
                    if new_photo not in subset:
                        bigger = tuple(sorted((*subset, new_photo)))
                        edges.append((prob.argmax() == y, subsets[bigger].argmax() == y))
            before, after = np.asarray(edges, dtype=bool).T
            correction = float((~before & after).mean())
            regression = float((before & ~after).mean())
            assert len(edges) == math.comb(len(group["photos"]), k) * (len(group["photos"]) - k)
            if not np.isclose(correction - regression, means[k+1] - means[k], atol=1e-12):
                raise AssertionError("Nested-edge identity failed")
            edge_rows.append(dict(
                **base, budget_from=k, budget_to=k+1, edge_count=len(edges),
                before_accuracy=float(before.mean()), after_accuracy=float(after.mean()),
                correction=correction, regression=regression,
                net_gain=correction-regression, any_harmful=float(regression > 0),
                any_helpful=float(correction > 0)))
    return pd.DataFrame(budget_rows), pd.DataFrame(edge_rows)


def budget_ece(groups: dict[int, dict], ids: list[int], k: int, bins: int = 15) -> float:
    """Group-equal expectation over actual subset predictions, not ensembled probs."""
    totals = np.zeros((bins, 3))
    for obs in ids:
        g = groups[obs]
        ps = [p for s, p in g["subsets"].items() if len(s) == k]
        weight = 1 / (len(ids) * len(ps))
        for prob in ps:
            confidence = float(prob.max())
            b = min(int(confidence * bins), bins-1)
            totals[b] += weight * np.array([1., float(prob.argmax() == g["label"]), confidence])
    return float(np.abs(totals[:, 1] - totals[:, 2]).sum())


def summarize_run(groups: dict[int, dict], repeats: int = 2000,
                  seed: int = 20260908) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    budgets, edges = group_statistics(groups)
    summaries = {}
    for name, frame in (("eligible_per_budget", budgets),
                        ("fixed_five_photo_cohort", budgets[budgets.n_photos == 5])):
        summaries[name] = {}
        for k, part in frame.groupby("budget"):
            means = part.groupby("label").expected_accuracy.mean()
            summaries[name][str(k)] = dict(
                groups=len(part), taxa=len(means), observers=part.observer_id.nunique(),
                macro_expected_accuracy=float(means.mean()),
                micro_expected_accuracy=float(part.expected_accuracy.mean()),
                macro_accuracy_interval=cluster_interval(part, "expected_accuracy", repeats, seed),
                group_equal_expected_top5=float(part.expected_top5.mean()),
                group_equal_expected_nll=float(part.expected_nll.mean()),
                group_equal_expected_brier=float(part.expected_brier.mean()),
                group_equal_subset_ece=budget_ece(groups, part.observation_id.tolist(), int(k)),
                legacy_macro_ensemble_accuracy=float(part.groupby("label").legacy_all_subsets_ensemble_accuracy.mean().mean()))
    summaries["nested_transitions"] = {}
    for k, part in edges.groupby("budget_from"):
        macro = part.groupby("label").mean(numeric_only=True).mean()
        summaries["nested_transitions"][f"{k}_to_{k+1}"] = dict(
            groups=len(part), taxa=part.label.nunique(), observers=part.observer_id.nunique(),
            raw_edges=int(part.edge_count.sum()),
            macro_before=float(macro.before_accuracy), macro_after=float(macro.after_accuracy),
            macro_correction=float(macro.correction), macro_regression=float(macro.regression),
            macro_net_gain=float(macro.net_gain),
            micro_any_harmful_groups=float(part.any_harmful.mean()),
            macro_any_harmful_groups=float(macro.any_harmful),
            regression_given_correct=float(macro.regression/macro.before_accuracy) if macro.before_accuracy else None,
            correction_given_wrong=float(macro.correction/(1-macro.before_accuracy)) if macro.before_accuracy < 1 else None,
            net_gain_interval=cluster_interval(part, "net_gain", repeats, seed),
            regression_interval=cluster_interval(part, "regression", repeats, seed),
            correction_interval=cluster_interval(part, "correction", repeats, seed))
    summaries["integrity"] = dict(
        exhaustive_coverage_passed=True, nested_identity_passed=True,
        groups=len(groups), subset_rows=sum(len(g["subsets"]) for g in groups.values()))
    return summaries, budgets, edges
