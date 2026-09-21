from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np


def softmax_numpy(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("logits must be a non-empty [N,C] array")
    shifted = values - values.max(axis=1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=1, keepdims=True)


def topk_accuracy(probabilities: np.ndarray, labels: np.ndarray, k: int) -> float:
    probabilities = np.asarray(probabilities)
    labels = np.asarray(labels, dtype=np.int64)
    if probabilities.ndim != 2 or len(probabilities) != len(labels) or len(labels) == 0:
        raise ValueError("probabilities and labels must be aligned non-empty arrays")
    if int(k) < 1:
        raise ValueError("k must be positive")
    k = min(int(k), probabilities.shape[1])
    top = np.argpartition(probabilities, -k, axis=1)[:, -k:]
    return float(np.mean(np.any(top == labels[:, None], axis=1)))


def macro_top1(probabilities: np.ndarray, labels: np.ndarray) -> float:
    predictions = np.asarray(probabilities).argmax(axis=1)
    labels = np.asarray(labels, dtype=np.int64)
    values = [float(np.mean(predictions[labels == label] == label)) for label in np.unique(labels)]
    return float(np.mean(values))


def negative_log_likelihood(probabilities: np.ndarray, labels: np.ndarray) -> float:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    selected = probabilities[np.arange(len(labels)), labels]
    return float(-np.log(np.clip(selected, 1e-12, 1.0)).mean())


def brier_score(probabilities: np.ndarray, labels: np.ndarray) -> float:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    targets = np.zeros_like(probabilities)
    targets[np.arange(len(labels)), labels] = 1.0
    return float(np.square(probabilities - targets).sum(axis=1).mean())


def expected_calibration_error(
    probabilities: np.ndarray, labels: np.ndarray, bins: int = 15
) -> float:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = predictions == labels
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    result = 0.0
    for index in range(int(bins)):
        if index == int(bins) - 1:
            selected = (confidence >= edges[index]) & (confidence <= edges[index + 1])
        else:
            selected = (confidence >= edges[index]) & (confidence < edges[index + 1])
        if selected.any():
            result += float(selected.mean()) * abs(
                float(correct[selected].mean()) - float(confidence[selected].mean())
            )
    return float(result)


def aggregate_subset_records(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for record in records:
        key = (int(record["observation_id"]), int(record["budget"]))
        grouped.setdefault(key, []).append(record)
    output: list[dict[str, Any]] = []
    for (observation_id, budget), rows in sorted(grouped.items()):
        labels = {int(row["label"]) for row in rows}
        if len(labels) != 1:
            raise ValueError(f"Label mismatch for observation {observation_id}, budget {budget}")
        probabilities = np.mean(
            np.stack([np.asarray(row["probabilities"], dtype=np.float64) for row in rows]), axis=0
        )
        output.append(
            {
                "observation_id": observation_id,
                "budget": budget,
                "label": labels.pop(),
                "probabilities": probabilities,
                "prediction": int(np.argmax(probabilities)),
                "subset_count": len(rows),
            }
        )
    return output


def budget_curve(
    records: Sequence[dict[str, Any]],
    ece_bins: int = 15,
    topk: Sequence[int] = (1, 5),
) -> dict[str, Any]:
    aggregated = aggregate_subset_records(records)
    if not aggregated:
        raise ValueError("No records for budget curve")
    budgets = sorted({int(row["budget"]) for row in aggregated})
    by_budget: dict[str, Any] = {}
    ids_by_budget = [
        {int(row["observation_id"]) for row in aggregated if int(row["budget"]) == budget}
        for budget in budgets
    ]
    common_ids = set.intersection(*ids_by_budget)
    if not common_ids:
        raise ValueError("No observation is shared across all requested budgets")
    common_by_budget: dict[str, Any] = {}
    common_macro_values: list[float] = []

    def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
        probabilities = np.stack([row["probabilities"] for row in rows])
        labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
        summary = {
            "groups": len(rows),
            "macro_top1": macro_top1(probabilities, labels),
            "micro_top1": topk_accuracy(probabilities, labels, 1),
            "nll": negative_log_likelihood(probabilities, labels),
            "brier": brier_score(probabilities, labels),
            "ece": expected_calibration_error(probabilities, labels, ece_bins),
        }
        for value in sorted({int(item) for item in topk}):
            summary[f"top{value}"] = topk_accuracy(probabilities, labels, value)
        return summary

    for budget in budgets:
        rows = [row for row in aggregated if int(row["budget"]) == budget]
        common_rows = [row for row in rows if int(row["observation_id"]) in common_ids]
        by_budget[str(budget)] = summarize(rows)
        common_by_budget[str(budget)] = summarize(common_rows)
        common_macro_values.append(common_by_budget[str(budget)]["macro_top1"])
    if len(budgets) == 1:
        area = common_macro_values[0]
    else:
        area = float(np.trapezoid(common_macro_values, budgets) / (budgets[-1] - budgets[0]))

    first = {row["observation_id"]: row for row in aggregated if row["budget"] == budgets[0]}
    last = {row["observation_id"]: row for row in aggregated if row["budget"] == budgets[-1]}
    shared = sorted(set(first) & set(last))
    corrections = sum(first[key]["prediction"] != first[key]["label"] and last[key]["prediction"] == last[key]["label"] for key in shared)
    regressions = sum(first[key]["prediction"] == first[key]["label"] and last[key]["prediction"] != last[key]["label"] for key in shared)
    return {
        "by_budget": by_budget,
        "common_cohort_by_budget": common_by_budget,
        "common_cohort_groups": len(common_ids),
        "normalized_macro_aubc": area,
        "first_to_last": {
            "shared_groups": len(shared),
            "corrections": int(corrections),
            "regressions": int(regressions),
            "net_corrections": int(corrections - regressions),
        },
    }


def stratified_group_bootstrap(
    rows: Sequence[dict[str, Any]],
    metric: Callable[[list[dict[str, Any]]], float],
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("Bootstrap requires at least one observation")
    if int(repeats) < 1:
        raise ValueError("Bootstrap repeats must be positive")
    by_label: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_label.setdefault(int(row["label"]), []).append(row)
    rng = np.random.default_rng(seed)
    samples = np.zeros(int(repeats), dtype=np.float64)
    labels = sorted(by_label)
    for repeat in range(int(repeats)):
        resampled: list[dict[str, Any]] = []
        for label in labels:
            values = by_label[label]
            indices = rng.integers(0, len(values), size=len(values))
            resampled.extend(values[int(index)] for index in indices)
        samples[repeat] = metric(resampled)
    point = float(metric(list(rows)))
    return {
        "point": point,
        "ci95": [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))],
        "repeats": int(repeats),
        "unit": "observation_group_stratified_by_species",
    }
