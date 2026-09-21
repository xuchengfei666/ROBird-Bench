from __future__ import annotations

import hashlib
import itertools
from collections.abc import Iterable, Sequence

import pandas as pd


def stable_key(seed: int, *parts: object) -> int:
    payload = "|".join([str(seed), *(str(part) for part in parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def enumerate_budget_subsets(
    photo_ids: Sequence[int],
    budget: int,
    max_subsets: int,
    seed: int,
    observation_id: int,
) -> list[tuple[int, ...]]:
    unique_ids = tuple(sorted({int(value) for value in photo_ids}))
    if budget < 1 or budget > len(unique_ids):
        raise ValueError(f"Budget {budget} invalid for {len(unique_ids)} photos")
    combinations = list(itertools.combinations(unique_ids, budget))
    if len(combinations) <= max_subsets:
        return combinations
    combinations.sort(key=lambda values: stable_key(seed, observation_id, budget, *values))
    return combinations[:max_subsets]


def build_budget_plan(
    frame: pd.DataFrame,
    budgets: Iterable[int],
    max_subsets: int,
    seed: int,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for observation_id, group in frame.groupby("observation_id", sort=True):
        photo_ids = sorted(int(value) for value in group["photo_id"])
        for budget in sorted({int(value) for value in budgets}):
            if budget > len(photo_ids):
                continue
            subsets = enumerate_budget_subsets(
                photo_ids, budget, max_subsets, seed, int(observation_id)
            )
            for replicate, subset in enumerate(subsets):
                records.append(
                    {
                        "observation_id": int(observation_id),
                        "budget": budget,
                        "replicate": replicate,
                        "photo_ids": ";".join(str(value) for value in subset),
                        "subset_key": stable_key(seed, observation_id, budget, *subset),
                    }
                )
    return pd.DataFrame.from_records(records)
