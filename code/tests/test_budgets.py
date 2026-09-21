from __future__ import annotations

import itertools

import pandas as pd

from robird.budgets import build_budget_plan, enumerate_budget_subsets


def test_small_budget_enumerates_every_combination() -> None:
    subsets = enumerate_budget_subsets([3, 1, 2], 2, max_subsets=10, seed=7, observation_id=9)
    assert subsets == list(itertools.combinations([1, 2, 3], 2))


def test_capped_budget_selection_is_deterministic_and_order_independent() -> None:
    first = enumerate_budget_subsets([1, 2, 3, 4, 5], 2, 3, 7, 9)
    second = enumerate_budget_subsets([5, 4, 3, 2, 1], 2, 3, 7, 9)
    assert first == second
    assert len(first) == 3


def test_budget_plan_skips_infeasible_group_budget() -> None:
    frame = pd.DataFrame(
        {
            "observation_id": [1, 1, 2, 2, 2],
            "photo_id": [11, 12, 21, 22, 23],
        }
    )
    plan = build_budget_plan(frame, budgets=[1, 3], max_subsets=10, seed=3)
    assert not ((plan["observation_id"] == 1) & (plan["budget"] == 3)).any()
    assert ((plan["observation_id"] == 2) & (plan["budget"] == 3)).sum() == 1
