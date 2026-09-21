import itertools
import json

import numpy as np
import pandas as pd
import pytest

from robird.budget_metrics_v2 import (
    budget_ece, cluster_interval, group_statistics, summarize_run, validate_records)


def fixture_data():
    manifest, rows = [], []
    for obs, label, ids in [(1, 0, [10, 11]), (2, 1, [20, 21, 22])]:
        for photo in ids:
            manifest.append(dict(observation_id=obs, observer_id=obs, taxon_id=label+100,
                                 class_index=label, photo_id=photo))
        for k in range(1, len(ids)+1):
            for s in itertools.combinations(ids, k):
                if obs == 1:
                    prob = [.9, .1] if s == (10,) else [.3, .7] if s == (11,) else [.8, .2]
                else:
                    prob = [.4, .6]
                rows.append(dict(observation_id=obs, budget=k, photo_ids=";".join(map(str, s)),
                                 label=label, prediction=int(np.argmax(prob)),
                                 probabilities_json=json.dumps(prob)))
    return pd.DataFrame(rows), pd.DataFrame(manifest)


def test_true_k_does_not_ensemble_all_singletons():
    frame, meta = fixture_data()
    budgets, edges = group_statistics(validate_records(frame, meta))
    one = budgets.query("observation_id == 1 and budget == 1").iloc[0]
    assert one.expected_accuracy == .5
    assert one.legacy_all_subsets_ensemble_accuracy == 1
    edge = edges.query("observation_id == 1").iloc[0]
    assert edge.correction == .5 and edge.regression == 0
    assert edge.net_gain == edge.after_accuracy-edge.before_accuracy
    assert edge.edge_count == 2


@pytest.mark.parametrize("fault", ["missing", "duplicate", "foreign", "label", "argmax", "nan", "sum", "selected"])
def test_fail_closed(fault):
    frame, meta = fixture_data()
    if fault == "missing":
        frame = frame.iloc[1:]
    elif fault == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]])
    elif fault == "foreign":
        frame.loc[0, "photo_ids"] = "20"
    elif fault == "label":
        frame.loc[0, "label"] = 1
    elif fault == "argmax":
        frame.loc[0, "prediction"] = 1
    elif fault == "nan":
        frame.loc[0, "probabilities_json"] = "[NaN, 0.1]"
    elif fault == "sum":
        frame.loc[0, "probabilities_json"] = "[0.9, 0.2]"
    elif fault == "selected":
        frame["selected_photo_ids"] = frame.photo_ids
        frame.loc[0, "selected_photo_ids"] = "999"
    with pytest.raises(ValueError):
        validate_records(frame, meta)


def test_all_edges_equal_expected_difference():
    frame, meta = fixture_data()
    groups = validate_records(frame, meta)
    rng = np.random.default_rng(71)
    for g in groups.values():
        g["subsets"] = {s: rng.dirichlet([1, 1]) for s in g["subsets"]}
    _, edges = group_statistics(groups)
    np.testing.assert_allclose(edges.net_gain, edges.after_accuracy-edges.before_accuracy)
    assert edges.edge_count.tolist() == [2, 6, 3]


def test_probability_tie_uses_argmax_not_truth_oracle():
    frame, meta = fixture_data()
    row = frame.index[frame.label == 1][0]
    frame.loc[row, "probabilities_json"] = "[0.5,0.5]"
    frame.loc[row, "prediction"] = 0
    budgets, _ = group_statistics(validate_records(frame, meta))
    assert budgets.query("observation_id == 2 and budget == 1").iloc[0].expected_accuracy == pytest.approx(2/3)


def test_equal_group_weight_and_subset_ece():
    frame, meta = fixture_data()
    groups = validate_records(frame, meta)
    summary, _, _ = summarize_run(groups, repeats=25)
    assert summary["eligible_per_budget"]["1"]["micro_expected_accuracy"] == .75
    assert summary["eligible_per_budget"]["1"]["macro_expected_accuracy"] == .75
    assert 0 <= budget_ece(groups, [1,2], 1) <= 1
    assert summary["fixed_five_photo_cohort"] == {}


def test_observer_cluster_shared_across_taxa_and_deterministic():
    frame = pd.DataFrame(dict(label=[0,0,1,1], observer_id=[1,2,1,3], x=[0.,1.,0.,1.]))
    a = cluster_interval(frame, "x", repeats=300)
    b = cluster_interval(frame, "x", repeats=300)
    assert a == b and a["clusters"] == 3 and a["point"] == .5
