import itertools
import json

import numpy as np
import pandas as pd
import pytest
import torch

from robird.explanatory_suite_v1 import (GateStop, aligned_features, decompose, eligible_support,
    regroup, validate_mapping, mapping_balance, paired_inference, scalar_summary, singleton_probs)
from robird.rsos_suite_v1 import make_model
from robird.budget_metrics_v2_1 import validate_records


def fixture_frame():
    rows = []
    for taxon in [10, 20]:
        for n in [2, 3, 5]:
            for i in range(5):
                obs = taxon*1000+n*100+i
                for j in range(n):
                    rows.append(dict(observation_id=obs, observer_id=obs//2, taxon_id=taxon,
                        class_index=taxon//10-1, photo_id=obs*10+j, difficulty_bin=j % 3,
                        difficulty=float(j % 3)+i*.01, split='development_test'))
    return pd.DataFrame(rows)


@pytest.mark.parametrize('matched', [False, True])
def test_regroup_twenty_repeats_all_invariants(matched):
    frame = fixture_frame()
    fingerprints = set()
    for repeat in range(20):
        pseudo, mapping = regroup(frame, 2026091900+repeat, matched)
        assert validate_mapping(frame, pseudo, matched)
        assert not pseudo.photo_id.duplicated().any()
        assert (mapping.slot_observation_id != mapping.source_observation_id).all()
        assert mapping.groupby('slot_observation_id').source_observation_id.nunique().min() >= 2
        origin = frame.set_index('photo_id')
        assert np.array_equal(mapping.source_observer_id, origin.loc[mapping.photo_id].observer_id)
        assert np.array_equal(mapping.source_observation_id, origin.loc[mapping.photo_id].observation_id)
        balance, summary = mapping_balance(frame, pseudo, mapping)
        if matched:
            assert balance.bin_match_fraction.eq(1).all()
        assert 0 <= summary['same_source_observer_fraction'] <= 1
        fingerprints.add(tuple(pseudo.photo_id))
    assert len(fingerprints) > 1


def test_matching_reproducible_with_same_seed():
    a, b = regroup(fixture_frame(), 19, True)
    c, d = regroup(fixture_frame(), 19, True)
    pd.testing.assert_frame_equal(a, c)
    pd.testing.assert_frame_equal(b, d)


def test_attrition_whole_group_never_threshold_rescue():
    frame = fixture_frame()
    first = frame.observation_id.min()
    frame.loc[frame.observation_id.eq(first), 'difficulty_bin'] = 2
    common, attrition = eligible_support(frame)
    assert first not in set(common.observation_id)
    assert attrition.observation_id.nunique() == frame.observation_id.nunique()
    assert attrition.loc[attrition.observation_id.eq(first), 'stratum_groups'].item() == 1
    assert common.groupby('observation_id').size().isin([2, 3, 5]).all()


@pytest.mark.parametrize('bad', ['pool', 'self', 'label', 'metadata', 'observer'])
def test_mapping_rejects_corruption(bad):
    real = fixture_frame()
    pseudo, _ = regroup(real, 19, True)
    if bad == 'pool':
        pseudo.loc[0, 'photo_id'] = pseudo.loc[1, 'photo_id']
    elif bad == 'self':
        origin = real.set_index('photo_id')
        pseudo.loc[0, 'observation_id'] = origin.loc[pseudo.loc[0, 'photo_id'], 'observation_id']
    elif bad == 'label':
        pseudo.loc[0, 'class_index'] = 99
    elif bad == 'metadata':
        pseudo.loc[0, 'difficulty_bin'] = 8
    else:
        pseudo.loc[0, 'observer_id'] = 123
    with pytest.raises(GateStop):
        validate_mapping(real, pseudo, True)


def test_small_stratum_fails_closed():
    real = fixture_frame()
    real = real[real.observation_id.isin(real.observation_id.unique()[:2])]
    with pytest.raises(GateStop):
        regroup(real, 1, True)


def toy_group():
    # Singleton1 correct, singleton2 wrong; native pair wrong, probability average correct.
    return {1: dict(photos=(10, 11), observer_id=7, taxon_id=12, label=0,
        subsets={(10,): np.array([.9, .1]), (11,): np.array([.4, .6]),
                 (10, 11): np.array([.3, .7])})}


def test_error_structure_and_singleton_not_ensemble():
    rows, edges, _ = decompose(toy_group())
    a = rows[(rows.method == 'native') & (rows.budget == 2)].iloc[0]
    b = rows[(rows.method == 'probability_pool') & (rows.budget == 2)].iloc[0]
    assert a.singleton_accuracy == .5
    assert a.accuracy == 0 and b.accuracy == 1
    assert a.any_single_correct_set_wrong == 1
    assert a.all_single_wrong_set_correct == 0
    assert a.joint_wrong_pair == 0 and a.same_wrong_pair == 0
    assert rows[rows.budget.eq(1)].accuracy.eq(.5).all()
    edge = edges[edges.method.eq('native')].iloc[0]
    assert edge.regression == .5 and edge.correction == 0 and edge.net_gain == -.5


def test_all_singletons_wrong_can_be_corrected_by_set():
    groups = toy_group()
    groups[1]['subsets'] = {(10,): np.array([.2, .8]), (11,): np.array([.3, .7]),
                            (10, 11): np.array([.8, .2])}
    rows, _, _ = decompose(groups)
    a = rows[(rows.method == 'native') & (rows.budget == 2)].iloc[0]
    assert a.all_single_wrong_set_correct == 1
    assert a.joint_wrong_pair == 1 and a.same_wrong_pair == 1


def test_probability_mlp_pool_equivalence_and_validated_coverage():
    torch.manual_seed(41)
    torch.set_num_threads(2)
    model = make_model('probability_mlp', 8).eval()
    feats = np.random.default_rng(41).normal(size=(3, 8)).astype('float32')
    singles = singleton_probs(model, feats)
    records = []
    for k in [1, 2, 3]:
        for s in itertools.combinations(range(3), k):
            x = torch.tensor(feats[list(s)])[None]
            with torch.no_grad():
                p = model(x, torch.ones((1, k), dtype=torch.bool)).logits.softmax(-1).numpy()[0]
            np.testing.assert_allclose(p, singles[list(s)].mean(0), atol=2e-7)
            records.append(dict(observation_id=1, label=0, photo_ids=';'.join(str(i+10) for i in s),
                budget=k, prediction=int(p.argmax()), probabilities_json=json.dumps(p.tolist())))
    manifest = pd.DataFrame([dict(photo_id=i+10, observation_id=1, taxon_id=10,
                                  observer_id=7, class_index=0) for i in range(3)])
    groups = validate_records(pd.DataFrame(records), manifest, num_classes=100)
    rows, _, maximum = decompose(groups)
    assert maximum < 2e-7
    assert len(scalar_summary(rows)) == 6


def test_feature_alignment_not_positional():
    matrix = np.array([[2, 2], [3, 3], [1, 1]])
    index = pd.DataFrame(dict(photo_id=[20, 30, 10], feature_row=[0, 1, 2]))
    frame = pd.DataFrame(dict(photo_id=[10, 20]))
    np.testing.assert_array_equal(aligned_features(frame, matrix, index), [[1, 1], [2, 2]])
    with pytest.raises(GateStop):
        aligned_features(frame, matrix, index.assign(feature_row=[0, 0, 2]))


def test_cluster_inference_rejects_pseudo_observers():
    frame = pd.DataFrame(dict(observation_id=[1, 2, 3, 4], observer_id=[5, 5, 6, 7],
                              label=[0, 1, 0, 1], delta=[.1, .2, -.1, .4]))
    cfg = dict(signflip_repeats=99, bootstrap_repeats=100, statistics_seed=19)
    out = paired_inference(frame, cfg)
    assert out['observers'] == 3 and out['observers_spanning_taxa'] == 1
    assert out['delta'] == pytest.approx(.15)
    assert out['p'] > 0
    with pytest.raises(GateStop):
        paired_inference(frame.assign(observer_id=-1), cfg)


@pytest.mark.parametrize('has_controls', [False, True])
def test_final_analysis_full16_family_and_resume(tmp_path, has_controls):
    from robird.explanatory_suite_v1 import final_analysis
    from robird.io import atomic_write_csv, atomic_write_json
    from robird.rsos_suite_v1 import finish
    from robird.artifact_tree_v1 import verify_tree
    report, data, source = tmp_path/'report', tmp_path/'data', tmp_path/'source'
    report.mkdir(); data.mkdir(); source.mkdir()
    frame = fixture_frame()
    if has_controls:
        for condition in ['unrestricted_common', 'difficulty_matched']:
            for repeat in range(2):
                pseudo, mapping = regroup(frame, 19+repeat, condition == 'difficulty_matched')
                balance, stats = mapping_balance(frame, pseudo, mapping)
                directory = data/'maps'/condition/f'{repeat:02d}'
                atomic_write_csv(directory/'balance.csv', balance)
                finish(directory/'done.json', 'toy', [directory/'balance.csv'], **stats)
    groups = {}
    for obs, part in frame.groupby('observation_id'):
        y = int(part.class_index.iloc[0])
        photos = tuple(sorted(part.photo_id))
        p = np.full(100, .1/99); p[y] = .9
        groups[int(obs)] = dict(photos=photos, observer_id=int(part.observer_id.iloc[0]),
            taxon_id=int(part.taxon_id.iloc[0]), label=y,
            subsets={s: p.copy() for k in range(1, len(photos)+1) for s in itertools.combinations(photos, k)})
    rows, nested, _ = decompose(groups)
    runs = []
    e3 = pd.DataFrame([dict(repeat=r, budget=k, macro_real=.5, macro_shuffled=.6,
                            difference=.1, groups=10, taxa=2) for r in range(20) for k in range(1, 6)])
    for backbone, models in [('dinov2', ['mean_feature', 'probability_mlp', 'deepsets', 'set_transformer']),
                             ('resnet50', ['probability_mlp', 'deepsets'])]:
        for model in models:
            for seed in [20260819, 20260820, 20260821]:
                run_id = f'{backbone}-{model}-all-seed{seed}'
                runs.append(dict(run_id=run_id, backbone=backbone, model=model, seed=seed))
                out = report/'real'/run_id
                atomic_write_csv(out/'groups.csv', rows)
                atomic_write_csv(out/'nested.csv', nested)
                finish(out/'done.json', 'toy', [out/'groups.csv', out/'nested.csv'])
                if has_controls:
                    for condition in ['unrestricted_common', 'difficulty_matched']:
                        for repeat in range(2):
                            out = report/'controls'/run_id/condition/f'{repeat:02d}'
                            atomic_write_json(out/'summary.json', dict(cells=scalar_summary(rows)))
                            finish(out/'done.json', 'toy', [out/'summary.json'])
                if backbone == 'dinov2':
                    atomic_write_csv(source/'runs'/run_id/'e3_randomizations.csv', e3)
                else:
                    out = report/'resnet_e3'/run_id
                    atomic_write_csv(out/'e3_randomizations.csv', e3)
                    finish(out/'e3_done.json', 'toy', [out/'e3_randomizations.csv'])
    for name in ['mappings_done.json', 'difficulty_done.json', 'resnet_features_done.json']:
        finish(report/name, 'toy')
    cfg = dict(runs=runs, source_report_absolute=str(source), bootstrap_repeats=100,
               signflip_repeats=99, statistics_seed=19, randomizations=2)
    common = frame if has_controls else frame.iloc[:0]
    final_analysis(frame, common, data, report, cfg, 'toy')
    test = pd.read_csv(report/'paired_tests_holm16.csv')
    assert len(test) == 16 and test.p_holm16.eq(1).all()
    assert (report/'RESULTS.md').exists()
    verify_tree(report/'analysis_done.json')
    before = (report/'analysis_done.json').read_bytes()
    final_analysis(frame, common, data, report, cfg, 'toy')
    assert before == (report/'analysis_done.json').read_bytes()
    if has_controls:
        controls = pd.read_csv(report/'control_distributions_not_CI.csv')
        assert controls.accuracy_delta_mean.eq(0).all()
        assert (report/'matching_balance_summary.csv').exists()
