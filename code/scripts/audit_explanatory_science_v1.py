"""Independent, read-only arithmetic/hash audit of finished explanatory outputs."""
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
import time

os.environ.update(OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4')
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'code/src'))
from robird.io import atomic_write_csv, atomic_write_json, sha256_file
from robird.rsos_suite_v1 import finish

REPORT = ROOT/'code/results/explanatory_suite_v1_1'
OUT = ROOT/'code/results/explanatory_scientific_audit_v1'
F = ROOT/'FROZEN_EXPLANATORY_SUITE_V1_1.json'
CFG = json.loads((REPORT/'resolved_config.json').read_text())
DATA = Path(CFG['data_root'])
metrics = ['accuracy', 'singleton_accuracy', 'joint_wrong_pair', 'same_wrong_pair',
           'any_single_correct_set_wrong', 'all_single_wrong_set_correct']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def close(a, b, message, tol=2e-10):
    if not np.allclose(a, b, atol=tol, rtol=0, equal_nan=True):
        raise AssertionError(message)


def integrity():
    expected = {}
    walked = set()
    def add(path, digest):
        path = Path(path).resolve()
        if path in expected and expected[path] != digest:
            raise AssertionError('Conflicting hash '+str(path))
        expected[path] = digest
    def descend(path):
        path = Path(path).resolve()
        if path in walked: return
        walked.add(path)
        value = read(path)
        if not isinstance(value, dict): return
        for name, digest in value.get('artifacts', {}).items():
            add(name, digest)
            if Path(name).suffix == '.json': descend(name)
    for name, digest in read(F)['files'].items(): add(name, digest)
    descend(REPORT/'automatic_done.json')
    total_bytes = 0
    for path, digest in expected.items():
        if sha256_file(path) != digest: raise AssertionError('Hash drift '+str(path))
        total_bytes += path.stat().st_size
    result = dict(status='PASS', frozen_inputs=len(read(F)['files']), checked_unique_files=len(expected),
                  checked_bytes=total_bytes, freeze_sha256=sha256_file(F),
                  automatic_done_sha256=sha256_file(REPORT/'automatic_done.json'))
    atomic_write_json(OUT/'integrity.json', result, refuse_if_exists=True)
    print('INTEGRITY', result, flush=True)


def raw_recompute(raw_path, stored_path, all_budgets=False):
    raw = pd.read_csv(raw_path)
    if not all_budgets: raw = raw[raw.budget.le(2)].copy()
    raw = raw.reset_index(drop=True)
    subsets = [tuple(map(int, x.split(';'))) for x in raw.photo_ids.astype(str)]
    probabilities = np.array([json.loads(x) for x in raw.probabilities_json])
    assert probabilities.shape == (len(raw), 100)
    assert np.isfinite(probabilities).all() and probabilities.min() >= 0
    close(probabilities.sum(1), 1, 'probability sum', 1e-5)
    labels = raw.label.to_numpy()
    guesses = probabilities.argmax(1)
    assert np.array_equal(guesses, raw.prediction)
    singleton = {s[0]: probabilities[i] for i, s in enumerate(subsets) if len(s) == 1}
    singles_prediction = {p: int(prob.argmax()) for p, prob in singleton.items()}
    rows = []
    for method in ['native', 'probability_pool']:
        probs = probabilities if method == 'native' else np.array([np.mean([singleton[p] for p in s], axis=0) for s in subsets])
        correct = probs.argmax(1) == labels
        expected_single = np.array([np.mean([singles_prediction[p] == labels[i] for p in s]) for i, s in enumerate(subsets)])
        anycorrect = np.array([any(singles_prediction[p] == labels[i] for p in s) for i, s in enumerate(subsets)])
        joint, same = [], []
        for i, s in enumerate(subsets):
            pairs = list(itertools.combinations(s, 2))
            joint.append(np.mean([singles_prediction[p] != labels[i] and singles_prediction[q] != labels[i] for p, q in pairs]) if pairs else np.nan)
            same.append(np.mean([singles_prediction[p] == singles_prediction[q] and singles_prediction[p] != labels[i] for p, q in pairs]) if pairs else np.nan)
        values = raw[['observation_id', 'budget']].copy()
        values['accuracy'] = correct.astype(float)
        values['singleton_accuracy'] = expected_single
        values['joint_wrong_pair'] = joint
        values['same_wrong_pair'] = same
        values['any_single_correct_set_wrong'] = (anycorrect & ~correct).astype(float)
        values['all_single_wrong_set_correct'] = (~anycorrect & correct).astype(float)
        rows.append(values.groupby(['observation_id', 'budget']).mean().reset_index().assign(method=method))
    recalculated = pd.concat(rows, ignore_index=True)
    original = pd.read_csv(stored_path)
    if not all_budgets: original = original[original.budget.le(2)]
    keys = ['method', 'observation_id', 'budget']
    a = recalculated.set_index(keys).sort_index()
    b = original.set_index(keys).sort_index()
    assert a.index.equals(b.index)
    close(a[metrics], b[metrics], 'raw metric mismatch '+str(raw_path))
    k2 = a.reset_index().query('budget==2')
    close(k2.accuracy, 1-k2.joint_wrong_pair-k2.any_single_correct_set_wrong+k2.all_single_wrong_set_correct,
          'two-photo error partition identity')
    return dict(raw_file=str(raw_path), rows=len(raw), groups=int(raw.observation_id.nunique()),
                budgets=sorted(raw.budget.unique().astype(int).tolist()), status='PASS')


def paired_check():
    real = pd.read_csv(REPORT/'real_group_all_seeds.csv')
    tests = pd.read_csv(REPORT/'paired_tests_holm16.csv')
    records = []
    for t in tests.itertuples():
        s = real[(real.backbone == t.backbone) & (real.model == t.model)]
        assert s.seed.nunique() == 3
        x = s.groupby(['method', 'budget', 'observation_id', 'observer_id', 'label']).accuracy.mean().reset_index()
        if t.contrast == 'native_minus_pool':
            a = x[(x.method == 'native') & (x.budget == t.budget)]
            b = x[(x.method == 'probability_pool') & (x.budget == t.budget)]
        else:
            a = x[(x.method == 'native') & (x.budget == t.budget)]
            b = x[(x.method == 'native') & (x.budget == t.budget-1)]
        pair = a.merge(b[['observation_id', 'accuracy']], on='observation_id', suffixes=('_a', '_b'), validate='one_to_one')
        pair['delta'] = pair.accuracy_a-pair.accuracy_b
        counts = pair.groupby('label').delta.transform('size')
        weights = 1/(pair.label.nunique()*counts)
        contributions = (pair.delta*weights).groupby(pair.observer_id).sum().to_numpy()
        estimate = float(contributions.sum())
        close(estimate, t.delta, 'paired effect')
        rng = np.random.default_rng(20260919); exceed = 0
        for start in range(0, 9999, 500):
            signs = rng.integers(0, 2, (min(500, 9999-start), len(contributions)))*2-1
            exceed += int((np.abs(signs@contributions) >= abs(estimate)-1e-12).sum())
        p = (1+exceed)/10000
        close(p, t.p, 'signflip p')
        centered = (pair.delta-pair.groupby('label').delta.transform('mean'))*weights
        cluster = centered.groupby(pair.observer_id).sum().to_numpy()
        rng = np.random.default_rng(20260919); boot = []
        for start in range(0, 5000, 100):
            draws = rng.multinomial(len(cluster), np.ones(len(cluster))/len(cluster), size=min(100, 5000-start))
            boot.extend(estimate+draws@cluster)
        ci = np.quantile(boot, [.025, .975])
        close(ci, [t.lower95, t.upper95], 'cluster interval')
        assert (len(pair), pair.label.nunique(), pair.observer_id.nunique()) == (t.groups, t.taxa, t.observers)
        records.append(dict(backbone=t.backbone, model=t.model, contrast=t.contrast, budget=t.budget,
                            estimate=estimate, p=p, lower95=ci[0], upper95=ci[1]))
    order = np.argsort([r['p'] for r in records], kind='stable')
    adjusted = np.empty(len(records))
    largest = 0.
    for rank, i in enumerate(order):
        largest = max(largest, (len(records)-rank)*records[i]['p'])
        adjusted[i] = min(1., largest)
    close(adjusted, tests.p_holm16, 'Holm16')
    for row, p in zip(records, adjusted): row['holm16'] = p
    atomic_write_csv(OUT/'paired_recomputed.csv', pd.DataFrame(records), refuse_if_exists=True)


def mapping_check(common, pseudo, mapping):
    assert set(common.photo_id) == set(pseudo.photo_id) and not pseudo.photo_id.duplicated().any()
    assert set(common.observation_id) == set(pseudo.observation_id)
    original = common.set_index('photo_id')
    src = original.loc[mapping.photo_id]
    assert np.array_equal(src.observation_id, mapping.source_observation_id)
    assert np.array_equal(src.observer_id, mapping.source_observer_id)
    assert np.array_equal(pseudo.set_index('photo_id').loc[mapping.photo_id].observation_id, mapping.slot_observation_id)
    assert (mapping.source_observation_id != mapping.slot_observation_id).all()
    assert mapping.groupby('slot_observation_id').source_observation_id.nunique().min() >= 2
    for obs, p in pseudo.groupby('observation_id'):
        old = common[common.observation_id == obs]
        assert len(old) == len(p)
        assert original.loc[p.photo_id].taxon_id.eq(old.taxon_id.iloc[0]).all()


def main():
    if (OUT/'audit_done.json').exists(): raise FileExistsError('Audit already sealed')
    OUT.mkdir(parents=True, exist_ok=True)
    integrity()
    paired_check()
    print('PAIRED_16_PASS', flush=True)
    frame = pd.read_csv(ROOT/CFG['source_report']/'primary_manifest.csv')
    common = pd.read_csv(REPORT/'common_manifest.csv')
    maps = []
    for condition in ['unrestricted_common', 'difficulty_matched']:
        for repeat in range(20):
            directory = DATA/'maps'/condition/f'{repeat:02d}'
            pseudo = pd.read_csv(directory/'manifest.csv'); mapping = pd.read_csv(directory/'mapping.csv')
            mapping_check(common, pseudo, mapping)
            if condition == 'difficulty_matched':
                a = common.groupby('observation_id').difficulty_bin.agg(lambda x: tuple(sorted(x)))
                b = pseudo.groupby('observation_id').difficulty_bin.agg(lambda x: tuple(sorted(x)))
                assert a.equals(b)
            maps.append(dict(condition=condition, repeat=repeat, groups=pseudo.observation_id.nunique(), status='PASS'))
    raw_audits = []
    for spec in CFG['runs']:
        rid = spec['run_id']
        raw_root = Path(CFG['source_data'])/'runs' if spec['backbone'] == 'dinov2' else Path('E:/Datasets/ROBird-Bench/explanatory_suite_v1/real')
        raw_audits.append(raw_recompute(raw_root/rid/'predictions.csv', REPORT/'real'/rid/'groups.csv', True))
        for condition in ['unrestricted_common', 'difficulty_matched']:
            for repeat in range(20):
                rel = Path('controls')/rid/condition/f'{repeat:02d}'
                raw_audits.append(raw_recompute(DATA/rel/'predictions.csv', REPORT/rel/'groups.csv'))
        print('RAW_PASS', rid, len(raw_audits), flush=True)
    atomic_write_csv(OUT/'raw_probability_audit.csv', pd.DataFrame(raw_audits), refuse_if_exists=True)
    # ResNet full-support donor mapping and per-repeat k2 raw native point.
    full_audits = []
    for spec in CFG['runs']:
        if spec['backbone'] != 'resnet50': continue
        rid = spec['run_id']; comparisons = pd.read_csv(REPORT/'resnet_e3'/rid/'e3_randomizations.csv')
        for repeat in range(20):
            directory = DATA/'resnet_e3'/rid
            mapping = pd.read_csv(directory/f'e3_repeat_{repeat:02d}_mapping.csv')
            selected = frame[frame.observation_id.isin(mapping.slot_observation_id)]
            pseudo = frame.set_index('photo_id').loc[mapping.photo_id].reset_index()
            pseudo['observation_id'] = mapping.slot_observation_id.to_numpy()
            mapping_check(selected, pseudo, mapping)
            pred = pd.read_csv(directory/f'e3_repeat_{repeat:02d}_predictions.csv')
            pred = pred[pred.budget.eq(2)].copy()
            pred['correct'] = [int(np.argmax(json.loads(x))) == y for x, y in zip(pred.probabilities_json, pred.label)]
            by_group = pred.groupby(['observation_id', 'label']).correct.mean().reset_index()
            point = by_group.groupby('label').correct.mean().mean()
            original = comparisons[(comparisons['repeat'] == repeat) & (comparisons.budget == 2)].iloc[0]
            close(point, original.macro_shuffled, 'ResNet full E3 k2')
            full_audits.append(dict(run_id=rid, repeat=repeat, groups=len(by_group), status='PASS'))
    atomic_write_csv(OUT/'mapping_audit.csv', pd.DataFrame(maps), refuse_if_exists=True)
    atomic_write_csv(OUT/'resnet_full_e3_audit.csv', pd.DataFrame(full_audits), refuse_if_exists=True)
    finish(OUT/'audit_done.json', sha256_file(F), [*sorted(OUT.glob('*.csv')), OUT/'integrity.json', Path(__file__)],
           status='PASS_SPECIFIED_RECOMPUTATIONS', real_raw_files=18, control_raw_files=720,
           control_raw_budgets=[1, 2], real_raw_budgets=[1, 2, 3, 4, 5],
           paired_tests=16, full_e3_maps_and_k2_points=120, new_inference=0, new_training=0)
    print('AUDIT_COMPLETE', sha256_file(OUT/'audit_done.json'), flush=True)


if __name__ == '__main__': main()
