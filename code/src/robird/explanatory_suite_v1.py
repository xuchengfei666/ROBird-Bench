"""Post-result explanatory controls. No training or network entry point."""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from robird.io import atomic_write_csv, atomic_write_json, sha256_file
from robird.rsos_suite_v1 import (atomic_binary, finish, marker, make_model,
                                  prediction_frame, shuffle_membership)
from robird.budget_metrics_v2_1 import validate_records, group_statistics, cluster_interval
from robird.external_cohort_v1_1 import GateStop, holm
from robird.external_completion_v1 import observer_signflip
from robird.artifact_tree_v1 import verify_tree

KEYS = ['observation_id', 'observer_id', 'taxon_id', 'label', 'n_photos', 'budget']
SCALARS = ['accuracy', 'singleton_accuracy', 'joint_wrong_pair', 'same_wrong_pair',
           'any_single_correct_set_wrong', 'all_single_wrong_set_correct', 'nll', 'brier', 'top5']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def checkpoint_model(spec, dim, device='cuda'):
    path = Path(spec['checkpoint'])
    if sha256_file(path) != spec['checkpoint_sha256']:
        raise GateStop('Checkpoint drift: ' + str(path))
    model = make_model(spec['model'], dim).to(device).eval()
    model.load_state_dict(torch.load(path, map_location='cpu', weights_only=False)['model_state'], strict=True)
    return model


@torch.no_grad()
def singleton_probs(model, features, batch_size=128):
    device = next(model.parameters()).device
    result = []
    for pos in range(0, len(features), batch_size):
        x = torch.as_tensor(np.asarray(features[pos:pos+batch_size]).copy(), dtype=torch.float32,
                            device=device)[:, None, :]
        mask = torch.ones(x.shape[:2], dtype=torch.bool, device=device)
        result.append(model(x, mask).logits.softmax(-1).cpu().numpy())
    p = np.concatenate(result)
    if p.shape != (len(features), 100) or not np.isfinite(p).all():
        raise GateStop('Invalid singleton probabilities')
    return p


def aligned_features(manifest, matrix, index):
    if index.photo_id.duplicated().any() or len(index) != len(matrix):
        raise GateStop('Invalid feature index')
    table = index.set_index('photo_id').feature_row
    if not set(manifest.photo_id).issubset(set(table.index)):
        raise GateStop('Missing feature photos')
    if table.duplicated().any() or table.min() < 0 or table.max() >= len(matrix):
        raise GateStop('Invalid feature row range')
    return np.asarray(matrix[table.loc[manifest.photo_id].astype(int).to_numpy()])


def extract_resnet(frame, data, report, config, contract):
    done = report/'resnet_features_done.json'
    directory = data/'resnet50_features'
    if marker(done, contract):
        verify_tree(done)
        return aligned_features(frame, np.load(directory/'features.npy', mmap_mode='r'),
                                pd.read_csv(directory/'index.csv'))
    from torchvision.models import resnet50, ResNet50_Weights
    weight = Path(config['resnet_weights'])
    if sha256_file(weight) != config['resnet_weights_sha256']:
        raise GateStop('Backbone weight drift')
    model = resnet50(weights=None)
    model.load_state_dict(torch.load(weight, map_location='cpu', weights_only=True), strict=True)
    model.fc = torch.nn.Identity()
    model = model.eval().cuda()
    transform = ResNet50_Weights.IMAGENET1K_V2.transforms()
    shards = []
    for start in range(0, len(frame), 512):
        part = frame.iloc[start:start+512]
        path = directory/f'chunk_{start:06d}.npy'
        stamp = path.with_suffix('.json')
        previous = marker(stamp, contract)
        if previous:
            if previous['photo_ids'] != part.photo_id.astype(int).tolist():
                raise GateStop('Feature shard order changed')
        else:
            values = []
            for pos in range(0, len(part), 32):
                images = []
                for row in part.iloc[pos:pos+32].itertuples():
                    if sha256_file(Path(row.local_path)) != row.sha256:
                        raise GateStop('Image bytes changed')
                    with Image.open(row.local_path) as im:
                        images.append(transform(im.convert('RGB')))
                with torch.no_grad():
                    f = torch.nn.functional.normalize(model(torch.stack(images).cuda()), dim=1)
                    values.append(f.cpu().numpy())
            features = np.concatenate(values)
            if features.shape != (len(part), 2048) or not np.isfinite(features).all():
                raise GateStop('Invalid ResNet features')
            atomic_binary(path, lambda f: np.save(f, features))
            finish(stamp, contract, [path], photo_ids=part.photo_id.astype(int).tolist())
        shards.append(stamp)
    features = np.concatenate([np.load(p.with_suffix('.npy')) for p in shards])
    atomic_binary(directory/'features.npy', lambda f: np.save(f, features))
    atomic_write_csv(directory/'index.csv', pd.DataFrame(dict(photo_id=frame.photo_id,
                                                            feature_row=np.arange(len(frame)))))
    finish(done, contract, [*shards, directory/'features.npy', directory/'index.csv', weight],
           transform=str(transform), l2_normalized=True, shape=list(features.shape),
           peak_cuda_bytes=int(torch.cuda.max_memory_allocated()))
    del model
    torch.cuda.empty_cache()
    return features


def difficulty_stage(root, frame, features, data, report, config, contract):
    done = report/'difficulty_done.json'
    if marker(done, contract):
        return pd.read_csv(report/'photo_difficulty.csv')
    dev = pd.read_csv(root/config['development_manifest'])
    splits = pd.read_csv(root/config['development_splits'])
    dev = dev.merge(splits[['observation_id', 'observer_id', 'taxon_id', 'split']],
                    on=['observation_id', 'observer_id', 'taxon_id'], validate='many_to_one')
    val = dev[dev.split.eq('validation')].copy()
    if val.empty or set(val.observer_id) & set(frame.observer_id):
        raise GateStop('Missing validation or external observer overlap')
    directory = Path(config['development_data'])/'resnet50_features'
    val_features = aligned_features(val, np.load(directory/'features.npy', mmap_mode='r'),
                                    pd.read_csv(directory/'index.csv'))
    pval, pext = [], []
    for spec in config['proxy_runs']:
        model = checkpoint_model(spec, 2048)
        pval.append(singleton_probs(model, val_features))
        pext.append(singleton_probs(model, features))
        del model
    pval, pext = np.mean(pval, axis=0), np.mean(pext, axis=0)
    val_d = -np.log(np.clip(pval[np.arange(len(val)), val.class_index.to_numpy()], 1e-12, 1))
    ext_d = -np.log(np.clip(pext[np.arange(len(frame)), frame.class_index.to_numpy()], 1e-12, 1))
    edges = np.quantile(val_d, config['proxy_quantiles'], method='linear')
    if not np.isfinite(edges).all() or edges[0] >= edges[1]:
        raise GateStop('Degenerate validation difficulty cutpoints; no adaptive rebins')
    out = frame[['photo_id', 'observation_id', 'observer_id', 'taxon_id', 'class_index']].copy()
    out['difficulty'] = ext_d
    out['difficulty_bin'] = np.searchsorted(edges, ext_d, side='right')
    validation = val[['photo_id', 'observation_id', 'observer_id', 'taxon_id', 'class_index']].copy()
    validation['difficulty'] = val_d
    atomic_write_csv(report/'photo_difficulty.csv', out)
    atomic_write_csv(report/'validation_difficulty.csv', validation)
    atomic_binary(data/'proxy_probabilities.npz', lambda f: np.savez_compressed(
        f, validation=pval, external=pext, validation_photo_ids=val.photo_id.to_numpy(),
        external_photo_ids=frame.photo_id.to_numpy()))
    atomic_write_json(report/'difficulty_cutpoints.json', dict(
        cutpoints=edges.tolist(), validation_photos=len(val), validation_observations=val.observation_id.nunique(),
        definition='minus_log_mean_seed_true_class_probability', split='validation',
        external_labels_used_for_proxy_only=True, target_correctness_used_for_matching=False))
    finish(done, contract, [report/'photo_difficulty.csv', report/'validation_difficulty.csv',
        report/'difficulty_cutpoints.json', data/'proxy_probabilities.npz'])
    torch.cuda.empty_cache()
    return out


def eligible_support(frame, minimum=3):
    rows = []
    for obs, p in frame.groupby('observation_id', sort=True):
        rows.append(dict(observation_id=int(obs), taxon_id=int(p.taxon_id.iloc[0]),
            observer_id=int(p.observer_id.iloc[0]), n=len(p),
            signature=';'.join(map(str, sorted(p.difficulty_bin.astype(int))))))
    groups = pd.DataFrame(rows)
    groups['stratum_groups'] = groups.groupby(['taxon_id', 'n', 'signature']).observation_id.transform('size')
    groups['included'] = groups.stratum_groups.ge(minimum)
    groups['reason'] = np.where(groups.included, 'MATCHABLE', 'LT3_SAME_TAXON_CARDINALITY_SIGNATURE')
    return frame[frame.observation_id.isin(groups.loc[groups.included, 'observation_id'])].copy(), groups


def validate_mapping(real, pseudo, matched=False):
    if real.photo_id.duplicated().any() or pseudo.photo_id.duplicated().any():
        raise GateStop('Duplicate image use')
    if set(real.photo_id) != set(pseudo.photo_id) or len(real) != len(pseudo):
        raise GateStop('Photo pool changed')
    if set(real.observation_id) != set(pseudo.observation_id):
        raise GateStop('Observation support changed')
    origin = real.set_index('photo_id')
    if not pseudo.observer_id.eq(-1).all():
        raise GateStop('Pseudo-set cannot claim a real observer')
    for obs, part in pseudo.groupby('observation_id'):
        source = origin.loc[part.photo_id]
        target = real[real.observation_id.eq(obs)]
        if len(target) != len(part) or source.observation_id.eq(obs).any():
            raise GateStop('Cardinality/self donor violation')
        if source.observation_id.nunique() < 2:
            raise GateStop('Fewer than two donor observations')
        if (not source.taxon_id.eq(target.taxon_id.iloc[0]).all()
                or not part.taxon_id.eq(target.taxon_id.iloc[0]).all()
                or not part.class_index.eq(target.class_index.iloc[0]).all()):
            raise GateStop('Cross-taxon assignment')
        for name in ['difficulty', 'difficulty_bin', 'sha256', 'local_path']:
            if name in real and not np.array_equal(part[name].to_numpy(), source[name].to_numpy()):
                raise GateStop('Photo metadata changed: ' + name)
        if matched and sorted(source.difficulty_bin.tolist()) != sorted(target.difficulty_bin.tolist()):
            raise GateStop('Difficulty signature changed')
    return True


def regroup(frame, seed, matched):
    frame = frame.copy()
    frame['n'] = frame.groupby('observation_id').photo_id.transform('size')
    signatures = frame.groupby('observation_id').difficulty_bin.agg(lambda s: ';'.join(map(str, sorted(s))))
    frame['signature'] = frame.observation_id.map(signatures)
    cols = ['taxon_id', 'n', 'signature'] if matched else ['taxon_id', 'n']
    rng = np.random.default_rng(seed)
    shuffled, mappings = [], []
    for _, part in frame.groupby(cols, sort=True):
        ids = np.sort(part.observation_id.unique())
        if len(ids) < 3:
            raise GateStop('Unmatchable stratum entered regrouping')
        rng.shuffle(ids)
        photos = {}
        for obs in ids:
            p = part[part.observation_id.eq(obs)].sort_values('photo_id')
            p = p.iloc[rng.permutation(len(p))]
            if matched:
                p = p.sort_values('difficulty_bin', kind='stable')
            photos[int(obs)] = p.to_dict('records')
        for i, obs in enumerate(ids):
            for j in range(int(part.n.iloc[0])):
                donor = int(ids[(i+1+j % (len(ids)-1)) % len(ids)])
                row = dict(photos[donor][j])
                mappings.append(dict(slot_observation_id=int(obs), photo_id=int(row['photo_id']),
                    source_observation_id=donor, source_observer_id=int(row['observer_id']),
                    slot_observer_id=int(part.loc[part.observation_id.eq(obs), 'observer_id'].iloc[0])))
                row.update(observation_id=int(obs), observer_id=-1)
                shuffled.append(row)
    pseudo = pd.DataFrame(shuffled).drop(columns=['n', 'signature'])
    validate_mapping(frame, pseudo, matched)
    return pseudo, pd.DataFrame(mappings)


def mapping_balance(real, pseudo, mapping):
    rows = []
    for obs, part in pseudo.groupby('observation_id'):
        a = real[real.observation_id.eq(obs)]
        x, y = np.sort(a.difficulty.to_numpy()), np.sort(part.difficulty.to_numpy())
        rows.append(dict(observation_id=int(obs), taxon_id=int(a.taxon_id.iloc[0]),
            rank_difficulty_abs=float(np.abs(x-y).mean()),
            mean_difficulty_abs=float(abs(x.mean()-y.mean())),
            bin_match_fraction=float(np.mean(np.sort(a.difficulty_bin)==np.sort(part.difficulty_bin)))))
    out = pd.DataFrame(rows)
    return out, dict(same_source_observer_fraction=float(
        mapping.source_observer_id.eq(mapping.slot_observer_id).mean()))


def mappings_stage(frame, difficulty, data, report, config, contract):
    done = report/'mappings_done.json'
    if marker(done, contract):
        verify_tree(done)
        return pd.read_csv(report/'common_manifest.csv')
    enriched = frame.merge(difficulty[['photo_id', 'difficulty', 'difficulty_bin']], on='photo_id',
                           validate='one_to_one')
    common, attrition = eligible_support(enriched, config['minimum_stratum_observations'])
    atomic_write_csv(report/'common_manifest.csv', common)
    atomic_write_csv(report/'matching_attrition.csv', attrition)
    summary = attrition.groupby('taxon_id').agg(total=('observation_id', 'size'), retained=('included', 'sum')).reset_index()
    atomic_write_csv(report/'matching_attrition_by_taxon.csv', summary)
    artifacts = [report/'common_manifest.csv', report/'matching_attrition.csv', report/'matching_attrition_by_taxon.csv']
    if len(common):
        for condition in ['unrestricted_common', 'difficulty_matched']:
            for repeat in range(config['randomizations']):
                directory = data/'maps'/condition/f'{repeat:02d}'
                stamp = directory/'done.json'
                if not marker(stamp, contract):
                    pseudo, mapping = regroup(common, config['randomization_seed']+repeat,
                                               condition == 'difficulty_matched')
                    balance, stats = mapping_balance(common, pseudo, mapping)
                    atomic_write_csv(directory/'manifest.csv', pseudo)
                    atomic_write_csv(directory/'mapping.csv', mapping)
                    atomic_write_csv(directory/'balance.csv', balance)
                    finish(stamp, contract, [directory/p for p in ('manifest.csv', 'mapping.csv', 'balance.csv')],
                           condition=condition, repeat=repeat, **stats)
                artifacts.append(stamp)
    finish(done, contract, artifacts, photos=len(common), observations=int(common.observation_id.nunique()),
           taxa=int(common.taxon_id.nunique()), status='MATCHABLE' if len(common) else 'NOT_ESTIMABLE')
    return common


def decompose(groups):
    rows, nested = [], []
    max_pool_difference = 0.
    for obs, g in groups.items():
        subsets, y = g['subsets'], g['label']
        singleton = {p: subsets[(p,)] for p in g['photos']}
        pooled = {s: np.stack([singleton[p] for p in s]).mean(0) for s in subsets}
        for s in subsets:
            max_pool_difference = max(max_pool_difference, float(np.max(np.abs(subsets[s]-pooled[s]))))
        base = dict(observation_id=obs, observer_id=g['observer_id'], taxon_id=g['taxon_id'],
                    label=y, n_photos=len(g['photos']))
        for method, probs in [('native', subsets), ('probability_pool', pooled)]:
            accuracy = {}
            for k in range(1, len(g['photos'])+1):
                selected = [s for s in subsets if len(s) == k]
                p = np.stack([probs[s] for s in selected])
                correct = p.argmax(1) == y
                single_correct = np.array([[singleton[j].argmax() == y for j in s] for s in selected])
                joint, same = [], []
                if k > 1:
                    for s in selected:
                        guesses = [int(singleton[j].argmax()) for j in s]
                        pairs = list(itertools.combinations(guesses, 2))
                        joint.append(np.mean([a != y and b != y for a, b in pairs]))
                        same.append(np.mean([a == b and a != y for a, b in pairs]))
                confidence = p.max(1)
                bins = np.minimum((confidence*15).astype(int), 14)
                residual = np.bincount(bins, weights=correct-confidence, minlength=15)/len(p)
                target = np.eye(p.shape[1])[y]
                accuracy[k] = float(correct.mean())
                rows.append(dict(base, method=method, budget=k, accuracy=accuracy[k],
                    singleton_accuracy=float(single_correct.mean()),
                    joint_wrong_pair=float(np.mean(joint)) if joint else np.nan,
                    same_wrong_pair=float(np.mean(same)) if same else np.nan,
                    any_single_correct_set_wrong=float((single_correct.any(1) & ~correct).mean()),
                    all_single_wrong_set_correct=float((~single_correct.any(1) & correct).mean()),
                    nll=float(-np.log(np.clip(p[:, y], 1e-12, 1)).mean()),
                    brier=float(np.square(p-target).sum(1).mean()),
                    top5=float(np.any(np.argsort(p, axis=1)[:, -5:] == y, axis=1).mean()),
                    **{f'ece_bin_{i:02d}': float(v) for i, v in enumerate(residual)}))
            for k in range(1, len(g['photos'])):
                edges = [(probs[s].argmax() == y, probs[tuple(sorted((*s, j)))].argmax() == y)
                         for s in subsets if len(s) == k for j in g['photos'] if j not in s]
                a, b = np.asarray(edges, dtype=bool).T
                correction, regression = float((~a & b).mean()), float((a & ~b).mean())
                if not np.isclose(correction-regression, accuracy[k+1]-accuracy[k], atol=1e-10):
                    raise GateStop('Nested-edge identity failed')
                nested.append(dict(base, method=method, budget_from=k, budget_to=k+1,
                    correction=correction, regression=regression, net_gain=correction-regression))
    return pd.DataFrame(rows), pd.DataFrame(nested), max_pool_difference


def scalar_summary(rows):
    output = []
    for (method, k), p in rows.groupby(['method', 'budget'], sort=True):
        values = p.groupby('label')[SCALARS].mean().mean()
        output.append(dict(method=method, budget=int(k), groups=len(p), taxa=int(p.label.nunique()),
            **{c: (None if pd.isna(v) else float(v)) for c, v in values.items()},
            micro_accuracy=float(p.accuracy.mean()),
            group_equal_nll=float(p.nll.mean()), group_equal_brier=float(p.brier.mean()),
            group_equal_top5=float(p.top5.mean()),
            group_equal_ece=float(np.abs(p[[f'ece_bin_{i:02d}' for i in range(15)]].mean()).sum())))
    return output


def save_decomposition(predictions, frame, out, contract, spec, artifacts, reference=None):
    groups = validate_records(predictions, frame, num_classes=100)
    rows, edges, difference = decompose(groups)
    if spec['model'] == 'probability_mlp' and difference > 2e-6:
        raise GateStop('ProbabilityMLP singleton-pool negative control failed')
    if spec['model'] == 'probability_mlp':
        a = rows[rows.method.eq('native')].accuracy.to_numpy()
        b = rows[rows.method.eq('probability_pool')].accuracy.to_numpy()
        if not np.array_equal(a, b):
            raise GateStop('ProbabilityMLP argmax differs despite numerical equivalence')
    if reference is not None:
        a = reference[(reference.budget == 1) & reference.method.eq('native')]
        b = rows[(rows.budget == 1) & rows.method.eq('native')]
        for col in ['accuracy', 'singleton_accuracy']:
            if not np.isclose(a.groupby('label')[col].mean().mean(),
                              b.groupby('label')[col].mean().mean(), atol=1e-7, rtol=0):
                raise GateStop('k1 photo-pool negative control failed')
    atomic_write_csv(out/'groups.csv', rows)
    atomic_write_csv(out/'nested.csv', edges)
    atomic_write_json(out/'summary.json', dict(spec=spec, cells=scalar_summary(rows),
        maximum_native_pool_probability_difference=difference, post_hoc=True))
    finish(out/'done.json', contract, [*artifacts, out/'groups.csv', out/'nested.csv', out/'summary.json'])


def real_stage(frame, feature_sets, root, data, report, config, contract, progress):
    for spec in config['runs']:
        run_id = spec['run_id']
        out, directory = report/'real'/run_id, data/'real'/run_id
        progress('real_native_and_pool', run_id=run_id)
        if marker(out/'done.json', contract):
            continue
        if spec['backbone'] == 'dinov2':
            path = Path(config['source_data'])/'runs'/run_id/'predictions.csv'
            predictions = pd.read_csv(path)
        else:
            path = directory/'predictions.csv'
            if marker(directory/'predictions_done.json', contract):
                predictions = pd.read_csv(path)
            else:
                features = feature_sets['resnet50']
                model = checkpoint_model(spec, features.shape[1])
                predictions = prediction_frame(model, frame.assign(split='development_test'), features)
                atomic_write_csv(path, predictions)
                finish(directory/'predictions_done.json', contract, [path, Path(spec['checkpoint'])])
                del model
                torch.cuda.empty_cache()
        save_decomposition(predictions, frame, out, contract, spec, [path, Path(spec['checkpoint'])])


def controls_stage(frame, common, feature_sets, data, report, config, contract, progress):
    if common.empty:
        return
    row_map = dict(zip(frame.photo_id.astype(int), range(len(frame))))
    for spec in config['runs']:
        run_id = spec['run_id']
        features = feature_sets[spec['backbone']]
        model = None
        for condition in ['unrestricted_common', 'difficulty_matched']:
            for repeat in range(config['randomizations']):
                out = report/'controls'/run_id/condition/f'{repeat:02d}'
                progress('common_support_controls', run_id=run_id, condition=condition, repeat=repeat)
                if marker(out/'done.json', contract):
                    continue
                maps = data/'maps'/condition/f'{repeat:02d}'
                pseudo = pd.read_csv(maps/'manifest.csv')
                validate_mapping(common, pseudo, condition == 'difficulty_matched')
                raw = data/'controls'/run_id/condition/f'{repeat:02d}'
                if marker(raw/'predictions_done.json', contract):
                    predictions = pd.read_csv(raw/'predictions.csv')
                else:
                    if model is None:
                        model = checkpoint_model(spec, features.shape[1])
                    aligned = features[[row_map[int(p)] for p in pseudo.photo_id]]
                    predictions = prediction_frame(model, pseudo.assign(split='development_test'), aligned)
                    atomic_write_csv(raw/'predictions.csv', predictions)
                    finish(raw/'predictions_done.json', contract,
                           [raw/'predictions.csv', maps/'done.json', Path(spec['checkpoint'])])
                real = pd.read_csv(report/'real'/run_id/'groups.csv')
                reference = real[real.observation_id.isin(common.observation_id)]
                save_decomposition(predictions, pseudo, out, contract, spec,
                                   [raw/'predictions_done.json', maps/'done.json'], reference=reference)
        del model
        torch.cuda.empty_cache()


def resnet_e3_stage(frame, features, data, report, config, contract, progress):
    from robird.rsos_suite_v1 import e3_run
    for spec in config['runs']:
        if spec['backbone'] != 'resnet50':
            continue
        run_id = spec['run_id']
        progress('resnet_full_support_e3', run_id=run_id)
        out = report/'resnet_e3'/run_id
        if marker(out/'e3_done.json', contract):
            continue
        # e3_run reads groups.csv and never writes training artifacts.
        real = pd.read_csv(report/'real'/run_id/'groups.csv')
        real = real[real.method.eq('native')].rename(columns={'accuracy': 'expected_accuracy'})
        atomic_write_csv(out/'groups.csv', real)
        e3_run(frame.assign(split='development_test'), features, spec, Path(spec['checkpoint']),
               data/'resnet_e3'/run_id, out, contract)


def paired_inference(frame, config):
    p = frame.copy()
    if p.observation_id.duplicated().any() or p.observer_id.lt(0).any():
        raise GateStop('Invalid real-observer inference')
    if not np.isfinite(p.delta).all():
        raise GateStop('Nonfinite paired effect')
    taxa = p.label.nunique()
    count = p.groupby('label').delta.transform('size')
    contribution = (p.delta/(taxa*count)).groupby(p.observer_id).sum()
    weights = (1/(taxa*count)).groupby(p.observer_id).sum()
    test = observer_signflip(contribution.to_numpy(), config['signflip_repeats'], config['statistics_seed'])
    interval = cluster_interval(p, 'delta', config['bootstrap_repeats'], config['statistics_seed'])
    return dict(groups=len(p), taxa=int(taxa), observers=len(contribution),
        delta=test['statistic'], p=test['p'], lower95=interval['interval95'][0], upper95=interval['interval95'][1],
        max_cluster_weight=float(weights.max()), max_cluster_groups=int(p.groupby('observer_id').size().max()),
        observers_spanning_taxa=int((p.groupby('observer_id').label.nunique()>1).sum()),
        method=test['method'], draws=test['draws'],
        independence_and_sign_exchangeability='ASSUMED_NOT_PROVED', post_hoc=True)


def final_analysis(frame, common, data, report, config, contract):
    done = report/'analysis_done.json'
    if marker(done, contract):
        return
    real, edges, controls, full_e3 = [], [], [], []
    artifacts = []
    for spec in config['runs']:
        identity = {k: spec[k] for k in ['run_id', 'backbone', 'model', 'seed']}
        out = report/'real'/spec['run_id']
        real.append(pd.read_csv(out/'groups.csv').assign(**identity))
        edges.append(pd.read_csv(out/'nested.csv').assign(**identity))
        artifacts.append(out/'done.json')
        if len(common):
            baseline = real[-1][real[-1].observation_id.isin(common.observation_id)]
            base = pd.DataFrame(scalar_summary(baseline))
            for condition in ['unrestricted_common', 'difficulty_matched']:
                for repeat in range(config['randomizations']):
                    ctrl = report/'controls'/spec['run_id']/condition/f'{repeat:02d}'
                    cells = pd.DataFrame(read(ctrl/'summary.json')['cells'])
                    merged = cells.merge(base, on=['method', 'budget'], suffixes=('_control', '_real'), validate='one_to_one')
                    for col in SCALARS:
                        merged[col+'_delta'] = merged[col+'_control']-merged[col+'_real']
                    controls.append(merged.assign(**identity, condition=condition, repeat=repeat))
                    artifacts.append(ctrl/'done.json')
        if spec['backbone'] == 'resnet50':
            out = report/'resnet_e3'/spec['run_id']
            full_e3.append(pd.read_csv(out/'e3_randomizations.csv').assign(**identity))
            artifacts.append(out/'e3_done.json')
    real, edges = pd.concat(real, ignore_index=True), pd.concat(edges, ignore_index=True)
    atomic_write_csv(report/'real_group_all_seeds.csv', real)
    atomic_write_csv(report/'real_nested_all_seeds.csv', edges)
    curves, comparisons = [], []
    for (backbone, model), part in real.groupby(['backbone', 'model'], sort=True):
        # Means of seed metrics; never probability ensembling between seeds.
        for seed, s in part.groupby('seed'):
            for row in scalar_summary(s):
                curves.append(dict(row, backbone=backbone, model=model, seed=int(seed)))
        averaged = part.groupby(KEYS+['method'], as_index=False).accuracy.mean()
        if part.seed.nunique() != 3:
            raise GateStop('Missing model seed')
        for k in [2, 3]:
            a = averaged[(averaged.method == 'native') & (averaged.budget == k)]
            b = averaged[(averaged.method == 'probability_pool') & (averaged.budget == k)]
            pair = a.merge(b[['observation_id', 'accuracy']], on='observation_id', suffixes=('_native', '_pool'), validate='one_to_one')
            pair['delta'] = pair.accuracy_native-pair.accuracy_pool
            comparisons.append(dict(backbone=backbone, model=model, contrast='native_minus_pool', budget=k,
                                    **paired_inference(pair, config)))
        if backbone == 'resnet50':
            for before, after in [(1, 2), (2, 3)]:
                a = averaged[(averaged.method == 'native') & (averaged.budget == before)]
                b = averaged[(averaged.method == 'native') & (averaged.budget == after)]
                pair = a.merge(b[['observation_id', 'accuracy']], on='observation_id', suffixes=('_before', '_after'), validate='one_to_one')
                pair['delta'] = pair.accuracy_after-pair.accuracy_before
                comparisons.append(dict(backbone=backbone, model=model, contrast=f'k{before}_to_k{after}', budget=after,
                                        **paired_inference(pair, config)))
    if len(comparisons) != 16:
        raise GateStop('Planned Holm16 family incomplete')
    for row, p in zip(comparisons, holm([r['p'] for r in comparisons])):
        row['p_holm16'] = p
    curves = pd.DataFrame(curves)
    atomic_write_csv(report/'real_seed_curves.csv', curves)
    atomic_write_csv(report/'real_mean_curves.csv', curves.groupby(['backbone', 'model', 'method', 'budget']).mean(numeric_only=True).drop(columns='seed').reset_index())
    atomic_write_csv(report/'paired_tests_holm16.csv', pd.DataFrame(comparisons))
    taxon = real.groupby(['backbone', 'model', 'seed', 'method', 'budget', 'taxon_id'])[SCALARS].mean().reset_index()
    atomic_write_csv(report/'real_per_taxon.csv', taxon)
    nested = edges.groupby(['backbone', 'model', 'seed', 'method', 'budget_from', 'taxon_id'])[['correction', 'regression', 'net_gain']].mean().reset_index()
    atomic_write_csv(report/'nested_per_taxon.csv', nested)
    atomic_write_csv(report/'nested_seed_macro.csv', nested.groupby(['backbone', 'model', 'seed', 'method', 'budget_from'])[['correction', 'regression', 'net_gain']].mean().reset_index())
    if controls:
        controls = pd.concat(controls, ignore_index=True)
        atomic_write_csv(report/'control_seed_repeat_effects.csv', controls)
        repeat = controls.groupby(['backbone', 'model', 'method', 'condition', 'repeat', 'budget']).mean(numeric_only=True).drop(columns='seed').reset_index()
        atomic_write_csv(report/'control_repeat_effects.csv', repeat)
        cols = [c+'_delta' for c in SCALARS]
        distribution = repeat.groupby(['backbone', 'model', 'method', 'condition', 'budget'])[cols].agg(['mean', 'std', 'min', 'max'])
        distribution.columns = ['_'.join(c) for c in distribution.columns]
        atomic_write_csv(report/'control_distributions_not_CI.csv', distribution.reset_index())
        balance_rows = []
        for condition in ['unrestricted_common', 'difficulty_matched']:
            for repeat_id in range(config['randomizations']):
                directory = data/'maps'/condition/f'{repeat_id:02d}'
                b = pd.read_csv(directory/'balance.csv')
                balance_rows.append(dict(condition=condition, repeat=repeat_id, groups=len(b),
                    rank_difficulty_abs_mean=float(b.rank_difficulty_abs.mean()),
                    rank_difficulty_abs_max=float(b.rank_difficulty_abs.max()),
                    mean_difficulty_abs_mean=float(b.mean_difficulty_abs.mean()),
                    bin_match_fraction=float(b.bin_match_fraction.mean()),
                    same_source_observer_fraction=m_read_fraction(directory)))
        atomic_write_csv(report/'matching_balance_summary.csv', pd.DataFrame(balance_rows))
    atomic_write_csv(report/'resnet_full_e3_seed_repeats.csv', pd.concat(full_e3, ignore_index=True))
    # Full-support DINO E3 is historical read-only; export on the identical old E3 estimand.
    dino = []
    for spec in config['runs']:
        if spec['backbone'] == 'dinov2':
            p = Path(config['source_report_absolute'])/'runs'/spec['run_id']/'e3_randomizations.csv'
            dino.append(pd.read_csv(p).assign(backbone='dinov2', model=spec['model'], seed=spec['seed']))
    full = pd.concat([*dino, *full_e3], ignore_index=True)
    atomic_write_csv(report/'full_support_e3_encoder_comparison.csv', full.groupby(
        ['backbone', 'model', 'repeat', 'budget'])[['macro_real', 'macro_shuffled', 'difference', 'groups', 'taxa']].mean().reset_index())
    write_results_note(report, comparisons, curves, common, config)
    artifacts.append(report/'RESULTS.md')
    artifacts.extend(sorted(p for p in report.glob('*.csv')))
    artifacts.extend([report/'mappings_done.json', report/'difficulty_done.json', report/'resnet_features_done.json'])
    finish(done, contract, artifacts, post_hoc=True, tests=16,
           regrouping_uncertainty='20_REPEAT_DISTRIBUTION_NOT_CI_NO_SLOT_OBSERVER_INFERENCE',
           matched_status='ESTIMABLE' if len(common) else 'NOT_ESTIMABLE', original_g6_pass=False)


def m_read_fraction(directory):
    return read(directory/'done.json')['same_source_observer_fraction']


def write_results_note(report, comparisons, curves, common, config):
    from robird.io import _atomic_text
    lines = ['# Explanatory suite v1 — automatic evidence summary', '',
        '这是post-hoc解释性补实验；完成不等于假设成立。旧v2.2和G6=false不改变。', '',
        f'难度匹配共同支持：{common.observation_id.nunique()} observations / {common.taxon_id.nunique()} taxa / {len(common)} photos。',
        '筛选损耗见matching_attrition_by_taxon.csv；代理为独立ResNet线性k1头，开发validation分箱。', '',
        '## Real-group paired tests', '',
        '三seed平均同组差值；百分点评估，observer cluster点态95%CI，新增Holm16家族。', '',
        '| Encoder | Model | Contrast | Budget | Effect pp | 95% CI pp | Holm16 p |',
        '|---|---|---|---|---:|---|---:|']
    for r in comparisons:
        lines.append(f"| {r['backbone']} | {r['model']} | {r['contrast']} | {r['budget']} | {100*r['delta']:+.3f} | [{100*r['lower95']:+.3f}, {100*r['upper95']:+.3f}] | {r['p_holm16']:.5f} |")
    lines += ['', '非显著不等于等效；Probability MLP原生与概率池化是同义负对照，不算独立方法成功。',
        '完整seed、逐类、nested regression/correction分别见real_seed_curves、real_per_taxon、nested_seed_macro。', '']
    if len(common):
        r = pd.read_csv(report/'control_repeat_effects.csv')
        lines += ['## Common-support grouping controls', '',
            '以下k2 native差值为control minus real，20重复先各平均3seed，再列重复范围；不是CI。', '',
            '| Encoder | Model | Control | Mean gain pp | Repeat range pp |', '|---|---|---|---:|---|']
        for (b, model, condition), p in r[(r.budget == 2) & r.method.eq('native')].groupby(['backbone', 'model', 'condition']):
            v = 100*p.accuracy_delta
            lines.append(f'| {b} | {model} | {condition} | {v.mean():+.3f} | [{v.min():+.3f}, {v.max():+.3f}] |')
        lines += ['', '难度匹配后差距保留只能说明超出该粗代理分箱的残余分组效应；',
            '差距缩小支持难度构成贡献，不能将残余全部归因错误依赖。',
            'joint_wrong_pair、same_wrong_pair和set-correction/regression见control_repeat_effects；',
            '残余难度差、同observer来源比例见matching_balance_summary，不得省略。', '']
    else:
        lines += ['难度匹配无支持：NOT_ESTIMABLE，未降低门槛，不能宣称该控制验证成功。', '']
    replication = [r for r in comparisons if r['contrast'].startswith('k')]
    failures = [r for r in replication if r['delta'] <= 0 or r['p_holm16'] >= .05]
    lines += ['## Boundaries and interpretation', '',
        f'ResNet两模型四个预定预算对比中，{len(failures)}项未同时满足正点估计与Holm16<.05；完整正负证据均在上表。',
        'Full-support E3编码器对比见full_support_e3_encoder_comparison.csv，不能与common-support数值直接相减。',
        '同平台、可用性选择56类而非无偏100类；AI辅助去重非独立人工gold standard。',
        '未做人工质量、未验证模糊/遮挡/光学因果、未提出或训练新算法。',
        '下一步应结合匹配损耗、连续难度残差、错误共现与聚合对照共同判断主线，不按单个p值改故事。', '']
    _atomic_text(report/'RESULTS.md', '\n'.join(lines), refuse_if_exists=False)
