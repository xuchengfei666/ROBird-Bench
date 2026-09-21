"""Explicit float32 serialization adapter; immutable v1 science implementation."""
import importlib.util
from pathlib import Path
import shutil
import numpy as np
import pandas as pd

from robird.io import atomic_write_json, sha256_file
from robird.rsos_suite_v1 import finish, marker
from robird.artifact_tree_v1 import verify_tree

_spec = importlib.util.spec_from_file_location('robird._explanatory_frozen_v1_continuation',
                                             Path(__file__).with_name('explanatory_suite_v1.py'))
legacy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(legacy)
GateStop = legacy.GateStop
_original_validate = legacy.validate_mapping


def canonical_frame(frame):
    result = frame.copy()
    value = result.difficulty.to_numpy(dtype=np.float64)
    if not np.isfinite(value).all():
        raise GateStop('Nonfinite difficulty')
    result['difficulty'] = value.astype(np.float32)
    if not np.isfinite(result.difficulty).all():
        raise GateStop('Difficulty outside float32 range')
    return result


def validate_mapping(real, pseudo, matched=False):
    return _original_validate(canonical_frame(real), canonical_frame(pseudo), matched)


legacy.validate_mapping = validate_mapping


def completed_markers(root, cfg):
    report = root/'code/results/explanatory_suite_v1'
    data = Path(cfg['data_root'])
    result = [report/name for name in ['resnet_features_done.json', 'difficulty_done.json', 'mappings_done.json']]
    result += [report/'real'/s['run_id']/'done.json' for s in cfg['runs']]
    result += [data/'maps'/c/f'{r:02d}'/'done.json' for c in ['unrestricted_common', 'difficulty_matched'] for r in range(20)]
    if len(result) != 61:
        raise GateStop('Expected 61 source completion markers')
    return result


def verify_precision_recovery(root, cfg):
    report = root/'code/results/explanatory_suite_v1'
    data = Path(cfg['data_root'])
    common = pd.read_csv(report/'common_manifest.csv')
    proxy = pd.read_csv(report/'photo_difficulty.csv')
    with np.load(data/'proxy_probabilities.npz') as saved:
        p = saved['external']
        ids = saved['external_photo_ids']
        if p.dtype != np.float32 or not np.array_equal(ids, proxy.photo_id):
            raise GateStop('Unexpected proxy precision or order')
        difficulty = -np.log(np.clip(p[np.arange(len(p)), proxy.class_index.to_numpy()], 1e-12, 1))
    if not np.array_equal(difficulty, proxy.difficulty.to_numpy(dtype=np.float32)):
        raise GateStop('Proxy difficulty no longer matches original probabilities')
    canonical = pd.Series(difficulty, index=ids)
    if not np.array_equal(common.difficulty.to_numpy(dtype=np.float32), canonical.loc[common.photo_id].to_numpy()):
        raise GateStop('Common difficulty not authoritative')
    edges = np.asarray(legacy.read(report/'difficulty_cutpoints.json')['cutpoints'])
    expected_bins = np.searchsorted(edges, difficulty, side='right')
    if not np.array_equal(expected_bins, proxy.difficulty_bin):
        raise GateStop('Proxy bins changed')
    if not np.array_equal(common.difficulty_bin, proxy.set_index('photo_id').loc[common.photo_id].difficulty_bin):
        raise GateStop('Common bins changed')
    evidence = []
    for condition in ['unrestricted_common', 'difficulty_matched']:
        for repeat in range(20):
            directory = data/'maps'/condition/f'{repeat:02d}'
            pseudo = pd.read_csv(directory/'manifest.csv')
            validate_mapping(common, pseudo, condition == 'difficulty_matched')
            if not np.array_equal(pseudo.difficulty.to_numpy(dtype=np.float32), canonical.loc[pseudo.photo_id].to_numpy()):
                raise GateStop('Map difficulty not authoritative')
            mapping = pd.read_csv(directory/'mapping.csv')
            source = common.set_index('photo_id').loc[mapping.photo_id]
            dest = pseudo.set_index('photo_id').loc[mapping.photo_id]
            if (not np.array_equal(mapping.source_observation_id, source.observation_id)
                    or not np.array_equal(mapping.source_observer_id, source.observer_id)
                    or not np.array_equal(mapping.slot_observation_id, dest.observation_id)):
                raise GateStop('Source identity mapping changed')
            before = common.set_index('photo_id').loc[pseudo.photo_id].difficulty.to_numpy()
            delta = np.abs(before-pseudo.difficulty.to_numpy())
            evidence.append(dict(condition=condition, repeat=repeat, photos=len(pseudo),
                float64_differences=int((delta != 0).sum()), max_float64_abs=float(delta.max()),
                authoritative_float32_exact=True, frozen_bins_preserved=True))
    return dict(status='PASS_EXACT_FLOAT32_SERIALIZATION_RECOVERY', maps=evidence,
                common_photos=len(common), common_groups=int(common.observation_id.nunique()),
                common_taxa=int(common.taxon_id.nunique()), no_allclose_tolerance=True)


def copy_exact(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha256_file(target) != sha256_file(source):
            raise GateStop('Existing continuation copy differs: ' + str(target))
    else:
        shutil.copyfile(source, target)
        if sha256_file(target) != sha256_file(source):
            raise GateStop('Copy did not preserve bytes')


def bootstrap_reuse(root, old_cfg, cfg, report, contract):
    if marker(report/'reuse_done.json', contract):
        verify_tree(report/'reuse_done.json')
        return
    old_report = root/'code/results/explanatory_suite_v1'
    old_data, data = Path(old_cfg['data_root']), Path(cfg['data_root'])
    artifacts = []
    def wrap(source_stamp, destination, names, source_dir, dest_dir):
        previous = marker(destination, contract)
        if previous:
            verify_tree(destination)
        else:
            copies = []
            for name in names:
                copy_exact(source_dir/name, dest_dir/name)
                copies.append(dest_dir/name)
            extra = {k: v for k, v in legacy.read(source_stamp).items() if k not in ['contract', 'artifacts']}
            finish(destination, contract, [source_stamp, *copies], **extra,
                   reused_from_v1=True, reuse_semantics='byte_identical_existing_outputs')
        artifacts.append(destination)
    wrap(old_report/'resnet_features_done.json', report/'resnet_features_done.json', [], old_report, report)
    wrap(old_report/'difficulty_done.json', report/'difficulty_done.json',
         ['photo_difficulty.csv', 'validation_difficulty.csv', 'difficulty_cutpoints.json'], old_report, report)
    for condition in ['unrestricted_common', 'difficulty_matched']:
        for repeat in range(20):
            relative = Path('maps')/condition/f'{repeat:02d}'
            wrap(old_data/relative/'done.json', data/relative/'done.json',
                 ['manifest.csv', 'mapping.csv', 'balance.csv'], old_data/relative, data/relative)
    wrap(old_report/'mappings_done.json', report/'mappings_done.json',
         ['common_manifest.csv', 'matching_attrition.csv', 'matching_attrition_by_taxon.csv'], old_report, report)
    for spec in cfg['runs']:
        relative = Path('real')/spec['run_id']
        wrap(old_report/relative/'done.json', report/relative/'done.json',
             ['groups.csv', 'nested.csv', 'summary.json'], old_report/relative, report/relative)
    finish(report/'reuse_done.json', contract, artifacts, real_checkpoints=18, maps=40,
           new_training=0, new_feature_extraction=0, old_stop_preserved=True)
