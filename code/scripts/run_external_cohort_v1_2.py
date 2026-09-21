"""Engineering continuation: unchanged v1.1 pipeline, corrected identity map."""
import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'code/src'))
from robird.external_cohort_v1_1 import read_json, GateStop
from robird.external_cohort_v1_2 import materialize
from robird.io import atomic_write_json, sha256_file

spec = importlib.util.spec_from_file_location('external_runner_frozen_v1_1', ROOT/'code/scripts/run_external_cohort_v1_1.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
runner.DATA = Path('E:/Datasets/ROBird-Bench/external_cohort_v1_2')
runner.REPORT = ROOT/'code/results/external_cohort_v1_2'
runner.CONFIG = ROOT/'code/configs/external_cohort_v1_2.json'
runner.FREEZE = ROOT/'FROZEN_EXTERNAL_COHORT_V1_2.json'
runner.materialize = materialize
os.environ['HF_HOME'] = str(runner.DATA/'hf_cache')
os.environ['TORCH_HOME'] = str(runner.DATA/'model_cache')


def prepare():
    if runner.CONFIG.exists() or runner.FREEZE.exists():
        raise FileExistsError('v1.2 already prepared')
    old_freeze = ROOT/'FROZEN_EXTERNAL_COHORT_V1_1.json'
    old_config = ROOT/'code/configs/external_cohort_v1_1.json'
    freeze = read_json(old_freeze)
    runner.verify_files(freeze['files'])
    config = read_json(old_config)
    config['data_root'] = str(runner.DATA)
    files = dict(freeze['files'])
    # Preserve the frozen failure evidence and every legacy input/code hash.
    for path in [old_freeze, old_config, Path(__file__),
        ROOT/'EXTERNAL_COHORT_CORRECTION_V1_2.md',
        ROOT/'code/src/robird/external_cohort_v1_2.py',
        ROOT/'code/tests/test_external_cohort_v1_2.py',
        ROOT/'code/results/external_cohort_v1_1/queue_status.json']:
        files[str(path)] = sha256_file(path)
    atomic_write_json(runner.CONFIG, config, refuse_if_exists=True)
    files[str(runner.CONFIG)] = sha256_file(runner.CONFIG)
    atomic_write_json(runner.FREEZE, dict(files=files, created_unix=time.time(),
        change='metadata identity normalization only', parent_sha256=sha256_file(old_freeze)), refuse_if_exists=True)
    print('PREPARED_V1_2', len(files), sha256_file(runner.FREEZE), flush=True)


def metadata_preflight():
    config = read_json(runner.CONFIG)
    runner.preflight(config)
    frame = materialize(ROOT, runner.DATA, runner.REPORT, config, sha256_file(runner.FREEZE))
    import pandas as pd
    old = pd.read_csv(ROOT/'code/data/manifests/external_cohort_v1.csv')
    cols = ['taxon_id','observation_id','observer_id','photo_id']
    a = frame[cols].sort_values(cols).reset_index(drop=True)
    b = old[cols].sort_values(cols).reset_index(drop=True)
    pd.testing.assert_frame_equal(a,b)
    atomic_write_json(runner.REPORT/'real_metadata_preflight.json', dict(
        status='PASS_EXACT_COHORT_IDENTITY', photos=len(frame), groups=int(frame.observation_id.nunique()),
        taxa=int(frame.taxon_id.nunique()), metadata_sha256=sha256_file(runner.REPORT/'metadata.csv'),
        all_original_members_preserved=True, checkpoint_load_count=24), refuse_if_exists=True)
    print('REAL_METADATA_PREFLIGHT_PASS', len(frame), frame.observation_id.nunique(), frame.taxon_id.nunique(), flush=True)


if __name__ == '__main__':
    if '--prepare' in sys.argv:
        prepare()
    elif '--metadata-preflight' in sys.argv:
        metadata_preflight()
    else:
        runner.main()
