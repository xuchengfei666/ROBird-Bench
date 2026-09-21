"""Versioned recovery of completed v1 artifacts; serial remaining evaluation only."""
import argparse
from datetime import datetime, timezone
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'code/src'))
os.environ.update(OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
                  HF_HUB_OFFLINE='1', PYTHONUNBUFFERED='1')
from robird import explanatory_suite_v1_1 as adapter
from robird.io import atomic_write_json, sha256_file
from robird.rsos_suite_v1 import finish, marker
from robird.artifact_tree_v1 import verify_tree
m = adapter.legacy
REPORT = ROOT/'code/results/explanatory_suite_v1_1'
OLD_REPORT = ROOT/'code/results/explanatory_suite_v1'
OLD_FREEZE = ROOT/'FROZEN_EXPLANATORY_SUITE_V1.json'
FREEZE = ROOT/'FROZEN_EXPLANATORY_SUITE_V1_1.json'
PYTHON = Path('C:/Users/Administrator/anaconda3/envs/pytorch1.0/python.exe')
OLD_HASH = '53061e367e466fa2bf89230a41502e1ee4e9adbf74f8b69484bfd11785670a25'


def now():
    return datetime.now(timezone.utc).isoformat()


def insert_hash(files, path, digest):
    path = str(Path(path).resolve())
    if path in files and files[path] != digest:
        raise m.GateStop('Conflicting inherited hashes: ' + path)
    files[path] = digest


def collect_tree(path, files, seen):
    path = Path(path).resolve()
    if path in seen:
        return
    seen.add(path)
    value = m.read(path)
    if not isinstance(value, dict):
        return
    for name, digest in value.get('artifacts', {}).items():
        insert_hash(files, name, digest)
        if Path(name).suffix == '.json':
            collect_tree(Path(name), files, seen)


def prepare():
    if FREEZE.exists():
        raise FileExistsError('v1.1 already frozen')
    if sha256_file(OLD_FREEZE) != OLD_HASH:
        raise m.GateStop('v1 anchor changed')
    cfg = m.read(OLD_REPORT/'resolved_config.json')
    files = dict(m.read(OLD_FREEZE)['files'])
    sources = adapter.completed_markers(ROOT, cfg)
    sources += [OLD_FREEZE, OLD_REPORT/'queue_status.json', OLD_REPORT/'host_status.json']
    seen = set()
    for path in sources:
        insert_hash(files, path, sha256_file(path))
        if path.suffix == '.json':
            collect_tree(path, files, seen)
    for name, digest in files.items():
        if sha256_file(Path(name)) != digest:
            raise m.GateStop('Input hash changed: ' + name)
    evidence = adapter.verify_precision_recovery(ROOT, cfg)
    # Original config is reused in all scientific fields; only output namespace differs.
    new = dict(cfg, version='explanatory_suite_v1_1',
               data_root='E:/Datasets/ROBird-Bench/explanatory_suite_v1_1')
    atomic_write_json(REPORT/'preflight.json', dict(evidence, at=now(), verified_inputs=len(files)), refuse_if_exists=True)
    atomic_write_json(REPORT/'resolved_config.json', new, refuse_if_exists=True)
    paths = [Path(__file__), ROOT/'code/src/robird/explanatory_suite_v1_1.py',
        ROOT/'code/tests/test_explanatory_suite_v1_1.py', ROOT/'EXPLANATORY_SUITE_V1_1_ENGINEERING_CONTINUATION.md',
        REPORT/'preflight.json', REPORT/'resolved_config.json']
    for path in paths:
        insert_hash(files, path, sha256_file(path))
    atomic_write_json(FREEZE, dict(version='explanatory_suite_v1_1', created_at=now(), files=files,
        parent=OLD_HASH, scientific_protocol_unchanged=True, original_g6_pass=False), refuse_if_exists=True)
    print(dict(status='FROZEN_V1_1', sha256=sha256_file(FREEZE), files=len(files),
               groups=evidence['common_groups'], taxa=evidence['common_taxa']), flush=True)


def lock(name):
    import msvcrt
    REPORT.mkdir(parents=True, exist_ok=True)
    f = (REPORT/name).open('a+b')
    if f.tell() == 0:
        f.write(b'0'); f.flush()
    f.seek(0); msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
    return f


def load_source():
    for name, digest in m.read(FREEZE)['files'].items():
        if sha256_file(Path(name)) != digest:
            raise m.GateStop('Frozen continuation input changed: ' + name)
    old = m.read(OLD_REPORT/'resolved_config.json')
    new = m.read(REPORT/'resolved_config.json')
    if new != dict(old, version='explanatory_suite_v1_1', data_root='E:/Datasets/ROBird-Bench/explanatory_suite_v1_1'):
        raise m.GateStop('Science config changed')
    # Original verification checks the full v2.2 completion tree, trained head provenance,
    # dataset gate and all 3090 image bytes. Its own disk source remains frozen.
    spec = importlib.util.spec_from_file_location('_verify_original_explanatory_runner',
                            ROOT/'code/scripts/run_explanatory_suite_v1.py')
    parent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parent)
    frame = parent.verify_inputs(old)
    adapter.verify_precision_recovery(ROOT, old)
    return old, new, frame


def run(smoke=False):
    guard = lock('worker.lock'); stage = 'verify_inherited_inputs'
    contract = sha256_file(FREEZE)
    def state(next_stage=None, status='RUNNING', **extra):
        nonlocal stage
        if next_stage:
            stage = next_stage
        value = dict(status=status, stage=stage, updated_at=now(), pid=os.getpid(),
                     contract=contract, original_g6_pass=False, post_hoc=True, **extra)
        atomic_write_json(REPORT/'queue_status.json', value)
        print(value, flush=True)
    try:
        state()
        old, cfg, frame = load_source()
        if marker(REPORT/'automatic_done.json', contract):
            verify_tree(REPORT/'automatic_done.json'); state('complete', status='EXPLANATORY_SUITE_COMPLETE'); return 0
        data = Path(cfg['data_root'])
        m.torch.set_num_threads(4)
        if not m.torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable')
        state('reuse_verified_v1_outputs')
        adapter.bootstrap_reuse(ROOT, old, cfg, REPORT, contract)
        common = m.pd.read_csv(REPORT/'common_manifest.csv')
        old_features = Path(old['data_root'])/'resnet50_features'
        resnet = m.aligned_features(frame, m.np.load(old_features/'features.npy', mmap_mode='r'),
                                   m.pd.read_csv(old_features/'index.csv'))
        dino = m.aligned_features(frame, m.np.load(Path(cfg['source_data'])/'features.npy', mmap_mode='r'),
                                 m.pd.read_csv(Path(cfg['source_data'])/'feature_index.csv'))
        features = dict(dinov2=dino, resnet50=resnet)
        if smoke:
            # Two actual first-repeat controls (one per condition), committed under the full
            # frozen contract and reused by the unchanged full queue. No result-based selection.
            sub = dict(cfg, runs=cfg['runs'][:1], randomizations=1)
            m.controls_stage(frame, common, features, data, REPORT, sub, contract, state)
            stamps = [REPORT/'controls'/cfg['runs'][0]['run_id']/c/'00/done.json'
                      for c in ['unrestricted_common', 'difficulty_matched']]
            if not marker(REPORT/'smoke_done.json', contract):
                finish(REPORT/'smoke_done.json', contract, stamps, actual_completed_controls=2,
                       full_protocol_unchanged=True)
            state('smoke_complete', status='PASS_TWO_REAL_CONTROLS'); return 0
        m.controls_stage(frame, common, features, data, REPORT, cfg, contract, state)
        m.resnet_e3_stage(frame, resnet, data, REPORT, cfg, contract, state)
        state('final_statistics_and_integrity')
        m.final_analysis(frame, common, data, REPORT, cfg, contract)
        verify_tree(REPORT/'analysis_done.json')
        finish(REPORT/'automatic_done.json', contract, [REPORT/'analysis_done.json', REPORT/'reuse_done.json',
               REPORT/'preflight.json'], status='EXPLANATORY_SUITE_COMPLETE', finished_at=now(),
               original_g6_pass=False, new_training=0, new_downloads=0, old_stop_preserved=True)
        state('complete', status='EXPLANATORY_SUITE_COMPLETE'); return 0
    except m.GateStop as exc:
        state(status='STOP_INPUT_OR_SCIENTIFIC_INTEGRITY', error=str(exc), traceback=traceback.format_exc()); return 2
    except Exception as exc:
        state(status='STOP_ENGINEERING_EXCEPTION', error=str(exc), traceback=traceback.format_exc()); return 1
    finally:
        guard.close()


def host():
    guard = lock('host.lock'); check = lock('worker.lock'); check.close()
    try:
        attempt = REPORT/'host'/f'{time.strftime("%Y%m%d_%H%M%S")}_{os.getpid()}'
        attempt.mkdir(parents=True, exist_ok=False)
        with (attempt/'stdout.log').open('ab', buffering=0) as out, (attempt/'stderr.log').open('ab', buffering=0) as err:
            child = subprocess.Popen([str(PYTHON), '-X', 'faulthandler', '-u', str(Path(__file__).resolve())],
                 cwd=ROOT, stdin=subprocess.DEVNULL, stdout=out, stderr=err, creationflags=subprocess.CREATE_NO_WINDOW)
            value = dict(status='CHILD_RUNNING', host_pid=os.getpid(), child_pid=child.pid,
                         attempt=str(attempt), started_at=now())
            atomic_write_json(REPORT/'host_status.json', value)
            code = child.wait()
        value.update(status='CHILD_EXITED', exit_code=code, finished_at=now())
        atomic_write_json(attempt/'exit.json', value, refuse_if_exists=True)
        atomic_write_json(REPORT/'host_status.json', value)
    except BaseException as exc:
        atomic_write_json(REPORT/'host_exception.json', dict(error=str(exc), traceback=traceback.format_exc()))
        raise
    finally:
        guard.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); args_group = parser.add_mutually_exclusive_group()
    args_group.add_argument('--prepare', action='store_true')
    args_group.add_argument('--smoke-control', action='store_true')
    args_group.add_argument('--host', action='store_true')
    args = parser.parse_args()
    if args.prepare: prepare()
    elif args.host: host()
    else: sys.exit(run(args.smoke_control))
