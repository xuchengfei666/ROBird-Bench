"""Frozen explanatory queue: one detached worker, no training/downloads/polling agent."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
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
from robird import explanatory_suite_v1 as m
from robird.io import atomic_write_json, atomic_write_csv, sha256_file
from robird.rsos_suite_v1 import finish, marker
from robird.artifact_tree_v1 import verify_tree

REPORT = ROOT/'code/results/explanatory_suite_v1'
CONFIG = ROOT/'code/configs/explanatory_suite_v1.json'
FREEZE = ROOT/'FROZEN_EXPLANATORY_SUITE_V1.json'
PYTHON = Path('C:/Users/Administrator/anaconda3/envs/pytorch1.0/python.exe')


def now():
    return datetime.now(timezone.utc).isoformat()


def resolved():
    config = m.read(CONFIG)
    for path, expected in [(ROOT/config['parent_freeze'], config['parent_freeze_sha256']),
        (ROOT/config['source_report']/'automatic_done.json', config['parent_done_sha256'])]:
        if sha256_file(path) != expected:
            raise m.GateStop('Parent anchor changed: ' + str(path))
    parent = m.read(ROOT/config['source_report']/'resolved_config.json')
    specs = [dict(s, backbone='dinov2') for s in parent['runs'] if s['training_policy'] == 'all']
    dev_data = Path(config['development_data'])
    proxies = []
    for name, budget, dest in [('probability_mlp', 'all', specs), ('deepsets', 'all', specs),
                               ('mean_feature', 'k1', proxies)]:
        for seed in config['seeds']:
            run_id = f'resnet50-{name}-{budget}-seed{seed}'
            stamp = dev_data/'runs'/run_id/'train_done.json'
            train = m.read(stamp)
            verify_tree(stamp)
            expected = dict(backbone='resnet50', model=name, budget=1 if budget == 'k1' else None,
                            seed=seed, run_id=run_id)
            if train['spec'] != expected:
                raise m.GateStop('Checkpoint training spec mismatch')
            checkpoint = Path(train['checkpoint'])
            dest.append(dict(backbone='resnet50', model=name, seed=seed, run_id=run_id,
                training_policy=budget, checkpoint=str(checkpoint), checkpoint_sha256=sha256_file(checkpoint),
                training_marker=str(stamp)))
    expected = {(b, model, seed) for b, models in [('dinov2', config['dino_models']),
                 ('resnet50', config['resnet_models'])] for model in models for seed in config['seeds']}
    if len(specs) != 18 or {(s['backbone'], s['model'], s['seed']) for s in specs} != expected:
        raise m.GateStop('18-checkpoint coverage mismatch')
    weight = dev_data/'model_cache/checkpoints/resnet50-11ad3fa6.pth'
    config.update(runs=specs, proxy_runs=proxies, resnet_weights=str(weight),
        resnet_weights_sha256=sha256_file(weight), source_report_absolute=str(ROOT/config['source_report']))
    return config


def prepare():
    if FREEZE.exists():
        raise FileExistsError('Already frozen: do not overwrite')
    config = resolved()
    smoke = REPORT/'smoke.json'
    if not smoke.exists() or m.read(smoke)['status'] != 'PASS':
        raise m.GateStop('GPU/feature-parity smoke must pass before freezing')
    # Inherited map preserves its expected hashes, never recalculates them into a new truth.
    files = dict(m.read(ROOT/config['parent_freeze'])['files'])
    paths = [CONFIG, Path(__file__), ROOT/'EXPLANATORY_SUITE_V1_PROTOCOL.md',
        ROOT/'code/src/robird/explanatory_suite_v1.py', ROOT/'code/tests/test_explanatory_suite_v1.py',
        ROOT/config['parent_freeze'], ROOT/config['source_report']/'automatic_done.json',
        ROOT/config['source_report']/'primary_manifest.csv',
        ROOT/config['source_report']/'data_gate_done.json', ROOT/config['source_report']/'features_done.json',
        ROOT/config['development_manifest'], ROOT/config['development_splits'],
        Path(config['resnet_weights']), smoke,
        Path(config['development_data'])/'resnet50_features/done.json',
        Path(config['development_data'])/'resnet50_features/features.npy',
        Path(config['development_data'])/'resnet50_features/index.csv']
    paths.extend(ROOT.glob('code/src/robird/*.py'))
    for spec in [*config['runs'], *config['proxy_runs']]:
        paths.append(Path(spec['checkpoint']))
        if 'training_marker' in spec:
            paths.append(Path(spec['training_marker']))
    output = REPORT/'resolved_config.json'
    if output.exists():
        if m.read(output) != config:
            raise m.GateStop('Partial prepared config changed')
    else:
        atomic_write_json(output, config, refuse_if_exists=True)
    paths.append(output)
    for path in paths:
        name = str(path.resolve())
        digest = sha256_file(path)
        if name in files and files[name] != digest:
            raise m.GateStop('Inherited frozen file drift: ' + name)
        files[name] = digest
    for name, expected in files.items():
        if sha256_file(Path(name)) != expected:
            raise m.GateStop('Frozen source drift: ' + name)
    atomic_write_json(FREEZE, dict(version=config['version'], created_at=now(), files=files,
        post_hoc=True, original_g6_pass=False, training_authorized=False), refuse_if_exists=True)
    print(json.dumps(dict(status='FROZEN', sha256=sha256_file(FREEZE), files=len(files))), flush=True)


def acquire_lock(name):
    import msvcrt
    REPORT.mkdir(parents=True, exist_ok=True)
    f = (REPORT/name).open('a+b')
    if f.tell() == 0:
        f.write(b'0'); f.flush()
    f.seek(0)
    msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
    return f


def verify_inputs(config):
    for name, expected in m.read(FREEZE)['files'].items():
        if sha256_file(Path(name)) != expected:
            raise m.GateStop('Frozen input drift: ' + name)
    seen = set()
    verify_tree(ROOT/config['source_report']/'automatic_done.json', seen)
    verify_tree(Path(config['development_data'])/'resnet50_features/done.json', seen)
    for spec in config['runs']+config['proxy_runs']:
        if 'training_marker' in spec:
            verify_tree(Path(spec['training_marker']), seen)
    gate = m.read(ROOT/config['source_report']/'data_gate_done.json')
    if gate.get('status') != 'PASS_DERIVED_UNIQUE_FRAME_DATA_GATE':
        raise m.GateStop('Parent dataset gate not passed')
    frame = m.pd.read_csv(ROOT/config['source_report']/'primary_manifest.csv')
    if (len(frame), frame.observation_id.nunique(), frame.observer_id.nunique(), frame.taxon_id.nunique()) != (3090, 1121, 759, 56):
        raise m.GateStop('Unexpected cohort')
    if frame.photo_id.duplicated().any():
        raise m.GateStop('Duplicate photo identifier')
    for row in frame.itertuples():
        if sha256_file(Path(row.local_path)) != row.sha256:
            raise m.GateStop('Image bytes changed: ' + str(row.photo_id))
    return frame


def run():
    lock = acquire_lock('worker.lock')
    stage = 'verify_inputs'
    contract = sha256_file(FREEZE)
    def state(next_stage=None, status='RUNNING', **extra):
        nonlocal stage
        if next_stage:
            stage = next_stage
        value = dict(status=status, stage=stage, updated_at=now(), pid=os.getpid(), contract=contract,
                     post_hoc=True, original_g6_pass=False, **extra)
        atomic_write_json(REPORT/'queue_status.json', value)
        print(json.dumps(value), flush=True)
    try:
        state()
        config = m.read(REPORT/'resolved_config.json')
        frame = verify_inputs(config)
        if marker(REPORT/'automatic_done.json', contract):
            verify_tree(REPORT/'automatic_done.json')
            state('complete', status='EXPLANATORY_SUITE_COMPLETE'); return 0
        if not m.torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; no silent CPU fallback')
        m.torch.set_num_threads(4)
        data = Path(config['data_root'])
        data.mkdir(parents=True, exist_ok=True)
        state('resnet_external_features')
        resnet = m.extract_resnet(frame, data, REPORT, config, contract)
        dino = m.aligned_features(frame, m.np.load(Path(config['source_data'])/'features.npy', mmap_mode='r'),
                                 m.pd.read_csv(Path(config['source_data'])/'feature_index.csv'))
        state('validation_proxy_and_external_difficulty')
        difficulty = m.difficulty_stage(ROOT, frame, resnet, data, REPORT, config, contract)
        state('difficulty_matching_maps')
        common = m.mappings_stage(frame, difficulty, data, REPORT, config, contract)
        features = dict(dinov2=dino, resnet50=resnet)
        m.real_stage(frame, features, ROOT, data, REPORT, config, contract, state)
        m.controls_stage(frame, common, features, data, REPORT, config, contract, state)
        m.resnet_e3_stage(frame, resnet, data, REPORT, config, contract, state)
        state('final_statistics_and_integrity')
        m.final_analysis(frame, common, data, REPORT, config, contract)
        verify_tree(REPORT/'analysis_done.json')
        finish(REPORT/'automatic_done.json', contract, [REPORT/'analysis_done.json'],
            status='EXPLANATORY_SUITE_COMPLETE', finished_at=now(), matched_groups=int(common.observation_id.nunique()),
            matched_taxa=int(common.taxon_id.nunique()), target_checkpoints=18, new_training_runs=0,
            new_downloads=0, original_g6_pass=False, post_hoc=True)
        state('complete', status='EXPLANATORY_SUITE_COMPLETE'); return 0
    except m.GateStop as exc:
        state(status='STOP_INPUT_OR_SCIENTIFIC_INTEGRITY', error=str(exc), traceback=traceback.format_exc())
        return 2
    except Exception as exc:
        state(status='STOP_ENGINEERING_EXCEPTION', error=str(exc), traceback=traceback.format_exc())
        return 1
    finally:
        lock.close()


def host():
    lock = acquire_lock('host.lock')
    worker_lock = acquire_lock('worker.lock'); worker_lock.close()
    try:
        attempt = REPORT/'host'/f'{time.strftime("%Y%m%d_%H%M%S")}_{os.getpid()}'
        attempt.mkdir(parents=True, exist_ok=False)
        with (attempt/'stdout.log').open('ab', buffering=0) as out, (attempt/'stderr.log').open('ab', buffering=0) as err:
            child = subprocess.Popen([str(PYTHON), '-X', 'faulthandler', '-u', str(Path(__file__).resolve())],
                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                creationflags=subprocess.CREATE_NO_WINDOW)
            value = dict(status='CHILD_RUNNING', host_pid=os.getpid(), child_pid=child.pid,
                         attempt=str(attempt), started_at=now())
            atomic_write_json(REPORT/'host_status.json', value)
            code = child.wait()
        value.update(status='CHILD_EXITED', exit_code=code, finished_at=now())
        atomic_write_json(attempt/'exit.json', value, refuse_if_exists=True)
        atomic_write_json(REPORT/'host_status.json', value)
        # No blind retry loop on deterministic defects or scientific gates.
    except BaseException as exc:
        atomic_write_json(REPORT/'host_exception.json', dict(error=str(exc), traceback=traceback.format_exc(), at=now()))
        raise
    finally:
        lock.close()


def smoke():
    """Tiny transport/parity check, no full external controls or hypothesis selection."""
    import torchvision
    from torchvision.models import resnet50, ResNet50_Weights
    m.torch.set_num_threads(4)
    cfg = resolved()
    # Reproduce the original batch shape as well as weights/transform. With cuDNN
    # TF32, changing batch size can select a numerically different convolution path.
    dev = m.pd.read_csv(ROOT/cfg['development_manifest']).iloc[:32]
    weight = Path(cfg['resnet_weights'])
    model = resnet50(weights=None)
    model.load_state_dict(m.torch.load(weight, map_location='cpu', weights_only=True))
    model.fc = m.torch.nn.Identity(); model = model.eval().cuda()
    transform = ResNet50_Weights.IMAGENET1K_V2.transforms()
    images = []
    for row in dev.itertuples():
        if sha256_file(Path(row.local_path)) != row.sha256:
            raise m.GateStop('Development smoke image changed')
        with m.Image.open(row.local_path) as im:
            images.append(transform(im.convert('RGB')))
    with m.torch.no_grad():
        f = m.torch.nn.functional.normalize(model(m.torch.stack(images).cuda()), dim=1).cpu().numpy()
    cache = Path(cfg['development_data'])/'resnet50_features'
    old = m.aligned_features(dev, m.np.load(cache/'features.npy', mmap_mode='r'), m.pd.read_csv(cache/'index.csv'))
    difference = float(m.np.max(m.np.abs(old-f)))
    if difference > 2e-6:
        raise m.GateStop('ResNet transform/cache parity failure')
    del model
    # All checkpoint shapes exercised on synthetic 2-photo inputs, not used to choose design.
    tested = []
    for spec in cfg['runs']+cfg['proxy_runs']:
        dim = 384 if spec['backbone'] == 'dinov2' else 2048
        model = m.checkpoint_model(spec, dim)
        x = m.torch.zeros(2, 2, dim, device='cuda')
        with m.torch.no_grad():
            p = model(x, m.torch.ones(2, 2, dtype=m.torch.bool, device='cuda')).logits.softmax(-1)
        if p.shape != (2, 100) or not m.torch.isfinite(p).all():
            raise m.GateStop('Checkpoint shape failure')
        tested.append(spec['run_id'])
        del model
    atomic_write_json(REPORT/'smoke.json', dict(status='PASS', at=now(), checkpoints=tested,
        resnet_cache_max_abs=difference, torch=m.torch.__version__, torchvision=torchvision.__version__,
        cuda=m.torch.version.cuda, device=m.torch.cuda.get_device_name(0),
        peak_cuda_bytes=int(m.torch.cuda.max_memory_allocated()), hypothesis_result=False), refuse_if_exists=True)
    print(json.dumps(m.read(REPORT/'smoke.json')), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group()
    group.add_argument('--prepare', action='store_true')
    group.add_argument('--host', action='store_true')
    group.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    if args.prepare: prepare()
    elif args.host: host()
    elif args.smoke: smoke()
    else: sys.exit(run())
