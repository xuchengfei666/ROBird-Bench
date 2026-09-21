"""Detached, bounded serial expansion. No AI watcher and no model fitting."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT/'code/results/external_expansion_v2'
CONFIG = ROOT/'code/configs/external_expansion_v2.json'
FREEZE = ROOT/'FROZEN_EXTERNAL_EXPANSION_V2.json'
PYTHON = Path('C:/Users/Administrator/anaconda3/envs/pytorch1.0/python.exe')
sys.path.insert(0,str(ROOT/'code/src'))
os.environ['HF_HUB_CACHE']='C:/Users/Administrator/.cache/huggingface/hub'
os.environ['HF_HUB_OFFLINE']='1'
from robird import external_expansion_v2 as m
from robird.io import atomic_write_json,sha256_file
from robird.rsos_suite_v1 import finish,marker


def resolved_config():
    config = m.read_json(CONFIG)
    parent = m.read_json(ROOT/config['parent_config'])
    config['runs'] = parent['runs']
    extra = config['old_external_manifest']
    config['exclusion_manifests'] = list(dict.fromkeys(parent['exclusion_manifests']+[extra]))
    config['reference_manifests'] = list(dict.fromkeys(parent['reference_manifests']+[extra]))
    return config


def prepare():
    if FREEZE.exists():
        raise FileExistsError('v2 already frozen; do not overwrite')
    config = resolved_config()
    cache = Path(config['cached_census'])
    m.verify_tree(cache/'done.json')
    m.verify_tree(ROOT/'code/results/external_cohort_v1_5/automatic_done.json')
    old = m.read_json(cache/'done.json')
    if old['taxa']!=100 or old['individually_feasible']!=13:
        raise m.GateStop('Legacy external census changed')
    paths = [CONFIG,Path(__file__),ROOT/config['parent_config'],ROOT/config['taxon_manifest'],
             ROOT/'EXTERNAL_EXPANSION_V2_PROTOCOL.md']
    paths += list((ROOT/'code/src/robird').glob('*.py'))
    paths += list((ROOT/'code/src/robird/models').glob('*.py'))
    paths += list((ROOT/'code/tests').glob('test_external_expansion_v2.py'))
    paths += [ROOT/p for p in set(config['exclusion_manifests']+config['reference_manifests'])]
    paths += list(cache.glob('*.json'))
    paths += [p for p in ROOT.glob('FROZEN_*') if p != FREEZE]
    for spec in config['runs']:
        path=Path(spec['checkpoint'])
        if sha256_file(path)!=spec['checkpoint_sha256']:
            raise m.GateStop('Original checkpoint changed')
        paths.append(path)
    # Existing development metrics, for future read-only matched-label comparisons.
    for spec in config['runs']:
        directory=ROOT/'code/results/rsos_serial_v1/runs'/spec['run_id']
        paths += [directory/n for n in ('groups.csv','metrics.json')]
    files={str(p.resolve()):sha256_file(p) for p in dict.fromkeys(paths)}
    output=REPORT/'resolved_config.json'
    if output.exists():
        if m.read_json(output)!=config:
            raise m.GateStop('Partial prepare config changed')
    else:
        atomic_write_json(output,config,refuse_if_exists=True)
    files[str(output.resolve())]=sha256_file(output)
    atomic_write_json(FREEZE,dict(version='external_expansion_v2',created_at=m.utc_now(),files=files,
        original_g6_pass=False,new_model_predictions_before_freeze=0),refuse_if_exists=True)
    print(json.dumps(dict(status='FROZEN_V2',contract=sha256_file(FREEZE),files=len(files))),flush=True)


def acquire_lock(name):
    import msvcrt
    REPORT.mkdir(parents=True,exist_ok=True)
    handle=(REPORT/name).open('a+b')
    if handle.tell()==0:
        handle.write(b'0');handle.flush()
    handle.seek(0);msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
    return handle


def run(preflight_only=False):
    lock=acquire_lock('worker.lock')
    stage='verify_freeze'
    contract=sha256_file(FREEZE)
    def state(status='RUNNING',**extra):
        value=dict(status=status,stage=stage,contract=contract,pid=os.getpid(),
                   updated_at=m.utc_now(),original_g6_pass=False,**extra)
        atomic_write_json(REPORT/'queue_status.json',value)
        print(json.dumps(value),flush=True)
    try:
        state()
        m.verify_files(m.read_json(FREEZE)['files'])
        config=m.read_json(REPORT/'resolved_config.json')
        data=Path(config['data_root']);data.mkdir(parents=True,exist_ok=True)
        if config!=resolved_config():
            raise m.GateStop('Resolved config changed')
        expected={(model,policy,seed) for model in m.legacy.MODELS for policy in ('all','k1') for seed in m.legacy.SEEDS}
        if len(config['runs'])!=24 or {(s['model'],s['training_policy'],s['seed']) for s in config['runs']}!=expected:
            raise m.GateStop('Fixed model matrix changed')
        if preflight_only:
            state('PREFLIGHT_PASS');return 0
        if marker(REPORT/'automatic_done.json',contract):
            m.verify_tree(REPORT/'automatic_done.json')
            state('EXPANDED_EXTERNAL_VALIDATION_COMPLETE');return 0
        stage='metadata_census';state()
        summary=m.census(ROOT,data,REPORT,config,contract,state)
        stage='freeze_metadata_selection';state(eligible8=summary['eligible8'],eligible10=summary['eligible10'])
        frame=m.select_metadata(ROOT,REPORT,config,contract,summary)
        stage='download';state(photos=len(frame),groups=int(frame.observation_id.nunique()),taxa=int(frame.taxon_id.nunique()))
        frame=m.legacy.download(frame,data,REPORT,contract)
        stage='duplicate_audit';state()
        m.legacy.duplicate_audit(frame,ROOT,data,REPORT,config,contract)
        m.require_data_gates(frame,ROOT,data,REPORT,config,contract)
        import torch
        torch.set_num_threads(4)
        stage='frozen_dinov2_features';state()
        features=m.legacy.extract_dino(frame,data,REPORT,contract)
        stage='fixed_24_evaluations_then_e3';state()
        m.legacy.evaluate_external(frame,features,config,ROOT,data,REPORT,contract)
        evaluations=[]
        for spec in config['runs']:
            evaluations.append(REPORT/'runs'/spec['run_id']/'eval_done.json')
            if spec['training_policy']=='all':
                evaluations.append(REPORT/'runs'/spec['run_id']/'e3_done.json')
        stage='full8_and_strict10_statistics';state()
        m.statistics(frame,data,REPORT,config,contract)
        artifacts=[REPORT/p for p in ('metadata_done.json','download_done.json','duplicate_done.json',
                   'features_done.json','statistics_done.json')]+evaluations
        for path in artifacts:
            m.verify_tree(path)
        finish(REPORT/'automatic_done.json',contract,artifacts,
               status='EXPANDED_EXTERNAL_VALIDATION_COMPLETE',original_g6_pass=False,
               taxa=int(frame.taxon_id.nunique()),strict10_taxa=summary['eligible10'],
               groups=int(frame.observation_id.nunique()),photos=len(frame),new_training_runs=0)
        stage='complete';state('EXPANDED_EXTERNAL_VALIDATION_COMPLETE');return 0
    except m.GateStop as exc:
        state('STOP_SCIENTIFIC_OR_INPUT_GATE',reason=str(exc),traceback=traceback.format_exc());return 2
    except Exception as exc:
        state('STOP_ENGINEERING_EXCEPTION',reason=str(exc),traceback=traceback.format_exc());return 1
    finally:
        lock.close()


def host():
    lock=acquire_lock('host.lock')
    check=acquire_lock('worker.lock');check.close()
    try:
        for restart in range(3):
            attempt=REPORT/'host'/f'{time.strftime("%Y%m%d_%H%M%S")}_{os.getpid()}_{restart}'
            attempt.mkdir(parents=True,exist_ok=False)
            env=dict(os.environ);env.update(PYTHONUNBUFFERED='1',PYTHONFAULTHANDLER='1')
            with (attempt/'stdout.log').open('ab',buffering=0) as out,(attempt/'stderr.log').open('ab',buffering=0) as err:
                child=subprocess.Popen([str(PYTHON),'-X','faulthandler','-u',str(Path(__file__).resolve())],
                    cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,
                    creationflags=subprocess.CREATE_NO_WINDOW)
                state=dict(status='CHILD_RUNNING',host_pid=os.getpid(),child_pid=child.pid,
                           attempt=str(attempt),restart=restart,started_at=m.utc_now())
                atomic_write_json(REPORT/'host_status.json',state)
                code=child.wait()  # OS wait; no model/AI polling, no repeated log narration.
            queue=m.read_json(REPORT/'queue_status.json') if (REPORT/'queue_status.json').exists() else {}
            state.update(status='CHILD_EXITED',exit_code=code,queue=queue,finished_at=m.utc_now())
            atomic_write_json(attempt/'exit.json',state,refuse_if_exists=True)
            atomic_write_json(REPORT/'host_status.json',state)
            # Integrity/data/duplicate STOP is terminal; never retry as though it were a network error.
            if code==0 or code==2 or queue.get('status')=='STOP_SCIENTIFIC_OR_INPUT_GATE':
                return
            if restart<2:
                time.sleep(60)
    except BaseException as exc:
        atomic_write_json(REPORT/'host_exception.json',dict(error=str(exc),traceback=traceback.format_exc(),at=m.utc_now()))
        raise
    finally:
        lock.close()


if __name__=='__main__':
    ap=argparse.ArgumentParser();group=ap.add_mutually_exclusive_group()
    group.add_argument('--prepare',action='store_true');group.add_argument('--preflight-only',action='store_true')
    group.add_argument('--host',action='store_true');args=ap.parse_args()
    if args.prepare:
        prepare()
    elif args.host:
        host()
    else:
        sys.exit(run(args.preflight_only))
