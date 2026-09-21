"""Serial detached audited continuation. No downloader or training entry point."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code/src'))
os.environ.update(HF_HUB_CACHE='C:/Users/Administrator/.cache/huggingface/hub',
    HF_HUB_OFFLINE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
from robird import external_expansion_v2_2 as m
from robird.io import atomic_write_json,sha256_file
from robird.rsos_suite_v1 import finish,marker
from robird.artifact_tree_v1 import verify_tree

REPORT=ROOT/'code/results/external_expansion_v2_2'
CONFIG=ROOT/'code/configs/external_expansion_v2_2.json'
FREEZE=ROOT/'FROZEN_EXTERNAL_EXPANSION_V2_2.json'
PYTHON=Path('C:/Users/Administrator/anaconda3/envs/pytorch1.0/python.exe')

def resolved_config():
    authored=m.read_json(CONFIG)
    parent=m.read_json(ROOT/authored['source_report']/'resolved_config.json')
    for key in ('min_observers','sensitivity_min_observers'):
        if parent[key]!=authored[key]:raise m.GateStop('No new threshold relaxation')
    return dict(parent,**authored)

def check_matrix(config):
    expected={(model,policy,seed) for model in m.parent.legacy.MODELS
              for policy in ('all','k1') for seed in m.parent.legacy.SEEDS}
    actual={(s['model'],s['training_policy'],s['seed']) for s in config['runs']}
    if len(config['runs'])!=24 or actual!=expected:raise m.GateStop('Frozen 24-model matrix changed')

def prepare():
    if FREEZE.exists():raise FileExistsError('v2.2 already frozen')
    config=resolved_config();check_matrix(config)
    review=m.verify_parents(ROOT,config)
    files=dict(m.read_json(ROOT/config['parent_freeze'])['files'])
    output=REPORT/'resolved_config.json'
    if output.exists():
        if m.read_json(output)!=config:raise m.GateStop('Partial resolved config changed')
    else:atomic_write_json(output,config,refuse_if_exists=True)
    paths=[CONFIG,Path(__file__),ROOT/'EXTERNAL_EXPANSION_V2_2_PROTOCOL.md',
           ROOT/'code/src/robird/external_expansion_v2_2.py',
           ROOT/'code/tests/test_external_expansion_v2_2.py',output,
           ROOT/config['parent_freeze'],review/'review_done.json',review/'impact_done.json',
           ROOT/config['source_report']/'download_done.json',
           ROOT/config['source_report']/'metadata_done.json',
           Path(config['source_data'])/'reference_signatures_done.json']
    for path in paths:files[str(path.resolve())]=sha256_file(path)
    atomic_write_json(FREEZE,dict(version='external_expansion_v2_2',created_at=m.parent.utc_now(),
        files=files,parent_sha256=config['parent_sha256'],review_sha256=config['review_sha256'],
        new_model_predictions_before_freeze=0,original_g6_pass=False),refuse_if_exists=True)
    print(dict(status='FROZEN_UNIQUE_FRAME_V2_2',sha256=sha256_file(FREEZE),inputs=len(files)),flush=True)

def acquire_lock(name):
    import msvcrt
    REPORT.mkdir(parents=True,exist_ok=True)
    f=(REPORT/name).open('a+b')
    if f.tell()==0:f.write(b'0');f.flush()
    f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
    return f

def run(data_only=False):
    lock=acquire_lock('worker.lock');stage='verify_freeze'
    contract=sha256_file(FREEZE)
    def state(status='RUNNING',**kwargs):
        value=dict(status=status,stage=stage,contract=contract,pid=os.getpid(),
                   updated_at=m.parent.utc_now(),original_g6_pass=False,**kwargs)
        atomic_write_json(REPORT/'queue_status.json',value);print(value,flush=True)
    try:
        state();m.parent.verify_files(m.read_json(FREEZE)['files'])
        config=m.read_json(REPORT/'resolved_config.json');check_matrix(config)
        if config!=resolved_config():raise m.GateStop('Resolved configuration changed')
        audit_seen=set()
        m.verify_parents(ROOT,config,seen=audit_seen)
        if marker(REPORT/'automatic_done.json',contract):
            verify_tree(REPORT/'automatic_done.json');state('EXPANDED_EXTERNAL_UNIQUE_FRAME_VALIDATION_COMPLETE');return 0
        data=Path(config['data_root']);data.mkdir(parents=True,exist_ok=True)
        stage='unique_frame_data_gate';state()
        primary,sensitivity=m.materialize_gate(ROOT,REPORT,config,contract,seen=audit_seen)
        m.require_gate(ROOT,REPORT,config,contract,primary,seen=audit_seen)
        if data_only:state('DATA_GATE_PASS',primary=m.counts(primary),sensitivity=m.counts(sensitivity));return 0
        import torch
        torch.set_num_threads(4)
        stage='frozen_dinov2_features';state(photos=len(primary))
        features=m.parent.legacy.extract_dino(primary,data,REPORT,contract)
        verify_tree(REPORT/'features_done.json')
        stage='fixed_24_evaluations_then_12x20_e3';state()
        m.parent.legacy.evaluate_external(primary,features,config,ROOT,data,REPORT,contract)
        evaluations=[]
        for spec in config['runs']:
            evaluations.append(REPORT/'runs'/spec['run_id']/'eval_done.json')
            if spec['training_policy']=='all':evaluations.append(REPORT/'runs'/spec['run_id']/'e3_done.json')
        for path in evaluations:verify_tree(path)
        stage='primary_full8_strict10_statistics';state()
        m.parent.statistics(primary,data,REPORT,config,contract)
        stage='exclude_affected_prediction_reuse';state()
        m.prepare_sensitivity_predictions(primary,sensitivity,data,REPORT,config,contract)
        stage='exclude_affected_full8_strict10_statistics';state()
        m.parent.statistics(sensitivity,data/'exclude_affected',REPORT/'exclude_affected',config,contract)
        stage='joint_holm128_and_source_gap';state()
        m.joint_tests(REPORT,contract)
        m.descriptive_source_gaps(primary,sensitivity,ROOT,REPORT,config,contract)
        artifacts=evaluations+[REPORT/p for p in ('data_gate_done.json','features_done.json','statistics_done.json',
            'sensitivity_predictions_done.json','exclude_affected/statistics_done.json','joint_tests_done.json',
            'descriptive_done.json')]
        for path in artifacts:verify_tree(path)
        finish(REPORT/'automatic_done.json',contract,artifacts,
            status='EXPANDED_EXTERNAL_UNIQUE_FRAME_VALIDATION_COMPLETE',primary=m.counts(primary),
            sensitivity=m.counts(sensitivity),new_training_runs=0,new_downloads=0,original_g6_pass=False,
            human_quality_ratings='NOT_PERFORMED_EXCLUDED_BY_USER_SCOPE_CHANGE')
        stage='complete';state('EXPANDED_EXTERNAL_UNIQUE_FRAME_VALIDATION_COMPLETE');return 0
    except m.GateStop as exc:
        state('STOP_SCIENTIFIC_OR_INPUT_GATE',reason=str(exc),traceback=traceback.format_exc());return 2
    except Exception as exc:
        state('STOP_ENGINEERING_EXCEPTION',reason=str(exc),traceback=traceback.format_exc());return 1
    finally:lock.close()

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
                status=dict(status='CHILD_RUNNING',host_pid=os.getpid(),child_pid=child.pid,
                            attempt=str(attempt),restart=restart,started_at=m.parent.utc_now())
                atomic_write_json(REPORT/'host_status.json',status)
                code=child.wait()
            queue=m.read_json(REPORT/'queue_status.json') if (REPORT/'queue_status.json').exists() else {}
            status.update(status='CHILD_EXITED',exit_code=code,queue=queue,finished_at=m.parent.utc_now())
            atomic_write_json(attempt/'exit.json',status,refuse_if_exists=True)
            atomic_write_json(REPORT/'host_status.json',status)
            if code in (0,2) or queue.get('status')=='STOP_SCIENTIFIC_OR_INPUT_GATE':return
            if restart<2:time.sleep(30)
    except BaseException as exc:
        atomic_write_json(REPORT/'host_exception.json',dict(error=str(exc),traceback=traceback.format_exc(),at=m.parent.utc_now()))
        raise
    finally:lock.close()

if __name__=='__main__':
    ap=argparse.ArgumentParser();g=ap.add_mutually_exclusive_group()
    g.add_argument('--prepare',action='store_true');g.add_argument('--data-only',action='store_true')
    g.add_argument('--host',action='store_true');args=ap.parse_args()
    if args.prepare:prepare()
    elif args.host:host()
    else:sys.exit(run(args.data_only))
