"""Locked serial continuation, including an optional detached exit-recording host."""
from __future__ import annotations
import argparse
import os
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
DATA=Path('E:/Datasets/ROBird-Bench/external_cohort_v1_5')
REPORT=ROOT/'code/results/external_cohort_v1_5'
CONFIG=ROOT/'code/configs/external_cohort_v1_5.json'
FREEZE=ROOT/'FROZEN_EXTERNAL_COHORT_V1_5.json'
PYTHON=Path('C:/Users/Administrator/anaconda3/envs/pytorch1.0/python.exe')
# Reuse the already present encoder; offline forbids another backbone fetch.
os.environ['HF_HOME']=str(DATA/'hf_cache')
os.environ['HF_HUB_CACHE']='C:/Users/Administrator/.cache/huggingface/hub'
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TORCH_HOME']=str(DATA/'model_cache')
sys.path.insert(0,str(ROOT/'code/src'))
from robird.io import atomic_write_json,sha256_file
from robird.external_cohort_v1_1 import read_json,GateStop,verify_tree
from robird.rsos_suite_v1 import marker,finish
from robird import external_cohort_v1_5 as m


def verify_files(files):
    checked={}
    for name,expected in files.items():
        path=(ROOT/name).resolve()
        if path not in checked: checked[path]=sha256_file(path)
        if checked[path]!=expected: raise GateStop('Frozen input changed: '+str(path))


def prepare():
    if FREEZE.exists(): raise FileExistsError('v1.5 already frozen')
    parent=ROOT/'FROZEN_EXTERNAL_COHORT_V1_4.json'
    if sha256_file(parent)!=m.PARENT_HASH: raise GateStop('Parent hash mismatch')
    inherited=read_json(parent)['files'];verify_files(inherited)
    config=read_json(ROOT/'code/configs/external_cohort_v1_4.json')
    config.update(data_root=str(DATA),input_data_root=str(m.PARENT_DATA),
                  input_manifest='code/results/external_cohort_v1_4/downloaded.csv',
                  sensitivity_mode='exclude_two_complete_shared_observations_inference_only',
                  no_photo_download=True,sensitivity_groups=337,sensitivity_relations=964)
    frame,review=m.inspect_inputs(ROOT,config)
    # A interrupted prepare may reuse byte-identical authored outputs, not overwrite.
    for path,value in ((CONFIG,config),(REPORT/'review.json',review)):
        if path.exists():
            if read_json(path)!=value: raise GateStop('Partial preparation changed: '+str(path))
        else: atomic_write_json(path,value,refuse_if_exists=True)
    files=dict(inherited)
    extra=[parent,CONFIG,REPORT/'review.json',Path(__file__),m.INDEX,
           ROOT/'EXTERNAL_COHORT_REVIEWED_CONTINUATION_V1_5.md',
           ROOT/'code/src/robird/external_cohort_v1_5.py',
           ROOT/'code/configs/external_duplicate_decisions_v1_5.json',
           ROOT/'code/tests/test_external_cohort_v1_5.py',
           m.PARENT_DATA/'reference_signatures.csv',m.PARENT_DATA/'reference_signatures_done.json',
           ROOT/'code/results/external_cohort_v1_3/downloaded.csv',
           ROOT/'code/results/external_cohort_v1_3/duplicate_review.csv']
    previous=ROOT/'code/results/external_cohort_v1_4'
    extra += [previous/name for name in ('metadata_done.json','download_done.json','metadata.csv',
              'downloaded.csv','duplicate_review.csv','duplicate_audit.json','queue_status.json',
              'unique_images/download_done.json','image_relation_mapping.csv')]
    extra += sorted(m.PARENT_DATA.joinpath('download_ledger').glob('*.json'))
    extra += sorted(m.INDEX.parent.glob('pairs_*.jpg'))
    for row in review['pairs']:
        extra += [Path(row[k]) for k in ('path_a','path_b','path_a_review_path','path_b_review_path')]
    for p in extra: files[str(p.resolve())]=sha256_file(p)
    atomic_write_json(FREEZE,dict(files=files,parent_sha256=m.PARENT_HASH,created_unix=time.time(),
        change='Explicit complete adjudication; reuse verified bytes; correct whole-cluster sensitivity',
        original_g6_pass=False),refuse_if_exists=True)
    print(json.dumps(dict(status='PREPARED_V1_5',files=len(files),contract=sha256_file(FREEZE),
          decisions=review['decision_counts'],new_image_downloads=0)),flush=True)


def preflight(config):
    import torch
    from robird.rsos_suite_v1 import make_model
    verify_files(read_json(FREEZE)['files'])
    expected={(model,policy,seed) for model in m.legacy.MODELS for policy in ('all','k1') for seed in m.legacy.SEEDS}
    if len(config['runs'])!=24 or {(s['model'],s['training_policy'],s['seed']) for s in config['runs']}!=expected:
        raise GateStop('Checkpoint matrix changed')
    for spec in config['runs']:
        if sha256_file(Path(spec['checkpoint']))!=spec['checkpoint_sha256']: raise GateStop('Checkpoint changed')
        model=make_model(spec['model'],384)
        model.load_state_dict(torch.load(spec['checkpoint'],map_location='cpu',weights_only=False)['model_state'],strict=True)
        del model
    if not torch.cuda.is_available(): raise GateStop('CUDA unavailable')
    # Small CPU ops otherwise oversubscribe Windows BLAS threads on this machine.
    torch.set_num_threads(4)


def acquire_lock(name):
    import msvcrt
    REPORT.mkdir(parents=True,exist_ok=True)
    handle=(REPORT/name).open('a+b')
    if handle.tell()==0:handle.write(b'0');handle.flush()
    handle.seek(0);msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
    return handle


def run(preflight_only=False):
    lock=acquire_lock('worker.lock')
    config=read_json(CONFIG);contract=sha256_file(FREEZE);stage='freeze_and_checkpoints'
    def state(status='RUNNING',**extra):
        value=dict(status=status,stage=stage,contract=contract,pid=os.getpid(),
                   updated_unix=time.time(),original_g6_pass=False,**extra)
        atomic_write_json(REPORT/'queue_status.json',value);print(json.dumps(value),flush=True)
    try:
        state();preflight(config)
        stage='reviewed_byte_provenance_gate';state()
        frame=m.data_gate(ROOT,DATA,REPORT,config,contract)
        if preflight_only:
            state('PREFLIGHT_PASS',relations=len(frame),unresolved_pairs=0);return
        if marker(REPORT/'automatic_done.json',contract):
            verify_tree(REPORT/'automatic_done.json')
            stage='human_quality_annotation_boundary'
            state('AUTOMATIC_COMPLETE_HUMAN_QUALITY_ANNOTATION_PENDING');return
        stage='unique_image_features';state()
        features=m.relations.extract_dino(frame,DATA,REPORT,contract)
        stage='A_B_C_full_then_cluster_exclusion';state(feature_shape=list(features.shape))
        m.evaluate_both(frame,features,config,ROOT,DATA,REPORT,contract)
        stage='shared_input_diagnostic';state()
        m.shared_diagnostic(frame,DATA,REPORT,config,contract)
        stage='D_quality_proxies';state()
        m.quality_both(frame,DATA,REPORT,contract)
        stage='E_statistics';state()
        m.statistics_both(ROOT,REPORT,config,contract)
        artifacts=[REPORT/p for p in ('duplicate_done.json','features_done.json','evaluation_both_done.json',
                   'shared_inputs_done.json','quality_both_done.json','statistics_done.json',
                   'sensitivity_no_cluster/statistics_done.json','comparison_done.json')]
        for path in artifacts: verify_tree(path)
        finish(REPORT/'automatic_done.json',contract,artifacts,new_image_downloads=0,
            original_files_deleted=0,full_relations=970,sensitivity_relations=964,
            full_groups=339,sensitivity_groups=337,original_g6_pass=False,
            human_quality_labels_pending=True)
        stage='human_quality_annotation_boundary'
        state('AUTOMATIC_COMPLETE_HUMAN_QUALITY_ANNOTATION_PENDING')
    except GateStop as exc:
        state('STOP_DATA_OR_REVIEW_GATE',reason=str(exc));raise
    except Exception as exc:
        state('STOP_ENGINEERING_EXCEPTION',reason=str(exc),traceback=traceback.format_exc());raise
    finally:lock.close()


def host():
    lock=acquire_lock('host.lock')
    check=acquire_lock('worker.lock');check.close()
    attempt=REPORT/'host'/f'{time.strftime("%Y%m%d_%H%M%S")}_{os.getpid()}'
    attempt.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ);env.update(PYTHONPATH=str(ROOT/'code/src'),PYTHONUNBUFFERED='1',PYTHONFAULTHANDLER='1')
    child=None
    def state(status,**extra):
        atomic_write_json(REPORT/'host_status.json',dict(status=status,host_pid=os.getpid(),
            child_pid=child.pid if child else None,attempt=str(attempt),updated_unix=time.time(),**extra))
    try:
        with (attempt/'worker.stdout.log').open('ab',buffering=0) as out,(attempt/'worker.stderr.log').open('ab',buffering=0) as err:
            child=subprocess.Popen([str(PYTHON),'-X','faulthandler','-u',str(Path(__file__).resolve())],
                cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
            state('CHILD_RUNNING')
            while True:
                try: code=child.wait(timeout=15);break
                except subprocess.TimeoutExpired:
                    queue=read_json(REPORT/'queue_status.json') if (REPORT/'queue_status.json').exists() else {}
                    state('CHILD_RUNNING',queue=queue,
                          feature_shards=len(list((DATA/'unique_features/feature_shards').glob('*.npy'))),
                          full_evaluations=len(list((REPORT/'runs').glob('*/eval_done.json'))),
                          sensitivity_evaluations=len(list((REPORT/'sensitivity_no_cluster/runs').glob('*/eval_done.json'))))
            queue=read_json(REPORT/'queue_status.json') if (REPORT/'queue_status.json').exists() else {}
            success=code==0 and queue.get('status')=='AUTOMATIC_COMPLETE_HUMAN_QUALITY_ANNOTATION_PENDING'
            result=dict(exit_code=code,exit_code_hex=f'0x{code & 0xffffffff:08X}',decision=queue,
                        status='AUTOMATIC_DONE_HUMAN_ANNOTATIONS_PENDING' if success else 'CHILD_STOP_OR_UNEXPECTED_EXIT')
            atomic_write_json(attempt/'exit.json',result,refuse_if_exists=True);state(**result)
    except BaseException as exc:
        state('HOST_EXCEPTION',error=str(exc),traceback=traceback.format_exc());raise
    finally:lock.close()


if __name__=='__main__':
    ap=argparse.ArgumentParser();group=ap.add_mutually_exclusive_group()
    group.add_argument('--prepare',action='store_true');group.add_argument('--preflight-only',action='store_true')
    group.add_argument('--host',action='store_true');args=ap.parse_args()
    if args.prepare:prepare()
    elif args.host:host()
    else:run(args.preflight_only)
