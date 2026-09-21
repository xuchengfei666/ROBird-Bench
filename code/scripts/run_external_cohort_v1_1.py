"""Versioned serial external evaluation; --prepare binds inputs, default executes."""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
DATA=Path('E:/Datasets/ROBird-Bench/external_cohort_v1_1')
REPORT=ROOT/'code/results/external_cohort_v1_1'
CONFIG=ROOT/'code/configs/external_cohort_v1_1.json'
FREEZE=ROOT/'FROZEN_EXTERNAL_COHORT_V1_1.json'
os.environ['HF_HOME']=str(DATA/'hf_cache')
os.environ['TORCH_HOME']=str(DATA/'model_cache')
sys.path.insert(0,str(ROOT/'code/src'))
from robird.io import atomic_write_json, sha256_file
from robird.rsos_suite_v1 import finish, marker, make_model
from robird.external_cohort_v1_1 import (MODELS,SEEDS,SCOPE,GateStop,read_json,
    verify_tree, materialize, download, duplicate_audit,extract_dino,
    evaluate_external,quality,statistics)


def verify_files(files):
    for name,expected in files.items():
        if sha256_file(ROOT/name)!=expected:
            raise GateStop('Frozen input/code changed: '+name)


def prepare():
    if CONFIG.exists() or FREEZE.exists():
        raise FileExistsError('Prepared config/freeze already exists; do not replace')
    legacy=ROOT/'FROZEN_RSOS_SERIAL_SUITE_V1_2.json'
    inherited=read_json(legacy)['files']
    verify_files(inherited)
    old_config=read_json(ROOT/'code/configs/rsos_serial_v1_2.json')
    census=Path(old_config['data_root'])/'holdout_metadata'
    verify_tree(census/'done.json')
    materialized=read_json(ROOT/'code/results/external_cohort_v1/audit.json')
    for key,path in [('manifest_sha256','code/data/manifests/external_cohort_v1.csv'),
                     ('feasibility_sha256','code/results/external_cohort_v1/feasibility_100_taxa.csv')]:
        if sha256_file(ROOT/path)!=materialized[key]:
            raise GateStop('Prior materialization changed')
    runs=[]; extras=[]
    for model in MODELS:
        for policy in ('all','k1'):
            for seed in SEEDS:
                run_id=f'dinov2-{model}-{policy}-seed{seed}'
                train_done=Path(old_config['data_root'])/'runs'/run_id/'train_done.json'
                old=marker(train_done,sha256_file(legacy))
                if old is None:
                    raise GateStop('Missing frozen checkpoint marker: '+run_id)
                checkpoint=Path(old['checkpoint'])
                runs.append(dict(model=model,training_policy=policy,seed=seed,run_id=run_id,
                                 checkpoint=str(checkpoint),checkpoint_sha256=sha256_file(checkpoint)))
                extras += [train_done,checkpoint]
    exclusions=old_config['exclusion_manifests']
    references=[p for p in exclusions if 'p0_metadata_photos' not in p]+['code/data/manifests/external_g6x_inat2021_v1_1.csv']
    config=dict(scope=SCOPE,source='same_iNaturalist_platform',census=str(census),
                data_root=str(DATA),runs=runs,exclusion_manifests=exclusions,reference_manifests=references,
                original_g6_pass=False,expected_photos=970,expected_groups=339,expected_taxa=13)
    paths=[ROOT/'EXTERNAL_COHORT_EXECUTION_V1_1.md',Path(__file__),
        ROOT/'code/src/robird/external_cohort_v1_1.py',ROOT/'code/tests/test_external_cohort_v1_1.py',
        ROOT/'code/data/manifests/external_cohort_v1.csv',ROOT/'code/results/external_cohort_v1/audit.json',
        ROOT/'code/results/external_cohort_v1/feasibility_100_taxa.csv',census/'done.json',legacy,
        *[ROOT/p for p in references],*extras]
    for directory in (ROOT/'code/results/rsos_serial_v1/runs').iterdir():
        paths += [directory/'groups.csv',directory/'metrics.json']
    paths += list((ROOT/'code/src/robird').glob('*.py'))
    files=dict(inherited)
    for path in paths:
        absolute=path.resolve()
        files[str(absolute)]=sha256_file(absolute)
    atomic_write_json(CONFIG,config,refuse_if_exists=True)
    files[str(CONFIG)]=sha256_file(CONFIG)
    atomic_write_json(FREEZE,dict(files=files,scope=SCOPE,created_unix=time.time()),refuse_if_exists=True)
    print(json.dumps(dict(status='PREPARED',files=len(files),runs=len(runs),freeze_sha256=sha256_file(FREEZE))))


def preflight(config):
    import torch
    verify_files(read_json(FREEZE)['files'])
    for spec in config['runs']:
        checkpoint=torch.load(spec['checkpoint'],map_location='cpu',weights_only=False)
        model=make_model(spec['model'],384)
        model.load_state_dict(checkpoint['model_state'],strict=True)
        if 'spec' in checkpoint:
            if checkpoint['spec']['run_id']!=spec['run_id']:
                raise GateStop('Checkpoint run ID mismatch')
        else:
            p=checkpoint['provenance']
            if p['seed']!=spec['seed'] or p['model_config']['name']!=spec['model']:
                raise GateStop('Reused checkpoint provenance mismatch')
        del model,checkpoint
    if not torch.cuda.is_available():
        raise GateStop('CUDA unavailable')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--prepare',action='store_true')
    ap.add_argument('--preflight-only',action='store_true')
    args=ap.parse_args()
    if args.prepare:
        prepare(); return
    config=read_json(CONFIG)
    if args.preflight_only:
        preflight(config)
        print(json.dumps(dict(status='PREFLIGHT_PASS',checkpoints=24))); return
    REPORT.mkdir(parents=True,exist_ok=True); DATA.mkdir(parents=True,exist_ok=True)
    import msvcrt
    lock=(REPORT/'worker.lock').open('a+b')
    if lock.tell()==0:
        lock.write(b'0'); lock.flush()
    lock.seek(0); msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    contract=sha256_file(FREEZE)
    stage='preflight'
    def status(value='RUNNING',**extra):
        obj=dict(status=value,stage=stage,pid=os.getpid(),updated_unix=time.time(),contract=contract,**extra)
        atomic_write_json(REPORT/'queue_status.json',obj)
        print(json.dumps(obj),flush=True)
    try:
        if marker(REPORT/'automatic_done.json',contract):
            status('AUTOMATIC_COMPLETE_HUMAN_QUALITY_ANNOTATION_PENDING'); return
        if (REPORT/'duplicate_audit.json').exists() and read_json(REPORT/'duplicate_audit.json')['pairs']:
            raise GateStop('Unresolved duplicate review; separately versioned adjudication required')
        status(); preflight(config)
        stage='metadata_provenance'; status()
        frame=materialize(ROOT,DATA,REPORT,config,contract)
        stage='download'; status(total_photos=970)
        frame=download(frame,DATA,REPORT,contract)
        stage='byte_provenance_duplicate_gate'; status()
        duplicate_audit(frame,ROOT,DATA,REPORT,config,contract)
        stage='features'; status()
        features=extract_dino(frame,DATA,REPORT,contract)
        stage='A_B_C_external_evaluation_and_membership'; status(checkpoints=24)
        evaluate_external(frame,features,config,ROOT,DATA,REPORT,contract)
        stage='D_automatic_quality_proxies'; status()
        quality(frame,DATA,REPORT,contract)
        stage='E_statistical_robustness'; status()
        statistics(ROOT,REPORT,config,contract)
        artifacts=[REPORT/'metadata_done.json',REPORT/'download_done.json',REPORT/'duplicate_done.json',
                   REPORT/'features_done.json',REPORT/'quality_auto_done.json',REPORT/'statistics_done.json']
        for spec in config['runs']:
            artifacts.append(REPORT/'runs'/spec['run_id']/'eval_done.json')
            if spec['training_policy']=='all':
                artifacts.append(REPORT/'runs'/spec['run_id']/'e3_done.json')
        for path in artifacts:
            verify_tree(path)
        finish(REPORT/'automatic_done.json',contract,artifacts,scope=SCOPE,
            A='COMPLETE',B='COMPLETE',C='COMPLETE_OR_EXPLICIT_NOT_ESTIMABLE',
            D='AUTOMATIC_PROXIES_COMPLETE_HUMAN_ANNOTATIONS_PENDING',E='COMPLETE',
            new_trainings=0,original_g6_pass=False)
        stage='human_quality_annotation_boundary'
        status('AUTOMATIC_COMPLETE_HUMAN_QUALITY_ANNOTATION_PENDING')
    except GateStop as exc:
        status('STOP_DATA_OR_REVIEW_GATE',reason=str(exc),original_g6_pass=False)
    except Exception as exc:
        status('STOP_ENGINEERING_EXCEPTION',reason=str(exc),traceback=traceback.format_exc())
        raise
    finally:
        lock.close()

if __name__=='__main__':
    main()
