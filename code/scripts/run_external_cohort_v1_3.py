"""Serial full-source and exclusion-sensitivity evaluation, no photo deletion."""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
DATA=Path('E:/Datasets/ROBird-Bench/external_cohort_v1_3')
REPORT=ROOT/'code/results/external_cohort_v1_3'
CONFIG=ROOT/'code/configs/external_cohort_v1_3.json'
FREEZE=ROOT/'FROZEN_EXTERNAL_COHORT_V1_3.json'
os.environ['HF_HOME']=str(DATA/'hf_cache')
os.environ['TORCH_HOME']=str(DATA/'model_cache')
sys.path.insert(0,str(ROOT/'code/src'))
from robird.io import atomic_write_json,sha256_file
from robird.external_cohort_v1_1 import read_json,GateStop,verify_tree
from robird.rsos_suite_v1 import marker,finish
from robird.external_cohort_v1_3 import (materialize,download,duplicate_audit,extract_dino,
    evaluate_external,quality,statistics,SHARED_PHOTO,SHARED_HASH,REVIEW)


def verify_files(files):
    checked={}
    for name,expected in files.items():
        path=(ROOT/name).resolve()
        actual=checked.get(str(path))
        if actual is None:
            actual=sha256_file(path);checked[str(path)]=actual
        if actual!=expected: raise GateStop('Frozen input changed: '+name)


def prepare():
    if FREEZE.exists() or CONFIG.exists(): raise FileExistsError('v1.3 already frozen')
    parent=ROOT/'FROZEN_EXTERNAL_COHORT_V1_2.json'
    inherited=read_json(parent)['files'];verify_files(inherited)
    config=read_json(ROOT/'code/configs/external_cohort_v1_2.json')
    config.update(data_root=str(DATA),photo_relations=970,unique_photos=969,
        shared_photo_id=SHARED_PHOTO,shared_photo_sha256=SHARED_HASH,
        main_mode='full_source_relations',sensitivity_mode='exclude_shared_inference_only',
        shared_scene_review='code/results/external_shared_scene_review_v1/review.json')
    review=read_json(ROOT/config['shared_scene_review'])
    if sha256_file(REVIEW)!=review['image_sha256'] or review['image_sha256']!=SHARED_HASH:
        raise GateStop('Review image changed')
    extra=[parent,ROOT/'code/configs/external_cohort_v1_2.json',Path(__file__),
        ROOT/'EXTERNAL_COHORT_SHARED_SCENE_V1_3.md',ROOT/'code/src/robird/external_cohort_v1_3.py',
        ROOT/'code/tests/test_external_cohort_v1_3.py',ROOT/config['shared_scene_review'],REVIEW,
        ROOT/'code/results/external_cohort_v1_2/metadata_preflight_failure.json',
        ROOT/'code/results/external_cohort_v1_2/queue_status.json']
    files=dict(inherited)
    for path in extra: files[str(path.resolve())]=sha256_file(path)
    atomic_write_json(CONFIG,config,refuse_if_exists=True);files[str(CONFIG)]=sha256_file(CONFIG)
    atomic_write_json(FREEZE,dict(files=files,parent_sha256=sha256_file(parent),created_unix=time.time(),
        change='User-authorized shared-scene retention plus paired exclusion sensitivity'),refuse_if_exists=True)
    print(json.dumps(dict(status='PREPARED_V1_3',bound_paths=len(files),contract=sha256_file(FREEZE))),flush=True)


def preflight(config):
    import torch
    from robird.rsos_suite_v1 import make_model
    verify_files(read_json(FREEZE)['files'])
    for spec in config['runs']:
        if sha256_file(Path(spec['checkpoint']))!=spec['checkpoint_sha256']:
            raise GateStop('Checkpoint changed')
        model=make_model(spec['model'],384)
        model.load_state_dict(torch.load(spec['checkpoint'],map_location='cpu',weights_only=False)['model_state'],strict=True)
        del model
    if not torch.cuda.is_available(): raise GateStop('CUDA unavailable')


def metadata_check(config,contract):
    import pandas as pd
    frame=materialize(ROOT,DATA,REPORT,config,contract)
    original=pd.read_csv(ROOT/'code/data/manifests/external_cohort_v1.csv')
    keys=['taxon_id','observation_id','observer_id','photo_id']
    pd.testing.assert_frame_equal(frame[keys].sort_values(keys).reset_index(drop=True),
                                  original[keys].sort_values(keys).reset_index(drop=True))
    if frame.photo_id.nunique()!=969 or len(frame)!=970: raise GateStop('Expected 970/969')
    path=REPORT/'real_metadata_preflight.json'
    if not path.exists():
        atomic_write_json(path,dict(status='PASS_ALL_SOURCE_RELATIONS_PRESERVED',relations=970,unique_photo_ids=969,
            groups=339,taxa=13,shared_scene_rows=2,metadata_sha256=sha256_file(REPORT/'metadata.csv')),refuse_if_exists=True)
    return frame


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true')
    ap.add_argument('--metadata-preflight',action='store_true');args=ap.parse_args()
    if args.prepare: prepare();return
    config=read_json(CONFIG);contract=sha256_file(FREEZE)
    if args.metadata_preflight:
        preflight(config);metadata_check(config,contract)
        print('REAL_METADATA_PREFLIGHT_PASS_970_RELATIONS_969_PHOTOS',flush=True);return
    REPORT.mkdir(parents=True,exist_ok=True);DATA.mkdir(parents=True,exist_ok=True)
    import msvcrt
    lock=(REPORT/'worker.lock').open('a+b')
    if lock.tell()==0: lock.write(b'0');lock.flush()
    lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    stage='preflight'
    def status(state='RUNNING',**extra):
        result=dict(status=state,stage=stage,contract=contract,pid=os.getpid(),updated_unix=time.time(),**extra)
        atomic_write_json(REPORT/'queue_status.json',result);print(json.dumps(result),flush=True)
    try:
        if marker(REPORT/'automatic_done.json',contract):
            status('AUTOMATIC_COMPLETE_HUMAN_QUALITY_ANNOTATION_PENDING');return
        status();preflight(config)
        stage='metadata_provenance';status()
        frame=metadata_check(config,contract)
        stage='download_unique_images';status(total_images=969,source_relations=970)
        frame=download(frame,DATA,REPORT,contract)
        stage='duplicate_audit';status()
        duplicate_audit(frame,ROOT,DATA,REPORT,config,contract)
        stage='unique_image_features';status()
        features=extract_dino(frame,DATA,REPORT,contract)
        stage='A_B_C_full_and_sensitivity';status()
        evaluate_external(frame,features,config,ROOT,DATA,REPORT,contract)
        stage='D_quality_proxies_both_modes';status()
        quality(frame,DATA,REPORT,contract)
        stage='E_statistics_both_modes';status()
        statistics(ROOT,REPORT,config,contract)
        paths=[REPORT/'metadata_done.json',REPORT/'download_done.json',REPORT/'duplicate_done.json',
            REPORT/'features_done.json',REPORT/'sensitivity_evaluation_done.json',
            REPORT/'quality_both_modes_done.json',REPORT/'both_modes_comparison_done.json']
        for spec in config['runs']:
            paths.append(REPORT/'runs'/spec['run_id']/'eval_done.json')
            if spec['training_policy']=='all':paths.append(REPORT/'runs'/spec['run_id']/'e3_done.json')
        for path in paths:verify_tree(path)
        finish(REPORT/'automatic_done.json',contract,paths,original_g6_pass=False,original_files_deleted=0,
            full_relations=970,sensitivity_relations=968,unique_images=969,
            quality_true_motion_blur_and_occlusion_labels='PENDING_HUMAN_ANNOTATION')
        stage='human_quality_annotation_boundary'
        status('AUTOMATIC_COMPLETE_HUMAN_QUALITY_ANNOTATION_PENDING')
    except GateStop as exc:
        status('STOP_DATA_OR_REVIEW_GATE',reason=str(exc),original_g6_pass=False)
    except Exception as exc:
        status('STOP_ENGINEERING_EXCEPTION',reason=str(exc),traceback=traceback.format_exc());raise
    finally:lock.close()


if __name__=='__main__':main()
