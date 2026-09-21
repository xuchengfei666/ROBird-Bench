"""Local outcome-blinded human review. Never auto-fills or impersonates a rater."""
from __future__ import annotations
import argparse
import io
import json
import mimetypes
import os
import sys
import time
import uuid
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse,parse_qs
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code/src'))
import numpy as np
import pandas as pd
from PIL import Image
from robird.io import atomic_write_json,atomic_write_csv,sha256_file
from robird.external_cohort_v1_1 import read_json,GateStop,verify_tree
from robird.rsos_suite_v1 import finish
SOURCE=ROOT/'code/results/external_cohort_v1_5'
DATA=Path('E:/Datasets/ROBird-Bench/external_completion_v1/human_quality')
ASSET=ROOT/'code/assets/quality_review_v1.html'
REPORT=ROOT/'code/results/external_completion_v1'
FIELDS=('bbox_valid','motion_blur_present','occlusion_fraction','background_complexity_1_5',
        'confidence_1_5','target_ambiguous','notes')


def prepare():
    verify_tree(SOURCE/'automatic_done.json')
    frame=pd.read_csv(SOURCE/'reused_source_relations.csv').drop_duplicates('photo_id').sort_values('photo_id')
    quality=pd.read_csv(SOURCE/'quality_proxies.csv').drop_duplicates('photo_id').set_index('photo_id')
    if len(frame)!=969:raise GateStop('Expected969 unique photo IDs')
    records=[]
    for row in frame.itertuples():
        q=quality.loc[row.photo_id]
        records.append(dict(item_id=f'item_{len(records):04d}',photo_id=int(row.photo_id),
            local_path=row.local_path,sha256=row.sha256,width=int(row.width),height=int(row.height),
            bbox=json.loads(q.bbox_json) if pd.notna(q.bbox_json) else None))
    queue=dict(schema='human_quality_v1',source_sha256=sha256_file(SOURCE/'reused_source_relations.csv'),
               items=records,orders={role:np.random.default_rng(seed).permutation(len(records)).tolist()
                 for role,seed in [('A',202609091),('B',202609092)]},
               labels_auto_generated=False)
    path=DATA/'queue_private.json'
    if path.exists():
        if read_json(path)!=queue:raise GateStop('Review queue changed')
    else:atomic_write_json(path,queue,refuse_if_exists=True)
    print(json.dumps(dict(status='HUMAN_REVIEW_PACKAGE_READY_NOT_ANNOTATED',photos=969,queue=str(path))),flush=True)


def validate_answer(value,item):
    if value.get('reviewer_type')!='human' or value.get('independent_attestation') is not True:
        raise ValueError('Requires an actual human and explicit independent-review attestation')
    name=str(value.get('rater_id','')).strip()
    if len(name)<2 or len(name)>100:raise ValueError('Enter your own reviewer name/identifier')
    if value.get('item_id')!=item['item_id'] or value.get('sha256')!=item['sha256']:
        raise ValueError('Item identity mismatch')
    if value.get('bbox_valid') not in ('yes','no','uncertain','not_detected'):raise ValueError('bbox_valid missing/invalid')
    if item['bbox'] is None and value['bbox_valid']!='not_detected':raise ValueError('No proposed box: select not_detected')
    if item['bbox'] is not None and value['bbox_valid']=='not_detected':raise ValueError('Box present: judge yes/no/uncertain')
    for field in ('motion_blur_present','target_ambiguous'):
        if value.get(field) not in ('yes','no','uncertain'):raise ValueError(field+' missing/invalid')
    occlusion=value.get('occlusion_fraction')
    if occlusion!='uncertain':
        if isinstance(occlusion,bool) or not isinstance(occlusion,(int,float)) or not np.isfinite(occlusion) or not 0<=occlusion<=1:
            raise ValueError('Occlusion must be0--1 or uncertain')
    for field in ('background_complexity_1_5','confidence_1_5'):
        if value.get(field)=='uncertain':continue
        if type(value.get(field)) is not int or not 1<=value[field]<=5:raise ValueError(field+' must be1--5 or uncertain')
    if len(str(value.get('notes','')))>4000:raise ValueError('Notes too long')
    return dict(item_id=item['item_id'],photo_id=item['photo_id'],sha256=item['sha256'],
        rater_id=name,reviewer_type='human',independent_attestation=True,
        **{field:value.get(field,'') for field in FIELDS})


class ReviewStore:
    def __init__(self,root=DATA):
        self.root=Path(root);self.queue=read_json(self.root/'queue_private.json')
        self.items={item['item_id']:item for item in self.queue['items']}
        self.queue_hash=sha256_file(self.root/'queue_private.json')

    def role(self,role):
        if role not in ('A','B'):raise ValueError('Invalid rater slot')

    def latest_path(self,role,item_id):
        self.role(role)
        if item_id not in self.items:raise ValueError('Unknown item')
        return self.root/'annotations'/role/'latest'/f'{item_id}.json'

    def save(self,role,value):
        self.role(role)
        item=self.items.get(value.get('item_id'))
        if item is None:raise ValueError('Unknown image')
        answer=validate_answer(value,item)
        answer.update(slot=role,saved_unix=time.time(),queue_sha256=self.queue_hash)
        path=self.root/'annotations'/role/'versions'/item['item_id']/f'{time.time_ns()}_{uuid.uuid4().hex}.json'
        atomic_write_json(path,answer,refuse_if_exists=True)
        atomic_write_json(self.latest_path(role,item['item_id']),answer)
        return answer

    def export(self,role):
        self.role(role);records=[]
        for item in self.items.values():
            path=self.latest_path(role,item['item_id'])
            if path.exists():records.append(read_json(path))
        return dict(schema='human_quality_v1',slot=role,queue_sha256=self.queue_hash,
                    expected_items=len(self.items),completed_items=len(records),records=records)

    def item(self,role,position):
        self.role(role);order=self.queue['orders'][role]
        if not 0<=position<len(order):raise ValueError('Position out of range')
        item=self.queue['items'][order[position]]
        public={k:item[k] for k in ('item_id','sha256','width','height','bbox')}
        path=self.latest_path(role,item['item_id'])
        public.update(position=position,total=len(order),image_url='/image/'+item['item_id'],
                      saved=read_json(path) if path.exists() else None)
        return public


def make_handler(store):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def respond(self,payload,content_type='application/json; charset=utf-8',code=200,filename=None):
            if not isinstance(payload,bytes):payload=json.dumps(payload,ensure_ascii=False).encode('utf-8')
            self.send_response(code);self.send_header('Content-Type',content_type)
            self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff')
            self.send_header('Content-Length',str(len(payload)))
            if filename:self.send_header('Content-Disposition','attachment; filename="'+filename+'"')
            self.end_headers();self.wfile.write(payload)
        def do_GET(self):
            if self.headers.get('Host')!='127.0.0.1:'+str(self.server.server_port):
                return self.respond(dict(error='Loopback host only'),code=403)
            try:
                path=urlparse(self.path);q=parse_qs(path.query)
                if path.path=='/':return self.respond(ASSET.read_bytes(),'text/html; charset=utf-8')
                if path.path=='/api/item':return self.respond(store.item(q.get('slot',['A'])[0],int(q.get('position',['0'])[0])))
                if path.path=='/api/progress':
                    slot=q.get('slot',['A'])[0];data=store.export(slot)
                    return self.respond({k:data[k] for k in ('slot','expected_items','completed_items')})
                if path.path=='/api/export':
                    slot=q.get('slot',['A'])[0]
                    return self.respond(store.export(slot),filename=f'human_quality_{slot}.json')
                if path.path.startswith('/image/'):
                    item_id=path.path.removeprefix('/image/')
                    if item_id not in store.items:raise ValueError('Unknown image')
                    item=store.items[item_id];file=Path(item['local_path'])
                    if sha256_file(file)!=item['sha256']:raise ValueError('Frozen image hash changed')
                    with Image.open(file) as image:fmt=image.format
                    mime=Image.MIME.get(fmt,'application/octet-stream')
                    return self.respond(file.read_bytes(),mime)
                return self.respond(dict(error='Not found'),code=404)
            except (ValueError,KeyError) as exc:self.respond(dict(error=str(exc)),code=400)
        def do_POST(self):
            if self.headers.get('Host')!='127.0.0.1:'+str(self.server.server_port):
                return self.respond(dict(error='Loopback host only'),code=403)
            try:
                expected='http://127.0.0.1:'+str(self.server.server_port)
                if self.headers.get('Origin')!=expected:raise ValueError('Local same-origin requests only')
                if self.headers.get('Content-Type','').split(';')[0]!='application/json':raise ValueError('JSON required')
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<=32000:raise ValueError('Invalid request length')
                path=urlparse(self.path);q=parse_qs(path.query)
                if path.path!='/api/save':return self.respond(dict(error='Not found'),code=404)
                payload=json.loads(self.rfile.read(size));store.save(q.get('slot',['A'])[0],payload)
                return self.respond(dict(status='SAVED'))
            except (ValueError,KeyError,TypeError) as exc:self.respond(dict(error=str(exc)),code=400)
    return Handler


def validate_export(payload,store,slot):
    if payload.get('schema')!='human_quality_v1' or payload.get('slot')!=slot or payload.get('queue_sha256')!=store.queue_hash:
        raise GateStop('Export schema/queue/slot mismatch')
    rows=payload.get('records',[])
    ids=[r.get('item_id') for r in rows]
    if len(ids)!=len(set(ids)) or set(ids)!=set(store.items):raise GateStop('Human annotation coverage incomplete/duplicated')
    validated=[]
    for value in rows:validated.append(validate_answer(value,store.items[value['item_id']]))
    frame=pd.DataFrame(validated)
    if frame.rater_id.nunique()!=1:raise GateStop('One slot must represent one actual rater')
    return frame


def analyze(a_path,b_path):
    from sklearn.metrics import cohen_kappa_score
    store=ReviewStore();a=validate_export(read_json(a_path),store,'A');b=validate_export(read_json(b_path),store,'B')
    if a.rater_id.iloc[0]==b.rater_id.iloc[0]:raise GateStop('Two different actual independent raters required')
    dest=REPORT/'human_quality_results'
    if (dest/'done.json').exists():raise FileExistsError('Human result already finalized')
    joined=a.merge(b,on=['photo_id','sha256'],suffixes=('_A','_B'),validate='one_to_one')
    results=[]
    for field in ('bbox_valid','motion_blur_present','background_complexity_1_5','target_ambiguous'):
        x,y=joined[field+'_A'],joined[field+'_B'];keep=~x.eq('uncertain')&~y.eq('uncertain')
        x=x[keep].astype(str);y=y[keep].astype(str)
        kappa=None
        if len(x) and len(set(x)|set(y))>1:
            kappa=float(cohen_kappa_score(x,y,weights='quadratic' if field=='background_complexity_1_5' else None))
            if not np.isfinite(kappa):kappa=None
        results.append(dict(field=field,paired_available=len(x),paired_uncertain=len(joined)-len(x),
            agreement=float(x.eq(y).mean()) if len(x) else None,kappa=kappa))
    x=pd.to_numeric(joined.occlusion_fraction_A,errors='coerce');y=pd.to_numeric(joined.occlusion_fraction_B,errors='coerce')
    valid=x.notna()&y.notna()
    results.append(dict(field='occlusion_fraction',paired_available=int(valid.sum()),
        mean_absolute_rater_difference=float((x[valid]-y[valid]).abs().mean()) if valid.any() else None))
    artifacts=[]
    atomic_write_json(dest/'rater_agreement.json',dict(results=results,independence='human_self_attestation_not_externally_verified',
        visual_ratings_not_physical_ground_truth=True));artifacts.append(dest/'rater_agreement.json')
    for role,ratings in [('A',a),('B',b)]:
        ratings=ratings.copy()
        ratings['motion_visible']=ratings.motion_blur_present.map({'yes':1.,'no':0.})
        ratings['occlusion']=pd.to_numeric(ratings.occlusion_fraction,errors='coerce')
        ratings['background']=pd.to_numeric(ratings.background_complexity_1_5,errors='coerce')
        for mode in ('full','sensitivity_no_cluster'):
            manifest=pd.read_csv(SOURCE/('reused_source_relations.csv' if mode=='full' else 'sensitivity_relations.csv'))
            rel=manifest[['photo_id','observation_id','observer_id','taxon_id']].merge(ratings,on='photo_id',validate='many_to_one')
            group=rel.groupby('observation_id')[['motion_visible','occlusion','background']].mean().reset_index()
            path=dest/f'{role}_{mode}_group_ratings.csv';atomic_write_csv(path,group);artifacts.append(path)
            source=SOURCE if mode=='full' else SOURCE/mode;rows=[]
            for run in sorted((source/'runs').iterdir()):
                edge=pd.read_csv(run/'nested.csv').merge(group,on='observation_id',validate='many_to_one')
                for k,part in edge.groupby('budget_from'):
                    for field in ('motion_visible','occlusion','background'):
                        avail=part.dropna(subset=[field]);bins=pd.qcut(avail[field],4,duplicates='drop') if avail[field].nunique()>1 else pd.Series(['single_observed_value']*len(avail),index=avail.index)
                        for level,piece in avail.groupby(bins,observed=True):
                            rows.append(dict(run_id=run.name,budget_from=int(k),field=field,stratum=str(level),
                                groups=len(piece),missing_groups=len(part)-len(avail),taxa=piece.label.nunique(),
                                macro_regression=float(piece.groupby('label').regression.mean().mean()),
                                macro_net_gain=float(piece.groupby('label').net_gain.mean().mean()),causal_claim=False))
            path=dest/f'{role}_{mode}_failure_strata.csv';atomic_write_csv(path,pd.DataFrame(rows));artifacts.append(path)
    input_record=dest/'inputs.json';atomic_write_json(input_record,dict(files={str(Path(a_path).resolve()):sha256_file(a_path),
        str(Path(b_path).resolve()):sha256_file(b_path)},queue_sha256=store.queue_hash));artifacts.append(input_record)
    contract=sha256_file(ROOT/'FROZEN_EXTERNAL_COMPLETION_V1.json')
    finish(dest/'done.json',contract,artifacts,independent_rater_exports_validated=True,human_quality_complete=True,
        uncertainty_retained=True,physical_ground_truth=False,original_g6_pass=False)
    print('HUMAN_RATING_ANALYSIS_COMPLETE',flush=True)


def serve(port):
    freeze=ROOT/'FROZEN_EXTERNAL_COMPLETION_V1.json'
    for name,expected in read_json(freeze)['files'].items():
        if sha256_file(ROOT/name)!=expected:raise GateStop('Frozen source changed: '+name)
    store=ReviewStore();server=ThreadingHTTPServer(('127.0.0.1',port),make_handler(store))
    atomic_write_json(DATA/'server_status.json',dict(pid=os.getpid(),port=port,host='127.0.0.1',started_unix=time.time()))
    print(f'Local human-review interface: http://127.0.0.1:{port}/',flush=True)
    server.serve_forever(poll_interval=.5)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--serve',action='store_true')
    ap.add_argument('--port',type=int,default=8766);ap.add_argument('--rater-a',type=Path);ap.add_argument('--rater-b',type=Path);args=ap.parse_args()
    if args.prepare:prepare()
    elif args.serve:serve(args.port)
    elif args.rater_a and args.rater_b:analyze(args.rater_a,args.rater_b)
    else:ap.error('Choose --prepare, --serve, or both --rater-a and --rater-b')
