"""Read-only legacy inputs, isolated external cohort and serial audit/analysis stages."""
from __future__ import annotations
import hashlib
import io
import itertools
import json
import math
import os
import time
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

import numpy as np
import pandas as pd
from PIL import Image

from robird.io import atomic_write_csv, atomic_write_json, sha256_file
from robird.rsos_suite_v1 import atomic_binary, finish, marker

SEEDS = (20260819, 20260820, 20260821)
MODELS = ('mean_feature', 'probability_mlp', 'deepsets', 'set_transformer')
SCOPE = 'SAME_PLATFORM_OUT_OF_DEVELOPMENT_METADATA_FEASIBLE_COHORT'


class GateStop(RuntimeError):
    pass


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def verify_tree(path, seen=None):
    seen = set() if seen is None else seen
    path = Path(path).resolve()
    if path in seen:
        return
    seen.add(path)
    value = read_json(path)
    for artifact, expected in value.get('artifacts', {}).items():
        p = Path(artifact)
        if sha256_file(p) != expected:
            raise GateStop(f'Input artifact changed: {p}')
        if p.suffix == '.json':
            verify_tree(p, seen)


def large_url(url):
    value = urlsplit(str(url))
    if value.scheme != 'https' or value.hostname not in (
        'inaturalist-open-data.s3.amazonaws.com', 'static.inaturalist.org'):
        raise GateStop('Unapproved image URL host or scheme')
    head, leaf = value.path.rsplit('/', 1)
    stem, dot, extension = leaf.partition('.')
    if stem not in ('square', 'small', 'medium', 'large', 'original') or not dot:
        raise GateStop('Unrecognized image size URL')
    if extension.lower() not in ('', 'jpg', 'jpeg', 'png', 'gif', 'webp'):
        raise GateStop('Unsupported image URL extension')
    return urlunsplit((value.scheme, value.netloc, head+'/large.'+extension, '', ''))


def image_signature(path):
    with Image.open(path) as img:
        img.load()
        width, height = img.size
        if width < 1 or height < 1:
            raise GateStop('Empty image')
        gray = np.asarray(img.convert('L').resize((9, 8)), dtype=np.int16)
    bits = (gray[:, :-1] > gray[:, 1:]).ravel()
    value = sum(int(b) << (63-i) for i, b in enumerate(bits))
    return dict(sha256=sha256_file(path), width=width, height=height, dhash=f'{value:016x}')


def assert_metadata(frame, history):
    if frame.photo_id.duplicated().any():
        raise GateStop('New cohort repeats a photo ID')
    for obs, part in frame.groupby('observation_id'):
        if not 2 <= len(part) <= 5 or any(part[c].nunique() != 1 for c in
                                        ('observer_id', 'taxon_id', 'class_index')):
            raise GateStop(f'Observation identity/cardinality: {obs}')
    for df in history:
        for col in ('photo_id', 'observation_id', 'observer_id'):
            if col in df and set(frame[col].astype(int)) & set(df[col].dropna().astype(int)):
                raise GateStop('History identity overlap: '+col)


def materialize(root, data, report, config, contract):
    done = report/'metadata_done.json'
    path = report/'metadata.csv'
    if marker(done, contract):
        return pd.read_csv(path, keep_default_na=False)
    meta = Path(config['census'])
    verify_tree(meta/'done.json')
    census = read_json(meta/'done.json')
    if census['taxa'] != 100 or census['individually_feasible'] != 13:
        raise GateStop('Census coverage changed')
    old = pd.read_csv(root/'code/data/manifests/external_cohort_v1.csv')
    dev = pd.read_csv(root/'code/data/manifests/development_photos_v5_3.csv')
    names = dev[['taxon_id', 'class_index', 'scientific_name']].drop_duplicates().set_index('taxon_id')
    identities = set(zip(old.taxon_id, old.observation_id, old.photo_id))
    records, feasibility = [], []
    for p in sorted(meta.glob('taxon_*.json')):
        result = read_json(p)['result']
        taxon = int(result['taxon_id'])
        feasible = result['independent_observers'] >= 10
        feasibility.append(dict(taxon_id=taxon, scientific_name=names.loc[taxon, 'scientific_name'],
                                independent_observers=result['independent_observers'],
                                candidate_groups=result['candidate_groups'],
                                selected_by_metadata=feasible, shortfall=result['shortfall']))
        if not feasible:
            continue
        for group in result['candidates']:
            for photo in group['photos']:
                if (taxon, group['observation_id'], photo['id']) not in identities:
                    raise GateStop('Source materialization membership mismatch')
                if photo.get('license_code') not in ('cc0', 'cc-by', 'cc-by-sa') or not photo.get('attribution'):
                    raise GateStop('Invalid license/attribution')
                records.append(dict(taxon_id=taxon, class_index=int(names.loc[taxon, 'class_index']),
                    scientific_name=names.loc[taxon, 'scientific_name'],
                    observation_id=group['observation_id'], observer_id=group['observer_id'],
                    photo_id=photo['id'], license_code=photo['license_code'], attribution=photo['attribution'],
                    observed_on=group.get('observed_on'), created_at=group.get('created_at'),
                    original_url=photo['url'], url=large_url(photo['url']),
                    observation_url=f"https://www.inaturalist.org/observations/{group['observation_id']}",
                    cohort=SCOPE, selection_basis='metadata_feasibility_after_census',
                    split='external_feasibility_test'))
    frame = pd.DataFrame(records).sort_values(['taxon_id','observation_id','photo_id']).reset_index(drop=True)
    if len(frame) != 970 or frame.observation_id.nunique() != 339 or frame.taxon_id.nunique() != 13:
        raise GateStop('Expected all 339 groups/970 photos/13 taxa')
    history = [pd.read_csv(root/p) for p in config['exclusion_manifests']]
    assert_metadata(frame, history)
    atomic_write_csv(path, frame)
    atomic_write_csv(report/'feasibility_100_taxa.csv', pd.DataFrame(feasibility))
    atomic_write_csv(report/'cardinality_coverage.csv', frame.groupby(['taxon_id','observation_id']).size()
                     .rename('photos').reset_index().groupby(['taxon_id','photos']).size().rename('groups').reset_index())
    finish(done, contract, [path, report/'feasibility_100_taxa.csv', report/'cardinality_coverage.csv'],
           scope=SCOPE, original_g6_pass=False, photos=970, groups=339, taxa=13)
    return frame


def download(frame, data, report, contract):
    path, done = report/'downloaded.csv', report/'download_done.json'
    if marker(done, contract):
        return pd.read_csv(path, keep_default_na=False, dtype={'dhash':str})
    rows = []
    for i, row in enumerate(frame.to_dict('records')):
        image_path = data/'images'/str(row['taxon_id'])/f"{row['photo_id']}.image"
        ledger = data/'download_ledger'/f"{row['photo_id']}.json"
        saved = marker(ledger, contract)
        if saved:
            if saved['url'] != row['url']:
                raise GateStop('Download URL changed')
            details = saved['details']
        else:
            # An incomplete file from a killed download has no committed ledger.
            for attempt in range(3):
                try:
                    req = Request(row['url'], headers={'User-Agent':'ROBird-research-image-audit/1.1'})
                    with urlopen(req, timeout=45) as response:
                        content = response.read(32*1024*1024+1)
                        if len(content) > 32*1024*1024:
                            raise GateStop('Unexpected image larger than 32MB')
                        if urlsplit(response.url).hostname not in ('inaturalist-open-data.s3.amazonaws.com', 'static.inaturalist.org'):
                            raise GateStop('Image redirect outside approved source')
                    with Image.open(io.BytesIO(content)) as img:
                        img.verify()
                    atomic_binary(image_path, lambda f: f.write(content))
                    details = image_signature(image_path)
                    details.update(local_path=str(image_path), downloaded_unix=time.time())
                    finish(ledger, contract, [image_path], url=row['url'], details=details)
                    break
                except (HTTPError, URLError, TimeoutError, ConnectionError, OSError) as exc:
                    transient = not isinstance(exc, HTTPError) or exc.code == 429 or exc.code >= 500
                    if not transient or attempt == 2:
                        atomic_write_json(report/'download_failure.json', dict(photo_id=row['photo_id'],
                                          url=row['url'], error=str(exc), attempt=attempt+1, images_committed=len(rows)))
                        raise
                    time.sleep((2, 5)[attempt])
            time.sleep(.1)
        rows.append(dict(**row, **details))
        if (i+1) % 25 == 0:
            atomic_write_json(report/'download_progress.json', dict(completed=i+1,total=len(frame)))
    frame = pd.DataFrame(rows)
    atomic_write_csv(path, frame)
    finish(done, contract, [path], photos=len(frame), image_ledgers=str(data/'download_ledger'))
    return frame


def resolve_history_path(path, root):
    p = Path(path)
    if p.exists():
        return p
    old = 'C:/Users/Administrator/Desktop/Optics-Aware-Bird-Recognition-Research/'
    normalized = str(p).replace('\\', '/')
    if normalized.startswith(old):
        alternative = root.parent/normalized[len(old):]
        if alternative.exists():
            return alternative
    raise GateStop('Missing historical image needed for duplicate audit: '+str(p))


def duplicate_audit(frame, root, data, report, config, contract):
    done = report/'duplicate_done.json'
    if marker(done, contract):
        return
    audit = report/'duplicate_audit.json'
    if audit.exists():
        raise GateStop('Duplicate review already requested; requires separately versioned adjudication')
    ref_path, ref_done = data/'reference_signatures.csv', data/'reference_signatures_done.json'
    if not marker(ref_done, contract):
        refs = []
        seen = set()
        for relative in config['reference_manifests']:
            df = pd.read_csv(root/relative, keep_default_na=False)
            if 'local_path' not in df and 'image_path' in df:
                df = df.rename(columns={'image_path':'local_path'})
            if 'local_path' not in df:
                continue
            for row in df.to_dict('records'):
                if not row.get('local_path'):
                    continue
                p = resolve_history_path(row['local_path'], root)
                if str(p) in seen:
                    continue
                seen.add(str(p))
                sig = image_signature(p)
                if row.get('sha256') and sig['sha256'] != row['sha256']:
                    raise GateStop('Historical image hash changed: '+str(p))
                refs.append(dict(local_path=str(p), photo_id=row.get('photo_id', ''),
                                 observation_id=row.get('observation_id', ''), **sig))
        if not refs:
            raise GateStop('Empty duplicate reference set')
        atomic_write_csv(ref_path, pd.DataFrame(refs))
        finish(ref_done, contract, [ref_path], count=len(refs))
    refs = pd.read_csv(ref_path, keep_default_na=False, dtype={'dhash':str}).to_dict('records')
    pairs = []
    def compare(a, b, source):
        distance = (int(a['dhash'],16)^int(b['dhash'],16)).bit_count()
        same = a['sha256'] == b['sha256']
        if same or distance <= 4:
            pairs.append(dict(photo_id=a['photo_id'], observation_id=a['observation_id'],
                path_a=a['local_path'], path_b=b['local_path'], reference_photo_id=b['photo_id'],
                reference_observation_id=b.get('observation_id',''), source=source,
                exact_bytes=same, dhash_distance=distance, decision='PENDING', rationale=''))
    new = frame.to_dict('records')
    for i, a in enumerate(new):
        for b in refs:
            compare(a,b,'history')
        for b in new[:i]:
            compare(a,b,'within_new_cohort')
    columns = ['photo_id','observation_id','path_a','path_b','reference_photo_id',
               'reference_observation_id','source','exact_bytes','dhash_distance','decision','rationale']
    atomic_write_csv(report/'duplicate_review.csv', pd.DataFrame(pairs, columns=columns))
    result = dict(scope=SCOPE, references=len(refs), candidate_photos=len(frame),
                  pairs=len(pairs), exact_pairs=sum(p['exact_bytes'] for p in pairs),
                  status='PENDING_DUPLICATE_ADJUDICATION' if pairs else 'PASS_DUPLICATE_SCREEN',
                  all_duplicates_proven_absent=False, original_g6_pass=False)
    atomic_write_json(audit, result)
    if pairs:
        raise GateStop(f'{len(pairs)} duplicate candidates require adjudication before features')
    finish(done, contract, [audit,report/'duplicate_review.csv',ref_path], **result)


def extract_dino(frame, data, report, contract):
    path, done = data/'features.npy', report/'features_done.json'
    if marker(done, contract):
        return np.load(path)
    import torch
    import timm
    from torchvision.transforms import v2, InterpolationMode
    if not torch.cuda.is_available():
        raise GateStop('CUDA required')
    model = timm.create_model('vit_small_patch14_dinov2.lvd142m', pretrained=True, num_classes=0, img_size=224)
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        v = tensor.detach().cpu().contiguous()
        digest.update(name.encode()); digest.update(str(v.dtype).encode('ascii'))
        digest.update(np.asarray(v.shape,dtype=np.int64).tobytes()); digest.update(v.numpy().tobytes())
    state = digest.hexdigest()
    if state != '90383f444f1260497bbd963fdf6d1277cbc3e6597c5d33475d060c3bc65e37d5':
        raise GateStop('DINO state differs from original encoder')
    model = model.eval().cuda()
    transform = v2.Compose([v2.Resize(256,interpolation=InterpolationMode.BICUBIC,antialias=True),
        v2.CenterCrop(224),v2.ToImage(),v2.ToDtype(torch.float32,scale=True),
        v2.Normalize(mean=[.485,.456,.406],std=[.229,.224,.225])])
    parts=[]
    for start in range(0,len(frame),64):
        part=frame.iloc[start:start+64]
        shard=data/'feature_shards'/f'{start:05d}.npy'
        mark=shard.with_suffix('.json')
        if not marker(mark,contract):
            images=[]
            for row in part.itertuples():
                if sha256_file(Path(row.local_path)) != row.sha256:
                    raise GateStop('New image changed after data gate')
                with Image.open(row.local_path) as image:
                    images.append(transform(image.convert('RGB')))
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
                x=model(torch.stack(images).cuda()).float()
            x=torch.nn.functional.normalize(x,dim=1).cpu().numpy()
            if x.shape != (len(part),384) or not np.isfinite(x).all():
                raise GateStop('Invalid DINO features')
            atomic_binary(shard,lambda f:np.save(f,x))
            finish(mark,contract,[shard],photo_ids=part.photo_id.tolist())
        elif read_json(mark)['photo_ids'] != part.photo_id.tolist():
            raise GateStop('Feature row order changed')
        parts.append(np.load(shard))
    features=np.concatenate(parts)
    atomic_binary(path,lambda f:np.save(f,features))
    atomic_write_csv(data/'feature_index.csv',frame[['photo_id','observation_id','class_index']].reset_index(names='feature_row'))
    finish(done,contract,[path,data/'feature_index.csv'],state_sha256=state,shape=list(features.shape))
    del model
    torch.cuda.empty_cache()
    return features


def paired_budgets(budgets, before, after):
    a = budgets[budgets.budget == before]
    b = budgets[budgets.budget == after][['observation_id','expected_accuracy']]
    paired = a.merge(b,on='observation_id',suffixes=('_before','_after'),validate='one_to_one')
    paired['delta'] = paired.expected_accuracy_after-paired.expected_accuracy_before
    return paired


def evaluate_external(frame, features, config, root, data, report, contract):
    import torch
    from robird.rsos_suite_v1 import make_model, prediction_frame, e3_run
    from robird.budget_metrics_v2_1 import validate_records, summarize_run, cluster_interval
    # The legacy loader uses this split token only as a row filter; saved metadata
    # and scope retain external_feasibility_test, never include training rows.
    inference = frame.assign(split='development_test')
    for spec in config['runs']:
        run_id=spec['run_id']; out=report/'runs'/run_id; raw=data/'runs'/run_id
        checkpoint=Path(spec['checkpoint'])
        if not marker(out/'eval_done.json',contract):
            if not marker(raw/'predictions_done.json',contract):
                model=make_model(spec['model'],384).eval().cuda()
                model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['model_state'],strict=True)
                predictions=prediction_frame(model,inference,features)
                atomic_write_csv(raw/'predictions.csv',predictions)
                finish(raw/'predictions_done.json',contract,[raw/'predictions.csv',checkpoint])
                del model
                torch.cuda.empty_cache()
            predictions=pd.read_csv(raw/'predictions.csv')
            groups=validate_records(predictions,inference,num_classes=100)
            summary,budgets,edges=summarize_run(groups,repeats=5000)
            summary.update(spec=spec,scope=SCOPE,output_classes=100,original_g6_pass=False)
            additional={}
            for k in (3,4,5):
                paired=paired_budgets(budgets,2,k)
                additional[f'2_to_{k}'] = (dict(groups=len(paired),taxa=paired.label.nunique(),
                    macro_before=float(paired.groupby('label').expected_accuracy_before.mean().mean()),
                    macro_after=float(paired.groupby('label').expected_accuracy_after.mean().mean()),
                    delta=cluster_interval(paired,'delta',5000)) if len(paired)
                    else dict(status='NO_ELIGIBLE_OBSERVATIONS'))
            summary['paired_2_to_larger']=additional
            atomic_write_json(out/'metrics.json',summary)
            atomic_write_csv(out/'groups.csv',budgets)
            atomic_write_csv(out/'nested.csv',edges)
            atomic_write_csv(out/'per_taxon.csv',budgets.groupby(['taxon_id','label','budget']).agg(
                groups=('observation_id','size'),observers=('observer_id','nunique'),
                top1=('expected_accuracy','mean'),nll=('expected_nll','mean'),brier=('expected_brier','mean')).reset_index())
            finish(out/'eval_done.json',contract,[out/'metrics.json',out/'groups.csv',out/'nested.csv',out/'per_taxon.csv',raw/'predictions.csv'])
        atomic_write_json(report/'evaluation_progress.json',dict(run_id=run_id,status='EVALUATED'))
    # All real predictions completed before the label-informed shuffled controls.
    eligibility=frame.groupby(['taxon_id','observation_id']).size().rename('photos').reset_index()
    eligible=eligibility.groupby(['taxon_id','photos']).filter(lambda x:len(x)>=3)
    atomic_write_csv(report/'e3_eligibility.csv',eligible)
    for spec in config['runs']:
        if spec['training_policy'] != 'all':
            continue
        out=report/'runs'/spec['run_id']; raw=data/'runs'/spec['run_id']
        if not len(eligible):
            if not marker(out/'e3_done.json',contract):
                finish(out/'e3_done.json',contract,status='NOT_ESTIMABLE_NO_ELIGIBLE_STRATA')
            continue
        e3_run(inference,features,spec,Path(spec['checkpoint']),raw,out,contract)
        atomic_write_json(report/'e3_progress.json',dict(run_id=spec['run_id'],status='20_RANDOMIZATIONS_DONE'))


def photo_proxies(image):
    gray=np.asarray(image.convert('L').resize((256,256)),dtype=np.float64)/255.
    gx,gy=np.gradient(gray)
    lap=(gray[1:-1,:-2]+gray[1:-1,2:]+gray[:-2,1:-1]+gray[2:,1:-1]-4*gray[1:-1,1:-1])
    energy=float(np.mean(gx*gx+gy*gy))
    xx,yy,xy=float(np.mean(gx*gx)),float(np.mean(gy*gy)),float(np.mean(gx*gy))
    return dict(laplacian_variance=float(lap.var()),gradient_energy=energy,
                dark_fraction=float(np.mean(gray<=.05)),bright_fraction=float(np.mean(gray>=.95)),
                gradient_anisotropy=float(math.sqrt((xx-yy)**2+4*xy**2)/(xx+yy+1e-12)))


def detector_proxies(boxes, scores, labels, width, height, gray):
    keep=(labels==16)&(scores>=.5)  # COCO contiguous category 16 is bird, asserted at load.
    selected=boxes[keep]; selected_scores=scores[keep]
    result=dict(bird_detection_count=len(selected),bird_detection_score=None,
                bbox_area_fraction=None,background_gradient_energy=None,bbox_json=None)
    if not len(selected):
        return result
    top=selected[int(selected_scores.argmax())].copy()
    top[[0,2]]=np.clip(top[[0,2]],0,width); top[[1,3]]=np.clip(top[[1,3]],0,height)
    result.update(bird_detection_score=float(selected_scores.max()),
        bbox_area_fraction=float(max(0,top[2]-top[0])*max(0,top[3]-top[1])/(width*height)),
        bbox_json=json.dumps(top.tolist()))
    background=np.ones(gray.shape,dtype=bool)
    h,w=gray.shape
    for box in selected:
        a,b,c,d=box
        a,c=np.clip(np.array([a,c])*w/width,0,w).astype(int)
        b,d=np.clip(np.array([b,d])*h/height,0,h).astype(int)
        background[b:d,a:c]=False
    gx,gy=np.gradient(gray)
    if background.any():
        result['background_gradient_energy']=float(np.mean((gx*gx+gy*gy)[background]))
    return result


def quality(frame, data, report, contract):
    import torch
    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
    from torchvision.transforms.functional import pil_to_tensor
    done=report/'quality_auto_done.json'
    if marker(done,contract):
        return
    weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
    if weights.meta['categories'][16] != 'bird':
        raise GateStop('Detector label mapping changed')
    torch.hub.set_dir(str(data/'model_cache'))
    model=fasterrcnn_resnet50_fpn_v2(weights=weights).eval().cuda()
    rows=[]
    for row in frame.to_dict('records'):
        path=data/'quality_ledger'/f"{row['photo_id']}.json"
        saved=marker(path,contract)
        if saved:
            rows.append(saved['row']); continue
        with Image.open(row['local_path']) as img:
            image=img.convert('RGB')
        if sha256_file(Path(row['local_path'])) != row['sha256']:
            raise GateStop('Image changed before quality audit')
        proxies=photo_proxies(image)
        # Default torchvision inference resize is part of the fixed model.
        with torch.inference_mode():
            pred=model([pil_to_tensor(image).float().div(255).cuda()])[0]
        gray=np.asarray(image.convert('L').resize((256,256)),dtype=float)/255.
        proxies.update(detector_proxies(pred['boxes'].cpu().numpy(),pred['scores'].cpu().numpy(),
                       pred['labels'].cpu().numpy(),image.width,image.height,gray))
        entry=dict(photo_id=row['photo_id'],observation_id=row['observation_id'],taxon_id=row['taxon_id'],
                   observer_id=row['observer_id'],class_index=row['class_index'],width=image.width,height=image.height,
                   local_path=row['local_path'],sha256=row['sha256'],**proxies)
        finish(path,contract,[Path(row['local_path'])],row=entry)
        rows.append(entry)
    q=pd.DataFrame(rows)
    atomic_write_csv(report/'quality_proxies.csv',q)
    # No model predictions, species accuracy or proposed mechanism is shown to raters.
    columns=['photo_id','local_path','sha256']
    template=q[columns].sample(frac=1,random_state=20260908).reset_index(drop=True)
    for c in ('rater_id','bbox_valid','motion_blur_present','occlusion_fraction','background_complexity_1_5','confidence_1_5','notes'):
        template[c]=''
    for rater in ('A','B'):
        atomic_write_csv(report/f'quality_rater_{rater}_template.csv',template)
    atomic_write_json(report/'quality_scope.json',dict(status='AUTOMATIC_PROXIES_COMPLETE_HUMAN_LABELS_PENDING',
        photos=len(q),undetected=int(q.bbox_area_fraction.isna().sum()),
        motion_blur_ground_truth=False,occlusion_ground_truth=False,bbox_ground_truth=False,
        instructions='Two independent raters, outcomes blinded; missing/uncertain remain NA. No inferred optical cause.'))
    covariates=['laplacian_variance','gradient_energy','dark_fraction','bright_fraction','gradient_anisotropy',
                'bbox_area_fraction','background_gradient_energy']
    group=q.groupby('observation_id')[covariates].mean().reset_index()
    artifacts=[report/'quality_proxies.csv',report/'quality_scope.json',report/'quality_rater_A_template.csv',report/'quality_rater_B_template.csv']
    correlations=[]; strata=[]
    for directory in sorted((report/'runs').iterdir()):
        edges=pd.read_csv(directory/'nested.csv')
        merged=edges.merge(group,on='observation_id',validate='many_to_one')
        atomic_write_csv(directory/'quality_nested.csv',merged); artifacts.append(directory/'quality_nested.csv')
        for k,part in merged.groupby('budget_from'):
            for field in covariates:
                available=part.dropna(subset=[field,'regression'])
                x=available[field]-available.groupby('label')[field].transform('mean')
                y=available.regression-available.groupby('label').regression.transform('mean')
                corr=x.corr(y) if len(x)>2 and x.std()>0 and y.std()>0 else None
                correlations.append(dict(run_id=directory.name,budget_from=int(k),covariate=field,
                    groups_available=len(available),groups_missing=len(part)-len(available),
                    within_taxon_correlation=float(corr) if corr is not None and np.isfinite(corr) else None))
                if len(available) and available[field].nunique()>1:
                    bins=pd.qcut(available[field],4,duplicates='drop')
                    for interval,binpart in available.groupby(bins,observed=True):
                        strata.append(dict(run_id=directory.name,budget_from=int(k),covariate=field,
                            descriptive_quantile=str(interval),groups=len(binpart),taxa=binpart.label.nunique(),
                            macro_regression=float(binpart.groupby('label').regression.mean().mean()),causal_claim=False))
    atomic_write_csv(report/'quality_correlations.csv',pd.DataFrame(correlations)); artifacts.append(report/'quality_correlations.csv')
    atomic_write_csv(report/'quality_strata.csv',pd.DataFrame(strata)); artifacts.append(report/'quality_strata.csv')
    weight_file=Path(torch.hub.get_dir())/'checkpoints'/Path(weights.url).name
    finish(done,contract,[*artifacts,weight_file],status='AUTOMATIC_ONLY_HUMAN_ANNOTATION_PENDING',weight_url=weights.url)
    del model
    torch.cuda.empty_cache()


def taxon_interval(frame, column, repeats=5000, seed=20260908):
    values=frame.groupby('label')[column].mean().to_numpy()
    if not len(values):
        return dict(status='NO_ELIGIBLE_TAXA')
    rng=np.random.default_rng(seed)
    samples=values[rng.integers(0,len(values),(repeats,len(values)))].mean(1)
    return dict(point=float(values.mean()),interval95=np.quantile(samples,[.025,.975]).tolist(),
                taxa=len(values),method='taxon_resampling_conditional_on_available_taxa',repeats=repeats)


def signflip(frame, column):
    values=frame.groupby('label')[column].mean().to_numpy()
    n=len(values)
    if n>16:
        signs=np.random.default_rng(20260908).choice([-1,1],size=(50000,n))
        count=np.count_nonzero(np.abs(signs@values/n)>=abs(values.mean())-1e-12)
        return float((1+count)/(1+len(signs)))
    signs=np.array(list(itertools.product((-1.,1.),repeat=n)))
    return float(np.mean(np.abs(signs@values/n)>=abs(values.mean())-1e-12))


def holm(values):
    p=np.asarray(values,float); order=np.argsort(p)
    adjusted=np.empty(len(p)); running=0.
    for rank,i in enumerate(order):
        running=max(running,(len(p)-rank)*p[i]); adjusted[i]=min(1.,running)
    return adjusted.tolist()


def statistics(root, report, config, contract):
    from robird.budget_metrics_v2_1 import cluster_interval
    done=report/'statistics_done.json'
    if marker(done,contract):
        return
    records=[]; all_edges=[]; per_seed=[]
    for spec in config['runs']:
        directory=report/'runs'/spec['run_id']
        group=pd.read_csv(directory/'groups.csv'); edge=pd.read_csv(directory/'nested.csv')
        for df,target in ((group,records),(edge,all_edges)):
            target.append(df.assign(model=spec['model'],training_policy=spec['training_policy'],seed=spec['seed']))
        for k,part in group.groupby('budget'):
            per_seed.append(dict(run_id=spec['run_id'],model=spec['model'],training_policy=spec['training_policy'],
                seed=spec['seed'],budget=int(k),groups=len(part),taxa=part.label.nunique(),
                macro=float(part.groupby('label').expected_accuracy.mean().mean())))
    group=pd.concat(records,ignore_index=True); edges=pd.concat(all_edges,ignore_index=True)
    key=['model','training_policy','observation_id','label','observer_id','taxon_id','n_photos','budget']
    if not (group.groupby(key).seed.nunique()==3).all():
        raise GateStop('Incomplete three-seed coverage')
    columns=['expected_accuracy','expected_nll','expected_brier']
    averaged=group.groupby(key)[columns].mean().reset_index()
    contrasts=[]; budget_tests=[]; coverage=[]; summaries=[]; source_gaps=[]
    for (model,policy),g in averaged.groupby(['model','training_policy']):
        for k,part in g.groupby('budget'):
            summaries.append(dict(model=model,training_policy=policy,budget=int(k),groups=len(part),taxa=part.label.nunique(),
                observer_cluster=cluster_interval(part,'expected_accuracy',5000),taxon_resampling=taxon_interval(part,'expected_accuracy')))
        for before,after in ((1,2),(2,3),(2,4),(2,5)):
            paired=paired_budgets(g,before,after)
            budget_tests.append(dict(model=model,training_policy=policy,before=before,after=after,groups=len(paired),
                observer_cluster=cluster_interval(paired,'delta',5000) if len(paired) else None,
                taxon_resampling=taxon_interval(paired,'delta')))
        at2=g[g.budget==2]; taxa=sorted(at2.label.unique())
        for omit in taxa:
            part=at2[at2.label!=omit]
            coverage.append(dict(model=model,training_policy=policy,mode='leave_one_taxon_out',
                omitted_label=int(omit),taxa=len(taxa)-1,repeat=0,macro=float(part.groupby('label').expected_accuracy.mean().mean())))
        means=at2.groupby('label').expected_accuracy.mean()
        rng=np.random.default_rng(20260908)
        for size in (5,8,10,13):
            if size>len(taxa): continue
            for repeat in range(100):
                selected=rng.choice(taxa,size,replace=False)
                coverage.append(dict(model=model,training_policy=policy,mode='random_taxa_subset',omitted_label=None,
                    taxa=size,repeat=repeat,macro=float(means.loc[selected].mean())))
        # Existing development results, matched labels; external images NEVER train models.
        dev=[]
        for seed in SEEDS:
            run_id=f'dinov2-{model}-{policy}-seed{seed}'
            d=pd.read_csv(root/'code/results/rsos_serial_v1/runs'/run_id/'groups.csv')
            dev.append(d[d.label.isin(taxa)].assign(seed=seed))
        d=pd.concat(dev).groupby(['observation_id','label','n_photos','budget']).expected_accuracy.mean().reset_index()
        for k,e in g.groupby('budget'):
            v=d[d.budget==k]
            common=sorted(set(e.label)&set(v.label))
            e=e[e.label.isin(common)]; v=v[v.label.isin(common)]
            cells=[]
            for (label,n),part in e.groupby(['label','n_photos']):
                match=v[(v.label==label)&(v.n_photos==n)]
                if len(match): cells.append(dict(label=label,n=n,weight=len(part),ext=part.expected_accuracy.mean(),dev=match.expected_accuracy.mean()))
            cell=pd.DataFrame(cells)
            weighted=[]
            if len(cell):
                for label,part in cell.groupby('label'):
                    weighted.append(float(np.average(part.ext-part.dev,weights=part.weight)))
            source_gaps.append(dict(model=model,training_policy=policy,budget=int(k),common_taxa=len(common),
                raw_gap=float(e.groupby('label').expected_accuracy.mean().mean()-v.groupby('label').expected_accuracy.mean().mean()),
                cardinality_standardized_gap=float(np.mean(weighted)) if weighted else None,
                matched_cells=len(cell),matched_external_groups=int(cell.weight.sum()) if len(cell) else 0,
                causal_claim=False))
    for policy in ('all','k1'):
        for a,b in itertools.combinations(MODELS,2):
            sub=averaged[(averaged.training_policy==policy)&(averaged.budget==2)]
            x=sub[sub.model==a].merge(sub[sub.model==b][['observation_id','expected_accuracy']],on='observation_id',
                                    suffixes=('_a','_b'),validate='one_to_one')
            x['delta']=x.expected_accuracy_a-x.expected_accuracy_b
            contrasts.append(dict(training_policy=policy,model_a=a,model_b=b,delta_a_minus_b=cluster_interval(x,'delta',5000),
                taxon_resampling=taxon_interval(x,'delta'),taxon_signflip_p=signflip(x,'delta')))
    for record,p in zip(contrasts,holm([r['taxon_signflip_p'] for r in contrasts])):
        record['holm_p_12_comparisons']=p
    atomic_write_json(report/'statistical_tests.json',dict(scope=SCOPE,seed_mean_accuracy=summaries,
        paired_budget_tests=budget_tests,model_contrasts=contrasts,
        caveats=['Taxon sign-flip assumes exchangeability/independence of taxon effects',
                 'Observer interval is linearized and fixed-taxa conditional',
                 'Availability-selected taxa are not a random bird population sample',
                 'Three training seeds are not three independent test datasets']))
    atomic_write_csv(report/'source_gaps.csv',pd.DataFrame(source_gaps))
    atomic_write_csv(report/'coverage_sensitivity.csv',pd.DataFrame(coverage))
    atomic_write_csv(report/'per_seed_accuracy.csv',pd.DataFrame(per_seed))
    atomic_write_csv(report/'seed_summary.csv',pd.DataFrame(per_seed).groupby(['model','training_policy','budget']).macro.agg(['mean','std','min','max']).reset_index())
    atomic_write_csv(report/'seed_mean_groups.csv',averaged)
    edgekey=[x for x in key if x!='budget']+['budget_from','budget_to']
    edge_avg=edges.groupby(edgekey)[['before_accuracy','after_accuracy','correction','regression','net_gain']].mean().reset_index()
    atomic_write_csv(report/'seed_mean_nested.csv',edge_avg)
    # Correct old conversational summaries without editing historical results.
    legacy=[]
    for directory in sorted((root/'code/results/rsos_serial_v1/runs').iterdir()):
        m=read_json(directory/'metrics.json'); spec=m['spec']
        for transition,value in m['nested_transitions'].items():
            legacy.append(dict(run_id=directory.name,backbone=spec['backbone'],model=spec['model'],
                training_policy='k1' if spec['budget']==1 else 'all',seed=spec['seed'],transition=transition,
                groups=value['groups'],taxa=value['taxa'],macro_before=value['macro_before'],
                macro_after=value['macro_after'],macro_net_gain=value['macro_net_gain']))
    atomic_write_csv(report/'development_matched_budget_correction.csv',pd.DataFrame(legacy))
    artifacts=[report/n for n in ('statistical_tests.json','source_gaps.csv','coverage_sensitivity.csv','per_seed_accuracy.csv',
        'seed_summary.csv','seed_mean_groups.csv','seed_mean_nested.csv','development_matched_budget_correction.csv')]
    finish(done,contract,artifacts,scope=SCOPE,no_new_training=True)
