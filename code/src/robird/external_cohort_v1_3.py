"""Retain reviewed shared scene; images and observation-photo relations are distinct."""
from __future__ import annotations
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
from PIL import Image
from robird import external_cohort_v1_1 as legacy
from robird.external_cohort_v1_1 import (
    GateStop, SCOPE, read_json, verify_tree, large_url, image_signature,
    resolve_history_path, photo_proxies, detector_proxies,
)
from robird.external_cohort_v1_2 import canonical_taxon_map
from robird.io import atomic_write_json, atomic_write_csv, sha256_file
from robird.rsos_suite_v1 import finish, marker, atomic_binary

SHARED_PHOTO = 690953732
SHARED_OBSERVATIONS = {377824923, 377824924}
SHARED_HASH = '22c70532c327a11907346b2ff990a2b64d03111e9d7bb8d972d40f639dd9afe3'
REVIEW = Path('E:/Datasets/ROBird-Bench/external_shared_photo_review_v1/690953732.jpg')


def assert_relation_metadata(frame, history):
    if frame[['observation_id','photo_id']].duplicated().any():
        raise GateStop('Duplicate observation-photo relation')
    shared = frame[frame.photo_id.duplicated(keep=False)]
    if set(shared.photo_id) != {SHARED_PHOTO} or len(shared) != 2:
        raise GateStop('Unreviewed shared photo identity')
    if set(shared.observation_id) != SHARED_OBSERVATIONS or set(shared.observer_id) != {2132783}:
        raise GateStop('Reviewed shared source identity changed')
    labels = dict(zip(shared.observation_id, shared.taxon_id))
    if labels != {377824923:4328, 377824924:4381}:
        raise GateStop('Shared photo taxon relation changed')
    if shared.url.nunique() != 1:
        raise GateStop('Shared photo URLs differ')
    for obs, part in frame.groupby('observation_id'):
        if not 2 <= len(part) <= 5 or part.photo_id.duplicated().any() or any(
            part[c].nunique()!=1 for c in ('observer_id','taxon_id','class_index')):
            raise GateStop(f'Invalid observation identity/cardinality: {obs}')
    for df in history:
        for col in ('photo_id','observation_id','observer_id'):
            if col in df and set(frame[col].astype(int)) & set(df[col].dropna().astype(int)):
                raise GateStop('History identity overlap: '+col)


def exclusion_sensitivity(frame):
    result = frame[frame.photo_id != SHARED_PHOTO].copy()
    if set(result.observation_id) != set(frame.observation_id):
        raise GateStop('Exclusion sensitivity lost observations')
    counts = result.groupby('observation_id').size()
    if not counts.between(2,5).all():
        raise GateStop('Exclusion sensitivity violates photo budget')
    return result.reset_index(drop=True)


def accepted_shared_pair(a, b, source):
    return (source == 'within_new_cohort' and
            int(a['photo_id']) == int(b['photo_id']) == SHARED_PHOTO and
            {int(a['observation_id']),int(b['observation_id'])} == SHARED_OBSERVATIONS and
            a['sha256'] == b['sha256'] == SHARED_HASH)


def relation_quality_row(cached, relation):
    # Cache is image-keyed, but its first observation must never leak into another.
    result = dict(cached)
    for key in ('photo_id','observation_id','taxon_id','observer_id','class_index'):
        result[key] = relation[key]
    return result


def download(frame, data, report, contract):
    done=report/'download_done.json'; manifest=report/'downloaded.csv'
    if marker(done,contract):
        return pd.read_csv(manifest,keep_default_na=False,dtype={'dhash':str})
    unique=frame.drop_duplicates('photo_id').reset_index(drop=True)
    # Reuse the exact inspected image; never request it twice or change its source.
    shared=unique[unique.photo_id==SHARED_PHOTO].iloc[0]
    if sha256_file(REVIEW)!=SHARED_HASH:
        raise GateStop('Inspected shared image changed')
    ledger=data/'download_ledger'/f'{SHARED_PHOTO}.json'
    if not marker(ledger,contract):
        details=image_signature(REVIEW)
        details.update(local_path=str(REVIEW),downloaded_unix=REVIEW.stat().st_mtime)
        finish(ledger,contract,[REVIEW],url=shared.url,details=details,
               reused_review_image=True)
    images=legacy.download(unique,data,report/'unique_images',contract)
    if images.photo_id.duplicated().any():
        raise GateStop('Unique image download ledger duplicates photo IDs')
    detail_columns=['photo_id','local_path','sha256','width','height','dhash','downloaded_unix']
    relations=frame.merge(images[detail_columns],on='photo_id',how='left',validate='many_to_one',sort=False)
    if len(relations)!=len(frame) or relations.sha256.isna().any():
        raise GateStop('Downloaded image to relation join failed')
    atomic_write_csv(manifest,relations)
    atomic_write_csv(report/'image_relation_mapping.csv',relations[
        ['relation_id','photo_id','observation_id','observer_id','taxon_id','class_index','shared_scene','local_path']])
    finish(done,contract,[manifest,report/'image_relation_mapping.csv',report/'unique_images/download_done.json'],
           relations=len(relations),unique_images=len(images),groups=relations.observation_id.nunique())
    return relations


def expand_features(frame, unique, features):
    if unique.photo_id.duplicated().any() or len(unique)!=len(features):
        raise GateStop('Invalid unique-image feature index')
    mapping=dict(zip(unique.photo_id,range(len(unique))))
    positions=[mapping[int(photo)] for photo in frame.photo_id]
    return np.asarray(features[positions])


def extract_dino(frame,data,report,contract):
    done=report/'features_done.json'; path=data/'relation_features.npy'
    if marker(done,contract):
        return np.load(path)
    unique=frame.drop_duplicates('photo_id').reset_index(drop=True)
    features=legacy.extract_dino(unique,data/'unique_features',report/'unique_images',contract)
    expanded=expand_features(frame,unique,features)
    if expanded.shape!=(len(frame),384):
        raise GateStop('Relation feature expansion failed')
    atomic_binary(path,lambda f:np.save(f,expanded))
    atomic_write_csv(data/'relation_feature_index.csv',frame[
        ['relation_id','photo_id','observation_id','taxon_id','class_index']].reset_index(names='feature_row'))
    finish(done,contract,[path,data/'relation_feature_index.csv',report/'unique_images/features_done.json'],
           unique_images=len(unique),relations=len(frame),features_extracted_once_per_photo=True)
    return expanded


def report_shared_singletons(report,data,config,contract):
    path=report/'shared_singleton_diagnostic.csv'; done=report/'shared_singletons_done.json'
    if marker(done,contract):
        return
    rows=[]
    for spec in config['runs']:
        p=pd.read_csv(data/'runs'/spec['run_id']/'predictions.csv')
        shared=p[(p.budget==1)&p.photo_ids.astype(str).eq(str(SHARED_PHOTO))]
        if len(shared)!=2 or set(shared.observation_id)!=SHARED_OBSERVATIONS:
            raise GateStop('Shared singleton coverage changed')
        probabilities=np.stack(shared.probabilities_json.map(json.loads))
        if not np.allclose(probabilities[0],probabilities[1],rtol=1e-5,atol=1e-6):
            raise GateStop('Same singleton image has different deterministic predictions')
        for row in shared.itertuples():
            rows.append(dict(run_id=spec['run_id'],model=spec['model'],training_policy=spec['training_policy'],
                seed=spec['seed'],observation_id=row.observation_id,photo_id=SHARED_PHOTO,
                label=row.label,prediction=row.prediction,correct=row.label==row.prediction,
                target_ambiguity=True,interpret_as_optical_robustness_failure=False))
    atomic_write_csv(path,pd.DataFrame(rows))
    finish(done,contract,[path],expected_relations_per_model=2,co_observer_cluster=True)


def evaluate_external(frame,features,config,root,data,report,contract):
    legacy.evaluate_external(frame,features,config,root,data,report,contract)
    report_shared_singletons(report,data,config,contract)
    keep=frame.photo_id.ne(SHARED_PHOTO).to_numpy()
    sensitivity=exclusion_sensitivity(frame)
    legacy.evaluate_external(sensitivity,np.asarray(features[keep]),config,root,
        data/'sensitivity_no_shared',report/'sensitivity_no_shared',contract)
    done=report/'sensitivity_evaluation_done.json'
    if not marker(done,contract):
        artifacts=[report/'shared_singletons_done.json']
        for spec in config['runs']:
            directory=report/'sensitivity_no_shared/runs'/spec['run_id']
            artifacts.append(directory/'eval_done.json')
            if spec['training_policy']=='all': artifacts.append(directory/'e3_done.json')
        finish(done,contract,artifacts,full_relations=len(frame),sensitivity_relations=len(sensitivity),
               original_files_deleted=0,same_observations=True)


def quality(frame,data,report,contract):
    quality_mode(frame,data,report,contract)
    quality_mode(exclusion_sensitivity(frame),data,report/'sensitivity_no_shared',contract)
    done=report/'quality_both_modes_done.json'
    if not marker(done,contract):
        finish(done,contract,[report/'quality_auto_done.json',report/'sensitivity_no_shared/quality_auto_done.json'],
               original_files_deleted=0,human_labels='PENDING')


def statistics(root,report,config,contract):
    legacy.statistics(root,report,config,contract)
    legacy.statistics(root,report/'sensitivity_no_shared',config,contract)
    done=report/'both_modes_comparison_done.json'
    if marker(done,contract): return
    results=[]; seeds=[]
    for spec in config['runs']:
        a=pd.read_csv(report/'runs'/spec['run_id']/'groups.csv')
        b=pd.read_csv(report/'sensitivity_no_shared/runs'/spec['run_id']/'groups.csv')
        for k,part in a.groupby('budget'):
            other=b[b.budget==k]
            pair=part.merge(other[['observation_id','expected_accuracy']],on='observation_id',
                suffixes=('_full','_sensitivity'),validate='one_to_one')
            pair['delta']=pair.expected_accuracy_sensitivity-pair.expected_accuracy_full
            results.append(dict(run_id=spec['run_id'],model=spec['model'],training_policy=spec['training_policy'],
                seed=spec['seed'],budget=int(k),full_eligible_groups=len(part),sensitivity_eligible_groups=len(other),
                common_groups=len(pair),common_taxa=pair.label.nunique(),
                delta_sensitivity_minus_full_common=float(pair.groupby('label').delta.mean().mean()) if len(pair) else None,
                ambiguity_affected_groups=int(pair.observation_id.isin(SHARED_OBSERVATIONS).sum())))
            if len(pair):
                seeds.append(pair[['observation_id','observer_id','label','budget','delta']].assign(
                    model=spec['model'],training_policy=spec['training_policy'],seed=spec['seed']))
    atomic_write_csv(report/'full_vs_sensitivity_per_seed.csv',pd.DataFrame(results))
    from robird.budget_metrics_v2_1 import cluster_interval
    merged=pd.concat(seeds)
    avg=merged.groupby(['model','training_policy','budget','observation_id','observer_id','label']).delta.mean().reset_index()
    summary=[]
    for (model,policy,k),part in avg.groupby(['model','training_policy','budget']):
        summary.append(dict(model=model,training_policy=policy,budget=int(k),groups=len(part),
            observer_cluster_delta=cluster_interval(part,'delta',5000),
            interpretation='Common observations only; full is primary, sensitivity is not a replacement'))
    atomic_write_json(report/'full_vs_sensitivity_summary.json',summary)
    artifacts=[report/'statistics_done.json',report/'sensitivity_no_shared/statistics_done.json',
        report/'full_vs_sensitivity_per_seed.csv',report/'full_vs_sensitivity_summary.json',
        report/'sensitivity_evaluation_done.json',report/'quality_both_modes_done.json']
    finish(done,contract,artifacts,original_files_deleted=0,original_g6_pass=False,
           primary_mode='full_source_relations',human_quality_labels_pending=True)

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
    names = canonical_taxon_map(dev)
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
    assert_relation_metadata(frame, history)
    if frame.photo_id.nunique() != 969:
        raise GateStop('Expected 969 unique images')
    frame['shared_scene'] = frame.photo_id.eq(SHARED_PHOTO)
    frame['relation_id'] = frame.observation_id.astype(str)+':'+frame.photo_id.astype(str)
    atomic_write_csv(path, frame)
    atomic_write_csv(report/'feasibility_100_taxa.csv', pd.DataFrame(feasibility))
    atomic_write_csv(report/'cardinality_coverage.csv', frame.groupby(['taxon_id','observation_id']).size()
                     .rename('photos').reset_index().groupby(['taxon_id','photos']).size().rename('groups').reset_index())
    finish(done, contract, [path, report/'feasibility_100_taxa.csv', report/'cardinality_coverage.csv'],
           scope=SCOPE, original_g6_pass=False, photos=970, groups=339, taxa=13)
    return frame


def duplicate_audit(frame, root, data, report, config, contract):
    done = report/'duplicate_done.json'
    if marker(done, contract):
        return
    audit = report/'duplicate_audit.json'
    if audit.exists():
        raise GateStop('Duplicate review already requested; inspect unresolved pairs')
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
                exact_bytes=same, dhash_distance=distance,
                decision='RETAIN_REVIEWED_SHARED_SCENE' if accepted_shared_pair(a,b,source) else 'PENDING',
                rationale='Reviewed same-platform multi-species shared photo; no relabeling'
                    if accepted_shared_pair(a,b,source) else ''))
    new = frame.to_dict('records')
    for i, a in enumerate(new):
        for b in refs:
            compare(a,b,'history')
        for b in new[:i]:
            compare(a,b,'within_new_cohort')
    columns = ['photo_id','observation_id','path_a','path_b','reference_photo_id',
               'reference_observation_id','source','exact_bytes','dhash_distance','decision','rationale']
    atomic_write_csv(report/'duplicate_review.csv', pd.DataFrame(pairs, columns=columns))
    pending = sum(p['decision']=='PENDING' for p in pairs)
    result = dict(scope=SCOPE, references=len(refs), candidate_photos=len(frame),
                  pairs=len(pairs), unresolved_pairs=pending, reviewed_shared_pairs=len(pairs)-pending, exact_pairs=sum(p['exact_bytes'] for p in pairs),
                  status='PENDING_DUPLICATE_ADJUDICATION' if pending else 'PASS_WITH_REVIEWED_SHARED_SCENE',
                  all_duplicates_proven_absent=False, original_g6_pass=False)
    atomic_write_json(audit, result)
    if pending:
        raise GateStop(f'{pending} unreviewed duplicate candidates require adjudication before features')
    finish(done, contract, [audit,report/'duplicate_review.csv',ref_path], **result)


def quality_mode(frame, data, report, contract):
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
            rows.append(relation_quality_row(saved['row'],row)); continue
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
    template=q[columns].drop_duplicates('photo_id').sample(frac=1,random_state=20260908).reset_index(drop=True)
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
