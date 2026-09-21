"""Audited unique-frame cohort; frozen source images and old STOP stay read-only."""
from __future__ import annotations
import json
from pathlib import Path
from decimal import Decimal
from collections import Counter
import numpy as np
import pandas as pd
from robird import external_expansion_v2 as parent
from robird.io import atomic_write_json, atomic_write_csv, sha256_file
from robird.rsos_suite_v1 import marker, finish
from robird.artifact_tree_v1 import verify_tree

GateStop=parent.GateStop
read_json=parent.read_json
DUPLICATES={'EXACT_WITHIN_OBSERVATION','NEAR_DUPLICATE_WITHIN_OBSERVATION'}
DISTINCT={'DISTINCT_HISTORY','DISTINCT_CROSS_OBSERVATION','WITHIN_OBSERVATION_REDUNDANCY'}
ID_COLUMNS=('photo_id','observation_id','observer_id','taxon_id','class_index')

def integer(value):
    d=Decimal(str(value))
    if not d.is_finite() or d!=d.to_integral_value():raise GateStop('Invalid integer identity')
    return int(d)

def read_manifest(path):
    frame=pd.read_csv(path,keep_default_na=False,dtype={'dhash':str})
    for col in ID_COLUMNS:frame[col]=frame[col].map(integer)
    return frame

def strict_support(frame,threshold=10):
    result=frame.copy()
    support=result.groupby('taxon_id').observer_id.nunique()
    result['strict10_taxon']=result.taxon_id.map(support).ge(threshold)
    return result.sort_values(['taxon_id','observation_id','photo_id']).reset_index(drop=True)

def counts(frame):
    support=frame.groupby('taxon_id').observer_id.nunique()
    return dict(photos=len(frame),groups=int(frame.observation_id.nunique()),
                observers=int(frame.observer_id.nunique()),taxa=len(support),
                strict10_taxa=int((support>=10).sum()))

def build_cohorts(source,index,notes):
    if notes.get('reviewer_type')!='AI_ASSISTED_VISUAL_REVIEW' or notes.get('independent_human_adjudication') is not False:
        raise GateStop('Review provenance mismatch')
    if notes.get('outcome_blinded') is not True:raise GateStop('Review not outcome blinded')
    pairs=index['pairs']; records=notes['pairs']
    if len(records)!=len(pairs) or sorted(x['pair_id'] for x in records)!=list(range(len(pairs))):
        raise GateStop('Incomplete review')
    if [p['pair_id'] for p in pairs]!=list(range(len(pairs))):raise GateStop('Pair ordering changed')
    if source.photo_id.duplicated().any():raise GateStop('Repeated source photo ID')
    by_id=source.set_index('photo_id')
    lookup={x['pair_id']:x for x in records}
    uf={integer(p):integer(p) for p in source.photo_id}
    def find(p):
        if p not in uf:raise GateStop('Duplicate edge outside source')
        while uf[p]!=p:uf[p]=uf[uf[p]];p=uf[p]
        return p
    affected=set();audit=[]
    for p in pairs:
        note=lookup[p['pair_id']];d=note['decision']
        a=integer(p['photo_id']);b=integer(p['reference_photo_id']) if p['reference_photo_id'] is not None else None
        if d not in DUPLICATES|DISTINCT or not note['rationale'].strip():raise GateStop('Unresolved/unsafe duplicate review')
        same=p['same_observation'];hist=p['source']=='history';exact=p['exact_bytes']
        valid=(d=='EXACT_WITHIN_OBSERVATION' and same and exact and not hist or
               d=='NEAR_DUPLICATE_WITHIN_OBSERVATION' and same and not exact and not hist or
               d=='WITHIN_OBSERVATION_REDUNDANCY' and same and not exact and not hist or
               d=='DISTINCT_HISTORY' and hist and not exact or
               d=='DISTINCT_CROSS_OBSERVATION' and not hist and not same and not exact)
        if not valid:raise GateStop('Review classification incompatible with source')
        if a not in by_id.index:raise GateStop('Review source photo missing')
        ar=by_id.loc[a]
        if integer(ar.observation_id)!=integer(p['observation_id']) or ar.sha256!=p['path_a_sha256']:
            raise GateStop('Review/source identity or hash mismatch')
        if Path(ar.local_path).resolve()!=Path(p['path_a']).resolve():raise GateStop('Review/source path mismatch')
        if not hist:
            if b not in by_id.index:raise GateStop('Within-cohort reference missing')
            br=by_id.loc[b]
            if integer(br.observation_id)!=integer(p['reference_observation_id']) or br.sha256!=p['path_b_sha256']:
                raise GateStop('Reference source mismatch')
            if Path(br.local_path).resolve()!=Path(p['path_b']).resolve():raise GateStop('Reference path mismatch')
            if (ar.observation_id==br.observation_id)!=same or (ar.sha256==br.sha256)!=exact:
                raise GateStop('Source exact/same-observation mismatch')
        if d in DUPLICATES:
            x,y=find(a),find(b);uf[max(x,y)]=min(x,y);affected.add(integer(ar.observation_id))
        audit.append(dict(pair_id=p['pair_id'],decision=d,photo_id=a,reference_photo_id=b,source=p['source']))
    mapping=source[list(ID_COLUMNS)].copy()
    mapping['representative_photo_id']=[find(integer(p)) for p in source.photo_id]
    mapping['representative']=mapping.photo_id.eq(mapping.representative_photo_id)
    primary=source[source.photo_id.isin(mapping.loc[mapping.representative,'photo_id'])].copy()
    sizes=primary.groupby('observation_id').size()
    small=set(int(x) for x in sizes[sizes<2].index)
    primary=primary[~primary.observation_id.isin(small)]
    sensitivity=source[~source.observation_id.isin(affected)].copy()
    mapping['retained_primary']=mapping.photo_id.isin(primary.photo_id)
    mapping['retained_exclude_affected']=~mapping.observation_id.isin(affected)
    mapping['reason']=np.where(mapping.retained_primary,'RETAINED',
        np.where(~mapping.representative,'SAME_FRAME_NONREPRESENTATIVE','LT2_UNIQUE_PHOTOS'))
    kept=set(primary.photo_id)
    for row in audit:
        row['both_retained_primary']=row['photo_id'] in kept and (row['source']=='history' or row['reference_photo_id'] in kept)
        if row['decision'] in DUPLICATES and row['both_retained_primary']:
            raise GateStop('Duplicate edge survived unique-frame mapping')
    return strict_support(primary),strict_support(sensitivity),mapping,pd.DataFrame(audit),sorted(affected),sorted(small)

def validate_cohort(frame,source,taxa,excluded,config):
    if not len(frame) or frame.photo_id.duplicated().any():raise GateStop('Empty/repeated photo cohort')
    if not set(frame.photo_id)<=set(source.photo_id):raise GateStop('Non-source photo')
    # No path, label, license, URL or source identity can be rewritten.
    cols=[c for c in source if c!='strict10_taxon']
    a=frame.set_index('photo_id').sort_index()
    b=source.set_index('photo_id').loc[a.index].sort_index()
    try:pd.testing.assert_frame_equal(a[[c for c in cols if c!='photo_id']],b[[c for c in cols if c!='photo_id']],check_dtype=False)
    except AssertionError as exc:raise GateStop('Source fields changed') from exc
    parent.legacy.assert_metadata(frame,[])
    for col,ids in excluded.items():
        if set(frame[col])&ids:raise GateStop('History identity overlap: '+col)
    if not frame.license_code.isin(parent.LICENSES).all() or not frame.attribution.map(lambda x:bool(str(x).strip())).all():
        raise GateStop('License/attribution invalid')
    for row in frame.itertuples():
        if parent.legacy.large_url(row.original_url)!=row.url:raise GateStop('Source URL normalization changed')
    mapping=dict(zip(taxa.taxon_id,taxa.class_index))
    if any(mapping.get(t)!=c for t,c in zip(frame.taxon_id,frame.class_index)):raise GateStop('100-way taxon map mismatch')
    sizes=frame.groupby(['taxon_id','observer_id']).observation_id.nunique()
    if (sizes>config['max_groups_per_observer_taxon']).any():raise GateStop('Observer group cap changed')
    support=frame.groupby('taxon_id').observer_id.nunique()
    if (support<config['min_observers']).any():raise GateStop('Taxon support below frozen minimum')
    expected=frame.taxon_id.map(support).ge(config['sensitivity_min_observers'])
    if not np.array_equal(expected.to_numpy(),frame.strict10_taxon.astype(str).str.lower().eq('true').to_numpy()):
        raise GateStop('Strict10 flag does not match actual support')
    if frame.sha256.duplicated().any():raise GateStop('Exact bytes still duplicated')
    return counts(frame)

def check_counts(actual,expected):
    if actual!=expected:raise GateStop('Derived cohort differs from audited expectation: '+str(actual))

def verify_parents(root,config,verify_reference_bytes=False,seen=None):
    parent_freeze=root/config['parent_freeze'];review=root/config['review_report']
    for path,digest in [(parent_freeze,config['parent_sha256']),(review/'review_done.json',config['review_sha256']),
                        (review/'impact_done.json',config['impact_sha256'])]:
        if sha256_file(path)!=digest:raise GateStop('Parent/review anchor changed')
    parent.verify_files(read_json(parent_freeze)['files'])
    seen=set() if seen is None else seen
    for path in (review/'review_done.json',review/'impact_done.json',
                 root/config['source_report']/'metadata_done.json',
                 root/config['source_report']/'download_done.json',
                 Path(config['source_data'])/'reference_signatures_done.json'):
        verify_tree(path,seen)
    if verify_reference_bytes:
        refs=pd.read_csv(Path(config['source_data'])/'reference_signatures.csv',keep_default_na=False)
        for row in refs.itertuples():
            if sha256_file(Path(row.local_path))!=row.sha256:raise GateStop('History reference bytes changed')
    return review

def verify_source_bytes(source,config):
    for row in source.itertuples():
        ledger=marker(Path(config['source_data'])/'download_ledger'/f'{row.photo_id}.json',config['parent_sha256'])
        if not ledger or ledger['details']['sha256']!=row.sha256 or ledger['url']!=row.url:
            raise GateStop('Source ledger mismatch')
        sig=parent.legacy.image_signature(Path(row.local_path))
        if any(str(sig[k])!=str(getattr(row,k)) for k in ('sha256','dhash','width','height')):
            raise GateStop('Source byte/signature drift')

def materialize_gate(root,report,config,contract,seen=None):
    done=report/'data_gate_done.json'
    if marker(done,contract):
        verify_tree(done,seen)
        return read_manifest(report/'primary_manifest.csv'),read_manifest(report/'exclude_affected_manifest.csv')
    review=verify_parents(root,config,verify_reference_bytes=True,seen=seen)
    source=read_manifest(root/config['source_report']/'downloaded.csv')
    verify_source_bytes(source,config)
    index=read_json(review/'index.json')
    notes=read_json(root/'code/configs/external_expansion_duplicate_decisions_v1.json')
    if len(index['pairs'])!=config['expected_pairs']:raise GateStop('Candidate count changed')
    primary,sensitivity,mapping,resolutions,affected,small=build_cohorts(source,index,notes)
    if len(affected)!=config['expected_affected_observations']:raise GateStop('Affected group count changed')
    _,excluded=parent.history_ids(root,config)
    taxa=pd.read_csv(root/config['taxon_manifest'])
    pc=validate_cohort(primary,source,taxa,excluded,config)
    sc=validate_cohort(sensitivity,source,taxa,excluded,config)
    check_counts(pc,config['expected_primary']);check_counts(sc,config['expected_exclude_affected'])
    # Sensitivity must be unchanged entire observations, not arbitrary photo deletion.
    filter_prediction_manifest(primary,sensitivity)
    coverage=source[['taxon_id','scientific_name']].drop_duplicates().sort_values('taxon_id')
    for name,frame in [('source',source),('primary',primary),('exclude_affected',sensitivity)]:
        support=frame.groupby('taxon_id').observer_id.nunique()
        coverage[name+'_observers']=coverage.taxon_id.map(support).fillna(0).astype(int)
    objects={'primary_manifest.csv':primary,'exclude_affected_manifest.csv':sensitivity,
             'photo_disposition.csv':mapping,'candidate_resolution.csv':resolutions,'taxon_support.csv':coverage}
    for name,frame in objects.items():
        path=report/name
        if path.exists():
            actual=pd.read_csv(path,keep_default_na=False,dtype=str)
            # Compare serialized bytes without overwriting committed files.
            if path.read_bytes()!=frame.to_csv(index=False).encode('utf-8'):
                raise GateStop('Partial data-gate output changed: '+name)
        else:atomic_write_csv(path,frame,refuse_if_exists=True)
    audit=dict(status='PASS_DERIVED_UNIQUE_FRAME_DATA_GATE',primary=pc,exclude_affected=sc,
        affected_observations=affected,lt2_unique_observations=small,deleted_images=0,
        original_g6_pass=False,independent_human_adjudication=False,
        duplicate_detection_scope='same frozen hash screen, outcome-blinded AI-assisted candidate adjudication',
        source_images_reused_read_only=True,new_downloads=0)
    audit_path=report/'data_gate_audit.json'
    if audit_path.exists():
        if read_json(audit_path)!=audit:raise GateStop('Partial gate audit changed')
    else:atomic_write_json(audit_path,audit,refuse_if_exists=True)
    finish(done,contract,[report/n for n in objects]+[audit_path,review/'review_done.json',
        review/'impact_done.json',root/config['source_report']/'download_done.json',
        root/config['source_report']/'metadata_done.json'],
        status=audit['status'],original_g6_pass=False,features_authorized_for_this_version=True)
    return primary,sensitivity

def require_gate(root,report,config,contract,frame,seen=None):
    if not marker(report/'data_gate_done.json',contract):raise GateStop('Model blocked: missing unique-frame data gate')
    verify_tree(report/'data_gate_done.json',seen)
    expected=read_manifest(report/'primary_manifest.csv')
    try:pd.testing.assert_frame_equal(frame,expected)
    except AssertionError as exc:raise GateStop('Model manifest differs from committed primary') from exc
    for row in frame.itertuples():
        if sha256_file(Path(row.local_path))!=row.sha256:raise GateStop('Image changed after data gate')

def filter_prediction_manifest(primary,sensitivity):
    if not set(sensitivity.observation_id)<=set(primary.observation_id):raise GateStop('Foreign sensitivity observation')
    cols=['observation_id','photo_id','class_index','taxon_id','observer_id','sha256']
    a=primary[primary.observation_id.isin(sensitivity.observation_id)][cols].sort_values('photo_id').reset_index(drop=True)
    b=sensitivity[cols].sort_values('photo_id').reset_index(drop=True)
    try:pd.testing.assert_frame_equal(a,b,check_dtype=False)
    except AssertionError as exc:raise GateStop('Sensitivity changed a retained observation') from exc

def filtered_predictions(predictions,primary,sensitivity):
    filter_prediction_manifest(primary,sensitivity)
    from robird.budget_metrics_v2_1 import validate_records
    validate_records(predictions,primary,num_classes=100)
    filtered=predictions[predictions.observation_id.isin(sensitivity.observation_id)].copy()
    validate_records(filtered,sensitivity,num_classes=100)
    return filtered

def prepare_sensitivity_predictions(primary,sensitivity,data,report,config,contract):
    done=report/'sensitivity_predictions_done.json'
    if marker(done,contract):verify_tree(done);return
    artifacts=[]
    for spec in config['runs']:
        name=spec['run_id'];source=data/'runs'/name/'predictions_done.json'
        if not marker(source,contract):raise GateStop('Primary predictions incomplete')
        target=data/'exclude_affected'/'runs'/name
        mark=target/'predictions_done.json'
        if not marker(mark,contract):
            predictions=pd.read_csv(data/'runs'/name/'predictions.csv')
            filtered=filtered_predictions(predictions,primary,sensitivity)
            atomic_write_csv(target/'predictions.csv',filtered)
            finish(mark,contract,[target/'predictions.csv',source,report/'data_gate_done.json'],
                   inference_reused=True,unchanged_whole_observations_only=True)
        artifacts.append(mark)
    finish(done,contract,artifacts,models_rerun=0,e3_sensitivity_performed=False)

def joint_tests(report,contract):
    done=report/'joint_tests_done.json'
    if marker(done,contract):verify_tree(done);return
    rows=[]
    for branch,base in [('primary',report),('exclude_affected',report/'exclude_affected')]:
        rows.extend(dict(row,analysis_branch=branch) for row in read_json(base/'paired_budget_tests.json'))
    if len(rows)!=128:raise GateStop('Combined sensitivity comparison family incomplete')
    for key in ('observer_signflip_p','taxon_signflip_p'):
        for row,p in zip(rows,parent.legacy.holm([r[key] for r in rows])):row[key+'_holm128']=float(p)
    atomic_write_json(report/'paired_budget_tests_holm128.json',rows)
    finish(done,contract,[report/'paired_budget_tests_holm128.json',report/'statistics_done.json',
                          report/'exclude_affected/statistics_done.json'],family_size=128)

def descriptive_source_gaps(primary,sensitivity,root,report,config,contract):
    """Read fixed per-group metrics; no new tests or causal interpretation."""
    done=report/'descriptive_done.json'
    if marker(done,contract):verify_tree(done);return
    collected=[];source_files=[]
    for spec in config['runs']:
        path=report/'runs'/spec['run_id']/'groups.csv'
        f=pd.read_csv(path).assign(model=spec['model'],training_policy=spec['training_policy'],seed=spec['seed'])
        collected.append(f);source_files.append(path)
    all_rows=pd.concat(collected,ignore_index=True);seed_rows=[];gaps=[]
    for branch,frame in [('primary',primary),('exclude_affected',sensitivity)]:
        for mode in ('full8','strict10'):
            subset=frame if mode=='full8' else frame[frame.strict10_taxon.astype(str).str.lower().eq('true')]
            ext=all_rows[all_rows.observation_id.isin(subset.observation_id)]
            for (model,policy,seed,k),g in ext.groupby(['model','training_policy','seed','budget']):
                seed_rows.append(dict(branch=branch,mode=mode,model=model,training_policy=policy,
                    seed=int(seed),budget=int(k),groups=len(g),taxa=int(g.label.nunique()),
                    macro_accuracy=float(g.groupby('label').expected_accuracy.mean().mean())))
            mean=ext.groupby(['model','training_policy','observation_id','label','n_photos','budget']).expected_accuracy.mean().reset_index()
            for (model,policy),g in mean.groupby(['model','training_policy']):
                dev=[]
                for seed in parent.legacy.SEEDS:
                    path=root/'code/results/rsos_serial_v1/runs'/f'dinov2-{model}-{policy}-seed{seed}'/'groups.csv'
                    dev.append(pd.read_csv(path));source_files.append(path)
                d=pd.concat(dev).groupby(['observation_id','label','n_photos','budget']).expected_accuracy.mean().reset_index()
                for k,e in g.groupby('budget'):
                    v=d[d.budget==k];common=set(e.label)&set(v.label);e=e[e.label.isin(common)];v=v[v.label.isin(common)]
                    cells=[]
                    for (label,n),part in e.groupby(['label','n_photos']):
                        match=v[(v.label==label)&(v.n_photos==n)]
                        if len(match):cells.append(dict(label=label,weight=len(part),delta=float(part.expected_accuracy.mean()-match.expected_accuracy.mean())))
                    cell=pd.DataFrame(cells);weighted=[]
                    if len(cell):
                        weighted=[np.average(p.delta,weights=p.weight) for _,p in cell.groupby('label')]
                    gaps.append(dict(branch=branch,mode=mode,model=model,training_policy=policy,budget=int(k),
                        common_taxa=len(common),raw_gap=float(e.groupby('label').expected_accuracy.mean().mean()-v.groupby('label').expected_accuracy.mean().mean()),
                        cardinality_standardized_gap=float(np.mean(weighted)) if weighted else None,
                        matched_external_groups=int(cell.weight.sum()) if len(cell) else 0,
                        causal_claim=False))
    atomic_write_csv(report/'per_seed_accuracy.csv',pd.DataFrame(seed_rows))
    atomic_write_csv(report/'source_gaps.csv',pd.DataFrame(gaps))
    atomic_write_csv(report/'seed_summary.csv',pd.DataFrame(seed_rows).groupby(
        ['branch','mode','model','training_policy','budget']).macro_accuracy.agg(['mean','std','min','max']).reset_index())
    finish(done,contract,[report/p for p in ('per_seed_accuracy.csv','source_gaps.csv','seed_summary.csv')]
           +list(dict.fromkeys(source_files)),no_new_inference=True,causal_claim=False)
