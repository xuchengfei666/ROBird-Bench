"""Read-only results interpretation and independent point checks; no models."""
from pathlib import Path
import os,sys,json,argparse,itertools
os.environ.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code/src'))
import numpy as np
import pandas as pd
from robird.io import atomic_write_json,atomic_write_csv,sha256_file
from robird.artifact_tree_v1 import verify_tree
from robird.external_expansion_v2 import verify_files
from robird.budget_metrics_v2_1 import validate_records,group_statistics
from robird.external_completion_v1 import components,point_metrics
from robird.external_cohort_v1_1 import holm
from robird.rsos_suite_v1 import finish
R=ROOT/'code/results/external_expansion_v2_2'
D=Path('E:/Datasets/ROBird-Bench/external_expansion_v2_2')
OUT=ROOT/'code/results/external_expansion_v2_2_analysis_v1'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def close(a,b):np.testing.assert_allclose(a,b,atol=1e-10,rtol=0)
def write(name,frame):atomic_write_csv(OUT/name,frame,refuse_if_exists=True)

def integrity():
    freeze=ROOT/'FROZEN_EXTERNAL_EXPANSION_V2_2.json'
    verify_files(read(freeze)['files'])
    print('Freeze inputs verified',flush=True)
    verify_tree(R/'automatic_done.json')
    print('Full completion artifact tree verified',flush=True)
    atomic_write_json(OUT/'integrity.json',dict(status='PASS',freeze_sha256=sha256_file(freeze),
        automatic_done_sha256=sha256_file(R/'automatic_done.json'),frozen_inputs=len(read(freeze)['files']),
        full_recursive_hash_verification=True),refuse_if_exists=True)

def main():
    cfg=read(R/'resolved_config.json')
    primary=pd.read_csv(R/'primary_manifest.csv');sensitivity=pd.read_csv(R/'exclude_affected_manifest.csv')
    manifest={'primary':primary,'exclude_affected':sensitivity}
    intervals={branch:pd.read_csv(base/'all_metric_intervals.csv') for branch,base in
               [('primary',R),('exclude_affected',R/'exclude_affected')]}
    tests=read(R/'paired_budget_tests_holm128.json')
    assert len(tests)==128
    allgroups=[];alledges=[];e3=[];mapping_audits=[];inputs=[R/'automatic_done.json']
    for spec in cfg['runs']:
        name=spec['run_id'];base=R/'runs'/name
        raw=pd.read_csv(D/'runs'/name/'predictions.csv')
        budgets,edges=group_statistics(validate_records(raw,primary,num_classes=100))
        saved=pd.read_csv(base/'groups.csv');saved_edges=pd.read_csv(base/'nested.csv')
        pd.testing.assert_frame_equal(budgets,saved,check_dtype=False,atol=1e-10,rtol=0)
        pd.testing.assert_frame_equal(edges,saved_edges,check_dtype=False,atol=1e-10,rtol=0)
        filtered=raw[raw.observation_id.isin(sensitivity.observation_id)].reset_index(drop=True)
        sr=pd.read_csv(D/'exclude_affected/runs'/name/'predictions.csv')
        pd.testing.assert_frame_equal(filtered,sr)
        for branch,frame in manifest.items():
            b=budgets[budgets.observation_id.isin(frame.observation_id)]
            e=edges[edges.observation_id.isin(frame.observation_id)]
            for mode in ('full8','strict10'):
                subset=frame if mode=='full8' else frame[frame.strict10_taxon]
                allgroups.append(b[b.observation_id.isin(subset.observation_id)].assign(
                    branch=branch,mode=mode,model=spec['model'],policy=spec['training_policy'],seed=spec['seed']))
                alledges.append(e[e.observation_id.isin(subset.observation_id)].assign(
                    branch=branch,mode=mode,model=spec['model'],policy=spec['training_policy'],seed=spec['seed']))
        packets=components(raw,primary)
        selected=intervals['primary']
        selected=selected[(selected.seed.astype(str)==str(spec['seed']))&
            (selected.model==spec['model'])&(selected.training_policy==spec['training_policy'])&(selected['mode']=='full8')]
        for k,(meta,values) in packets.items():
            for cohort in ('eligible_per_budget','fixed_five_photo_cohort'):
                mask=np.ones(len(meta),dtype=bool) if cohort=='eligible_per_budget' else meta.n_photos.eq(5).to_numpy()
                if not mask.any():continue
                point=point_metrics(meta[mask],values[mask])
                for metric,value in point.items():
                    target=selected[(selected.cohort==cohort)&(selected.budget==k)&(selected.metric==metric)]
                    assert len(target)==1;close(value,target.point.iloc[0])
        inputs += [base/'groups.csv',base/'nested.csv',D/'runs'/name/'predictions.csv']
        if spec['training_policy']=='all':
            e3.append(pd.read_csv(base/'e3_randomizations.csv').assign(model=spec['model'],seed=spec['seed']))
            source=primary.set_index('photo_id')
            slot=primary.groupby('observation_id').agg(taxon=('taxon_id','first'),n=('photo_id','size'),
                                                     observer=('observer_id','first'))
            eligible=slot.groupby(['taxon','n']).filter(lambda x:len(x)>=3)
            expected=set(primary[primary.observation_id.isin(eligible.index)].photo_id)
            for rep in range(20):
                mp=pd.read_csv(D/'runs'/name/f'e3_repeat_{rep:02d}_mapping.csv')
                assert set(mp.photo_id)==expected and not mp.photo_id.duplicated().any()
                assert set(mp.slot_observation_id)==set(eligible.index)
                actual_source=mp.photo_id.map(source.observation_id);actual_observer=mp.photo_id.map(source.observer_id)
                assert np.array_equal(actual_source,mp.source_observation_id)
                assert np.array_equal(actual_observer,mp.source_observer_id)
                assert (mp.slot_observation_id!=mp.source_observation_id).all()
                assert np.array_equal(mp.slot_observation_id.map(slot.taxon),mp.photo_id.map(source.taxon_id))
                assert np.array_equal(mp.slot_observation_id.map(slot.n),mp.source_observation_id.map(slot.n))
                count=mp.groupby('slot_observation_id').size()
                assert np.array_equal(count,slot.loc[count.index,'n'])
                assert (mp.groupby('slot_observation_id').source_observation_id.nunique()>=2).all()
                same_observer=(mp.slot_observation_id.map(slot.observer)==mp.source_observer_id).mean()
                mapping_audits.append(dict(model=spec['model'],seed=spec['seed'],repeat=rep,photos=len(mp),
                    groups=len(count),same_source_observer_fraction=float(same_observer)))
        print('Verified raw/group/nested/interval/reuse/E3',name,flush=True)
    g=pd.concat(allgroups,ignore_index=True);e=pd.concat(alledges,ignore_index=True)
    key=['branch','mode','model','policy','observation_id','label','observer_id','taxon_id','n_photos','budget']
    assert g.groupby(key).seed.nunique().eq(3).all()
    avg=g.groupby(key).expected_accuracy.mean().reset_index()
    paired=[];taxa_rows=[];seed_rows=[]
    for row in tests:
        a=avg[(avg.branch==row['analysis_branch'])&(avg['mode']==row['mode'])&
              (avg.model==row['model'])&(avg.policy==row['training_policy'])]
        p=a[a.budget==row['before']].merge(a[a.budget==row['after']][['observation_id','expected_accuracy']],
            on='observation_id',suffixes=('_before','_after'),validate='one_to_one')
        p['delta']=p.expected_accuracy_after-p.expected_accuracy_before
        macro=p.groupby('label').delta.mean()
        close(macro.mean(),row['delta_accuracy'])
        assert len(p)==row['groups'] and len(macro)==row['taxa'] and p.observer_id.nunique()==row['observers']
        rec={k:row[k] for k in ['analysis_branch','mode','model','training_policy','before','after','groups','taxa','observers',
            'delta_accuracy','observer_signflip_p','observer_signflip_p_holm64','observer_signflip_p_holm128',
            'taxon_signflip_p_holm128']}
        rec.update(macro_before=float(p.groupby('label').expected_accuracy_before.mean().mean()),
                   macro_after=float(p.groupby('label').expected_accuracy_after.mean().mean()),
                   lower95=row['observer_interval']['interval95'][0],upper95=row['observer_interval']['interval95'][1],
                   positive_taxa=int((macro>1e-12).sum()),negative_taxa=int((macro< -1e-12).sum()),
                   zero_taxa=int((abs(macro)<=1e-12).sum()),
                   leave_one_taxon_out_min=float(min(macro.drop(t).mean() for t in macro.index)),
                   leave_one_taxon_out_max=float(max(macro.drop(t).mean() for t in macro.index)))
        paired.append(rec)
        for label,v in macro.items():taxa_rows.append(dict(analysis_branch=row['analysis_branch'],mode=row['mode'],
            model=row['model'],policy=row['training_policy'],before=row['before'],after=row['after'],label=int(label),delta=float(v)))
    for field in ('observer_signflip_p','taxon_signflip_p'):
        close(holm([r[field] for r in tests]),[r[field+'_holm128'] for r in tests])
    for identity,part in g.groupby(['branch','mode','model','policy','seed']):
        for before,after in [(1,2),(2,3),(2,4),(2,5)]:
            p=part[part.budget==before].merge(part[part.budget==after][['observation_id','expected_accuracy']],
                on='observation_id',suffixes=('_before','_after'),validate='one_to_one')
            p['delta']=p.expected_accuracy_after-p.expected_accuracy_before
            seed_rows.append(dict(zip(['branch','mode','model','policy','seed'],identity),
                before=before,after=after,delta=float(p.groupby('label').delta.mean().mean())))
    edgekeys=['branch','mode','model','policy','budget_from','budget_to','seed']
    cols=['correction','regression','net_gain','any_harmful','any_helpful']
    harm=e.groupby(edgekeys+['label'])[cols].mean().groupby(edgekeys).mean().reset_index()
    harm=harm.groupby(edgekeys[:-1])[cols].mean().reset_index()
    close(harm.correction-harm.regression,harm.net_gain)
    ec=pd.concat(e3,ignore_index=True)
    assert len(ec)==4*3*20*5
    ec=ec.groupby(['model','repeat','budget']).agg(macro_real=('macro_real','mean'),
        macro_shuffled=('macro_shuffled','mean'),difference=('difference','mean'),groups=('groups','first'),taxa=('taxa','first')).reset_index()
    close(ec[ec.budget==1].difference,0)
    e3summary=ec.groupby(['model','budget']).agg(real=('macro_real','mean'),shuffled=('macro_shuffled','mean'),
        difference=('difference','mean'),minimum=('difference','min'),maximum=('difference','max'),
        positive_repeats=('difference',lambda x:int((x>1e-12).sum())),groups=('groups','first'),taxa=('taxa','first')).reset_index()
    p=pd.DataFrame(paired)
    keys=['mode','model','training_policy','before','after']
    compare=p[p.analysis_branch=='primary'].merge(p[p.analysis_branch=='exclude_affected'],on=keys,suffixes=('_primary','_exclude'))
    compare['delta_change']=compare.delta_accuracy_exclude-compare.delta_accuracy_primary
    compare['holm128_significance_changed']=(compare.observer_signflip_p_holm128_primary<.05)!=(compare.observer_signflip_p_holm128_exclude<.05)
    # Development and previous 13-taxon cohort are contextual replication, not equally sampled datasets.
    replication=[]
    for cohort,base in [('development',ROOT/'code/results/rsos_serial_v1/runs'),
                        ('previous13',ROOT/'code/results/external_cohort_v1_5/runs')]:
        for model in cfg['runs']:
            if model['training_policy']!='all':continue
            fpath=base/model['run_id']/'groups.csv'
            b=pd.read_csv(fpath)
            z=b[b.budget==1].merge(b[b.budget==2][['observation_id','expected_accuracy']],on='observation_id',suffixes=('_one','_two'))
            z['delta']=z.expected_accuracy_two-z.expected_accuracy_one
            replication.append(dict(cohort=cohort,model=model['model'],seed=model['seed'],
                groups=len(z),taxa=int(z.label.nunique()),delta=float(z.groupby('label').delta.mean().mean())))
            inputs.append(fpath)
    tables={'paired_effects.csv':p,'per_taxon_deltas.csv':pd.DataFrame(taxa_rows),'seed_paired_effects.csv':pd.DataFrame(seed_rows),
        'harmful_additions.csv':harm,'e3_repeat_means.csv':ec,'e3_summary.csv':e3summary,
        'e3_mapping_checks.csv':pd.DataFrame(mapping_audits),'duplicate_sensitivity_comparison.csv':compare,
        'historical_replication_context.csv':pd.DataFrame(replication)}
    for name,frame in tables.items():write(name,frame)
    sizes=primary.groupby('observation_id').size()
    observers=primary[['observation_id','observer_id','taxon_id']].drop_duplicates().groupby('observer_id')
    audit=dict(status='POINT_AND_DESIGN_CHECKS_PASS',raw_runs_recomputed=24,sensitivity_prediction_reuse_verified=24,
        e3_mappings_checked=len(mapping_audits),paired_effects_recomputed=128,
        max_abs_holm128_point_error=0.,e3_k1_invariance=True,
        source_observer_clusters=int(primary.observer_id.nunique()),
        observers_multiple_groups=int((observers.observation_id.nunique()>1).sum()),
        observers_multiple_taxa=int((observers.taxon_id.nunique()>1).sum()),
        max_observations_per_observer=int(observers.observation_id.nunique().max()),
        cardinality=dict((str(k),int(v)) for k,v in sizes.value_counts().sort_index().items()),
        independence_or_sign_exchangeability_proven=False,normality_test='NOT_APPLICABLE_NO_NEW_PARAMETRIC_TEST',
        new_training=False,new_inference=False,new_hypothesis_tests=False,
        automatic_done_sha256=sha256_file(R/'automatic_done.json'))
    atomic_write_json(OUT/'point_audit.json',audit,refuse_if_exists=True)
    finish(OUT/'analysis_done.json',audit['automatic_done_sha256'],
        [OUT/n for n in tables]+[OUT/'point_audit.json',Path(__file__),ROOT/'docs/EXTERNAL_EXPANSION_V2_2_ANALYSIS_PLAN_20260919.md']
        +list(dict.fromkeys(inputs)),read_only_reporting=True)
    print(json.dumps(audit),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--integrity-only',action='store_true');a=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if a.integrity_only:integrity()
    else:main()
