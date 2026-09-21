"""Read-only counterfactual duplicate handling, never an evaluation manifest."""
from pathlib import Path
import sys, json
from collections import Counter
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code/src'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import pandas as pd
import numpy as np
from PIL import Image
from robird.io import atomic_write_csv,atomic_write_json,sha256_file
from robird.rsos_suite_v1 import finish
from robird.artifact_tree_v1 import verify_tree
from robird.external_cohort_v1_1 import GateStop
from prepare_expansion_duplicate_review_v1 import identity
REPORT=ROOT/'code/results/external_expansion_duplicate_review_v1'
SOURCE=ROOT/'code/results/external_expansion_v2_1'

def components(photo_ids, edges):
    parent={int(p):int(p) for p in photo_ids}
    def find(p):
        if p not in parent:raise GateStop('Edge photo outside observation')
        while parent[p]!=p:
            parent[p]=parent[parent[p]];p=parent[p]
        return p
    for a,b in edges:
        x,y=find(int(a)),find(int(b))
        parent[max(x,y)]=min(x,y)
    groups={}
    for p in parent:groups.setdefault(find(p),[]).append(p)
    return sorted([sorted(v) for v in groups.values()])

def summarize(frame, taxa):
    groups=frame.groupby(['taxon_id','observation_id','observer_id']).size().reset_index(name='photos')
    support=groups.groupby('taxon_id').observer_id.nunique().reindex(taxa,fill_value=0)
    return dict(photos=len(frame),observations=len(groups),observers=int(frame.observer_id.nunique()),
                taxa_present=int(frame.taxon_id.nunique()),taxa_ge8=int((support>=8).sum()),
                taxa_ge10=int((support>=10).sum()),
                groups_ge_k={str(k):int((groups.photos>=k).sum()) for k in range(1,6)},
                support={str(int(k)):int(v) for k,v in support.items()})

def main():
    if (REPORT/'impact_done.json').exists():raise FileExistsError('Impact already sealed')
    verify_tree(REPORT/'review_done.json');verify_tree(SOURCE/'download_done.json')
    pairs=pd.read_csv(REPORT/'adjudicated_pairs.csv',keep_default_na=False,dtype=str)
    for column in ('pair_id','observation_id','reference_observation_id','photo_id','reference_photo_id'):
        pairs[column]=pd.Series([identity(v) for v in pairs[column]],dtype=object)
    frame=pd.read_csv(SOURCE/'downloaded.csv',keep_default_na=False)
    for column in ('taxon_id','observation_id','observer_id','photo_id'):
        frame[column]=frame[column].astype('int64')
    if frame.photo_id.duplicated().any():raise GateStop('Unexpected repeated photo IDs in source')
    if (frame.groupby('observation_id').taxon_id.nunique()!=1).any():raise GateStop('Ambiguous taxon')
    if (frame.groupby('observation_id').observer_id.nunique()!=1).any():raise GateStop('Ambiguous observer')
    for r in frame.itertuples():
        if sha256_file(Path(r.local_path))!=r.sha256:raise GateStop('Source image bytes changed')
    taxa=sorted(frame.taxon_id.unique());base=summarize(frame,taxa)
    scenario_names={'exact_only':{'EXACT_WITHIN_OBSERVATION'},
                    'exact_and_visual_same_frame':{'EXACT_WITHIN_OBSERVATION','NEAR_DUPLICATE_WITHIN_OBSERVATION'}}
    summaries={};rows=[];maps=[]
    for scenario,labels in scenario_names.items():
        flagged=pairs[pairs.decision.isin(labels)]
        kept_ids=set();affected=set();ineligible=set()
        for obs,g in frame.groupby('observation_id',sort=True):
            edges=flagged[flagged.observation_id.astype('int64')==int(obs)]
            comps=components(g.photo_id.tolist(),[(int(r.photo_id),int(r.reference_photo_id)) for r in edges.itertuples()])
            if len(comps)<len(g):
                affected.add(int(obs))
                rows.append(dict(scenario=scenario,observation_id=int(obs),taxon_id=int(g.taxon_id.iloc[0]),
                    scientific_name=g.scientific_name.iloc[0],observer_id=int(g.observer_id.iloc[0]),
                    photos_before=len(g),unique_components=len(comps),retained_if_min2=len(comps)>=2,
                    pair_ids=';'.join(str(int(x)) for x in edges.pair_id),
                    components=json.dumps(comps)))
            for comp in comps:
                for p in comp:
                    maps.append(dict(scenario=scenario,observation_id=int(obs),photo_id=p,
                                     representative_photo_id=min(comp),same_frame_component_size=len(comp),
                                     hypothetical_only=True))
            if len(comps)>=2:kept_ids.update(min(c) for c in comps)
            else:ineligible.add(int(obs))
        collapsed=frame[frame.photo_id.isin(kept_ids)]
        excluded=frame[~frame.observation_id.isin(affected)]
        sums=summarize(collapsed,taxa)
        sums.update(affected_observations=sorted(affected),ineligible_observations=sorted(ineligible),
                    repeated_photo_slots=len(frame)-sum(len(json.loads(r['components'])) for r in rows if r['scenario']==scenario)
                    -len(frame[~frame.observation_id.isin(affected)]),
                    entire_affected_observation_exclusion=summarize(excluded,taxa))
        summaries[scenario]=sums
    pixel_checks=[]
    for r in pairs[pairs.decision=='NEAR_DUPLICATE_WITHIN_OBSERVATION'].itertuples():
        with Image.open(r.path_a) as im:a=np.asarray(im.convert('RGB')).copy()
        with Image.open(r.path_b) as im:b=np.asarray(im.convert('RGB')).copy()
        same_shape=a.shape==b.shape
        pixel_checks.append(dict(pair_id=int(r.pair_id),same_rgb_shape=same_shape,
            decoded_rgb_equal=bool(same_shape and np.array_equal(a,b)),
            mean_abs_rgb_difference=float(np.abs(a.astype(float)-b.astype(float)).mean()) if same_shape else None,
            maximum_rgb_difference=int(np.abs(a.astype(int)-b.astype(int)).max()) if same_shape else None,
            automated_metric_used_as_decision_threshold=False))
    atomic_write_csv(REPORT/'duplicate_observation_impact.csv',pd.DataFrame(rows),refuse_if_exists=True)
    atomic_write_csv(REPORT/'hypothetical_photo_components.csv',pd.DataFrame(maps),refuse_if_exists=True)
    support_rows=[]
    for tid in taxa:
        rec=dict(taxon_id=int(tid),scientific_name=frame[frame.taxon_id==tid].scientific_name.iloc[0],
                 original_observers=base['support'][str(tid)])
        for name,s in summaries.items():
            rec[name+'_observers']=s['support'][str(tid)]
            rec[name+'_exclude_groups_observers']=s['entire_affected_observation_exclusion']['support'][str(tid)]
        support_rows.append(rec)
    atomic_write_csv(REPORT/'hypothetical_taxon_support.csv',pd.DataFrame(support_rows),refuse_if_exists=True)
    result=dict(status='READ_ONLY_DUPLICATE_IMPACT_COMPLETE',baseline=base,scenarios=summaries,
                near_duplicate_pixel_checks=pixel_checks,source_manifest_modified=False,
                deleted_images=0,model_features_extracted=0,evaluations_run=0,original_g6_pass=False,
                handling_policy_enacted=False,review_done_sha256=sha256_file(REPORT/'review_done.json'),
                source_manifest_sha256=sha256_file(SOURCE/'downloaded.csv'))
    atomic_write_json(REPORT/'impact_summary.json',result,refuse_if_exists=True)
    finish(REPORT/'impact_done.json',sha256_file(REPORT/'review_done.json'),
           [REPORT/'review_done.json',SOURCE/'download_done.json',SOURCE/'downloaded.csv',Path(__file__)]
           +[REPORT/p for p in ('duplicate_observation_impact.csv','hypothetical_photo_components.csv',
                               'hypothetical_taxon_support.csv','impact_summary.json')],
           read_only_counterfactual=True,model_evaluation_authorized=False)
    print(json.dumps({k:v for k,v in result.items() if k not in ('baseline','scenarios')},default=int))
    for name,value in [('baseline',base)]+list(summaries.items()):
        print(name,json.dumps({k:v for k,v in value.items() if k not in ('support','entire_affected_observation_exclusion')},default=int))

if __name__=='__main__':main()
