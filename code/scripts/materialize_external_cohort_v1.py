from __future__ import annotations
import json, hashlib
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
META = Path('E:/Datasets/ROBird-Bench/rsos_serial_v1/holdout_metadata')
OUT = ROOT/'code/results/external_cohort_v1'
MANIFEST = ROOT/'code/data/manifests/external_cohort_v1.csv'
FEAS = OUT/'feasibility_100_taxa.csv'
AUDIT = OUT/'audit.json'

def sha(p):
    h=hashlib.sha256();
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def main():
    if AUDIT.exists() or MANIFEST.exists(): raise FileExistsError('external cohort outputs already exist')
    rows=[]; selected=[]
    for p in sorted(META.glob('taxon_*.json')):
        j=json.loads(p.read_text(encoding='utf-8'))['result']
        obs=int(j.get('independent_observers',0)); feasible=obs>=10
        rows.append({'taxon_id':int(j['taxon_id']),'candidate_groups':int(j.get('candidate_groups',0)),
                     'independent_observers':obs,'shortfall':int(j.get('shortfall',max(0,10-obs))),
                     'metadata_feasible':feasible,'source_file':str(p)})
        if feasible: selected.append(j)
    table=pd.DataFrame(rows).sort_values('taxon_id')
    if len(table)!=100 or len(selected)!=13: raise RuntimeError(f'unexpected census shape {len(table)} / {len(selected)}')
    OUT.mkdir(parents=True,exist_ok=True); FEAS.parent.mkdir(parents=True,exist_ok=True)
    table.to_csv(FEAS,index=False)
    # Keep one row per observation/photo, preserving the source observation identity.
    out=[]
    for j in sorted(selected,key=lambda x:int(x['taxon_id'])):
        for g in j.get('candidates',[]):
            photos=g.get('photos',[])
            if not isinstance(photos,list) or not 2<=len(photos)<=5: continue
            for pos,photo in enumerate(photos):
                out.append({'taxon_id':int(j['taxon_id']),'observation_id':int(g['observation_id']),
                            'observer_id':int(g['observer_id']),'photo_position':pos,
                            'photo_id':int(photo['id']),'license_code':str(photo.get('license_code','')),
                            'url':str(photo.get('url','')),'width':int(photo.get('original_dimensions',{}).get('width',0)),
                            'height':int(photo.get('original_dimensions',{}).get('height',0)),
                            'selection_basis':'metadata_feasibility_after_census'})
    manifest=pd.DataFrame(out).sort_values(['taxon_id','observation_id','photo_position'])
    manifest.to_csv(MANIFEST,index=False)
    audit={'status':'MATERIALIZED_EXTERNAL_FEASIBLE_COHORT_V1','claim_scope':'EXPLORATORY_PROSPECTIVE_EXTERNAL_COHORT',
           'taxa_total':100,'feasible_taxa':len(selected),'photos_manifested':len(manifest),
           'groups_manifested':int(manifest.observation_id.nunique()),'shortfall_taxa':int((~table.metadata_feasible).sum()),
           'feasibility_sha256':sha(FEAS),'manifest_sha256':sha(MANIFEST),'images_downloaded':0}
    AUDIT.write_text(json.dumps(audit,indent=2),encoding='utf-8'); print(json.dumps(audit))
if __name__=='__main__': main()
