"""Verify present public files and headline claims, explicitly reporting absent bulk files."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def main():
    m=json.loads((ROOT/'FILE_MANIFEST.json').read_text());checked=0;absent=0
    for item in m['files']:
        p=ROOT/item['path']
        if not p.exists():
            assert item.get('asset'),'Missing Git-tree file: '+item['path'];absent+=1;continue
        assert sha(p)==item['sha256'],'Hash mismatch: '+item['path'];checked+=1
    expected={'development_photos.csv':(14092,5000,100),'external_primary.csv':(3090,1121,56),'external_exclude_affected.csv':(3063,1112,56),'matched_common.csv':(1025,474,53)}
    for name,counts in expected.items():
        d=pd.read_csv(ROOT/'data/manifests'/name)
        assert (len(d),d.observation_id.nunique(),d.taxon_id.nunique())==counts,(name,counts)
        assert d.license_code.notna().all() and d.attribution.notna().all()
    classes=pd.read_csv(ROOT/'data/manifests/classes100.csv');assert len(classes)==100
    gains=pd.read_csv(ROOT/'figure_source_data/gains.csv');assert len(gains)==6 and gains.delta.gt(0).all()
    tests=pd.read_csv(ROOT/'results/explanatory_suite_v1_1/paired_tests_holm16.csv')
    ds=tests.query("backbone=='dinov2' and model=='deepsets' and contrast=='native_minus_pool' and budget==2").iloc[0]
    assert np.isclose(ds.delta*100,.145,atol=.001) and ds.p_holm16==1
    curves=pd.read_csv(ROOT/'figure_source_data/curves.csv').query("backbone=='dinov2' and model=='deepsets' and budget==2").set_index('method')
    rescue=curves.all_single_wrong_set_correct;destruction=curves.any_single_correct_set_wrong
    assert np.isclose((rescue['native']-rescue['probability_pool'])-(destruction['native']-destruction['probability_pool']),ds.delta,atol=1e-10)
    result=dict(status='PASS_PRESENT_FILES_AND_SPECIFIED_HEADLINE_CHECKS',checked_files=checked,bulk_files_not_downloaded=absent,cohort_supports=expected,full_retraining_test=False)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
