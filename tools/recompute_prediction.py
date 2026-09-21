"""Portable CPU replay of fixed-subset accuracy; no fitted models or new inference."""
from pathlib import Path
import argparse,sys
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code/src'))
from robird.budget_metrics_v2_1 import validate_records,group_statistics
def main():
    p=argparse.ArgumentParser();p.add_argument('--predictions',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Choose a new output; existing results are not overwritten')
    raw=pd.read_csv(a.predictions);manifest=pd.read_csv(a.manifest)
    groups,_=group_statistics(validate_records(raw,manifest,num_classes=100))
    result=groups.groupby(['budget','label']).expected_accuracy.mean().groupby('budget').mean().to_frame('species_macro_accuracy')
    result['observation_micro_accuracy']=groups.groupby('budget').expected_accuracy.mean()
    for metric in ('top5','nll','brier'):
        result['observation_equal_'+metric]=groups.groupby('budget')['expected_'+metric].mean()
    assert result.shape==(5,5),'Unexpected budget metric shape'
    result.to_csv(a.output);print(result.to_string())
if __name__=='__main__':main()
