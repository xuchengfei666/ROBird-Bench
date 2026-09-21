"""Seal explicit visual decisions without altering the stopped experiment's gate."""
from pathlib import Path
import sys,json
from collections import Counter
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code/src'))
import pandas as pd
from robird.io import atomic_write_json,atomic_write_csv,sha256_file
from robird.rsos_suite_v1 import finish
from robird.external_cohort_v1_1 import GateStop
from robird.artifact_tree_v1 import verify_tree
from robird.external_expansion_v2 import verify_files
REPORT=ROOT/'code/results/external_expansion_duplicate_review_v1'
NOTES=ROOT/'code/configs/external_expansion_duplicate_decisions_v1.json'

def validate(index,notes):
    if notes['reviewer_type']!='AI_ASSISTED_VISUAL_REVIEW' or notes['independent_human_adjudication']:
        raise GateStop('Reviewer source must not pretend human ground truth')
    if not notes['outcome_blinded']:raise GateStop('Review must be outcome blinded')
    entries=notes['pairs']; n=len(index['pairs'])
    if len(entries)!=n or sorted(r['pair_id'] for r in entries)!=list(range(n)):
        raise GateStop('Missing/repeated/foreign pair decisions')
    lookup={r['pair_id']:r for r in entries}; records=[]
    for pair in index['pairs']:
        note=lookup[pair['pair_id']]; decision=note['decision']
        history=pair['source']=='history'; exact=pair['exact_bytes']; same=pair['same_observation']
        valid={
            'EXACT_WITHIN_OBSERVATION':exact and same and not history,
            'EXACT_CROSS_OBSERVATION':exact and not same and not history,
            'EXACT_HISTORY':exact and history,
            'DISTINCT_HISTORY':not exact and history,
            'DISTINCT_CROSS_OBSERVATION':not exact and not history and not same,
            'WITHIN_OBSERVATION_REDUNDANCY':not exact and not history and same,
            'NEAR_DUPLICATE_WITHIN_OBSERVATION':not exact and not history and same,
            'SUSPECT_CROSS_DUPLICATE':not exact and not same,
            'UNRESOLVED':not exact,
        }
        if not valid.get(decision,False) or not note['rationale'].strip():
            raise GateStop('Decision incompatible with pair or empty rationale: '+str(pair['pair_id']))
        records.append(dict(pair,decision=decision,rationale=note['rationale'],
                            reviewer_type=notes['reviewer_type'],independent_human_adjudication=False,
                            inspected_evidence=note.get('inspected_evidence',pair['sheet'])))
    return records

def verify_visual_evidence(index,details,records):
    panels={p['pair_id']:p for p in details['panels']}
    if len(panels)!=len(details['panels']):raise GateStop('Repeated detail panel IDs')
    for path,digest in index['sheets'].items():
        if sha256_file(Path(path))!=digest:raise GateStop('Review sheet changed')
    for p in panels.values():
        if sha256_file(Path(p['path']))!=p['sha256']:raise GateStop('Detail panel changed')
    for row in records:
        allowed={Path(row['sheet']).resolve()}
        if row['pair_id'] in panels:allowed.add(Path(panels[row['pair_id']]['path']).resolve())
        if Path(row['inspected_evidence']).resolve() not in allowed:
            raise GateStop('Evidence does not belong to pair: '+str(row['pair_id']))

def main():
    if (REPORT/'review_done.json').exists():raise FileExistsError('Review already sealed')
    index=json.loads((REPORT/'index.json').read_text(encoding='utf-8'))
    notes=json.loads(NOTES.read_text(encoding='utf-8'))
    records=validate(index,notes)
    details=json.loads((REPORT/'detail_index.json').read_text(encoding='utf-8'))
    if sha256_file(REPORT/'first_visual_pass.json')!=details['first_pass_sha256']:
        raise GateStop('First-pass record changed')
    verify_visual_evidence(index,details,records)
    freeze=ROOT/'FROZEN_EXTERNAL_EXPANSION_V2_1.json'
    if sha256_file(freeze)!=index['parent_freeze_sha256']:raise GateStop('Parent freeze changed')
    verify_files(json.loads(freeze.read_text(encoding='utf-8'))['files'])
    source=ROOT/'code/results/external_expansion_v2_1'
    for marker in ('metadata_done.json','download_done.json'):verify_tree(source/marker)
    if sha256_file(Path(index['source_csv']))!=index['source_sha256']:
        raise GateStop('Source candidate list changed')
    checked={}
    for row in records:
        for key in ('path_a','path_b'):
            if row[key] not in checked:checked[row[key]]=sha256_file(Path(row[key]))
            if checked[row[key]]!=row[key+'_sha256']:raise GateStop('Reviewed bytes changed')
    atomic_write_csv(REPORT/'adjudicated_pairs.csv',pd.DataFrame(records),refuse_if_exists=True)
    counts=dict(Counter(r['decision'] for r in records))
    summary=dict(status='REVIEW_COMPLETE_WITH_ACTION_ITEMS',pairs=len(records),decision_counts=counts,
        unresolved_pair_ids=[r['pair_id'] for r in records if r['decision'] in ('UNRESOLVED','SUSPECT_CROSS_DUPLICATE')],
        exact_duplicate_observations=sorted({r['observation_id'] for r in records if r['decision']=='EXACT_WITHIN_OBSERVATION'}),
        near_duplicate_observations=sorted({r['observation_id'] for r in records if r['decision']=='NEAR_DUPLICATE_WITHIN_OBSERVATION'}),
        reviewed_visual_not_human_gold_standard=True,original_g6_pass=False,
        original_duplicate_gate_overwritten=False,features_authorized_by_this_audit=False,
        deleted_images=0,notes_sha256=sha256_file(NOTES),index_sha256=sha256_file(REPORT/'index.json'),
        detail_index_sha256=sha256_file(REPORT/'detail_index.json'),detail_panels_verified=len(details['panels']),
        sources=dict(Counter(r['source'] for r in records)),
        cross_history_duplicates_confirmed=counts.get('EXACT_HISTORY',0),
        claim_limit='No duplicate confirmed among flagged cross-history/cross-observation pairs; not proof of exhaustive duplicate absence')
    atomic_write_json(REPORT/'review_summary.json',summary,refuse_if_exists=True)
    finish(REPORT/'review_done.json',sha256_file(NOTES),
        [REPORT/'index.json',REPORT/'preparation_audit.json',REPORT/'adjudicated_pairs.csv',REPORT/'review_summary.json',NOTES,
         ROOT/'EXTERNAL_EXPANSION_DUPLICATE_REVIEW_V1.md',Path(__file__),ROOT/'code/scripts/prepare_expansion_duplicate_review_v1.py',
         REPORT/'detail_index.json',REPORT/'first_visual_pass.json',
         ROOT/'code/scripts/render_expansion_review_details_v1.py',freeze,
         source/'metadata_done.json',source/'download_done.json',source/'duplicate_audit.json',
         source/'duplicate_review.csv',source/'queue_status.json']
        +[Path(p) for p in index['sheets']]+[Path(p['path']) for p in details['panels']],
        all_pairs_addressed=True,original_g6_pass=False,original_gate_unmodified=True)
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
