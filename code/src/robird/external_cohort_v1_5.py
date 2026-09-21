"""Outcome-blinded adjudication and byte-reusing continuation; no image download."""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import json
import numpy as np
import pandas as pd
from robird import external_cohort_v1_1 as legacy
from robird import external_cohort_v1_3 as relations
from robird.external_cohort_v1_1 import GateStop, read_json, verify_tree, image_signature
from robird.io import atomic_write_csv, atomic_write_json, sha256_file
from robird.rsos_suite_v1 import marker, finish

PARENT_HASH = '27b23425efe59fc2db951fd4610efeb4d086c01970afd9f6e75458a57f7f8f8c'
PARENT_DATA = Path('E:/Datasets/ROBird-Bench/external_cohort_v1_4')
INDEX = Path('E:/Datasets/ROBird-Bench/external_duplicate_review_v1/index.json')
SHARED_OBS = {377824923, 377824924}
SHARED_PHOTOS = {690953732, 690959646, 690959649, 690960036, 690960029}
COUNTS = {'FALSE_POSITIVE_HISTORY':50, 'RETAIN_WITHIN_OBSERVATION_REDUNDANCY':12,
          'RETAIN_REVIEWED_SHARED_SCENE':3}
KEYS = ['photo_id','observation_id','reference_photo_id','reference_observation_id',
        'source','exact_bytes','dhash_distance']


def validate_decision(row, verdict):
    """An annotation never authorizes a different source or exact-byte history pair."""
    exact = str(row['exact_bytes']).lower() == 'true'
    obs, ref = str(row['observation_id']), str(row['reference_observation_id'])
    if verdict == 'FALSE_POSITIVE_HISTORY':
        valid = row['source'] == 'history' and not exact
    elif verdict == 'RETAIN_WITHIN_OBSERVATION_REDUNDANCY':
        valid = row['source'] == 'within_new_cohort' and obs == ref and not exact
    elif verdict == 'RETAIN_REVIEWED_SHARED_SCENE':
        valid = (row['source'] == 'within_new_cohort' and exact and
                 {int(obs),int(ref)} == SHARED_OBS and
                 int(row['photo_id']) in SHARED_PHOTOS and int(row['reference_photo_id']) in SHARED_PHOTOS)
    else:
        valid = False
    if not valid:
        raise GateStop('Annotation incompatible with source/identity: '+str(row))


def bind_review(candidates, old_candidates, index, notes, hasher=sha256_file):
    """Bind every explicit decision to both original reviewed and reused image bytes."""
    if len(candidates) != 65 or len(old_candidates) != 65 or len(index['pairs']) != 65:
        raise GateStop('Candidate set size changed')
    entries = notes['pairs']
    if len(entries) != 65 or sorted(d['pair_id'] for d in entries) != list(range(65)):
        raise GateStop('Missing or repeated pair decision')
    if notes['reviewer_type'] != 'AI_ASSISTED_VISUAL_REVIEW' or notes['independent_human_adjudication']:
        raise GateStop('Reviewer provenance changed')
    if not notes['outcome_blinded']:
        raise GateStop('Review not outcome-blinded')
    decisions = {d['pair_id']:d for d in entries}
    records = []
    checked = {}
    def digest(path):
        path = Path(path)
        if str(path) not in checked:
            checked[str(path)] = hasher(path)
        return checked[str(path)]
    for i, (row, old, seen) in enumerate(zip(candidates.to_dict('records'),
            old_candidates.to_dict('records'), index['pairs'])):
        if seen['pair_id'] != i or any(str(row[k]) != str(old[k]) for k in KEYS):
            raise GateStop('Review candidate order/identity changed')
        for key in ('photo_id','observation_id','reference_observation_id','source','exact_bytes','dhash_distance'):
            if str(seen[key]) != str(old[key]):
                raise GateStop('Contact sheet identity changed: '+key)
        decision = decisions[i]
        if not decision['rationale'].strip():
            raise GateStop('Empty pair rationale')
        validate_decision(row, decision['decision'])
        record = dict(row, pair_id=i, decision=decision['decision'], rationale=decision['rationale'],
                      reviewer_type=notes['reviewer_type'])
        for side in ('path_a','path_b'):
            if Path(old[side]).resolve() != Path(seen[side]).resolve():
                raise GateStop('Contact image path changed')
            expected = seen[side+'_sha256']
            if digest(old[side]) != expected or digest(row[side]) != expected:
                raise GateStop('Reviewed image bytes changed: '+str(row[side]))
            record[side+'_sha256'] = expected
            record[side+'_review_path'] = seen[side]
        if (record['path_a_sha256'] == record['path_b_sha256']) != (str(row['exact_bytes']).lower() == 'true'):
            raise GateStop('Exact-byte flag inconsistent')
        records.append(record)
    if dict(Counter(r['decision'] for r in records)) != COUNTS:
        raise GateStop('Unexpected review decision counts')
    return records


def exclude_cluster(frame, features=None):
    """Inference-only exclusion; same boolean mask for metadata and feature rows."""
    if features is not None and len(features) != len(frame):
        raise GateStop('Feature/manifest length mismatch')
    shared = frame[frame.observation_id.isin(SHARED_OBS)]
    if len(shared) != 6 or set(shared.photo_id) != SHARED_PHOTOS:
        raise GateStop('Shared cluster identity changed')
    if set(shared.observer_id) != {2132783}:
        raise GateStop('Shared observer changed')
    a, b = [shared[shared.observation_id == obs] for obs in sorted(SHARED_OBS)]
    if len(a) != 3 or len(b) != 3 or Counter(a.sha256) != Counter(b.sha256):
        raise GateStop('Shared cluster is not the same complete image-byte set')
    if {int(obs):int(part.taxon_id.iloc[0]) for obs,part in shared.groupby('observation_id')} != {377824923:4328,377824924:4381}:
        raise GateStop('Shared labels changed')
    keep = ~frame.observation_id.isin(SHARED_OBS).to_numpy()
    result = frame.loc[keep].reset_index(drop=True)
    return result if features is None else (result, np.asarray(features)[keep])


def inspect_inputs(root, config):
    """Read-only real-data integration gate, including every committed image ledger."""
    parent = root/'code/results/external_cohort_v1_4'
    if sha256_file(root/'FROZEN_EXTERNAL_COHORT_V1_4.json') != PARENT_HASH:
        raise GateStop('Parent freeze changed')
    for path in (parent/'metadata_done.json',parent/'download_done.json',PARENT_DATA/'reference_signatures_done.json'):
        if not marker(path,PARENT_HASH):
            raise GateStop('Missing completed parent stage')
        verify_tree(path)
    frame = pd.read_csv(parent/'downloaded.csv',keep_default_na=False,dtype={'dhash':str})
    metadata = pd.read_csv(parent/'metadata.csv',keep_default_na=False)
    original = pd.read_csv(root/'code/results/external_cohort_v1_3/downloaded.csv',keep_default_na=False,dtype={'dhash':str})
    pd.testing.assert_frame_equal(frame[metadata.columns],metadata)
    fields = [c for c in frame.columns if c not in ('local_path','downloaded_unix')]
    pd.testing.assert_frame_equal(frame[fields],original[fields])
    if (len(frame),frame.photo_id.nunique(),frame.observation_id.nunique(),frame.taxon_id.nunique(),frame.sha256.nunique()) != (970,969,339,13,967):
        raise GateStop('Unexpected source counts')
    history = [pd.read_csv(root/p,keep_default_na=False) for p in config['exclusion_manifests']]
    relations.assert_relation_metadata(frame,history)
    for row in frame.drop_duplicates('photo_id').itertuples():
        ledger = PARENT_DATA/'download_ledger'/f'{row.photo_id}.json'
        value = marker(ledger,PARENT_HASH)
        if not value or value['url'] != row.url:
            raise GateStop('Missing or mismatched image ledger')
        details = value['details']
        for key in ('sha256','width','height','dhash','local_path'):
            if str(details[key]) != str(getattr(row,key)):
                raise GateStop('Ledger/manifest mismatch: '+key)
        actual = image_signature(Path(row.local_path))
        if any(str(actual[k]) != str(getattr(row,k)) for k in actual):
            raise GateStop('Byte/decode audit mismatch')
    candidates_path = parent/'duplicate_review.csv'
    old_path = root/'code/results/external_cohort_v1_3/duplicate_review.csv'
    index = read_json(INDEX)
    if index['source_sha256'] != sha256_file(old_path):
        raise GateStop('Reviewed candidate source changed')
    notes_path = root/'code/configs/external_duplicate_decisions_v1_5.json'
    records = bind_review(pd.read_csv(candidates_path,keep_default_na=False,dtype=str),
        pd.read_csv(old_path,keep_default_na=False,dtype=str),index,read_json(notes_path))
    subset = exclude_cluster(frame)
    if (len(subset),subset.observation_id.nunique(),subset.taxon_id.nunique()) != (964,337,13):
        raise GateStop('Unexpected exclusion cohort')
    coverage = frame.groupby('taxon_id').observer_id.nunique()
    other = subset.groupby('taxon_id').observer_id.nunique()
    if coverage.loc[4328] != 10 or other.loc[4328] != 9:
        raise GateStop('Unexpected sensitivity observer coverage')
    result = dict(status='PASS_REVIEWED_CANDIDATES_BYTE_PROVENANCE',pairs=records,
        decision_counts=COUNTS,unresolved_pairs=0,reviewer_type='AI_ASSISTED_VISUAL_REVIEW',
        independent_human_adjudication=False,outcome_blinded=True,
        source_csv_sha256=sha256_file(candidates_path),contact_index_sha256=sha256_file(INDEX),
        decisions_sha256=sha256_file(notes_path),source_manifest_sha256=sha256_file(parent/'downloaded.csv'),
        full_relations=970,unique_photo_ids=969,unique_byte_hashes=967,full_groups=339,taxa=13,
        sensitivity_relations=964,sensitivity_groups=337,all_duplicates_proven_absent=False,
        original_g6_pass=False,new_image_downloads=0,original_files_deleted=0,
        claim='No historical leakage identified among these flagged pairs; not exhaustive absence proof')
    return frame,result


def data_gate(root, data, report, config, contract):
    done = report/'duplicate_done.json'
    frame, review = inspect_inputs(root,config)
    if review != read_json(report/'review.json'):
        raise GateStop('Frozen adjudication no longer matches inputs')
    if not marker(done,contract):
        atomic_write_csv(report/'reused_source_relations.csv',frame)
        atomic_write_csv(report/'sensitivity_relations.csv',exclude_cluster(frame))
        atomic_write_csv(report/'reviewed_pairs.csv',pd.DataFrame(review['pairs']))
        finish(done,contract,[report/'review.json',report/'reused_source_relations.csv',
            report/'sensitivity_relations.csv',report/'reviewed_pairs.csv'],
            status=review['status'],unresolved_pairs=0,new_image_downloads=0)
    return frame


def evaluate_both(frame, features, config, root, data, report, contract):
    legacy.evaluate_external(frame,features,config,root,data,report,contract)
    subset, aligned = exclude_cluster(frame,features)
    legacy.evaluate_external(subset,aligned,config,root,data/'sensitivity_no_cluster',
                             report/'sensitivity_no_cluster',contract)
    done = report/'evaluation_both_done.json'
    if not marker(done,contract):
        artifacts=[]
        for location in (report,report/'sensitivity_no_cluster'):
            for spec in config['runs']:
                directory=location/'runs'/spec['run_id']
                artifacts.append(directory/'eval_done.json')
                if spec['training_policy']=='all': artifacts.append(directory/'e3_done.json')
        finish(done,contract,artifacts,full_groups=339,sensitivity_groups=337,output_classes=100)


def quality_both(frame,data,report,contract):
    # Same data directory deliberately shares image-keyed quality ledger and weights.
    relations.quality_mode(frame,data,report,contract)
    relations.quality_mode(exclude_cluster(frame),data,report/'sensitivity_no_cluster',contract)
    done=report/'quality_both_done.json'
    if not marker(done,contract):
        finish(done,contract,[report/'quality_auto_done.json',report/'sensitivity_no_cluster/quality_auto_done.json'],
               human_quality_labels_pending=True)


def shared_diagnostic(frame,data,report,config,contract):
    done=report/'shared_inputs_done.json'
    if marker(done,contract): return
    image_hash = dict(zip(frame.photo_id.astype(int),frame.sha256))
    rows=[]
    for spec in config['runs']:
        predictions=pd.read_csv(data/'runs'/spec['run_id']/'predictions.csv')
        selected=predictions[predictions.observation_id.isin(SHARED_OBS)].copy()
        if len(selected)!=14: raise GateStop('Shared subset coverage differs from 2x7')
        selected['bytes_key']=selected.photo_ids.map(lambda s: '|'.join(sorted(image_hash[int(p)] for p in str(s).split(';'))))
        for (budget,key),part in selected.groupby(['budget','bytes_key']):
            if len(part)!=2 or set(part.observation_id)!=SHARED_OBS:
                raise GateStop('Shared byte subset mismatch')
            probabilities=np.stack(part.probabilities_json.map(json.loads))
            max_difference=float(np.max(np.abs(probabilities[0]-probabilities[1])))
            if not np.allclose(probabilities[0],probabilities[1],atol=2e-4,rtol=2e-4):
                raise GateStop('Identical input subsets have inconsistent probabilities')
            for row in part.itertuples():
                rows.append(dict(run_id=spec['run_id'],budget=int(budget),observation_id=row.observation_id,
                    label=row.label,prediction=row.prediction,correct=row.prediction==row.label,
                    bytes_key=key,pair_probability_max_abs_difference=max_difference,
                    same_input_target_ambiguity=True,optical_failure_claim=False))
    path=report/'shared_inputs_diagnostic.csv'
    atomic_write_csv(path,pd.DataFrame(rows));finish(done,contract,[path],source_labels_unchanged=True)


def compare_modes(report,config,contract):
    done=report/'comparison_done.json'
    if marker(done,contract): return
    records=[]
    for spec in config['runs']:
        full=pd.read_csv(report/'runs'/spec['run_id']/'groups.csv')
        sensitivity=pd.read_csv(report/'sensitivity_no_cluster/runs'/spec['run_id']/'groups.csv')
        for k,a in full.groupby('budget'):
            b=sensitivity[sensitivity.budget==k]
            paired=a.merge(b[['observation_id','expected_accuracy']],on='observation_id',
                           suffixes=('_full','_sensitivity'),validate='one_to_one')
            expected=set(a.observation_id)-SHARED_OBS
            if set(b.observation_id)!=expected or len(paired)!=len(b):
                raise GateStop('Common-observation sensitivity coverage mismatch')
            delta=paired.expected_accuracy_sensitivity-paired.expected_accuracy_full
            records.append(dict(run_id=spec['run_id'],model=spec['model'],seed=spec['seed'],
                training_policy=spec['training_policy'],budget=int(k),full_groups=len(a),
                sensitivity_groups=len(b),common_groups=len(paired),
                full_macro=float(a.groupby('label').expected_accuracy.mean().mean()),
                sensitivity_macro=float(b.groupby('label').expected_accuracy.mean().mean()),
                common_max_abs_delta=float(delta.abs().max()) if len(delta) else None,
                common_macro_delta=float(delta.groupby(paired.label).mean().mean()) if len(delta) else None))
    path=report/'full_vs_cluster_exclusion.csv';atomic_write_csv(path,pd.DataFrame(records))
    finish(done,contract,[path],interpretation='Whole-cohort differences reflect composition; full remains primary',
           sensitivity_taxon4328_observers=9,sensitivity_independently_passes_original_feasibility=False)


def statistics_both(root,report,config,contract):
    legacy.statistics(root,report,config,contract)
    legacy.statistics(root,report/'sensitivity_no_cluster',config,contract)
    compare_modes(report,config,contract)
