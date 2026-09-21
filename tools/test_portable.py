"""Run self-contained tests; report legacy fixture-dependent cases separately."""
from pathlib import Path
import json,os,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
EXCLUSIONS=[
 'test_full_cluster_real_manifest','test_review_all_65_exact_real_identities',
 'test_review_fails_closed','test_real_frozen_development_mapping_has_exactly_100_classes',
 'test_p0_v4_table_is_exact_deterministic_replacement',
 'test_p0_v4_replacement_is_first_unused_feasible_audit_candidate',
 'test_v5_1_binds_the_immutable_86_taxon_checkpoint',
 'test_v5_1_has_no_dataset_decision_or_download_path',
 'test_v5_2_first_reserve_is_scheduled_from_complete_census',
 'test_v5_2_temporary_table_has_78_seed_and_22_expansion_taxa',
 'test_v5_2_freeze_binds_complete_v5_1_evidence',
 'test_census_pool_is_exact_current_plus_all_frozen_feasible_reserves',
 'test_census_bootstrap_is_exact_terminal_v4_prefix',
 'test_census_freeze_contract_binds_v4_terminal_evidence_and_own_script']
env=os.environ.copy();env['PYTHONPATH']=str(ROOT/'code/src')
print('Excluded only because they require historical non-bundled census/review fixtures:',json.dumps(EXCLUSIONS),flush=True)
result=subprocess.run([sys.executable,'-m','pytest','code/tests','-q','-k','not ('+' or '.join(EXCLUSIONS)+')'],cwd=ROOT,env=env)
raise SystemExit(result.returncode)
