"""Bind completed input-only native/score artifacts to the already fixed endpoint."""
import argparse,json,sys
from pathlib import Path
COMMON=Path(__file__).resolve().parents[1]/'public_phenotagger_biobert_selector';sys.path.insert(0,str(COMMON))
from prepare_pool import sha,ref,checked,read,write,sources
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--phase',choices=['predict_dev','finalize_dev'],required=True);a=ap.parse_args()
 R=Path(a.root).resolve();V=R/'ontology_rules/public-biobert-dev-transfer';N=R/'research/public-phenotagger-biobert-dev16-cache';B=R/'sequence_crf/selector-nested-dev-transfer';src=Path(__file__).resolve().parent
 fit=read(V/'fit-plan.json');protocol=read(V/'fixed-dev-protocol.json');old=read(B/'plan.json');sources(fit)
 assert all(sha(k)==v for k,v in protocol['source_sha256'].items())
 p={k:fit[k] for k in ['python','sources','old_full64_fit','reference','ontology_source','training_review']};p['sources']=dict(p['sources'])
 for n in ['predict_dev.py','finalize_dev.py','bind_plan.py']:p['sources'][str(src/n)]=sha(src/n)
 p.update(status='fixed_single_dev_endpoint_actual_artifacts_bound',phase=a.phase,CPU_cap_seconds=30,record_dir=str(V/(a.phase+'-job')),fixed_protocol=ref(V/'fixed-dev-protocol.json'),fit_plan=ref(V/'fit-plan.json'),model_manifest=ref(V/'fitted/manifest.json'),cut=.3,parameters=protocol['parameters'])
 if a.phase=='predict_dev':
  # Completion/contract/reuse artifacts must exist and bind the actual cache.
  final=N/'terminal/native-dev16-final-cache';cm=read(final/'manifest.json');assert cm['input_sha256']==old['input']['sha256']
  p.update(output_dir=str(V/'predicted'),input=old['input'],old_scores=old['scores'],cache={role:ref(final/name) for role,name in [('manifest','manifest.json'),('canonical','canonical-entities.jsonl'),('provenance','source-provenance.jsonl')]})
  p['native_completion']=ref(N/'completion.json');p['native_terminal_contract']=ref(N/'terminal-input-contract-check.json');p['native_reuse_check']=ref(N/'reuse-check.json')
  assert read(N/'terminal/dev16-cache/run-state.json')['status']=='complete_cache_without_quality_evaluation'
 else:
  p.update(output_dir=str(V/'terminal'),score_manifest=ref(V/'predicted/manifest.json'),baseline=protocol['baseline'],baseline_metrics=protocol['baseline_metrics'],baseline_freeze=protocol['baseline_freeze'],hpo=old['hpo'],gold=old['gold'],nested_source=str(COMMON/'dependencies/solutions/sequence_crf'),evaluation_source=str(COMMON/'dependencies/scripts'))
  # Copy the preexisting gold reference; do not open gold while binding this plan.
  assert read(checked(p['score_manifest']))['dev_gold_read'] is False
 sources(p);target=V/(a.phase+'-plan.json');write(target,p);print(json.dumps(ref(target)))
if __name__=='__main__':main()
