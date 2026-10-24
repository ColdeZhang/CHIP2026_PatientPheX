"""Two fixed final-epoch4 endpoints, common freeze before any new heldout quality."""
from pathlib import Path
import argparse,time,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from patient_first import validate_matrix
from common import *

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--controller',type=Path,required=True);a=ap.parse_args();p=read(a.plan);b=a.controller;start=time.time();deadline=start+120
 assert p['source_sha256']==ref(__file__)['sha256']
 for source in p['fixed_source_refs']:check(source)
 auth=read(b/'GPU-AUTHORIZATION.json');assert auth.get('F1_inference_and_quality_authorized') is True
 for k in ['strong_N','strong_metrics','matched_baseline','matched_metrics','old_P','old_P_metrics','prompts']:check(p[k])
 complete=read(b/'campaign-complete.json');assert complete['status']=='FOUR_COMPLETE_EPOCHS_64_STEPS_DURABLE_ACKS_VERIFIED'
 collection=read(b/'inference-collected.json');manifest=read(b/'inference-terminal-manifest.json');assert collection['manifest_sha256']==ref(b/'inference-terminal-manifest.json')['sha256']
 terminal=b/'inference-terminal'
 for r in manifest['files']:check(dict(path=str(terminal/r['name']),bytes=r['bytes'],sha256=r['sha256']))
 wrapper=read(terminal/'inference-control/wrapper-complete.json');assert wrapper['returncode']==0 and wrapper['complete80']
 raw=terminal/'inference-output/responses.jsonl';responses=docs(raw);prompts=docs(p['prompts']['path']);assert len(responses)==len(prompts)==80
 assert ref(raw)['sha256']==wrapper['responses_sha256']
 by={r['row_id']:r for r in responses};assert len(by)==80 and set(by)=={r['row_id'] for r in prompts}
 base=docs(p['strong_N']['path']);decisions={str(d['pmc_id']):{str(a['patient_id']):set() for a in d['association']} for d in base}
 assert len(base)==16 and sum(len(d['entities']) for d in base)==1189
 candidate_count=pair_count=0
 for row in prompts:
  r=by[row['row_id']];assert str(r['pmc_id'])==str(row['pmc_id']) and r['generated_token_ids'][-1] in (248044,248046) and not r['hit_token_limit']
  mapping=r['canonical_decisions'];assert set(mapping)==set(row['code_map'])
  for cid,pids in mapping.items():
   assert len(pids)==len(set(pids)) and set(pids)<=set(row['patient_ids'])
   for pid in pids:decisions[str(row['pmc_id'])][pid].add(row['code_map'][cid])
  candidate_count+=len(row['code_map']);pair_count+=len(row['code_map'])*len(row['patient_ids'])
 assert (candidate_count,pair_count)==(587,1557)
 out=b/'quality';out.mkdir(exist_ok=False);preds={}
 for arm,mapping in [('primary-N-union-P4',union(pair_map(base),decisions)),('diagnostic-P4-alone',decisions)]:
  folder=out/arm;folder.mkdir();preds[arm]=render(p['strong_N']['path'],mapping,folder/'pred.jsonl')
 save(out/'common-freeze.json',dict(at_unix=time.time(),plan=ref(a.plan),campaign=ref(b/'campaign-complete.json'),inference_manifest=ref(b/'inference-terminal-manifest.json'),predictions=preds,all_nonrelation_bytes_preserved=True,entities=1189,patients=39,documents=16,gold_read=False))
 save(out/'evaluation-started.json',dict(at_unix=time.time(),common_freeze=ref(out/'common-freeze.json'),new_calls=2,old_calls=0))
 n=read(p['strong_metrics']['path'])['metrics'];matched=read(p['matched_metrics']['path'])['metrics'];oldp=read(p['old_P_metrics']['path'])['metrics'];results={}
 for arm,item in preds.items():
  m=quality(p,item['path'],out/arm,deadline);metric=m['metrics'];assert metric['menF1']==n['menF1'] and metric['docF1']==n['docF1']
  results[arm]=dict(metrics=metric,counts=m['counts'],delta_vs_strong_N={k:v-n[k] for k,v in metric.items()},delta_vs_matched_N_P2={k:v-matched[k] for k,v in metric.items()},delta_vs_P2={k:v-oldp[k] for k,v in metric.items()},beats_strong=metric['Score']>n['Score'],metrics_ref=ref(out/arm/'metrics.json'))
 primary='primary-N-union-P4';diagnostic='diagnostic-P4-alone'
 mechanism=results[primary]['metrics']['Score']>max(n['Score'],matched['Score'])
 eligible=[arm for arm in [primary,diagnostic] if results[arm]['beats_strong'] and (arm!=primary or mechanism)]
 selected=max(eligible,key=lambda arm:(results[arm]['metrics']['Score'],arm==primary)) if eligible else None
 save(out/'result.json',dict(status='COMPLETE_FIXED_PATIENT4_F1',arms=results,primary_mechanism_pass=mechanism,selected_candidate=selected,keep=selected is not None,Score_only_macro_reported=True,strong_N_metrics=p['strong_metrics'],matched_N_P2_metrics=p['matched_metrics'],common_freeze=ref(out/'common-freeze.json'),new_quality_calls=2,old_quality_calls=0,automatic_F0_dev_official=False,seconds=time.time()-start));print(read(out/'result.json'))
if __name__=='__main__':main()
