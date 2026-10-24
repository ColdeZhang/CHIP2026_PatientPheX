"""Two predetermined complete58 decoded endpoints, common freeze then exactly two v2."""
from pathlib import Path
import argparse,time
from common import *

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--inference',type=Path,required=True);a=ap.parse_args();p=read(a.plan);B=a.inference;start=time.time();deadline=start+120
 assert p['source_sha256']==ref(__file__)['sha256']
 for s in p['fixed_source_refs']:check(s)
 for k in ['strong_N','strong_metrics','prompts','policy']:check(p[k])
 policy=read(p['policy']['path']);assert policy['primary']=='coded-alone' and policy['diagnostic']=='N-union-coded'
 plan=read(B/'plan.json');assert plan['training_completed_epochs']==2 and plan['training_global_step']==32 and plan['cached_original_indices']==[] and plan['fresh_rows']==58
 manifest=read(B/'terminal-manifest.json');assert manifest['plan_sha256']==ref(B/'plan.json')['sha256']
 T=B/'terminal'
 for item in manifest['files']:check(dict(path=str(T/item['name']),bytes=item['bytes'],sha256=item['sha256']))
 w=read(T/'wrapper-complete.json');summary=read(T/'summary.json');assert w['returncode']==0 and summary['fresh_rows']==58 and summary['cached_rows']==0 and summary['expected_rows']==58
 assert not read(T/'gpu-release.json')['target_compute_rows']
 assert summary['responses_sha256']==ref(T/'responses.jsonl')['sha256'] and summary['canonical_sha256']==ref(T/'canonical.jsonl')['sha256']
 assert summary['source_sha256']==plan['files']['infer_full58.py']['sha256'] and summary['plan_sha256']==ref(B/'plan.json')['sha256']
 prompts=docs(p['prompts']['path']);responses=docs(T/'responses.jsonl');canonical=docs(T/'canonical.jsonl');assert len(prompts)==len(responses)==len(canonical)==58
 rm={r['row_id']:r for r in responses};cm={r['row_id']:r for r in canonical};assert len(rm)==len(cm)==58 and set(rm)==set(cm)=={r['row_id'] for r in prompts}
 base=docs(p['strong_N']['path']);decisions={str(d['pmc_id']):{str(x['patient_id']):set() for x in d['association']} for d in base}
 assert (len(base),sum(len(d['entities']) for d in base),sum(len(d['association']) for d in base))==(16,947,47)
 candidates=pairs=abstentions=0
 for i,row in enumerate(prompts):
  c=cm[row['row_id']];r=rm[row['row_id']];assert str(c['pmc_id'])==str(row['pmc_id'])==str(r['pmc_id'])
  assert read(T/f'canonical-records/{i:03d}.json')==c and read(T/f'raw-responses/{i:03d}.json')==r
  ids=read(T/f'generated-ids/{i:03d}.json');assert ids['row_id']==row['row_id'] and ids['natural_eos']==r['natural_eos'] and ids['hit_token_limit']==r['hit_token_limit']
  assert len(ids['generated_ids'])==r['output_tokens'] and c['provenance']=='fresh'
  m=c['candidate_patients']
  if c['strict_complete']:
   assert c['abstention'] is None and r['natural_eos'] and not r['hit_token_limit'];assert set(m)==set(row['code_map'])
   for cid,pids in m.items():
    assert len(pids)==len(set(pids)) and set(pids)<=set(row['patient_ids'])
    for pid in pids:decisions[str(row['pmc_id'])][pid].add(row['code_map'][cid])
  else:assert m=={} and c['abstention'];abstentions+=1
  candidates+=len(row['code_map']);pairs+=len(row['code_map'])*len(row['patient_ids'])
 assert (candidates,pairs)==(415,1166) and abstentions==summary['whole_row_abstentions']
 out=B/'quality';out.mkdir(exist_ok=False);preds={}
 for arm,mapping in [('coded-alone',decisions),('N-union-coded',union(pair_map(base),decisions))]:
  folder=out/arm;folder.mkdir();preds[arm]=render(p['strong_N']['path'],mapping,folder/'pred.jsonl')
 save(out/'common-freeze.json',dict(at_unix=time.time(),plan=ref(a.plan),inference_manifest=ref(B/'terminal-manifest.json'),predictions=preds,whole_row_abstentions=abstentions,all_nonrelation_bytes_preserved=True,entities=947,patients=47,documents=16,gold_read=False))
 save(out/'evaluation-started.json',dict(at_unix=time.time(),common_freeze=ref(out/'common-freeze.json'),new_calls=2,old_calls=0))
 n=read(p['strong_metrics']['path'])['metrics'];assert n['Score']==policy['strong_N_Score'];results={}
 for arm,item in preds.items():
  q=quality(p,item['path'],out/arm,deadline);m=q['metrics'];assert m['menF1']==n['menF1'] and m['docF1']==n['docF1']
  results[arm]=dict(metrics=m,counts=q['counts'],delta_vs_strong_N={k:v-n[k] for k,v in m.items()},beats_strong=m['Score']>n['Score'],delta_vs_historical_fullsupport=m['Score']-policy['historical_fullsupport_Score'],delta_vs_root2of3=m['Score']-policy['root2of3_Score'],beats_root2of3=m['Score']>policy['root2of3_Score'],beats_historical_fullsupport=m['Score']>policy['historical_fullsupport_Score'],metrics_ref=ref(out/arm/'metrics.json'))
 eligible=[arm for arm in ['coded-alone','N-union-coded'] if results[arm]['beats_strong']];chosen=max(eligible,key=lambda arm:(results[arm]['metrics']['Score'],arm=='coded-alone')) if eligible else None
 save(out/'result.json',dict(status='COMPLETE_FIXED_RESPONSE_CODE_F0',arms=results,primary_pass=results['coded-alone']['beats_strong'],selected_candidate=chosen,keep=chosen is not None,Score_only_macro_reported=True,common_freeze=ref(out/'common-freeze.json'),new_quality_calls=2,old_quality_calls=0,automatic_otherfold_dev_official=False,seconds=time.time()-start));print(read(out/'result.json'))
if __name__=='__main__':main()
