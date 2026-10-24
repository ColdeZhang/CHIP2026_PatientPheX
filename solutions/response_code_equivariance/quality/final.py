"""Fixed cyclic1 endpoints; complete common freeze precedes any label access."""
from pathlib import Path
import argparse, time
from common import *

def intersect(a,b):
 assert set(a)==set(b)
 assert all(set(a[k])==set(b[k]) for k in a)
 return {k:{p:a[k][p]&b[k][p] for p in a[k]} for k in a}

def preserve_render(template,mapping,destination):
 lines=Path(template).read_text().splitlines(keepends=True); result=[]
 for line in lines:
  d=json.loads(line); pmc=str(d['pmc_id'])
  assert set(mapping[pmc])=={str(x['patient_id']) for x in d['association']}
  for item in d['association']:
   target=mapping[pmc][str(item['patient_id'])]; old=item['phenotype']
   item['phenotype']=[x for x in old if x in target]+sorted(target-set(old))
  begin,end=association_span(line)
  result.append(line[:begin]+json.dumps(d['association'],ensure_ascii=False)+line[end:])
 assert nonrelation_bytes(lines)==nonrelation_bytes(result)
 with Path(destination).open('x') as f:f.write(''.join(result));f.flush();os.fsync(f.fileno())
 return ref(destination)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True,type=Path);ap.add_argument('--inference',required=True,type=Path);a=ap.parse_args()
 started=time.time(); deadline=started+120; p=read(a.plan); B=a.inference; T=B/'terminal'
 assert ref(__file__)['sha256']==p['source_sha256']
 for s in p['fixed_source_refs']:check(s)
 for k in ['B4_pred','B4_metrics','original_C_strict_pred','prompts','policy','cache_refs']:check(p[k])
 policy=read(p['policy']['path']); assert policy['B4_Score']==.701325377212172
 plan=read(B/'plan.json'); assert (plan['fresh_rows'],len(plan['cached_original_indices']))==(53,27)
 manifest=read(B/'terminal-manifest.json'); assert manifest['plan_sha256']==ref(B/'plan.json')['sha256']
 for r in manifest['files']:check(dict(path=str(T/r['name']),bytes=r['bytes'],sha256=r['sha256']))
 wrapper=read(T/'wrapper-complete.json');s=read(T/'summary.json')
 assert wrapper['returncode']==0 and not read(T/'gpu-release.json')['target_compute_rows']
 assert (s['expected_rows'],s['fresh_rows'],s['cached_rows'])==(80,53,27)
 assert s['plan_sha256']==ref(B/'plan.json')['sha256'] and s['source_sha256']==plan['files']['infer_full80.py']['sha256']
 assert s['responses_sha256']==ref(T/'responses.jsonl')['sha256'] and s['canonical_sha256']==ref(T/'canonical.jsonl')['sha256']
 rows=docs(p['prompts']['path']); raws=docs(T/'responses.jsonl'); canon=docs(T/'canonical.jsonl')
 assert len(rows)==len(raws)==len(canon)==80 and len({r['row_id'] for r in canon})==80
 base=docs(p['B4_pred']['path']); bm=pair_map(base); old=pair_map(docs(p['original_C_strict_pred']['path']))
 assert set(old)==set(bm) and all(set(old[k])==set(bm[k]) for k in bm)
 assert (len(base),sum(len(d['entities']) for d in base),sum(len(d['association']) for d in base))==(16,1189,39)
 decisions={k:{pid:set() for pid in bm[k]} for k in bm}; cached=set(plan['cached_original_indices'])
 cref={r['original_index']:r for r in read(p['cache_refs']['path'])['cache']}; assert set(cref)==cached
 abstentions=candidates=pairs=0
 for i,(row,r,c) in enumerate(zip(rows,raws,canon)):
  assert row['row_id']==r['row_id']==c['row_id'];assert str(row['pmc_id'])==str(r['pmc_id'])==str(c['pmc_id'])
  assert read(T/f'raw-responses/{i:03d}.json')==r and read(T/f'canonical-records/{i:03d}.json')==c
  ids=read(T/f'generated-ids/{i:03d}.json');assert ids['row_id']==row['row_id'] and len(ids['generated_ids'])==r['output_tokens']
  assert ids['natural_eos']==r['natural_eos'] and ids['hit_token_limit']==r['hit_token_limit']
  if i in cached:
   assert len(row['patient_ids'])==1 and s['row_provenance'][str(i)]=='exact_original_C_single_patient_cache'
   for folder,src in cref[i]['files'].items():
    assert ref(T/f'{folder}/{i:03d}.json')['sha256']==src['sha256']
  else:assert c['provenance']=='fresh' and s['row_provenance'][str(i)]=='fresh_cyclic1'
  m=c['candidate_patients']
  if c['strict_complete']:
   assert c['abstention'] is None and r['natural_eos'] and not r['hit_token_limit'] and set(m)==set(row['code_map'])
   for cid,pids in m.items():
    assert len(pids)==len(set(pids)) and set(pids)<=set(row['patient_ids'])
    for pid in pids:decisions[str(row['pmc_id'])][pid].add(row['code_map'][cid])
  else:assert m=={} and c['abstention'];abstentions+=1
  candidates+=len(row['code_map']);pairs+=len(row['code_map'])*len(row['patient_ids'])
 assert (candidates,pairs)==(587,1557) and abstentions==s['whole_row_abstentions']
 maps={'primary-B4-union-agreement':union(bm,intersect(old,decisions)),'diagnostic-cyclic1-alone':decisions}
 out=B/'quality';out.mkdir(exist_ok=False);preds={};unchanged={}
 for arm,m in maps.items():
  folder=out/arm;folder.mkdir();unchanged[arm]=m==bm
  if m==bm:
   with (folder/'pred.jsonl').open('xb') as f:f.write(Path(p['B4_pred']['path']).read_bytes());f.flush();os.fsync(f.fileno())
   preds[arm]=ref(folder/'pred.jsonl')
  else:preds[arm]=preserve_render(p['B4_pred']['path'],m,folder/'pred.jsonl')
 save(out/'common-freeze.json',dict(at_unix=time.time(),plan=ref(a.plan),inference_manifest=ref(B/'terminal-manifest.json'),predictions=preds,primary_byteexact_B4=unchanged['primary-B4-union-agreement'],whole_row_abstentions=abstentions,all_nonrelation_bytes_preserved=True,entities=1189,patients=39,documents=16,gold_read=False,permutation='cyclic1_only'))
 count=1+int(not unchanged['primary-B4-union-agreement'])
 save(out/'evaluation-started.json',dict(at_unix=time.time(),common_freeze=ref(out/'common-freeze.json'),new_calls=count,old_calls=0))
 baseline=read(p['B4_metrics']['path']);n=baseline['metrics'];assert n['Score']==policy['B4_Score']
 assert baseline['prediction_sha256']==p['B4_pred']['sha256'] and baseline['gold_sha256']==p['gold']['sha256']
 results={}
 for arm,item in preds.items():
  inherited=arm=='primary-B4-union-agreement' and unchanged[arm]
  if inherited:q=baseline;save(out/arm/'inherited-metrics.json',dict(source=p['B4_metrics'],prediction=item,quality_calls=0,metrics=q['metrics'],counts=q['counts']))
  else:q=quality(p,item['path'],out/arm,deadline)
  m=q['metrics'];assert m['menF1']==n['menF1'] and m['docF1']==n['docF1']
  results[arm]=dict(metrics=m,counts=q['counts'],deltas={k:v-n[k] for k,v in m.items()},beats_B4=m['Score']>n['Score'],quality_calls=0 if inherited else 1,metrics_ref=ref(out/arm/('inherited-metrics.json' if inherited else 'metrics.json')))
 save(out/'result.json',dict(status='COMPLETE_FIXED_CYCLIC1_F1',arms=results,primary_pass=results['primary-B4-union-agreement']['beats_B4'],diagnostic_candidate_report_only=results['diagnostic-cyclic1-alone']['beats_B4'],common_freeze=ref(out/'common-freeze.json'),new_quality_calls=count,old_quality_calls=0,automatic_F0_dev_official=False,scope='mapping plus host/runtime development control',seconds=time.time()-started));print(json.dumps(read(out/'result.json')))

if __name__=='__main__':main()
