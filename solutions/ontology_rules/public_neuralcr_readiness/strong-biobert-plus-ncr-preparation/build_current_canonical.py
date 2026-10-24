"""Post-native public HPO mapping and exact-key NCR provenance, without labels or fitting."""
from pathlib import Path
import argparse,collections,datetime,gzip,hashlib,json,math,struct,time,traceback

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def ref(p):p=Path(p).resolve();return {'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)}
def check(r):p=Path(r['path']);assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256'];return p

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);args=ap.parse_args();started=time.monotonic();plan=json.loads(Path(args.plan).read_text());out=Path(plan['output_dir']);out.mkdir(parents=True,exist_ok=False);report={'status':'preparing_no_quality','started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source':ref(__file__),'plan':ref(args.plan),'NN_fit_quality':False}
 try:
  assert report['source']['sha256']==plan['source_sha256'];paths={n:check(r) for n,r in plan['refs'].items()};final=json.loads(paths['finalization'].read_text());collection=json.loads(paths['collection'].read_text());mapping=json.loads(paths['mapping'].read_text())
  assert final['status']=='complete_cache_verified_by_CPU_finalization_original_GPU_rc1_preserved' and collection['status'] in ['complete_all_compact_artifacts_local_full_SHA','complete_local_quality_inputs_remote_full_cache_proof']
  if collection['status']=='complete_local_quality_inputs_remote_full_cache_proof':assert collection['remote_full_cache_finalization_sha256']==plan['refs']['finalization']['sha256'] and collection['verified']['columns.json']['sha256']==final['outputs']['columns.json']['sha256']
  assert final['outputs']['native-raw-spans.jsonl.gz']['sha256']==plan['refs']['raw_spans']['sha256']==collection['verified']['native-raw-spans.jsonl.gz']['sha256']
  assert mapping['status']=='complete_ontology_only_original_NCR_head_mapping' and mapping['columns']==14261 and mapping['HPO_sha256']=='a5092cbdf605f568403cf7380d9173014015692433b2cc631bc5c1b053876b1b'
  assert mapping['original_order_sha256']==final['cache_format']['columns_order_sha256']=='bb2edeb6af490408a31580d73f8959a34a2738884b1f79ab6328fdf4cd68e030'
  byid={r['original_label']:r for r in mapping['mapping']};assert len(byid)==14261 and [r['column'] for r in mapping['mapping']]==list(range(14261))
  docs={d['pmc_id']:d for d in map(json.loads,paths['input'].read_text().splitlines())};assert len(docs)==64 and all(set(d)=={'pmc_id','pmid','patient','full_text'} for d in docs.values())
  assert plan['refs']['input']['sha256']=='ee2523843ff61f524db313fb2f9c37707a858e40f252870a7a05f4e7c1b3c246'
  grouped={};reasons=collections.Counter();raw_count=0;mapped_count=0
  with gzip.open(paths['raw_spans'],'rt',encoding='utf-8') as f:
   for line in f:
    z=json.loads(line);raw_count+=1;row=byid[z['raw_hpo_id']];target=row['current_label'];reasons[row['reason']]+=1
    bits=struct.unpack('<I',struct.pack('<f',z['score']))[0];assert bits==z['score_float32_bits'] and math.isfinite(z['score']) and 0<=z['score']<=1
    p=docs[z['pmc_id']]['full_text'][z['passage_index']];assert z['text']==p['text'][z['local_start']:z['local_end']] and z['offset']==p['offset']+z['local_start'] and z['length']==z['local_end']-z['local_start']
    if target is None:continue
    mapped_count+=1;k=(z['pmc_id'],z['offset'],z['length'],target)
    if k not in grouped:grouped[k]={'candidate':dict(pmc_id=k[0],offset=k[1],length=k[2],identifier=k[3],text=z['text'],ncr_present=True,ncr_score=z['score']),'origins':[],'score_float32_bits':bits}
    g=grouped[k];assert g['candidate']['text']==z['text'];g['origins'].append({'raw_hpo_id':z['raw_hpo_id'],'cache_row':z['cache_row'],'score':z['score'],'score_float32_bits':bits,'passage_index':z['passage_index'],'mapping_reason':row['reason']})
    if z['score']>g['candidate']['ncr_score']:g['candidate']['ncr_score']=z['score'];g['score_float32_bits']=bits
  assert raw_count==5341==final['coverage']['native_raw_spans'] and len(grouped)<=mapped_count<=raw_count
  canonical=out/'canonical-entities.jsonl';provenance=out/'source-provenance.jsonl.gz'
  with canonical.open('x',encoding='utf-8') as c,gzip.open(provenance,'xt',encoding='utf-8') as p:
   for g in grouped.values():
    x=g['candidate'];assert struct.unpack('<I',struct.pack('<f',x['ncr_score']))[0]==g['score_float32_bits'] and x['ncr_score']==max(z['score'] for z in g['origins']);c.write(json.dumps(x,ensure_ascii=False,separators=(',',':'))+'\n');p.write(json.dumps(dict(x,score_float32_bits=g['score_float32_bits'],origins=g['origins']),ensure_ascii=False,separators=(',',':'))+'\n')
  refs={**plan['refs'],'source':ref(__file__),'plan':ref(args.plan),'provenance':ref(provenance)}
  report.update(status='complete_NCR_current_canonical_from_verified_raw_cache_no_quality',canonical=ref(canonical),refs=refs,documents=64,raw_spans=raw_count,mapped_raw_spans=mapped_count,unmapped_raw_spans=raw_count-mapped_count,canonical_keys=len(grouped),canonical_collapse_count=mapped_count-len(grouped),mapping_reasons=dict(reasons),feature_names=['ncr_present','ncr_score'],score_policy='Maximum actual original float32 score on exact current key; preserved selected bits and all raw origins. No new threshold/overlap/renormalization.',order='First appearance in immutable raw native spans',current_HPO_mapping_after_native=True,public_model_training_on_task=False,no_gold=True,finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
 except BaseException as e:report.update(status='failed_current_canonical_preparation_preserved',error=repr(e),traceback=traceback.format_exc());raise
 finally:
  report['seconds']=time.monotonic()-started
  with (out/'manifest.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
  print(json.dumps({k:report.get(k) for k in ['status','raw_spans','canonical_keys','seconds','error']}))
if __name__=='__main__':main()
