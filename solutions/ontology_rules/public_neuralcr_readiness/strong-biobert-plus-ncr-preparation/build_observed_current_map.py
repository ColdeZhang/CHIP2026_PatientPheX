"""Project the same public HPO mapping onto actual immutable native raw-span IDs only."""
from pathlib import Path
import argparse,collections,datetime,gzip,hashlib,json,re,time
import build_ncr_current_map as public_map

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):p=Path(p).resolve();return {'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--raw-collection',required=True);ap.add_argument('--finalization',required=True);ap.add_argument('--hpo',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();start=time.monotonic();out=Path(a.output);assert not out.exists() and not out.with_suffix(out.suffix+'.summary.json').exists();collection=json.loads(Path(a.raw_collection).read_text());final=json.loads(Path(a.finalization).read_text())
 assert collection['status']=='complete_local_raw_spans_remote_full_cache_proof' and final['status']=='complete_cache_verified_by_CPU_finalization_original_GPU_rc1_preserved' and collection['remote_full_cache_finalization_sha256']==sha(a.finalization)
 item=collection['verified']['native-raw-spans.jsonl.gz'];raw=Path(item['local_path']);assert raw.stat().st_size==item['bytes']==192972 and sha(raw)==item['sha256']==final['outputs']['native-raw-spans.jsonl.gz']['sha256'];assert final['cache_format']['columns_order_sha256']=='bb2edeb6af490408a31580d73f8959a34a2738884b1f79ab6328fdf4cd68e030'
 hpo=Path(a.hpo);assert hpo.stat().st_size==public_map.HPO_BYTES and sha(hpo)==public_map.HPO_SHA256
 assert sha(public_map.__file__)=='33d1c0afd00a98f7d4cc2b6091cf6ef87f634eab840f4de66a6da439c0d2a978'
 docs=[json.loads(x) for x in gzip.decompress(raw.read_bytes()).decode().splitlines()];assert len(docs)==5341==final['coverage']['native_raw_spans'];ids=list(dict.fromkeys(x['raw_hpo_id'] for x in docs));assert all(re.fullmatch(r'HP:[0-9]{7}',x) and x!='HP:0000118' for x in ids)
 terms,allowed,alts=public_map.parse_ontology(hpo.read_text());assert len(allowed)==19120;mapping=[]
 for raw_id in ids:
  target,reason=public_map.map_label(raw_id,terms,allowed,alts);mapping.append({'original_label':raw_id,'current_label':target,'reason':reason})
 report={'status':'complete_observed_raw_ID_current_HPO_mapping','scope':'Only model HPO IDs actually present in the immutable5341 native raw spans. Same public map_label function; not a local complete-head coverage check. No new classification, threshold, overlap or model head changes.','raw_spans':5341,'observed_raw_ids':len(ids),'observed_label_reason_counts':dict(collections.Counter(x['reason'] for x in mapping)),'mapped_observed_current_ids':len({x['current_label'] for x in mapping if x['current_label']}),'original_order_sha256':final['cache_format']['columns_order_sha256'],'remote_head_columns_file_sha256':final['outputs']['columns.json']['sha256'],'full_head_locally_read_or_verified':False,'remote_full_cache_finalization_sha256':sha(a.finalization),'HPO_sha256':sha(hpo),'refs':{'source':ref(__file__),'unchanged_public_mapping_source':ref(public_map.__file__),'raw_spans':ref(raw),'raw_collection':ref(a.raw_collection),'finalization':ref(a.finalization),'hpo':ref(hpo)},'no_gold_NN_fit_or_quality':True,'seconds_before_serialization':time.monotonic()-start,'finished_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 out.parent.mkdir(parents=True,exist_ok=True)
 with out.open('x') as f:json.dump(dict(report,mapping=mapping),f,indent=2);f.write('\n')
 report['mapping_file']=ref(out)
 with out.with_suffix(out.suffix+'.summary.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
 print(json.dumps({k:report[k] for k in ['status','raw_spans','observed_raw_ids','observed_label_reason_counts','seconds_before_serialization']}))
if __name__=='__main__':main()
