"""Validate completed immutable NCR artifacts after the terminal metadata-path exception."""
from pathlib import Path
import datetime,gzip,hashlib,json,os,re,sys,time,traceback
R=Path(__file__).resolve().parent;start=time.monotonic();assert os.environ['CUDA_VISIBLE_DEVICES']=='' and not(R/'finalization-result.json').exists();fp=json.loads((R/'finalization-plan.json').read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def ref(p):p=Path(p);return {'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)}
def check(r):p=Path(r['path']);assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256'];return p
report={'status':'running_CPU_read_only_finalization','started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_sha256':sha(__file__),'finalization_plan_sha256':sha(R/'finalization-plan.json'),'new_NN_forward_calls':0,'fit_or_quality':False,'original_failed_result_preserved':True}
try:
 for r in fp['required_files'].values():check(r)
 original=json.loads((R/'cache-result.json').read_text());job=json.loads((R/'run-state.json').read_text());plan=json.loads((R/'plan.json').read_text());prep=json.loads((R/'preparation.json').read_text());bound=json.loads((R/'bound-plan.json').read_text())
 assert original['status']=='failed_partial_preserved_not_complete' and 'FileNotFoundError' in original['error'] and "columns.json'" in original['traceback'] and job['returncode']==1
 assert original['completed_batches']==798 and original['completed_rows']==408272 and original['new_NN_forward_calls']==795 and original['new_NN_rows']==406736 and original['reused_NN_rows']==1536 and original['native_decode']['raw_native_spans']==5341 and original['native_decode']['candidate_occurrences']==683438
 assert original['runtime']['TF32_enabled'] is False and 'TF32_enabled_at_end' not in original
 fields=[x.strip() for x in job['release']['gpu'].split(',')];assert fields[1]==plan['gpu_uuid'] and int(fields[2])==0 and not any(x.split(',')[0].strip()==plan['gpu_uuid'] for x in job['release']['compute_apps'].splitlines())
 assert sha(R/'plan.json')==bound['plan_sha256'] and sha(R/'preparation.json')==bound['preparation_sha256']
 queries_path=check(prep['artifacts']['queries']);columns_path=check(prep['artifacts']['columns']);input_path=check(plan['input']);native_source=check(plan['native_source'])
 assert prep['artifacts']['queries']['sha256']==bound['query_sha256']==plan['expected_keys']['sha256'];head=json.loads(columns_path.read_text());labels=head['labels'];assert len(labels)==len(set(labels))==14261 and labels[-1]=='None' and hashlib.sha256('\n'.join(labels).encode()).hexdigest()==bound['columns_order_sha256']
 queries=[json.loads(x)['query'] for x in queries_path.read_text().splitlines()];assert len(queries)==len(set(queries))==408272 and queries==sorted(queries)
 import numpy as np
 assert np.__version__=='1.23.5';dtype=np.dtype([('column','<u2'),('score','<f4')],align=False);cache=np.load(R/'top1.npy',allow_pickle=False,mmap_mode='r');assert cache.shape==(408272,) and cache.dtype==dtype and dtype.itemsize==6
 assert (cache['column']<14261).all() and np.isfinite(cache['score']).all() and ((cache['score']>=0)&(cache['score']<=1)).all()
 raw=(R/'top1.npy').read_bytes();assert gzip.decompress((R/'top1.npy.gz').read_bytes())==raw
 progress=[json.loads(x) for x in (R/'batch-progress.jsonl').read_text().splitlines()];assert len(progress)==798
 for i,z in enumerate(progress):assert z['batch']==i and z['start']==i*512 and z['end']==min((i+1)*512,408272) and z['rows']==z['end']-z['start'] and z['origin']==('verified_probe_reuse' if i<3 else 'new_NN') and z['score_float32_bits_preserved'] is True
 times=original['NN_batch_seconds'];assert len(times)==795 and all(t>0 for t in times) and times==[x['seconds'] for x in progress[3:]]
 docs={x['pmc_id']:x for x in map(json.loads,input_path.read_text().splitlines())};assert len(docs)==64 and sum(len(d['full_text']) for d in docs.values())==2301
 from prepare_inputs import extract_native
 _,tokenize,_=extract_native(native_source.read_text());seen=set();n=0
 with gzip.open(R/'native-raw-spans.jsonl.gz','rt',encoding='utf-8') as f:
  for line in f:
   z=json.loads(line);p=docs[z['pmc_id']]['full_text'][z['passage_index']];begin=z['local_start'];end=z['local_end'];index=z['cache_row'];assert 0<=begin<end<=len(p['text']) and 0<=index<408272
   assert z['text']==p['text'][begin:end] and z['passage_offset']==p['offset'] and z['offset']==p['offset']+begin and z['length']==end-begin and z['passage_type']==p['type']
   row=cache[index];assert labels[int(row['column'])]==z['raw_hpo_id'] and z['raw_hpo_id'] not in [head['root_id'],'None'] and int(row['score'].view(np.uint32))==z['score_float32_bits']==int(np.float32(z['score']).view(np.uint32)) and row['score']>.8
   key=' '.join(tokenize(re.sub(r'[\\/\r\n\t-]',' ',z['text']))[:49]);assert key==queries[index];identity=(z['pmc_id'],z['passage_index'],begin,end,z['raw_hpo_id']);assert identity not in seen;seen.add(identity);n+=1
 assert n==5341 and len(original['weight_checks'])==60 and all(x['numeric_equal'] and x['dtype']=='float32' for x in original['weight_checks']) and original['parameters']['total']==491068890
 assert not any(n=='tensorflow' or n.startswith('tensorflow.') or n=='fasttext' for n in sys.modules)
 outputs={n:ref(R/n) for n in ['top1.npy','top1.npy.gz','batch-progress.jsonl','native-raw-spans.jsonl.gz']};outputs['queries.jsonl']=ref(queries_path);outputs['columns.json']=ref(columns_path)
 report.update(status='complete_cache_verified_by_CPU_finalization_original_GPU_rc1_preserved',finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),outputs=outputs,cache_format={'rows':408272,'dtype_descr':dtype.descr,'itemsize':6,'payload_bytes':2449632,'query_order_sha256':bound['query_sha256'],'columns_order_sha256':bound['columns_order_sha256']},coverage={'batches':798,'reused_batches':3,'reused_rows':1536,'new_NN_batches':795,'new_NN_rows':406736,'all_rows':408272,'batch_intervals_exact':True,'native_raw_spans':n,'all_span_text_offsets_query_HPO_scorebits_exact':True,'strict_threshold_checked_in_original_numpy':np.__version__},original_GPU_job={'main_seconds':job['gpu_elapsed_seconds'],'returncode':1,'release':job['release'],'original_report_sha256':sha(R/'cache-result.json')},TF32={'observed_initial':False,'end_measurement':'not executed due final output-ref exception; no value fabricated','source_control':'same frozen native run source disables TF32 and has no later mutation'},parameters=original['parameters'],actual_weight_comparison='All60 shapes/dtype/finite/numeric np.array_equal in original GPU receipt; not a new runtime byte hash',native_decode=original['native_decode'],NN_timing={'get_match_count':len(times),'sum':sum(times),'min':min(times),'max':max(times),'mean':sum(times)/len(times),'load_seconds':original['load_seconds'],'NN_elapsed_including_cache_writes':original['NN_elapsed_seconds_including_cache_writes'],'peak_allocator':original['TF_allocator_peak_memory']},query_metadata_locations='Exact original preparation paths reused; no copied aliases used to hide original exception',original_failed_result_still_sha256=sha(R/'cache-result.json'),CPU_numpy=np.__version__)
except BaseException as e:report.update(status='failed_CPU_finalization_preserved',error=repr(e),traceback=traceback.format_exc());raise
finally:
 report['elapsed_seconds']=time.monotonic()-start
 with (R/'finalization-result.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
 print(json.dumps({k:report.get(k) for k in ['status','elapsed_seconds','error']}))
