"""Full frozen input-only NCR cache: fixed top1 numerics and unchanged native raw-span decoding."""
from pathlib import Path
import datetime,gzip,hashlib,json,os,re,sys,time,traceback,types
R=Path(__file__).resolve().parent;started=time.monotonic();plan=json.loads((R/'plan.json').read_text());bound=json.loads((R/'bound-plan.json').read_text())
assert os.environ.get('CUDA_VISIBLE_DEVICES')==plan['gpu_uuid'] and not(R/'cache-result.json').exists()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def ref(p):return {'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)}
def write(n,v):
 p=R/n;t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)
report={'status':'running_partial_not_complete','started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_sha256':sha(__file__),'plan_sha256':sha(R/'plan.json'),'bound_plan_sha256':sha(R/'bound-plan.json'),'no_gold_fit_or_quality':True,'completed_batches':0,'completed_rows':0,'new_NN_rows':0,'reused_NN_rows':0,'NN_batch_seconds':[]}
def save():report['elapsed_seconds']=time.monotonic()-started;write('cache-state.json',report)
save()
try:
 assert bound['plan_sha256']==sha(R/'plan.json') and bound['preparation_sha256']==sha(R/'preparation.json')
 prep=json.loads((R/'preparation.json').read_text());assert prep['status']=='complete_input_only_preparation'
 for role,item in plan['model_files'].items():assert Path(item['path']).stat().st_size==item['bytes'] and sha(item['path'])==item['sha256'],role
 for item in [prep['artifacts']['queries'],prep['artifacts']['columns'],plan['input']]:assert Path(item['path']).stat().st_size==item['bytes'] and sha(item['path'])==item['sha256']
 input_docs=input_docs;assert len(input_docs)==64 and all('type' in p and type(p['offset']) is int and isinstance(p['text'],str) for d in input_docs for p in d['full_text'])
 assert prep['artifacts']['queries']['sha256']==plan['expected_keys']['sha256']==bound['query_sha256']
 assert prep['head']['order_sha256']==bound['columns_order_sha256']
 queries=[json.loads(line)['query'] for line in Path(prep['artifacts']['queries']['path']).read_text().splitlines()];assert len(queries)==len(set(queries))==408272 and queries==sorted(queries)
 head=json.loads(Path(prep['artifacts']['columns']['path']).read_text());labels=head['labels'];assert len(labels)==len(set(labels))==14261 and labels[-1]=='None'
 assert hashlib.sha256('\n'.join(labels).encode()).hexdigest()==bound['columns_order_sha256'];indices={h:i for i,h in enumerate(labels)}
 import numpy as np
 assert np.__version__==prep['numpy']['version']=='1.23.5'
 dtype=np.dtype([('column','<u2'),('score','<f4')],align=False);assert dtype.itemsize==6
 cache=np.lib.format.open_memmap(R/'top1.partial.npy',mode='w+',dtype=dtype,shape=(408272,));cache['column']=65535;cache['score']=np.float32('nan');cache.flush()
 progress=(R/'batch-progress.jsonl').open('x');all_batch_seconds=[]
 def commit_batch(index,start,columns,scores,origin,seconds):
  end=start+len(columns);assert index==report['completed_batches'] and start==report['completed_rows'] and 0<end<=408272
  columns=np.asarray(columns,dtype=np.uint16);scores=np.asarray(scores,dtype=np.float32)
  assert (columns<14261).all() and np.isfinite(scores).all() and ((scores>=0)&(scores<=1)).all()
  cache['column'][start:end]=columns;cache['score'][start:end]=scores;cache.flush()
  assert np.array_equal(cache['column'][start:end],columns) and np.array_equal(cache['score'][start:end].copy().view(np.uint32),scores.view(np.uint32))
  row={'batch':index,'start':start,'end':end,'rows':len(columns),'origin':origin,'seconds':seconds,'score_float32_bits_preserved':True};progress.write(json.dumps(row,separators=(',',':'))+'\n');progress.flush();os.fsync(progress.fileno())
  report['completed_batches']+=1;report['completed_rows']=end;report['reused_NN_rows' if origin=='verified_probe_reuse' else 'new_NN_rows']+=len(columns);all_batch_seconds.append(seconds)
  if origin=='new_NN':report['NN_batch_seconds'].append(seconds)
  save()
 # Reuse only the already verified exact three native NCR batches; all other-model logits excluded.
 pr=plan['probe_result'];assert sha(pr['path'])==pr['sha256'];probe=json.loads(Path(pr['path']).read_text());assert probe['status']=='pass_fixed_cost_probe_no_quality' and probe['parameters']['total']==491068890 and probe['TF32_enabled_at_end'] is False
 probe_inputs=json.loads(Path(plan['probe_input']['path']).read_text());assert sha(plan['probe_input']['path'])==plan['probe_input']['sha256'] and sum(probe_inputs['batches'],[])==queries[:1536]
 assert len(probe['batches'])==len(probe_inputs['batches'])==3
 for i,item in enumerate(probe['batches']):
  assert i<3 and item['queries']==512;refs=item['outputs']
  for v in refs.values():assert Path(v['path']).stat().st_size==v['bytes'] and sha(v['path'])==v['sha256']
  array=np.load(refs['probabilities']['path'],allow_pickle=False,mmap_mode='r');rows=[json.loads(x) for x in Path(refs['top1']['path']).read_text().splitlines()];assert array.shape==(512,14261) and array.dtype==np.float32 and len(rows)==512
  assert np.isfinite(array).all() and ((array>=0)&(array<=1)).all() and np.max(np.abs(array.sum(1)-1))<1e-5
  cols=[];scores=[]
  for j,row in enumerate(rows):
   assert row['query']==queries[i*512+j];col=indices[row['HPO']];score=np.float32(row['score']);assert score.view(np.uint32)==array[j,col].view(np.uint32) and float(score)==float(array[j].max());cols.append(col);scores.append(score)
  commit_batch(i,i*512,cols,scores,'verified_probe_reuse',0.0)
 del array,rows
 sys.path.insert(0,plan['native_module_directory']);sys.path.insert(0,plan['fasttext_runtime'])
 import h5py,tensorflow as tf
 devices=tf.config.list_physical_devices('GPU');assert len(devices)==1;tf.config.experimental.set_memory_growth(devices[0],True)
 tf.keras.mixed_precision.set_global_policy('float32');tf.config.experimental.enable_tensor_float_32_execution(False);assert not tf.config.experimental.tensor_float_32_execution_enabled();tf.config.experimental.reset_memory_stats('GPU:0')
 import ncrmodel
 report['runtime']={'tensorflow':tf.__version__,'numpy':np.__version__,'mixed_precision_policy':tf.keras.mixed_precision.global_policy().name,'TF32_enabled':tf.config.experimental.tensor_float_32_execution_enabled(),'GPU_uuid':os.environ['CUDA_VISIBLE_DEVICES']};save()
 load=time.monotonic();model=ncrmodel.NCR.safeloadfromjson(plan['HP_directory'],plan['fasttext_file']);report['load_seconds']=time.monotonic()-load
 assert model.config.n_ensembles==len(model.ncr_cores)==10 and model.ensembled_ncr.count_params()==157705490 and model.word_model.get_dimension()==100
 assert list(model.ont.concepts)+['None']==labels and model.ont.root_id==head['root_id']
 checks=[];dec=lambda x:x.decode() if isinstance(x,bytes) else str(x)
 with h5py.File(Path(plan['HP_directory'])/'ncr_weights.h5','r') as h:
  records=[]
  for layer in h.attrs['layer_names']:
   group=h[dec(layer)]
   for n in group.attrs.get('weight_names',[]):records.append(group[dec(n)])
  assert len(records)==len(model.ensembled_ncr.weights)==60
  for ds,var in zip(records,model.ensembled_ncr.weights):
   expected=ds[()];actual=var.numpy();assert actual.dtype==expected.dtype==np.float32 and actual.shape==expected.shape and np.isfinite(actual).all() and np.array_equal(actual,expected)
   checks.append({'H5_dataset':ds.name,'variable':var.name,'shape':list(actual.shape),'dtype':str(actual.dtype),'numeric_equal':True})
 report['weight_checks']=checks;report['parameters']={'NN':157705490,'FT_both_matrices':333363400,'total':491068890};save()
 underlying=model.ensembled_ncr;capture={};forward_calls=0;max_probability_error=0.0
 class CaptureForward:
  def __call__(self,*args,**kwargs):
   global forward_calls,max_probability_error
   tensor=underlying(*args,**kwargs);array=tensor.numpy();assert 'GPU:0' in tensor.device and array.dtype==np.float32 and array.ndim==2 and array.shape[1]==14261
   assert np.isfinite(array).all() and ((array>=0)&(array<=1)).all();error=float(np.max(np.abs(array.sum(1)-1)));assert error<1e-5
   capture['array']=array;forward_calls+=1;max_probability_error=max(max_probability_error,error);return tensor
 model.ensembled_ncr=CaptureForward();inference=time.monotonic()
 for index,start in enumerate(range(1536,len(queries),512),start=3):
  batch=queries[start:start+512];before=forward_calls;begin=time.monotonic();matches=model.get_match(batch,1);seconds=time.monotonic()-begin;array=capture.pop('array')
  assert forward_calls==before+1 and array.shape==(len(batch),14261) and len(matches)==len(batch)
  cols=[];scores=[]
  for j,m in enumerate(matches):
   assert len(m)==1;h,p=m[0];col=indices[h];assert isinstance(p,np.float32) and p.view(np.uint32)==array[j,col].view(np.uint32) and float(p)==float(array[j].max());cols.append(col);scores.append(p)
  commit_batch(index,start,cols,scores,'new_NN',seconds);del array
 report['NN_elapsed_seconds_including_cache_writes']=time.monotonic()-inference
 assert report['completed_batches']==798 and report['completed_rows']==408272 and report['new_NN_rows']==406736 and report['reused_NN_rows']==1536 and forward_calls==795
 assert (cache['column']<14261).all() and np.isfinite(cache['score']).all() and ((cache['score']>=0)&(cache['score']<=1)).all();progress.close();cache.flush()
 report['max_probability_sum_error']=max_probability_error;report['new_NN_forward_calls']=forward_calls;report['TF_allocator_peak_memory']=tf.config.experimental.get_memory_info('GPU:0')
 # Replay the unchanged native decoder in the same NumPy runtime; no further neural calls.
 sys.path.insert(0,str(R));from prepare_inputs import extract_native
 enum,tokenize,annotate=extract_native(Path(plan['native_source']['path']).read_text());lookup={q:i for i,q in enumerate(queries)};docs=input_docs;decoded=0;seen_candidates=0;decoder_started=time.monotonic()
 class CacheOracle:
  ont=types.SimpleNamespace(root_id=head['root_id'])
  def get_match(self,raw_queries,count=1):
   global seen_candidates
   assert raw_queries and count==1;out=[]
   for q in raw_queries:
    row=cache[lookup[' '.join(tokenize(q)[:49])]];assert isinstance(row['score'],np.float32);out.append([(labels[int(row['column'])],row['score'])])
   seen_candidates+=len(out);return out
 oracle=CacheOracle();native_path=R/'native-raw-spans.jsonl.gz'
 with gzip.open(native_path,'xt',encoding='utf-8') as out:
  for doc in docs:
   for pi,p in enumerate(doc['full_text']):
    spans=annotate(oracle,p['text'])
    for start,end,h,score in spans:
     assert 0<=start<end<=len(p['text']) and h not in [head['root_id'],'None'] and score>.8
     key=' '.join(tokenize(re.sub(r'[\\/\r\n\t-]',' ',p['text'][start:end]))[:49]);key_index=lookup[key];stored=cache[key_index];assert labels[int(stored['column'])]==h and stored['score'].view(np.uint32)==score.view(np.uint32)
     row={'cache_row':key_index,'score_float32_bits':int(score.view(np.uint32)),'pmc_id':doc['pmc_id'],'passage_index':pi,'passage_offset':p['offset'],'passage_type':p['type'],'local_start':int(start),'local_end':int(end),'offset':p['offset']+int(start),'length':int(end-start),'text':p['text'][start:end],'raw_hpo_id':h,'score':float(score)};out.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n');decoded+=1
 assert seen_candidates==683438 and forward_calls==795
 report['native_decode']={'candidate_occurrences':seen_candidates,'raw_native_spans':decoded,'seconds':time.monotonic()-decoder_started,'original_AST_decoder_unchanged':True,'threshold':.8,'score_scalar_type':'numpy.float32','numpy_runtime':np.__version__,'all_spans_query_row_HPO_and_float32_bits_bound':True,'batch_shape_equivalence':'Uses fixed512 native get_match outputs; not claiming bitwise equivalence to every alternative original passage-sized NN batch','current_HPO_mapping_not_yet_applied':True,'NN_forward_calls_added':0}
 cache.flush();del cache;(R/'top1.partial.npy').rename(R/'top1.npy')
 # Deterministic gzip is optional transport encoding; canonical NPY remains immutable.
 with (R/'top1.npy').open('rb') as src,(R/'top1.npy.gz').open('xb') as dst:
  with gzip.GzipFile(filename='',mode='wb',fileobj=dst,mtime=0) as z:
   for b in iter(lambda:src.read(1048576),b''):z.write(b)
 report.update(status='complete_all_keys_top1_and_native_raw_span_cache_no_quality',finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),cache_format={'dtype_descr':dtype.descr,'itemsize':6,'rows':408272,'payload_bytes':2449632,'query_order_sha256':prep['artifacts']['queries']['sha256'],'columns_order_sha256':bound['columns_order_sha256'],'all_score_bits_preserved':True},outputs={n:ref(R/n) for n in ['top1.npy','top1.npy.gz','columns.json','queries.jsonl','batch-progress.jsonl','native-raw-spans.jsonl.gz']},TF32_enabled_at_end=tf.config.experimental.tensor_float_32_execution_enabled())
 assert report['TF32_enabled_at_end'] is False
 # Batch timings are full external evidence; keep the terminal result compact.
 times=report.pop('NN_batch_seconds');report['NN_timing_summary']={'batches':len(times),'sum_get_match_seconds':sum(times),'minimum':min(times),'maximum':max(times),'mean':sum(times)/len(times)}
 save();write('cache-result.json',report)
except BaseException as e:
 report.update(status='failed_partial_preserved_not_complete',error=repr(e),traceback=traceback.format_exc());save();write('cache-result.json',report);raise
