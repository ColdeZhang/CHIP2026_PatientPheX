"""One fixed three-batch native NCR GPU cost probe, then four CPU-reference synthetics."""
from pathlib import Path
import datetime,hashlib,json,os,sys,time,traceback
R=Path(__file__).resolve().parent;plan=json.loads((R/'plan.json').read_text());started=time.monotonic()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert os.environ.get('CUDA_VISIBLE_DEVICES')==plan['gpu_uuid']
assert not (R/'result.json').exists()
report={'status':'running','started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_sha256':sha(__file__),'plan_sha256':sha(R/'plan.json'),'competition_gold_or_quality':False,'fit':False,'batches':[],'synthetic_tests':[]}
def save():report['elapsed_seconds']=time.monotonic()-started;(R/'result.json').write_text(json.dumps(report,indent=2)+'\n')
def ref(p):return {'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)}
save()
try:
 for key,item in plan['required_files'].items():
  p=Path(item['path']);assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256'],key
 inputs=json.loads((R/'probe-inputs.json').read_text());batches=inputs['batches'];assert len(batches)==3 and all(len(b)==512 for b in batches)
 cpu=json.loads((R/'cpu-reference.json').read_text());assert cpu['status']=='pass_cpu_interface_only' and len(cpu['tests'])==4
 sys.path.insert(0,plan['native_module_directory']);sys.path.insert(0,plan['fasttext_runtime'])
 import numpy as np
 import h5py
 import tensorflow as tf
 devices=tf.config.list_physical_devices('GPU');assert len(devices)==1,devices
 tf.config.experimental.set_memory_growth(devices[0],True)
 tf.keras.mixed_precision.set_global_policy('float32')
 tf.config.experimental.enable_tensor_float_32_execution(False)
 assert not tf.config.experimental.tensor_float_32_execution_enabled()
 tf.config.experimental.reset_memory_stats('GPU:0')
 import ncrmodel
 report['runtime']={'tensorflow':tf.__version__,'numpy':np.__version__,'GPU_device_details':tf.config.experimental.get_device_details(devices[0]),'mixed_precision_policy':tf.keras.mixed_precision.global_policy().name,'TF32_enabled':tf.config.experimental.tensor_float_32_execution_enabled(),'GPU_visible_uuid':os.environ['CUDA_VISIBLE_DEVICES'],'CPU_threads':1};save()
 for b in batches:
  for q in b:assert ' '.join(ncrmodel.tokenize(q)[:49])==q
 load_start=time.monotonic();model=ncrmodel.NCR.safeloadfromjson(plan['HP_directory'],plan['fasttext_file']);report['model_load_seconds']=time.monotonic()-load_start
 assert model.config.n_ensembles==len(model.ncr_cores)==10 and model.ensembled_ncr.count_params()==157705490 and model.word_model.get_dimension()==100
 assert len(model.ont.concepts)==14260
 checks=[]
 with h5py.File(Path(plan['HP_directory'])/'ncr_weights.h5','r') as h:
  records=[]
  dec=lambda x:x.decode() if isinstance(x,bytes) else str(x)
  for layer in h.attrs['layer_names']:
   g=h[dec(layer)]
   for n in g.attrs.get('weight_names',[]):records.append(g[dec(n)])
  assert len(records)==len(model.ensembled_ncr.weights)==60
  for ds,var in zip(records,model.ensembled_ncr.weights):
   expected=ds[()];actual=var.numpy();assert actual.dtype==expected.dtype==np.float32 and actual.shape==expected.shape and np.isfinite(actual).all() and np.array_equal(actual,expected),ds.name
   checks.append({'H5_dataset':ds.name,'model_variable':var.name,'shape':list(actual.shape),'dtype':str(actual.dtype),'all_finite_and_numeric_equal':True})
 report['weight_checks']=checks;report['parameters']={'NN':157705490,'FT_input_and_output':333363400,'total':491068890,'ensembles':10};save()
 underlying=model.ensembled_ncr;captures=[]
 class CaptureForward:
  def __call__(self,*args,**kwargs):
   begin=time.monotonic();out=underlying(*args,**kwargs);array=out.numpy();elapsed=time.monotonic()-begin
   assert 'GPU:0' in out.device and array.dtype==np.float32 and array.ndim==2 and array.shape[1]==14261
   assert np.isfinite(array).all() and (array>=0).all() and (array<=1).all()
   error=float(np.abs(array.sum(axis=1)-1).max());assert error<1e-5
   captures.append({'array':array,'forward_and_sync_seconds':elapsed,'device':out.device,'probability_sum_max_error':error});return out
 model.ensembled_ncr=CaptureForward();indices={h:i for i,h in enumerate(model.ont.concepts)};indices['None']=len(model.ont.concepts)
 # First actual forward is exactly batch0; no preceding synthetic warm-up or repeated batches.
 for i,batch in enumerate(batches):
  before=len(captures);begin=time.monotonic();matches=model.get_match(batch,1);seconds=time.monotonic()-begin
  assert len(captures)==before+1 and len(matches)==512
  call=captures[-1];array=call.pop('array');assert array.shape==(512,14261)
  rows=[]
  for j,(query,m) in enumerate(zip(batch,matches)):
   assert len(m)==1;h,p=m[0];assert h in indices and float(p)==float(array[j,indices[h]])==float(array[j].max())
   rows.append({'query':query,'HPO':h,'score':float(p)})
  npy=R/f'batch{i}.probabilities.npy';np.save(npy,array,allow_pickle=False)
  out=R/f'batch{i}.top1.jsonl';out.write_text(''.join(json.dumps(x,separators=(',',':'))+'\n' for x in rows))
  report['batches'].append({'index':i,'phase':'first_forward_after_load' if i==0 else 'warm','queries':512,'native_get_match_seconds':seconds,'forward_and_sync_seconds':call['forward_and_sync_seconds'],'output_device':call['device'],'output_shape':list(array.shape),'output_dtype':str(array.dtype),'probability_sum_max_error':call['probability_sum_max_error'],'outputs':{'probabilities':ref(npy),'top1':ref(out)},'TF_allocator_memory':tf.config.experimental.get_memory_info('GPU:0')});del array;save()
 # Same four frozen CPU examples only; no gold HPO expectation.
 original=model.get_match;observed=[]
 def recorded(queries,count=1):
  result=original(queries,count);assert count==1
  observed.extend({'query':q,'HPO':m[0][0],'score':float(m[0][1])} for q,m in zip(queries,result));return result
 model.get_match=recorded
 for expected in cpu['tests']:
  qstart=len(observed);cstart=len(captures);begin=time.monotonic();spans=model.annotate_text(expected['text']);seconds=time.monotonic()-begin;actual=observed[qstart:]
  assert [x['query'] for x in actual]==[x['query'] for x in expected['actual_scored_queries']]
  assert [x['HPO'] for x in actual]==[x['HPO'] for x in expected['actual_scored_queries']]
  a=np.array([x['score'] for x in actual]);b=np.array([x['score'] for x in expected['actual_scored_queries']]);assert np.allclose(a,b,rtol=1e-4,atol=1e-5) and np.array_equal(a>.8,b>.8)
  assert [(int(s),int(e),h) for s,e,h,p in spans]==[(x['start'],x['end'],x['HPO']) for x in expected['spans']]
  span_rows=[]
  for s,e,h,p in spans:
   assert 0<=s<e<=len(expected['text']) and float(p)>.8
   span_rows.append({'start':int(s),'end':int(e),'text':expected['text'][s:e],'HPO':h,'score':float(p)})
  forward=[]
  for c in captures[cstart:]:
   array=c.pop('array');forward.append({**c,'shape':list(array.shape),'dtype':str(array.dtype)});del array
  report['synthetic_tests'].append({'text':expected['text'],'queries':actual,'spans':span_rows,'seconds':seconds,'CPU_score_max_abs_error':float(np.max(np.abs(a-b))),'CPU_top1_ID_query_order_threshold_and_span_coordinates_exact':True,'score_tolerance':{'rtol':1e-4,'atol':1e-5},'forwards':forward});save()
 assert len(report['batches'])==3 and len(report['synthetic_tests'])==4 and len(captures)==7
 warm=[x['native_get_match_seconds'] for x in report['batches'][1:]];cold=report['batches'][0]['native_get_match_seconds'];load=report['model_load_seconds']
 report['full408272_estimate']={'queries':408272,'batches_at512':798,'low_seconds':load+cold+797*min(warm),'high_seconds_with25percent_warm_margin_and60seconds_setup_io':load+cold+797*max(warm)*1.25+60,'formula':'low=measured_load+firstbatch+797*min(warm); high=measured_load+firstbatch+797*max(warm)*1.25+60','not_measured_full_run':True,'not_statistical_confidence_interval':True,'selection_bias':'deterministic first1536 keys are an a-prefix, not a representative content sample; actual full vocabulary/preprocessing/serialization may differ','batch_timing_scope':'original get_match incl phrase2vec, native FP32 NN, synchronized .numpy, extra full probability integrity check and native top1 argpartition; full probability file writing excluded; no 512b repetitions'}
 report['TF_allocator_peak_memory']=tf.config.experimental.get_memory_info('GPU:0');report['TF32_enabled_at_end']=tf.config.experimental.tensor_float_32_execution_enabled();assert not report['TF32_enabled_at_end']
 report['status']='pass_fixed_cost_probe_no_quality';report['finished_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()
except BaseException as error:
 report.update(status='failed_preserved_stop_no_extra_samples',error=repr(error),traceback=traceback.format_exc());save();raise
