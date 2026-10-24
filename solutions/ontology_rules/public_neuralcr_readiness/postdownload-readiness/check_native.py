"""One CPU-only strict NeuralCR load and four frozen synthetic interface cases."""
from pathlib import Path
import os,json,hashlib,time,traceback,sys,datetime,re
R=Path(__file__).resolve().parent
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
assert not (R/'native-cpu-result.json').exists()
plan=json.loads((R/'native-cpu-plan.json').read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
assert sha(Path(__file__))==plan['sources']['check_native.py']['sha256']
t=time.monotonic(); report={'status':'running','started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_sha256':sha(Path(__file__)),'plan_sha256':sha(R/'native-cpu-plan.json'),'competition_inputs_or_gold_read':False,'synthetic_interface_only':True,'tests':[]}
try:
 for name,expected in plan['required_files'].items():
  p=R/name;assert p.stat().st_size==expected['bytes'] and sha(p)==expected['sha256'],name
 runtime_overlay=Path(plan['runtime_overlay']).resolve();assert runtime_overlay.is_dir()
 sys.path.insert(0,str(runtime_overlay));sys.path.insert(0,str(R))
 import numpy as np
 import h5py
 import tensorflow.compat.v1 as tf
 assert tf.config.list_physical_devices('GPU')==[]
 import ncrmodel
 load_start=time.monotonic();model=ncrmodel.NCR.safeloadfromjson(str(R/'extracted/HP'),str(R/plan['fasttext_load_file']))
 report['load_seconds']=time.monotonic()-load_start
 inventory=json.loads((R/'shape-inventory.json').read_text());assert model.ensembled_ncr.count_params()==inventory['H5_unique_parameters']
 assert model.config.n_ensembles==10 and len(model.ncr_cores)==10
 assert model.word_model.get_dimension()==inventory['fastText']['dimension']==100
 assert len(model.ont.concepts)==inventory['ontology_JSON']['concepts']
 def dec(x):return x.decode() if isinstance(x,bytes) else str(x)
 records=[]
 with h5py.File(R/'extracted/HP/ncr_weights.h5','r') as h:
  for layer in h.attrs['layer_names']:
   g=h[dec(layer)]
   for name in g.attrs.get('weight_names',[]):
    ds=g[dec(name)];records.append((g.name+'/'+dec(name),ds[()]))
  assert len(records)==len(model.ensembled_ncr.weights), (len(records),len(model.ensembled_ncr.weights))
  variable_checks=[]
  for (name,value),variable in zip(records,model.ensembled_ncr.weights):
   actual=variable.numpy();assert actual.shape==value.shape and actual.dtype==value.dtype
   assert np.isfinite(actual).all() and np.array_equal(actual,value),name
   variable_checks.append({'H5_dataset':name,'model_variable':variable.name,'shape':list(value.shape),'dtype':str(value.dtype),'bit_exact':True})
  del records
 report['strict_weight_load']={'unique_parameters':model.ensembled_ncr.count_params(),'variables':len(variable_checks),'all_finite_bit_exact_to_H5':True,'details':variable_checks}
 underlying=model.ensembled_ncr; calls=[];head=len(model.ont.concepts)+1
 class CheckedForward:
  def __call__(self,*args,**kwargs):
   out=underlying(*args,**kwargs);a=out.numpy();assert a.ndim==2 and a.shape[1]==head
   assert np.isfinite(a).all() and (a>=0).all() and (a<=1).all()
   error=float(np.max(np.abs(a.sum(axis=1)-1)));assert error<1e-5
   calls.append({'shape':list(a.shape),'finite':True,'probability_sum_max_error':error});return out
 model.ensembled_ncr=CheckedForward()
 original_get_match=model.get_match; scored_queries=[]
 def checked_get_match(queries,count=1):
  result=original_get_match(queries,count)
  assert isinstance(queries,list) and count==1 and len(queries)==len(result)
  scored_queries.extend({'query':q,'HPO':r[0][0],'score':float(r[0][1])} for q,r in zip(queries,result))
  return result
 model.get_match=checked_get_match
 normalization=re.compile('[\\\\/\r\n\t-]')
 for text in plan['fixed_synthetic_texts']:
  before=len(calls);query_start=len(scored_queries);test_start=time.monotonic();spans=model.annotate_text(text)
  assert len(calls)>before
  rows=[]
  for start,end,hpo,score in spans:
   assert 0<=start<end<=len(text) and text[start:end] and hpo in model.ont.concepts
   assert hpo!=model.ont.root_id and float(score)>.8 and np.isfinite(score)
   normalized=normalization.sub(' ',text[start:end])
   matching=[i for i,q in enumerate(scored_queries[query_start:]) if q['query']==normalized and q['HPO']==hpo and q['score']==float(score)]
   assert matching, ('returned_offset_does_not_bind_to_actual_scored_query',start,end,hpo)
   rows.append({'start':int(start),'end':int(end),'text':text[start:end],'HPO':hpo,'score':float(score),'normalized_substring':normalized,'matching_actual_query_indices':matching})
  report['tests'].append({'text':text,'spans':rows,'actual_scored_queries':scored_queries[query_start:],'forward_calls':calls[before:],'seconds':time.monotonic()-test_start,'offsets_python_codepoints_end_exclusive':True,'returned_offsets_bound_to_actual_query_HPO_score':True})
 assert len(report['tests'])==4
 report['status']='pass_cpu_interface_only';report['fixed_policy']={'threshold':.8,'ngram_lengths':'1..7','ensembles':10,'empty_text':'upstream unsupported; not invoked and no decoder repair','no_expected_HPO_or_quality_assertions':True}
except Exception as e:
 report['status']='failed_preserved';report['error']=repr(e);report['traceback']=traceback.format_exc();raise
finally:
 report['elapsed_seconds']=time.monotonic()-t;report['finished_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();(R/'native-cpu-result.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
