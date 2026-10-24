"""Single fixed15 teacher supply, public R1; raw first, no training or quality."""
import argparse,datetime,gzip,hashlib,json,os,subprocess,time,traceback
from pathlib import Path
from evidence_contract import final_json,validate

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):p=Path(p).resolve();return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def checked(x):
 p=Path(x['path']);assert p.stat().st_size==x['bytes'] and sha(p)==x['sha256'];return p

def atomic(p,obj):
 p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp')
 with tmp.open('w') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(tmp,p);fd=os.open(p.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
def ids_sha(ids):return hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()
def admit(a):
 t=time.time();assert a['valid_from_unix']-5<=t<a['valid_until_unix'],'launch validity expired'

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--execution',required=True);ap.add_argument('--admission',required=True);z=ap.parse_args();pp=Path(z.execution).resolve();plan=json.loads(pp.read_text());a=json.loads(Path(z.admission).read_text());admit(a);assert sha(pp)==a['execution_sha256']
 out=pp.parent/'terminal';out.mkdir(exist_ok=False);start=time.monotonic();start_epoch=time.time();soft_deadline=start+2670;r=dict(status='RUNNING_TRAIN15_TEACHER_SUPPLY_NO_QUALITY',started_at=now(),source=ref(__file__),execution=ref(pp),admission=ref(z.admission),rows=[],no_student_training=True,no_quality=True);torch=None;model=None
 try:
  assert plan['fixed']['max_new_tokens']==3072 and plan['fixed']['pilot_seconds']==2700 and plan['fixed']['rows']==15
  assert os.environ['CUDA_VISIBLE_DEVICES']==plan['gpu_uuid']
  for name,expected in plan['sources'].items():assert sha(pp.parent/name)==expected,name
  for x in plan['references'].values():checked(x)
  assert plan['parent_plan_sha256']==sha(plan['references']['parent_plan']['path'])
  rows=[json.loads(l) for l in gzip.decompress(checked(plan['references']['inputs_gz']).read_bytes()).splitlines()];assert len(rows)==15 and len({x['pmc_id'] for x in rows})==15
  assert hashlib.sha256(gzip.decompress(checked(plan['references']['inputs_gz']).read_bytes())).hexdigest()==plan['input_uncompressed_sha256']
  encoded=[json.loads(l) for l in gzip.decompress(checked(plan['references']['encoded_gz']).read_bytes()).splitlines()];audits=json.loads(checked(plan['references']['audit']).read_text());selected=json.loads(checked(plan['references']['selection']).read_text());assert len(encoded)==len(audits)==len(selected)==15
  assert [x['row_id'] for x in rows]==[x['row_id'] for x in encoded]==[x['row_id'] for x in audits]
  assert sorted(plan['row_order'])==list(range(15)) and plan['row_order'][0]==max(range(15),key=lambda i:len(encoded[i]['input_ids']))
  modeldir=Path(plan['model_dir']);assets=json.loads(checked(plan['references']['asset_identity']).read_text());stat_before={}
  for x in assets['files']:
   p=modeldir/x['name'];st=p.stat();assert st.st_size==x['bytes'] and st.st_mtime_ns==x['mtime_ns'] and st.st_ino==x['inode']
   if not x['name'].endswith('.safetensors'):assert sha(p)==x['sha256']
   stat_before[x['name']]=dict(bytes=st.st_size,mtime_ns=st.st_mtime_ns,inode=st.st_ino,full_SHA_inherited=x['sha256'])
  r['asset_stat_before']=stat_before;r['weight_identity_policy']='Old fullSHA receipt bound to samehost fresh size/mtime/inode; no repeated16GB hash claim.'
  # Target actually has no GPU allocation before this admission/fresh-resource block.
  admit(a)
  q=subprocess.run(['nvidia-smi','--query-gpu=uuid,memory.used,memory.free','--format=csv,noheader,nounits'],check=True,capture_output=True,text=True).stdout
  gpu={l.split(',')[0].strip():{'used':int(l.split(',')[1]),'free':int(l.split(',')[2])} for l in q.splitlines()};apps=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],check=True,capture_output=True,text=True).stdout
  assert gpu[plan['gpu_uuid']]['used']<=64 and not any(l.split(',')[0].strip()==plan['gpu_uuid'] for l in apps.splitlines()),'assigned GPU not free'
  free_bytes=gpu[plan['gpu_uuid']]['free']*1024**2;assert free_bytes>plan['empirical_memory_proxy_bytes'],'free below nonbinding memory estimate'
  atomic(out/'resource-before-import.json',dict(at_utc=now(),uuid=plan['gpu_uuid'],used_MiB=gpu[plan['gpu_uuid']]['used'],free_MiB=gpu[plan['gpu_uuid']]['free'],empirical_proxy_bytes=plan['empirical_memory_proxy_bytes'],empirical_headroom_bytes=free_bytes-plan['empirical_memory_proxy_bytes'],proxy_is_not_upper_bound=True,no_compute_app=True))
  import torch as tm,transformers,tokenizers
  torch=tm
  runtime_root=Path(transformers.__file__).resolve().parent
  for relative,expected_sha in plan['runtime_source_refs'].items():assert sha(runtime_root/relative)==expected_sha,relative
  from transformers import PreTrainedTokenizerFast,AutoConfig,AutoModelForCausalLM,StoppingCriteria,StoppingCriteriaList
  torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
  assert torch.cuda.is_available() and torch.cuda.device_count()==1 and torch.cuda.is_bf16_supported()
  r['runtime']=dict(torch=torch.__version__,transformers=transformers.__version__,tokenizers=tokenizers.__version__,GPU=torch.cuda.get_device_name(0),uuid=plan['gpu_uuid'])
  tc=json.loads((modeldir/'tokenizer_config.json').read_text())
  def content(k):
   x=tc.get(k);return x.get('content') if isinstance(x,dict) else x
  raw=tokenizers.Tokenizer.from_file(str(modeldir/'tokenizer.json'));tok=PreTrainedTokenizerFast(tokenizer_file=str(modeldir/'tokenizer.json'),bos_token=content('bos_token'),eos_token=content('eos_token'),pad_token=content('pad_token'),unk_token=content('unk_token'),chat_template=tc['chat_template'],clean_up_tokenization_spaces=False,model_max_length=tc['model_max_length'])
  rb=json.loads(raw.to_str());tb=json.loads(tok.backend_tokenizer.to_str());assert all(rb.get(k)==tb.get(k) for k in ('model','normalizer','pre_tokenizer','post_processor','decoder','added_tokens'))
  for row,enc,audit in zip(rows,encoded,audits):
   text=tok.apply_chat_template(row['messages'],tokenize=False,add_generation_prompt=True);ids=tok.encode(text,add_special_tokens=False);assert ids==enc['input_ids']==raw.encode(text,add_special_tokens=False).ids
   assert ids_sha(ids)==audit['input_ids_sha256'] and hashlib.sha256(text.encode()).hexdigest()==audit['rendered_sha256']
   assert tok.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)==raw.decode(ids,skip_special_tokens=False)==text
  assert sum(len(x['input_ids']) for x in encoded)==82445 and max(len(x['input_ids']) for x in encoded)==9400
  original=json.loads((modeldir/'config.json').read_text());derived=json.loads(checked(plan['references']['derived_config']).read_text());expected=json.loads(json.dumps(original));expected['rope_scaling']['attention_factor']=1.0;assert derived==expected
  cfg=AutoConfig.from_pretrained(str(checked(plan['references']['derived_config'])),local_files_only=True,trust_remote_code=False);assert cfg.rope_parameters['attention_factor']==1.0
  # H69: again immediately before the first model load, after all CPU checks.
  admit(a);atomic(out/'progress.json',dict(r,phase='loading_public_R1_no_adapter'));tick=time.monotonic()
  model,loading=AutoModelForCausalLM.from_pretrained(modeldir,config=cfg,local_files_only=True,trust_remote_code=False,dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='sdpa',output_loading_info=True)
  model.eval();torch.cuda.synchronize();r['load_seconds']=time.monotonic()-tick;r['loading_information']={k:sorted(v) if isinstance(v,(set,frozenset)) else v for k,v in loading.items()}
  assert not any(loading.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs'))
  ps=dict(model.named_parameters());assert len(ps)==399 and sum(p.numel() for p in ps.values())==8190735360
  assert all(p.dtype==torch.bfloat16 and p.device.type=='cuda' and bool(torch.isfinite(p).all()) for p in ps.values());assert model.get_input_embeddings().weight.data_ptr()!=model.get_output_embeddings().weight.data_ptr()
  from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
  inv=model.model.rotary_emb.inv_freq.detach();orig=model.model.rotary_emb.original_inv_freq.detach();refgpu,scale=ROPE_INIT_FUNCTIONS['yarn'](model.config,device=inv.device);refcpu,cpuscale=ROPE_INIT_FUNCTIONS['yarn'](model.config,device=torch.device('cpu'))
  assert inv.dtype==orig.dtype==refgpu.dtype==refcpu.dtype==torch.float32 and inv.shape==orig.shape==refgpu.shape==refcpu.shape==(64,) and torch.equal(inv,orig)
  assert torch.allclose(inv,refgpu,rtol=8*torch.finfo(torch.float32).eps,atol=0) and torch.allclose(refgpu.cpu(),refcpu,rtol=8*torch.finfo(torch.float32).eps,atol=0)
  assert float(model.model.rotary_emb.attention_scaling)==float(scale)==float(cpuscale)==1.0
  assert hashlib.sha256(refcpu.numpy().tobytes()).hexdigest()=='427ed49dc1d6e18683800eca336467c70fc6817bfdae8a61166cbc7e01b05cb1'
  gen=dict(do_sample=True,temperature=.6,top_p=.95,top_k=0,repetition_penalty=1.,max_new_tokens=3072,num_return_sequences=1,use_cache=True,eos_token_id=tok.eos_token_id,pad_token_id=tok.pad_token_id,return_dict_in_generate=True,output_scores=False)
  r['generation_arguments']=gen;r['parameters']=8190735360;r['pre_generation_seconds']=time.monotonic()-start
  for i in plan['row_order']:
   if time.monotonic()>=soft_deadline:raise TimeoutError('fixed2700 budget exhausted before complete15')
   row=rows[i];enc=encoded[i];torch.manual_seed(20261024);torch.cuda.manual_seed_all(20261024);ids=torch.tensor([enc['input_ids']],device='cuda',dtype=torch.long);mask=torch.ones_like(ids);n=ids.shape[1];tick=time.monotonic()
   class StopAndPreserve(StoppingCriteria):
    def __init__(self):self.wall=False;self.last=0
    def __call__(self,input_ids,scores,**kwargs):
     count=input_ids.shape[1]-n;self.wall=time.monotonic()>=soft_deadline
     if count-self.last>=32 or self.wall:
      atomic(out/f'row{i}.partial.json',dict(row_id=row['row_id'],generated_ids=input_ids[0,n:].tolist(),generated_tokens=count,wall_stop=self.wall));self.last=count
     return self.wall
   stopper=StopAndPreserve()
   with torch.inference_mode():generated=model.generate(input_ids=ids,attention_mask=mask,stopping_criteria=StoppingCriteriaList([stopper]),**gen)
   torch.cuda.synchronize();elapsed=time.monotonic()-tick;full=generated.sequences[0].tolist();assert full[:n]==enc['input_ids'];new=full[n:];assert len(new)<=3072;eos=bool(new and new[-1]==tok.eos_token_id);cap=not eos and len(new)>=3072
   item=dict(row_id=row['row_id'],original_index=i,seed=20261024,input_tokens=n,input_ids_sha256=ids_sha(enc['input_ids']),generated_ids=new,generated_tokens=len(new),native_EOS_observed=eos,stopped_at_token_cap=cap,stopped_at_wall_budget=stopper.wall,raw_generated_text=tok.decode(new,skip_special_tokens=False,clean_up_tokenization_spaces=False),final_text_excluding_terminal_EOS=tok.decode(new[:-1] if eos else new,skip_special_tokens=False,clean_up_tokenization_spaces=False),generation_seconds=elapsed,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),gold_hint_provided=True,quality_evaluated=False)
   # Immutable original response BEFORE parsing, so any failure cannot lose it.
   atomic(out/f'row{i}.raw.json',item)
   try:
    obj=final_json(item['final_text_excluding_terminal_EOS'],eos_observed=eos,stopped_at_cap=cap);body=json.loads(row['messages'][1]['content'].split('\n',1)[1]);units={u['unit_id']:u for u in body['report']};target=next(p for p in body['all_patients'] if p['patient_id']==body['target_patient']);assert str(selected[i]['pmc_id'])==row['pmc_id'] and selected[i]['patient_id']==body['target_patient']
    parse=validate(obj,units=units,target_mentions=target['mention'],concept_mentions=selected[i]['choices']);parse['parsed_final']=obj
   except (ValueError,KeyError,StopIteration,TypeError) as e:parse=dict(status='MECHANICAL_REJECT',error_type=type(e).__name__,error=str(e),semantic_support='not_assessed')
   atomic(out/f'row{i}.parsed.json',parse);r['rows'].append(dict(index=i,row_id=row['row_id'],raw=ref(out/f'row{i}.raw.json'),parsed=ref(out/f'row{i}.parsed.json'),generated_tokens=len(new),EOS=eos,cap=cap,status=parse['status']));atomic(out/'progress.json',dict(r,phase='row_complete'));del generated,ids,mask
   if stopper.wall:raise TimeoutError('fixed soft deadline stopped row; no complete15 claim')
  assert len(r['rows'])==15
  bundle=[json.loads((out/f'row{i}.raw.json').read_text()) for i in range(15)];payload=''.join(json.dumps(x,ensure_ascii=False,separators=(',',':'))+'\n' for x in bundle).encode();p=out/'responses.raw.jsonl.gz'
  with p.open('xb') as f:f.write(gzip.compress(payload,mtime=0));f.flush();os.fsync(f.fileno())
  r.update(status='COMPLETE_FIXED15_TEACHER_RESPONSES_SEMANTIC_REVIEW_PENDING',raw_bundle=ref(p),raw_uncompressed_sha256=hashlib.sha256(payload).hexdigest(),mechanical_counts={s:sum(x['status']==s for x in r['rows']) for s in ('MECHANICAL_PASS_SEMANTIC_REVIEW_REQUIRED','TEACHER_ABSTAIN','MECHANICAL_REJECT')},accepted_semantic_count=None)
  for name,before in stat_before.items():
   st=(modeldir/name).stat();assert (st.st_size,st.st_mtime_ns,st.st_ino)==(before['bytes'],before['mtime_ns'],before['inode'])
 except BaseException as e:r.update(status='FAILED_ENGINEERING_OR_RESOURCE_PRESERVED',error=repr(e),traceback=traceback.format_exc());raise
 finally:
  r.update(finished_at=now(),program_seconds=time.monotonic()-start,start_epoch=start_epoch,process_pid=os.getpid())
  if torch is not None and torch.cuda.is_initialized():r['peak_memory']=dict(allocated=torch.cuda.max_memory_allocated(),reserved=torch.cuda.max_memory_reserved())
  atomic(out/'result.json',r);print(json.dumps({k:r.get(k) for k in ('status','program_seconds','error','mechanical_counts')}),flush=True)
if __name__=='__main__':main()
