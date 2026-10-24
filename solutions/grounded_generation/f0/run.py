"""Fixed complete F0 greedy generation after actual epoch2 durable confirmation.

This runner saves generated IDs before decoding and decoded raw before any local
interpretation. It does not call the mapper or evaluator and never regenerates.
"""
import argparse,gzip,hashlib,importlib.util,inspect,json,os,time
from pathlib import Path
from raw_store import publish_json,publish_bytes,row_path


def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()

def ref(p):p=Path(p).resolve();return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);a=ap.parse_args();p=json.loads(Path(a.plan).read_text());started=time.monotonic()
 assert p['schema']=='grounded-held47-inference-v1' and p['inference_authorized'] is True
 assert p['max_new_tokens']==1600 and p['all_rows_fresh'] and p['expected_rows']==47
 assets=p['assets']
 for name,item in assets.items():assert ref(item['path'])==item,name
 assert assets['runner']==ref(__file__)
 assert assets['raw_store']['sha256']==sha(inspect.getsourcefile(publish_json))=='d52bb0c186d2467b5ae5710a6b7df2ee271783c22ceadadc2e56299295e3af43'
 complete=json.loads(Path(assets['training_complete']['path']).read_text());checkpoint=json.loads(Path(assets['checkpoint']['path']).read_text());durable=json.loads(Path(assets['durability']['path']).read_text())
 wrapper=json.loads(Path(assets['training_wrapper']['path']).read_text())
 assert wrapper['returncode']==0 and not wrapper['timed_out'] and 'endpoint_error' not in wrapper and wrapper['all_epochs_persistent_and_readback_verified']
 assert set(wrapper['durability'])=={'1','2'} and wrapper['durability']['2']['receipt']==durable
 assert wrapper['training_complete_sha256']==assets['training_complete']['sha256']
 assert complete['epochs']==2 and complete['global_step']==22 and complete['rows']==87
 assert complete['source_sha256']==assets['trainer']['sha256']==checkpoint['fingerprint']['source_sha256']
 assert complete['both_epochs_file_directory_fsync_and_reopened_SHA_verified'] and complete['probe_state_used'] is False
 assert checkpoint['completed_epochs']==2 and checkpoint['global_step']==22 and checkpoint['smoke_only'] is False
 fingerprint=checkpoint['fingerprint']
 assert fingerprint['schema']=='grounded-whole-report87-standard-indexed-CE-v1' and fingerprint['configuration']['optimizer_calls']==22
 assert fingerprint['train_sha256']=='bf0091ef057ddb67f9906d035fdaf0ac7b41ae645f77432f0e5a007078291d08'
 assert fingerprint['indexed_head_sha256']=='0d28201832721c800696321c6afa1ef7831d63b697620a02290d65317a4428bb'
 assert durable['status']=='PASS_file_and_directory_fsync_rename_and_reopened_SHA'
 assert durable['completed_epochs']==2 and durable['global_step']==22 and durable['checkpoint_metadata_sha256']==assets['checkpoint']['sha256']
 assert durable['all_members_reopened_and_SHA_verified'] and durable['reopened_SHA_all_exact'] and durable['file_directory_fsync']
 ckpt=Path(p['checkpoint_directory']).resolve();root=Path(p['storage_parent']).resolve();out=Path(p['output_dir']).resolve();out.relative_to(root)
 assert sha(ckpt/'checkpoint.json')==assets['checkpoint']['sha256']
 assert ckpt.name=='epoch-002' and Path(p['adapter']).resolve()==ckpt/'adapter'
 # Explicit inference-only replica: original full optimizer/RNG remains A100.
 binding=json.loads(Path(assets['inference_cache_binding']['path']).read_text());ack=binding['ack']
 assert ack['status']=='PASS_adapter_only_second_persistent_copy' and ack['destination']==str(ckpt)
 assert ack['source_directory']==p['original_training_checkpoint'] and Path(durable['checkpoint']).as_posix()==p['original_training_checkpoint']
 assert Path(durable['persistent_root']).resolve()==root and ack['original_source_retained']
 assert ack['optimizer_state_copied'] is False and ack['GPU_or_model_calls']==0
 assert ack['source_fsync_and_reopen_SHA'] and ack['destination_all_files_fsync_and_reopen_SHA'] and ack['destination_ancestor_directories_fsynced']
 assert set(ack['files'])=={'checkpoint.json','adapter/adapter_model.safetensors','adapter/adapter_config.json','adapter/README.md'}
 for name,item in ack['files'].items():
  q=ckpt/name;assert q.stat().st_size==item['bytes'] and sha(q)==item['sha256']
  if name!='checkpoint.json':assert checkpoint['files'][name]==item['sha256']
 assert checkpoint['files']['training_state.pt']=='47ae70277d36f2bce2cddda271f2aa7bdfcb38158272e180aeb0c35d60898cb7'
 assert ack['files']['adapter/adapter_model.safetensors']['sha256']=='d52e2e6c74d3fe874940ddc9a882d3c0faa453ac362b118d1edb2782bae7c6ba'
 native_cpu=json.loads(Path(assets['native_cpu_result']['path']).read_text())
 assert native_cpu['status']=='PASS_NATIVE_COMPLETE47_F0_INPUT' and native_cpu['rows']==47 and native_cpu['all47_local_native_actual_IDs_rendering_decode_exact']
 assert native_cpu['encoded']['sha256']==assets['held_ids']['sha256'] and native_cpu['input_tokens']==289420 and native_cpu['max_input_tokens']==9935
 assert os.environ['PATIENTPHEX_DURABILITY_ROOT']==str(root) and os.environ['CUDA_VISIBLE_DEVICES']==p['gpu_uuid']
 assert not out.exists();out.mkdir(parents=True,exist_ok=False)
 rows=[json.loads(x) for x in Path(assets['prompts']['path']).read_text().splitlines()]
 ids_rows=json.loads(gzip.decompress(Path(assets['held_ids']['path']).read_bytes()))
 audit={r['row_id']:r for r in json.loads(Path(assets['held_audit']['path']).read_text())};ids={r['row_id']:r['input_ids'] for r in ids_rows}
 assert len(rows)==len(ids)==len(audit)==47 and len({r['pmc_id'] for r in rows})==16
 assert {r['row_id'] for r in rows}==set(ids)==set(audit)
 assert len({(r['pmc_id'],r['target_patient_id']) for r in rows})==47 and all('answer' not in r for r in rows)
 for row in rows:
  expected=audit[row['row_id']];tokens=ids[row['row_id']]
  assert hashlib.sha256(json.dumps(tokens).encode()).hexdigest()==expected['input_ids_sha256']
  assert len(tokens)==expected['input_tokens'] and len(tokens)<=9935
 import torch,transformers,peft,tokenizers
 from transformers import AutoTokenizer,Qwen3_5TextConfig,Qwen3_5ForCausalLM
 from peft import PeftModel
 from tokenizers import Tokenizer
 import transformers.models.qwen3_5.modeling_qwen3_5 as native
 assert torch.__version__=='2.7.1+cu126' and transformers.__version__=='5.20.2' and peft.__version__=='0.21.2' and tokenizers.__version__=='0.23.1'
 assert sha(native.__file__)=='b4b17fe08d8d54da34a0fce78639666279ae7fd0a9e4ec06fce6602774707ca3'
 assert importlib.util.find_spec('fla') is None and all(inspect.unwrap(getattr(native,n)).__module__==native.__name__ for n in ['torch_chunk_gated_delta_rule','torch_recurrent_gated_delta_rule'])
 assert torch.cuda.device_count()==1 and torch.cuda.is_bf16_supported();torch.set_num_threads(1)
 source=json.loads(Path(assets['model_manifest']['path']).read_text());assert source['hf_revision']=='c202236235762e1c871ad0ccb60c8ee5ba337b9a'
 modelpath=Path(p['model']);verified={}
 for item in source['files']:
  assert Path(item['name']).name==item['name'];q=modelpath/item['name'];assert q.stat().st_size==item['size'] and sha(q)==item['sha256'];verified[item['name']]=dict(bytes=item['size'],sha256=item['sha256'])
 tok=AutoTokenizer.from_pretrained(modelpath,local_files_only=True);native_tok=Tokenizer.from_file(str(modelpath/'tokenizer.json'))
 if tok.pad_token_id is None:tok.pad_token=tok.eos_token
 for row in rows:
  text=tok.apply_chat_template(row['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=False)
  assert hashlib.sha256(text.encode()).hexdigest()==audit[row['row_id']]['rendered_prefix_sha256']
 from transformers import LogitsProcessor,LogitsProcessorList
 class FiniteLogitsOnly(LogitsProcessor):
  def __call__(self,input_ids,scores):
   if not torch.isfinite(scores).all().item():
    raise FloatingPointError('Nonfinite generation logits; stop without retry')
   return scores
 finite_processors=LogitsProcessorList([FiniteLogitsOnly()])
 config=Qwen3_5TextConfig.from_pretrained(modelpath,local_files_only=True)
 model,loading=Qwen3_5ForCausalLM.from_pretrained(modelpath,config=config,dtype=torch.bfloat16,device_map='cuda:0',attn_implementation='sdpa',local_files_only=True,output_loading_info=True)
 publish_json(out/'loading.json',loading,root)
 assert all(not loading.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs'])
 assert sum(x.numel() for x in model.parameters())==8953803264
 model=PeftModel.from_pretrained(model,p['adapter'],is_trainable=False);model.eval()
 assert sum(x.numel() for x in model.parameters())==8997081600<10_000_000_000
 eos=model.generation_config.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos]);eos.add(tok.eos_token_id);eos.discard(None);assert eos
 publish_json(out/'run-manifest.json',dict(plan=ref(a.plan),source=ref(__file__),verified_model=verified,assets=assets,
  runtime=dict(torch=torch.__version__,transformers=transformers.__version__,peft=peft.__version__,tokenizers=tokenizers.__version__),
  rows=47,documents=16,finite_logits_guard_reused=True,eos_ids=sorted(eos),max_new_tokens=1600,all_fresh=True,normalization_and_quality_performed=False),root)
 records=[];generation_refs=[];raw_refs=[]
 for row in sorted(rows,key=lambda r:(len(ids[r['row_id']]),r['row_id'])):
  began=time.monotonic();tokens=ids[row['row_id']];inp=torch.tensor([tokens],device='cuda',dtype=torch.long)
  try:
   with torch.inference_mode():output=model.generate(input_ids=inp,attention_mask=torch.ones_like(inp),do_sample=False,num_beams=1,use_cache=True,max_new_tokens=1600,eos_token_id=sorted(eos),pad_token_id=tok.pad_token_id,logits_processor=finite_processors)
  except BaseException as error:
   publish_json(out/'generation-failure.json',dict(row_id=row['row_id'],type=type(error).__name__,message=str(error),no_retry=True,completed_rows=len(records),no_completed_suffix_available=True),root);raise
  suffix=output[0,len(tokens):].tolist();ended=bool(suffix and suffix[-1] in eos);capped=len(suffix)>=1600 and not ended
  gen=dict(row_id=row['row_id'],pmc_id=row['pmc_id'],target_patient_id=row['target_patient_id'],generated_token_ids=suffix,input_tokens=len(tokens),output_tokens=len(suffix),actual_last_token_eos=ended,hit_token_limit=capped,seconds=time.monotonic()-began,decoded_or_parsed=False)
  gref=publish_json(row_path(out/'generation',row['row_id']),gen,root);generation_refs.append(gref)
  text_ids=suffix[:-1] if ended else suffix
  response=native_tok.decode(text_ids,skip_special_tokens=False)
  hresponse=tok.decode(text_ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
  record=dict({k:v for k,v in gen.items() if k!='decoded_or_parsed'},response=response,published_native_decode=True,HF_decode_equal=response==hresponse,generation=gref,raw_decoded=True,parsed=False,raw_before_parse=True)
  rref=publish_json(row_path(out/'raw',row['row_id']),record,root);raw_refs.append(rref)
  assert response==hresponse,'Actual raw decoder differs; original response already saved'
  records.append(record);print(json.dumps(dict(completed=len(records),total=47,output_tokens=len(suffix),seconds=gen['seconds'])),flush=True)
 assert len(records)==47 and {r['row_id'] for r in records}==set(ids)
 output=publish_bytes(out/'responses.jsonl',b''.join((json.dumps(r,ensure_ascii=False)+'\n').encode() for r in records),root)
 publish_json(out/'complete.json',dict(status='COMPLETE_FRESH47_RAW_GENERATION_NO_QUALITY',rows=47,documents=16,
  generated=records and len(records),EOS_rows=sum(r['actual_last_token_eos'] for r in records),cap_rows=sum(r['hit_token_limit'] for r in records),
  source=ref(__file__),plan=ref(a.plan),responses=output,generation_refs=generation_refs,raw_refs=raw_refs,
  max_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),seconds=time.monotonic()-started,no_regeneration_or_answer_repair=True),root)

if __name__=='__main__':main()
