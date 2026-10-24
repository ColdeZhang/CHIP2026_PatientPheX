"""Fixed complete F1 greedy generation from verified two-epoch effective-delta average.

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
 assert p['schema']=='grounded-meanweights-held39-inference-v1' and p['inference_authorized'] is True
 assert p['max_new_tokens']==1600 and p['all_rows_fresh'] and p['expected_rows']==39
 assets=p['assets']
 for name,item in assets.items():assert ref(item['path'])==item,name
 assert assets['runner']==ref(__file__)
 assert assets['raw_store']['sha256']==sha(inspect.getsourcefile(publish_json))=='d52bb0c186d2467b5ae5710a6b7df2ee271783c22ceadadc2e56299295e3af43'
 synthesis=json.loads(Path(assets['synthesis']['path']).read_text())
 assert synthesis['status']=='COMPLETE_CPU_EXACT_EFFECTIVE_DELTA_MEAN_NOT_TRAINED_EPOCH'
 assert synthesis['new_r']==synthesis['new_alpha']==32 and synthesis['new_scaling']==1
 assert synthesis['adapter_parameters']==86556672 and synthesis['expected_full_text_parameters']==9040359936
 assert synthesis['original_tensors_bitexact_concatenated'] and synthesis['reopened_saved_tensor_values_bitexact']
 assert synthesis['no_optimizer_or_training_checkpoint_created'] and synthesis['no_GPU']
 assert ref(synthesis['new_adapter']['path'])==synthesis['new_adapter']
 assert ref(synthesis['new_config']['path'])==synthesis['new_config']
 assert ref(synthesis['tiny_PEFT']['path'])==synthesis['tiny_PEFT']
 tiny=json.loads(Path(synthesis['tiny_PEFT']['path']).read_text())
 assert tiny['status']=='PASS_REAL_PEFT0212_12FAMILY_CAT_AND_SAVE_LOAD_CPU'
 assert tiny['loaded_state_values_bitexact'] and tiny['new_r']==tiny['new_alpha']==32
 assert [x['epoch'] for x in synthesis['parents']]==[1,2]
 for parent in synthesis['parents']:
  for key in ['config','checkpoint','ACK']:assert ref(parent[key]['path'])==parent[key]
  checkpoint=json.loads(Path(parent['checkpoint']['path']).read_text())
  assert checkpoint['completed_epochs']==parent['epoch'] and checkpoint['global_step']==11*parent['epoch']
  assert checkpoint['fingerprint']['train_sha256']=='bf0091ef057ddb67f9906d035fdaf0ac7b41ae645f77432f0e5a007078291d08'
 root=Path(p['storage_parent']).resolve();out=Path(p['output_dir']).resolve();out.relative_to(root)
 assert Path(p['adapter']).resolve()==Path(synthesis['new_adapter']['path']).parent
 assert os.environ['PATIENTPHEX_DURABILITY_ROOT']==str(root) and os.environ['CUDA_VISIBLE_DEVICES']==p['gpu_uuid']
 assert not out.exists();out.mkdir(parents=True,exist_ok=False)
 rows=[json.loads(x) for x in Path(assets['prompts']['path']).read_text().splitlines()]
 ids_rows=json.loads(gzip.decompress(Path(assets['held_ids']['path']).read_bytes()))
 audit={r['row_id']:r for r in json.loads(Path(assets['held_audit']['path']).read_text())};ids={r['row_id']:r['input_ids'] for r in ids_rows}
 assert len(rows)==len(ids)==len(audit)==39 and len({r['pmc_id'] for r in rows})==16
 assert {r['row_id'] for r in rows}==set(ids)==set(audit)
 assert len({(r['pmc_id'],r['target_patient_id']) for r in rows})==39 and all('answer' not in r for r in rows)
 for row in rows:
  expected=audit[row['row_id']];tokens=ids[row['row_id']]
  assert hashlib.sha256(json.dumps(tokens).encode()).hexdigest()==expected['input_ids_sha256']
  assert len(tokens)==expected['input_tokens'] and len(tokens)<=10080
 import torch,transformers,peft,tokenizers
 from transformers import AutoTokenizer,Qwen3_5TextConfig,Qwen3_5ForCausalLM
 from peft import PeftModel
 from tokenizers import Tokenizer
 import transformers.models.qwen3_5.modeling_qwen3_5 as native
 assert torch.__version__=='2.6.0+cu124' and transformers.__version__=='5.20.2' and peft.__version__=='0.21.2' and tokenizers.__version__=='0.23.1'
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
 assert sum(x.numel() for x in model.parameters())==9040359936<10_000_000_000
 eos=model.generation_config.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos]);eos.add(tok.eos_token_id);eos.discard(None);assert eos
 publish_json(out/'run-manifest.json',dict(plan=ref(a.plan),source=ref(__file__),verified_model=verified,assets=assets,
  runtime=dict(torch=torch.__version__,transformers=transformers.__version__,peft=peft.__version__,tokenizers=tokenizers.__version__),
  rows=39,documents=16,finite_logits_guard_reused=True,eos_ids=sorted(eos),max_new_tokens=1600,all_fresh=True,normalization_and_quality_performed=False,synthesis=synthesis,model_parameters=9040359936),root)
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
  records.append(record);print(json.dumps(dict(completed=len(records),total=39,output_tokens=len(suffix),seconds=gen['seconds'])),flush=True)
 assert len(records)==39 and {r['row_id'] for r in records}==set(ids)
 output=publish_bytes(out/'responses.jsonl',b''.join((json.dumps(r,ensure_ascii=False)+'\n').encode() for r in records),root)
 publish_json(out/'complete.json',dict(status='COMPLETE_FRESH39_RAW_GENERATION_NO_QUALITY',rows=39,documents=16,
  generated=records and len(records),EOS_rows=sum(r['actual_last_token_eos'] for r in records),cap_rows=sum(r['hit_token_limit'] for r in records),
  source=ref(__file__),plan=ref(a.plan),responses=output,generation_refs=generation_refs,raw_refs=raw_refs,
  max_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),seconds=time.monotonic()-started,no_regeneration_or_answer_repair=True),root)

if __name__=='__main__':main()
