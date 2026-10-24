"""Native tokenize only578 added rows; preserve old2353 and1416 exact encoded bytes."""
import argparse,gzip,hashlib,json,math,os,random,sys,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent/'reuse/mention'))
from helpers import ref,write,validate_refs,digest
from cpu_contract import NativeTokenizer,encode
from mention_schema import MentionCodec,strict_parse

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);a=ap.parse_args();plan=json.loads(Path(a.plan).read_text());out=Path(plan['output']);out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
 try:
  assert os.environ['CUDA_VISIBLE_DEVICES']=='' and os.environ['OMP_NUM_THREADS']=='1';validate_refs(plan['files']);validate_refs({'old_complete':plan['old_complete'],'appended':plan['appended']})
  old=json.loads(Path(plan['old_complete']['path']).read_text());assert old['status']=='PASS_COMPLETE_MENTION_NATIVE_HF_INPUTS_AND_THREE_STATE_CONSTRAINT_CPU_ONLY' and old['train_rows']==2353 and old['held_rows']==1416
  import torch,transformers,tokenizers
  from transformers import AutoTokenizer
  torch.set_num_threads(1);assert not torch.cuda.is_available() and not torch.cuda.is_initialized();assert transformers.__version__==plan['runtime_versions']['transformers'] and tokenizers.__version__==plan['runtime_versions']['tokenizers']
  model=Path(plan['model_dir'])
  for name,x in plan['model_assets'].items():r=ref(model/name);assert r['sha256']==x['sha256'] and r['bytes']==x['bytes']
  validate_refs({'names':plan['public_names']});names=json.loads(Path(plan['public_names']['path']).read_text());assert len(names)==19119;tk=AutoTokenizer.from_pretrained(model,local_files_only=True);native=NativeTokenizer(model/'tokenizer.json',json.loads((model/'tokenizer_config.json').read_text())['chat_template'],plan['eos_token_id'],plan['runtime_versions']['tokenizers']);codec=MentionCodec(native.backend,names,plan['eos_token_id'])
  raw=gzip.decompress(Path(plan['appended']['path']).read_bytes());assert hashlib.sha256(raw).hexdigest()==plan['appended_raw_sha256'];rows=[json.loads(x) for x in raw.splitlines()];assert len(rows)==578
  encoded=[];audits=[];prefixes=0
  for i,row in enumerate(rows):
   assert row['target_status']=='NO' and row['answer']=='[]' and row['supervision_origin']=='OOF_annotation_disjoint_negative'
   prompt=tk.apply_chat_template(row['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=False);ids=tk.encode(prompt,add_special_tokens=False);assert ids==native.backend.encode(prompt,add_special_tokens=False).ids;assert tk.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)==native.backend.decode(ids,skip_special_tokens=False)==prompt
   value=encode(row,native);assert value==encode(row,tk);body=native.backend.encode(row['answer'],add_special_tokens=False).ids;assert strict_parse(row['answer'],row['quote'],names,eos=True,capped=False)=={'status':'NO','ids':[]}
   for k,token in enumerate(body+[plan['eos_token_id']]):assert token in codec.allowed(body[:k],row['quote']);prefixes+=1
   assert len(value['input_ids'])<=1024 and len(body)+1<=128;encoded.append(value);audits.append({'index':2353+i,'row_id':row['row_id'],'pmc_id':row['pmc_id'],'tokens':len(value['input_ids']),'supervised_tokens':value['supervised_tokens'],'input_ids_sha256':digest(value['input_ids']),'labels_sha256':digest(value['labels']),'native_HF_IDs_rendering_fullmask_exact':True})
  for split in ['train','dev']:validate_refs({split+'_encoded':old['counts'][split]['encoded'],split+'_audit':old['counts'][split]['audit']})
  original=[json.loads(x) for x in gzip.decompress(Path(old['counts']['train']['encoded']['path']).read_bytes()).splitlines()];assert len(original)==2353;allrows=original+encoded;assert len({r['row_id'] for r in allrows})==2931
  p=out/'train.encoded.jsonl.gz';p.write_bytes(gzip.compress(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in allrows).encode(),mtime=0));oldaudits=json.loads(Path(old['counts']['train']['audit']['path']).read_text());ar=write(out/'train.audit.json',oldaudits+audits);write(out/'appended.audit.json',audits)
  all_audit=oldaudits+audits;counts={'train':{'rows':2931,'tokens':sum(x['tokens'] for x in all_audit),'supervised_tokens':sum(x['supervised_tokens'] for x in all_audit),'max_tokens':max(x['tokens'] for x in all_audit),'max_supervised_tokens':max(x['supervised_tokens'] for x in all_audit),'encoded':ref(p),'audit':ar},'dev':old['counts']['dev']};byshape={}
  for r in all_audit:byshape.setdefault((r['tokens'],r['supervised_tokens']),r['index'])
  frontier=sorted((shape,i) for shape,i in byshape.items() if not any(q[0]>=shape[0] and q[1]>=shape[1] and q!=shape for q in byshape));covered=all(any(x['tokens']<=a and x['supervised_tokens']<=b for a,b in old['pareto_shapes']) for x in audits)
  order=list(range(2931));random.Random(20261023).shuffle(order);groups=[order[i:i+8] for i in range(0,2931,8)];assert len(groups)==367 and len(groups[-1])==3
  result={'status':'PASS_INCREMENTAL_HARD_NEGATIVE_NATIVE_INPUTS_ALL578','plan':ref(a.plan),'source':ref(__file__),'reused_original_CPU':plan['old_complete'],'runtime':{'torch':torch.__version__,'transformers':transformers.__version__,'tokenizers':tokenizers.__version__,'CUDA_initialized':torch.cuda.is_initialized()},'train_rows':2931,'held_rows':1416,'training_states':{'HPO':2222,'NIL':44,'NO':665},'new_native_rows':578,'new_prefixes_checked':prefixes,'old2353_encoded_objects_exact_and_same_order':allrows[:2353]==original,'old1416_encoded_reference_exact':True,'new_rows_all_original_char_quotes_no_truncation':True,'all_new_shapes_dominated_by_previous_actual_probe_frontier':covered,'appended_max_tokens':max(x['tokens'] for x in audits),'appended_max_supervised':max(x['supervised_tokens'] for x in audits),'counts':counts,'pareto_indices':[i for shape,i in frontier],'pareto_shapes':[list(shape) for shape,i in frontier],'stress_initial8':groups[0],'stress_accumulated3':groups[1][:3],'schedule':{'epochs':2,'updates_per_epoch':367,'updates':734,'last_group_size':3,'warmup_steps':37},'training_cap':1024,'inference_total_cap':1024,'max_new_tokens':128,'no_model_forward_generate_GPU_fit_quality':True,'old_processor_and_real_FB_proof_reused_not_rerun':True,'seconds':time.monotonic()-started}
  write(out/'complete.json',result);print(json.dumps(result),flush=True)
 except BaseException:write(out/'failure.json',{'seconds':time.monotonic()-started,'traceback':traceback.format_exc()});raise

if __name__=='__main__':main()
