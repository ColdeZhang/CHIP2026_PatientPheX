"""CPU-only A800 delta to the existing full124 .100 contract; no model weights."""
from pathlib import Path
import argparse,os,json,time,inspect
import train_medgemma32 as training
import medgemma_interface as interface

def main():
 p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--expected-files',required=True);p.add_argument('--reference',required=True);p.add_argument('--train',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 os.environ['CUDA_VISIBLE_DEVICES']='';started=time.perf_counter()
 import torch,transformers,peft,tokenizers
 from transformers import AutoConfig,AutoTokenizer,Gemma3ForConditionalGeneration
 root=Path(a.reference);old=json.loads((root/'run_manifest.json').read_text());audit=json.loads((root/'token-mask-audit.json').read_text())
 packages={'torch':torch.__version__,'transformers':transformers.__version__,'peft':peft.__version__,'tokenizers':tokenizers.__version__}
 assert all(old['fingerprint']['environment']['packages'][n]==v for n,v in packages.items())
 assert training.sha256(inspect.getsourcefile(Gemma3ForConditionalGeneration))==old['fingerprint']['environment']['modeling_source_sha256']
 assert training.sha256(a.train)==training.TRAIN_SHA
 assets=interface.model_files(a.model,a.expected_files,False)
 tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True,trust_remote_code=False)
 longest=max(audit['rows'],key=lambda r:r['full_tokens']);row=next(r for r in training.helpers.rows(a.train) if r['row_id']==longest['row_id']);e=interface.encode(row,tok)
 assert (len(e['input_ids']),e['prefix_tokens'],e['supervised_tokens'])==(longest['full_tokens'],longest['prefix_tokens'],longest['supervised_tokens'])
 cfg=AutoConfig.from_pretrained(a.model,local_files_only=True,trust_remote_code=False);cfg._attn_implementation='sdpa';cfg.text_config._attn_implementation='sdpa'
 with torch.device('meta'):
  model=Gemma3ForConditionalGeneration(cfg).to(dtype=torch.bfloat16);adapted,named,coverage=interface.attach_adapter(model,torch,peft)
 assert coverage['all_parameter_devices']==['meta'] and not torch.cuda.is_initialized()
 training.helpers.write(a.output,{'status':'CPU_runtime_delta_pass','reference_complete_sha256':training.sha256(root/'cpu_complete.json'),'modeling_source_sha256':training.sha256(inspect.getsourcefile(Gemma3ForConditionalGeneration)),
  'packages':packages,'python':os.sys.version,'same_native_library_versions_as_full100_CPU':True,'tokenizer_assets':assets,'longest_row_id':e['row_id'],'longest_full_tokens':len(e['input_ids']),'longest_supervised_tokens':e['supervised_tokens'],
  'base_parameters':coverage['base_parameters'],'adapter_parameters':coverage['adapter_parameters'],'total_parameters':coverage['total_parameters'],'CUDA_initialized':False,'all_parameters_meta':True,'elapsed_seconds':time.perf_counter()-started,
  'full124_encoding_not_repeated_here':'GPU probe still re-encodes exact124 and matches original encoded SHA before weight load; no old data/isolation/evaluator rerun.'})
 print('A800_CPU_DELTA_PASS_NO_WEIGHTS_NO_GPU')
if __name__=='__main__':main()
