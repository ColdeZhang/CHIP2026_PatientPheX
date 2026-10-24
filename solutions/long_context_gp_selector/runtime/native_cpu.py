"""New host runtime/64 native inputs; tiny CPU real encoder/head interface only."""
import gzip,hashlib,json,os,time
from pathlib import Path
from common import ref,save,sha

def main():
 start=time.monotonic();assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
 import torch,transformers,tokenizers,numpy,huggingface_hub,safetensors
 from transformers import AutoTokenizer,AutoModel
 from model import FullContextGP
 assert transformers.__version__=='4.50.0' and tokenizers.__version__=='0.21.4'
 torch.set_num_threads(1);root=Path(__file__).resolve().parent;cfg=json.load(open(root/'formal-plan.json'));data=json.loads((root/'data.json').read_bytes());assets={}
 for n,r in cfg['model_files'].items():
  path=Path(cfg['model'])/n;actual=ref(path);assert all(actual[k]==r[k] for k in ['bytes','sha256']);st=path.stat();actual['guard']=dict(inode=st.st_ino,mtime_ns=st.st_mtime_ns);assets[n]=actual
 assert ref(cfg['GP23_checkpoint']['path'])==cfg['GP23_checkpoint'];tok=AutoTokenizer.from_pretrained(cfg['model'],local_files_only=True,use_fast=True);checked=[]
 for row in data['native_contracts']:
  e=tok(row['text'],add_special_tokens=False,truncation=False,return_offsets_mapping=True);assert e['input_ids']==row['ids'];raw=e['offset_mapping'];assert hashlib.sha256(json.dumps(raw,separators=(',',':')).encode()).hexdigest()==row['raw_offsets_sha256'];lex=[]
  for a,b in raw:
   while a<b and row['text'][a].isspace():a+=1
   while b>a and row['text'][b-1].isspace():b-=1
   lex.append((a,b))
  assert hashlib.sha256(json.dumps(lex,separators=(',',':')).encode()).hexdigest()==row['lexical_offsets_sha256'];assert tok(row['text'],add_special_tokens=True,truncation=False)['input_ids']==[50281]+e['input_ids']+[50282];checked.append(dict(pmc_id=row['pmc_id'],tokens=len(e['input_ids'])))
 assert len(checked)==64
 enc,info=AutoModel.from_pretrained(cfg['model'],local_files_only=True,torch_dtype=torch.float32,attn_implementation='sdpa',reference_compile=False,output_loading_info=True)
 assert not info.get('missing_keys') and not info.get('mismatched_keys') and not info.get('error_msgs') and all(k.startswith(('head.','decoder.')) for k in info.get('unexpected_keys',[]));model=FullContextGP(enc);assert sum(p.numel() for p in model.parameters())==149112704
 row=dict(ids=[50281,100,50282],blocks=[dict(indices=[1],positions=[0],positive=[(0,0)])]);model.train();loss=model(row);assert torch.isfinite(loss);loss.backward();assert torch.isfinite(model.head.weight.grad).all() and model.head.weight.grad.abs().sum()>0
 from safetensors.torch import load_file
 gp=load_file(cfg['GP23_checkpoint']['path']);model.load_state_dict(gp,strict=True);assert all(torch.equal(v,gp[k]) for k,v in model.state_dict().items())
 assert not torch.cuda.is_initialized();save(root/'cpu-complete.json',dict(status='PASS_NATIVE64_AND_TARGET_RUNTIME_INTERFACE_NO_CUDA',plan=ref(root/'formal-plan.json'),data=ref(root/'data.json'),assets=assets,all_docs=checked,tiny_native_encoder_GP_forward_backward=True,synthetic_tokens=3,optimizer_updates=0,old_loss_math_reused=True,seconds=time.monotonic()-start,runtime={n:dict(version=x.__version__,path=x.__file__) for n,x in [('torch',torch),('transformers',transformers),('tokenizers',tokenizers),('numpy',numpy),('huggingface_hub',huggingface_hub),('safetensors',safetensors)]},no_GPU=True))
if __name__=='__main__':main()
