"""Necessary held quote queries against frozen public bank; no bank encode/fit."""
import argparse,hashlib,json,os,time
from pathlib import Path
import numpy as np
from strict import HPO,sha256
from semantic import PublicConceptBank,PublicSapEncoder,load_policy,precheck

def put(p,b):
 p=Path(p)
 with p.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
 d=p.parent
 while True:
  fd=os.open(d,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
  if d.parent==d:break
  d=d.parent
def save(p,v):put(p,(json.dumps(v,ensure_ascii=False,indent=2)+'\n').encode())
def ref(p):p=Path(p).resolve();return dict(path=str(p),bytes=p.stat().st_size,sha256=sha256(p))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);a=ap.parse_args();p=json.loads(Path(a.plan).read_text());start=time.monotonic()
 assert p['schema']=='grounded-fixed-held-public-queries-v1' and p['authorized'] and p['complete_necessary_queries']
 for name,item in p['files'].items():assert Path(name).stat().st_size==item['bytes'] and sha256(name)==item['sha256']
 policy=load_policy(p['policy']);assert sha256(p['policy'])=='b59eb3f8845ff297578241ed61192fb8ae0153587deab924c51b0a33a5e8bd20'
 queries=json.loads(Path(p['queries']).read_text());quotes=[q['quote'] for q in queries]
 assert quotes==sorted(set(quotes)) and len(quotes)==p['query_count'] and all(precheck(q,policy) is None for q in quotes)
 hpo=HPO(p['hpo']);bank=PublicConceptBank(hpo,p['bank'],p['labels'],p['bank_manifest']);assert len(bank.rows)==41099
 out=Path(p['output_dir']);out.mkdir(parents=True,exist_ok=False);save(out/'started.json',dict(plan=ref(a.plan),source=ref(__file__),queries=len(quotes),no_bank_reencoding=True,no_fit=True))
 assert os.environ['CUDA_VISIBLE_DEVICES']=='' and p['execution_device']=='cpu' and p['execution_dtype']=='float32' and p['threads']==16
 import torch
 torch.set_num_interop_threads(1)
 encoder=PublicSapEncoder(p['model_dir'],'cpu');torch.set_num_threads(16)
 assert all(x.device.type=='cpu' and x.dtype==torch.float32 for x in encoder.model.parameters())
 assert not torch.cuda.is_initialized() and torch.get_num_threads()==16
 encoder.manifest.update(dtype='torch.float32',threads=16,interop_threads=1,CUDA_initialized=False,original_GPU_parameter_dtype='torch.float16',FP16_to_FP32_amendment=p['normalization_amendment'])
 save(out/'encoder-runtime.json',encoder.manifest)
 assert all(encoder.manifest[k]==v for k,v in p['expected_encoder_runtime'].items())
 vectors,counts=encoder.encode(quotes)
 import io
 buf=io.BytesIO();np.save(buf,vectors,allow_pickle=False);put(out/'query-vectors.npy',buf.getvalue());save(out/'query-token-counts.json',dict(quotes=quotes,token_counts=counts,query_vectors=ref(out/'query-vectors.npy')))
 scores=[dict(quote=q,token_count=n,candidates=bank.rank(v) if n<=32 else []) for q,n,v in zip(quotes,counts,vectors)]
 save(out/'query-scores.json',scores)
 save(out/'complete.json',dict(status='COMPLETE_FIXED_PUBLIC_HELD_QUERIES',query_count=len(quotes),public_bank_rows=41099,queries=ref(p['queries']),policy=ref(p['policy']),plan=ref(a.plan),source=ref(__file__),seconds=time.monotonic()-start,scores=ref(out/'query-scores.json'),vectors=ref(out/'query-vectors.npy'),counts=ref(out/'query-token-counts.json'),encoder=ref(out/'encoder-runtime.json'),no_bank_reencoding=True,no_fit=True,no_quality=True,execution_device='cpu',execution_dtype='float32',threads=16,normalization_amendment=p['normalization_amendment']))
if __name__=='__main__':main()
