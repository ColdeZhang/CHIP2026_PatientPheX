"""Exact native HF IDs/offsets and selected-block GP math, CUDA hidden."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import time
from common import save,ref,sha


def main():
    start=time.monotonic();assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    import torch,transformers,tokenizers
    from transformers import AutoTokenizer
    from model import rotate,pair_loss
    torch.set_num_threads(1)
    root=Path(__file__).resolve().parent;cfg=json.loads((root/'execution-plan.json').read_text())
    data=json.loads(gzip.decompress((root/'data.json.gz').read_bytes()))
    assets={}
    for n,v in cfg['model_files'].items():
        p=Path(cfg['model'])/n;r=ref(p);assert r['sha256']==v['sha256'] and r['bytes']==v['bytes']
        st=p.stat();r['guard']={'inode':st.st_ino,'mtime_ns':st.st_mtime_ns};assets[n]=r
    tok=AutoTokenizer.from_pretrained(cfg['model'],local_files_only=True,use_fast=True)
    checked=[]
    for row in data['native_contracts']:
        e=tok(row['text'],add_special_tokens=False,truncation=False,return_offsets_mapping=True)
        assert e['input_ids']==row['ids'],row['pmc_id']
        raw=e['offset_mapping'];assert hashlib.sha256(json.dumps(raw,separators=(',',':')).encode()).hexdigest()==row['raw_offsets_sha256']
        lex=[]
        for a,b in raw:
            while a<b and row['text'][a].isspace():a+=1
            while b>a and row['text'][b-1].isspace():b-=1
            lex.append((a,b))
        assert hashlib.sha256(json.dumps(lex,separators=(',',':')).encode()).hexdigest()==row['lexical_offsets_sha256']
        # Fast-tokenizer native post_processor, not the generic Python helper
        # which may intentionally return an identity sequence.
        assert tok(row['text'],add_special_tokens=True,truncation=False)['input_ids']==[50281]+e['input_ids']+[50282]
        checked.append({'pmc_id':row['pmc_id'],'tokens':len(e['input_ids'])})
    assert len(checked)==48 and len(data['train'])==32 and len(data['held'])==17
    assert len(data['target_coverage']['accepted_keys'])==2266 and len(data['target_coverage']['aligned_keys'])==2265
    assert all('positive' not in b for row in data['held'] for b in row['blocks'])
    # A real differentiable head-only CPU contract for the newly partitioned
    # span matrix: cross-passage logits do not enter either reduction, nested
    # positives coexist, and a shared position shift preserves RoPE scores.
    g=torch.Generator().manual_seed(20261024);q=torch.randn(9,64,generator=g,requires_grad=True);k=torch.randn(9,64,generator=g,requires_grad=True)
    pos=torch.arange(9,dtype=torch.float32);dense=rotate(q,pos)@rotate(k,pos).T/8
    shifted=rotate(q,pos+200)@rotate(k,pos+200).T/8
    assert torch.allclose(dense,shifted,atol=3e-5,rtol=3e-5)
    groups=[[1,2,3],[5,6,7,8]];gold=[[(0,2),(1,2)],[]];a=[];b=[]
    for idx,target in zip(groups,gold):
        i=torch.tensor(idx);score=rotate(q[i],pos[i])@rotate(k[i],pos[i]).T/8
        assert torch.allclose(score,dense[i][:,i],atol=1e-6,rtol=1e-6)
        a.append(pair_loss(score,target));b.append(pair_loss(dense[i][:,i],target))
    loss=torch.stack(a).mean();assert torch.allclose(loss,torch.stack(b).mean(),atol=1e-6)
    loss.backward();assert torch.isfinite(q.grad).all() and q.grad[groups[0]].abs().sum()>0
    assert q.grad[0].count_nonzero()==0 and k.grad[4].count_nonzero()==0
    assert not torch.cuda.is_initialized()
    save(root/'cpu-complete.json',{'status':'PASS_ALL_NATIVE_IDS_LABELS_AND_LOCAL_GP_CONTRACT_NO_CUDA',
        'plan':ref(root/'execution-plan.json'),'data':ref(root/'data.json.gz'),'assets':assets,'all_docs':checked,
        'train_windows':32,'held_windows':17,'accepted':2266,'aligned':2265,'held_gold_reads':0,
        'model_loads':0,'head_only_CPU_contract':True,'seconds':time.monotonic()-start,
        'runtime':{'torch':torch.__version__,'transformers':transformers.__version__,'tokenizers':tokenizers.__version__}})


if __name__=='__main__':main()
