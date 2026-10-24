"""Load the exact final DocRE30 checkpoint; infer all F0 pairs once, no quality."""
import argparse
import gzip
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import time
import torch
from transformers import AutoModel
sys.path.insert(0,str(Path(__file__).parent/'vendor'))
from model import WindowDocRE
from driver import sha, write

def ref(path):
    p=Path(path);return {'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)}
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
    plan=json.loads(Path(a.plan).read_text());out=Path(a.output);out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert torch.cuda.is_available()
    docs=json.loads(gzip.decompress(Path(plan['encoded']).read_bytes()))
    assert len(docs)==16 and sorted(d['pmc_id'] for d in docs)==plan['held_ids'];assert all('labels' not in d for d in docs)
    keys=[tuple(k) for d in docs for k in d['keys']];assert len(keys)==len(set(keys))==plan['expected_pairs']
    encoder,loading=AutoModel.from_pretrained(plan['model'],local_files_only=True,attn_implementation='eager',output_loading_info=True)
    assert not loading.get('missing_keys') and not loading.get('mismatched_keys') and not loading.get('error_msgs')
    model=WindowDocRE(encoder)
    state=torch.load(plan['checkpoint']['path'],map_location='cpu',weights_only=False)
    assert state['epoch']==30 and state['global_step']==240 and len(state['model'])==205
    model.load_state_dict(state['model'],strict=True)
    assert all(torch.equal(v,state['model'][n]) for n,v in model.state_dict().items())
    del state
    parameters=sum(p.numel() for p in model.parameters());assert parameters==110277377<10_000_000_000
    model=model.cuda();model.eval()
    write(out/'manifest.json',{'plan':ref(a.plan),'checkpoint':plan['checkpoint'],'loaded_epoch':30,'loaded_global_step':240,'state_tensors':205,'strict_load_reopened_state_exact':True,'parameters':parameters,'loading':loading,'versions':{k:importlib.metadata.version(k) for k in ['torch','transformers','tokenizers']},'precision':'FP32 parameters + BF16 autocast, TF32 false','no_training_or_quality':True,'encoded':ref(plan['encoded'])})
    scores=out/'held-scores.jsonl';inference=[]
    with scores.open('x') as f:
        for doc in docs:
            begin=time.monotonic()
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):logits,diag=model(doc)
            values=logits.float().cpu().tolist();assert len(values)==len(doc['keys']) and all(math.isfinite(v) for v in values)
            for key,value in zip(doc['keys'],values):
                probability=1/(1+math.exp(-max(-80.,min(80.,value))))
                f.write(json.dumps({'key':key,'logit':value,'probability':probability},ensure_ascii=False)+'\n')
            f.flush();os.fsync(f.fileno())
            inference.append({'pmc_id':doc['pmc_id'],'pairs':len(values),**diag,'seconds':time.monotonic()-begin})
    assert sum(r['pairs'] for r in inference)==plan['expected_pairs']
    write(out/'complete.json',{'status':'complete_all_F0_pairs_no_training_no_quality','scores':ref(scores),'checkpoint':plan['checkpoint'],'inference':inference,'elapsed_seconds':time.monotonic()-started,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()})
if __name__=='__main__':main()
