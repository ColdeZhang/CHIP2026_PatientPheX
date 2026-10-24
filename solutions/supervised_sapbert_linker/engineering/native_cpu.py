"""All native tokenizer inputs and base-asset identity, CUDA hidden, no model."""
import gzip
import json
import os
from pathlib import Path
import time
from common import ref,save,sha


def main():
    t=time.monotonic();root=Path(__file__).resolve().parent
    cfg=json.loads((root/'execution-plan.json').read_text())
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    assert sha(root/'data.json.gz')==cfg['data']['sha256']
    import torch,transformers,tokenizers
    from transformers import AutoTokenizer
    torch.set_num_threads(1)
    assert not torch.cuda.is_initialized()
    actual={}
    for name,expected in cfg['model_files'].items():
        p=Path(cfg['model'])/name
        actual[name]=ref(p)
        assert actual[name]['sha256']==expected,(name,actual[name])
        st=p.stat();actual[name]['guard']=dict(inode=st.st_ino,mtime_ns=st.st_mtime_ns)
    with open(Path(cfg['model'])/'model.safetensors','rb') as f:
        header=json.loads(f.read(int.from_bytes(f.read(8),'little')))
    from collections import Counter
    dtypes=Counter(v['dtype'] for k,v in header.items() if k!='__metadata__')
    assert set(dtypes)=={'F16'},dtypes
    data=json.loads(gzip.decompress((root/'data.json.gz').read_bytes()))
    tok=AutoTokenizer.from_pretrained(cfg['model'],local_files_only=True,use_fast=True)
    groups={}
    allids={}
    for name in ['train','bank','held']:
        texts=[x['text'] for x in data[name]]
        encoded=tok(texts,add_special_tokens=True,padding=False,truncation=False)
        ids=encoded['input_ids']; expected=[x['ids'] for x in data[name]]
        assert ids==expected, name+' native/local token mismatch'
        assert len(ids)==cfg['counts'][name] and max(map(len,ids))<=32
        assert all(a[0]==tok.cls_token_id and a[-1]==tok.sep_token_id for a in ids)
        groups[name]=dict(rows=len(ids),tokens=sum(map(len,ids)),maximum=max(map(len,ids)),truncated=0,
            input_ids_sha256=__import__('hashlib').sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest())
        allids[name]=ids
    assert not torch.cuda.is_initialized()
    save(root/'cpu-complete.json',dict(status='PASS_ALL_NATIVE_IDS_NO_MODEL_OR_CUDA',elapsed_seconds=time.monotonic()-t,
        groups=groups,assets=actual,python=__import__('sys').version,torch=torch.__version__,transformers=transformers.__version__,
        tokenizers=tokenizers.__version__,tokenizer_class=type(tok).__name__,tokenizer_specials=tok.special_tokens_map,
        checkpoint_header_dtypes=dict(dtypes),
        data=ref(root/'data.json.gz'),plan=ref(root/'execution-plan.json')))


if __name__=='__main__':main()
