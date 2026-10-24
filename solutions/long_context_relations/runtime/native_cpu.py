"""Same native IDs, full candidate/owner coverage, and finite pooled-head gradient."""
import gzip,json,os
from pathlib import Path
from common import save,ref

def main():
 assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
 import torch,transformers
 from types import SimpleNamespace
 from model import FullContextRelations
 torch.set_num_threads(1);assert not torch.cuda.is_initialized()
 root=Path(__file__).resolve().parent;cfg=json.loads((root/'execution-plan.json').read_text())
 assert ref(root/'data.json.gz')['sha256']==cfg['data']['sha256']
 data=json.loads(gzip.decompress((root/'data.json.gz').read_bytes()))
 parent=json.loads(Path(cfg['parent_cpu']['path']).read_text());assert ref(cfg['parent_cpu']['path'])['sha256']==cfg['parent_cpu']['sha256']
 assert ref(cfg['parent_native_data']['path'])['sha256']==cfg['parent_native_data']['sha256']
 old=json.loads(gzip.decompress(Path(cfg['parent_native_data']['path']).read_bytes()));contracts={x['pmc_id']:x for x in old['native_contracts']}
 summary={}
 for split in ('train','held'):
  rows=data[split];assert len(rows)==(32 if split=='train' else 16)
  for row in rows:
   assert ('labels'in row)==(split=='train');restored=[None]*len(contracts[row['pmc_id']]['ids'])
   for window in row['windows']:
    lo,hi=window['body_range'];assert window['ids'][0]==50281 and window['ids'][-1]==50282
    for i,t in enumerate(window['ids'][1:-1],lo):assert restored[i] in (None,t);restored[i]=t
   assert restored==contracts[row['pmc_id']]['ids']
   for e in row['entities']:
    assert len(e['spans'])==len(set(tuple(x) for x in e['spans']))
    for i,j,owner,pi in e['spans']:
     lo,hi=row['windows'][owner]['body_range'];assert lo<=i<j<=hi
   assert len(row['pairs'])==len(row['keys']) and len({tuple(k) for k in row['keys']})==len(row['keys'])
  summary[split]=dict(documents=len(rows),pairs=sum(len(x['pairs']) for x in rows),windows=sum(len(x['windows']) for x in rows),physical_mentions=sum(len(e['spans']) for x in rows for e in x['entities']))
 assert summary['train']['pairs']==4495 and sum(sum(x['labels'])for x in data['train'])==727 and summary['held']['pairs']==2874
 class FakeEncoder(torch.nn.Module):
  def __init__(self):super().__init__();self.config=SimpleNamespace(hidden_size=4);self.embedding=torch.nn.Embedding(20,4)
  def forward(self,input_ids,**kw):
   assert not kw['output_attentions'] and not kw['output_hidden_states'];return SimpleNamespace(last_hidden_state=self.embedding(input_ids))
 torch.manual_seed(7);m=FullContextRelations(FakeEncoder());m.eval()
 doc=dict(windows=[dict(ids=[0,1,2,3,4,5,6,0],body_range=[0,6])],entities=[dict(spans=[(0,2,0,0),(3,4,0,0)]),dict(spans=[(4,6,0,0)]),dict(spans=[(4,6,0,0)])],pairs=[[0,1],[0,2]])
 states=m.entity_states(doc);values=m.encoder.embedding(torch.tensor([1,2,3,4,5,6]))
 expected=torch.stack([torch.logsumexp(torch.stack([values[:2].mean(0),values[3]]),0),values[4:6].mean(0),values[4:6].mean(0)])
 torch.testing.assert_close(states,expected,rtol=0,atol=0)
 logits=m(doc);assert torch.equal(logits[0],logits[1]),'same-span identities remain indistinguishable'
 torch.nn.functional.binary_cross_entropy_with_logits(logits,torch.tensor([1.,0.])).backward()
 for n,p in m.named_parameters():assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.count_nonzero()>0,n
 assets={}
 for name,r in parent['assets'].items():
  p=Path(cfg['model'])/name;st=p.stat();assert st.st_size==r['bytes'] and st.st_ino==r['guard']['inode'] and st.st_mtime_ns==r['guard']['mtime_ns'];assets[name]=r
 assert not torch.cuda.is_initialized()
 save(root/'cpu-complete.json',dict(status='PASS_FIXED_NATIVE_IDS_ALL_PAIR_POOLING_NO_CUDA',plan=ref(root/'execution-plan.json'),data=ref(root/'data.json.gz'),parent_cpu=cfg['parent_cpu'],assets=assets,summary=summary,positive_pairs=727,head_gradient_tensors=len(list(m.parameters())),tiny_pooling_exact=True,same_span_id_limit_preserved=True,torch=torch.__version__,transformers=transformers.__version__,pretrained_model_loads=0,held_gold_reads=0))
 print('CPU_COMPLETE')
if __name__=='__main__':main()
