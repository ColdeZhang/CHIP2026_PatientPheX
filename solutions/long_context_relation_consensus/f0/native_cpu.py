"""Full16 F0 native tokenizer/owner check; no pretrained model or CUDA."""
import gzip,hashlib,json,os,time
from pathlib import Path
from common import save,ref,sha

def main():
 start=time.monotonic();assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
 import torch,transformers,tokenizers
 from transformers import AutoTokenizer
 torch.set_num_threads(1);root=Path(__file__).resolve().parent;cfg=json.loads((root/'infer-plan.json').read_text())
 assert sha(root/'data.json.gz')==cfg['data']['sha256']
 old=json.loads(Path(cfg['parent_cpu']['path']).read_text());assert ref(cfg['parent_cpu']['path'])==cfg['parent_cpu']
 for n,v in old['assets'].items():
  p=Path(cfg['model'])/n;st=p.stat();assert st.st_size==v['bytes'] and st.st_ino==v['guard']['inode'] and st.st_mtime_ns==v['guard']['mtime_ns']
 data=json.loads(gzip.decompress((root/'data.json.gz').read_bytes()));assert len(data['held'])==len(data['native_contracts'])==16
 tok=AutoTokenizer.from_pretrained(cfg['model'],local_files_only=True,use_fast=True);seen=[]
 for doc,c in zip(data['held'],data['native_contracts']):
  assert doc['pmc_id']==c['pmc_id'] and 'labels'not in doc
  enc=tok(c['text'],add_special_tokens=False,truncation=False,return_offsets_mapping=True)
  assert enc['input_ids']==c['ids']
  assert hashlib.sha256(json.dumps(enc['offset_mapping'],separators=(',',':')).encode()).hexdigest()==c['raw_offsets_sha256']
  lex=[]
  for a,b in enc['offset_mapping']:
   while a<b and c['text'][a].isspace():a+=1
   while b>a and c['text'][b-1].isspace():b-=1
   lex.append((a,b))
  assert hashlib.sha256(json.dumps(lex,separators=(',',':')).encode()).hexdigest()==c['lexical_offsets_sha256']
  assert tok(c['text'],add_special_tokens=True,truncation=False)['input_ids']==[50281]+c['ids']+[50282]
  restored=[None]*len(c['ids'])
  for w in doc['windows']:
   lo,hi=w['body_range'];assert w['ids']==[50281]+c['ids'][lo:hi]+[50282] and len(w['ids'])<=8192
   for i,t in enumerate(w['ids'][1:-1],lo):assert restored[i]in(None,t);restored[i]=t
  assert restored==c['ids']
  for e in doc['entities']:
   assert len(e['spans'])==len(set(map(tuple,e['spans'])))
   for i,j,owner,pi in e['spans']:
    lo,hi=doc['windows'][owner]['body_range'];assert lo<=i<j<=hi
  seen.append(doc['pmc_id'])
 assert sum(len(d['pairs'])for d in data['held'])==2195
 assert sum(e['kind']=='patient'for d in data['held']for e in d['entities'])==47
 assert not torch.cuda.is_initialized()
 save(root/'cpu-complete.json',dict(status='PASS_ALL_F0_NATIVE_IDS_OWNERS_NO_CUDA',data=ref(root/'data.json.gz'),plan=ref(root/'infer-plan.json'),
  docs=16,patients=47,pairs=2195,windows=16,assets=old['assets'],model_calls=0,held_gold_reads=0,seconds=time.monotonic()-start,
  runtime=dict(torch=torch.__version__,transformers=transformers.__version__,tokenizers=tokenizers.__version__)))
 print('ALL_F0_NATIVE_CPU_PASS')

if __name__=='__main__':main()
