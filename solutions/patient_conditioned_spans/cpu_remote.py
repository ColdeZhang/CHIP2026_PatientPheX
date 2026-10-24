"""CUDA-hidden full source/token/interface checks; never loads model weights."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys,time,gzip
from common import ref,read,save,unpack

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--model',required=True);ap.add_argument('--hpo',required=True);a=ap.parse_args();D=Path(a.root);started=time.monotonic();assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
 assets={p.name:ref(p) for p in sorted(Path(a.model).iterdir()) if p.is_file()};assert assets['pytorch_model.bin']['sha256']=='ad7bbb66376cfd6b2db3447192b034efe016337cbef135c35c411fd61b13c193';assert assets['vocab.txt']['sha256']=='79489a52be45e6fa033521e8ce8e4f62aedc0a742ee2aa6fc04667e5b0b1454d';assert ref(a.hpo)['sha256']=='a5092cbdf605f568403cf7380d9173014015692433b2cc631bc5c1b053876b1b';save(D/'assets.json',{'files':assets,'hpo':ref(a.hpo),'weights_loaded':False})
 from contracts import run
 save(D/'autograd.json',run())
 expected=json.loads((D/'input/preparation.json').read_text());cmd=[sys.executable,str(Path(__file__).with_name('prepare.py')),'--train',str(D/'input/train.jsonl'),'--input',str(D/'input/held.jsonl'),'--provenance',str(D/'input/provenance.json'),'--vocab',str(Path(a.model)/'vocab.txt'),'--hpo',a.hpo,'--output',str(D/'encoded')];save(D/'prepare-command.json',cmd);subprocess.run(cmd,check=True)
 actual=json.loads((D/'encoded/preparation.json').read_text());checks={}
 for split in ['train','held']:
  assert actual['files'][split]['decoded_sha256']==expected['files'][split]['decoded_sha256'];assert actual['splits'][split]==expected['splits'][split];checks[split]={'payload_sha256':actual['files'][split]['decoded_sha256'],'whole_geometry_exact':True}
 from transformers import AutoTokenizer
 tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True,use_fast=True);seen=[]
 for split in ['train','held']:
  docs=read(D/'input'/('train.jsonl' if split=='train' else 'held.jsonl'));rows=unpack(D/'encoded'/(split+'.json.gz'))
  for doc,row in zip(docs,rows):
   body='\n\n'.join(p['text'] for p in doc['full_text']);e=tok(body,add_special_tokens=False,return_offsets_mapping=True,truncation=False);ids=e['input_ids'];assert hashlib.sha256(json.dumps(ids).encode()).hexdigest()==row['token_ids_sha256'];assert list(map(list,e['offset_mapping']))==row['offsets'];seen.append({'pmc_id':row['pmc_id'],'tokens':len(ids),'ids_sha256':row['token_ids_sha256']})
 import importlib.metadata as md
 save(D/'cpu-complete.json',{'status':'PASS_FULL48_TOKEN_GEOMETRY_AND_PRODUCTION_AUTOGRAD','seconds':time.monotonic()-started,'geometry':checks,'tokenizer_class':type(tok).__name__,'documents':seen,'assets':ref(D/'assets.json'),'autograd':ref(D/'autograd.json'),'versions':{k:md.version(k) for k in ['torch','transformers','tokenizers']},'weights_loaded':False,'cuda_hidden':True,'source':ref(__file__)})
if __name__=='__main__':main()
