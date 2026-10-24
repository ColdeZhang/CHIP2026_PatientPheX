"""Pure F0 geometry, independent native tokenizer full-input contract; no gold."""
from pathlib import Path
from collections import Counter
import argparse,json,time,hashlib
from tokenizers import BertWordPieceTokenizer
from transformers import AutoTokenizer
from prepare_geometry import geometry
from vendor.strict import HPO
from common import read,ref,save,packed

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('--input',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();p=json.loads(Path(a.plan).read_text());O=Path(a.output);O.mkdir(parents=True,exist_ok=False);t=time.monotonic();docs=read(a.input);assert ref(a.input)['sha256']==p['f0_input']['sha256'];assert len(docs)==16 and sum(len(d['patient']) for d in docs)==47 and all('entities' not in d and 'association' not in d for d in docs)
 ti={str(d['pmc_id']) for d in read(p['normalization']['train'])};hi={str(d['pmc_id']) for d in docs};assert len(ti)==32 and not ti&hi;assert ref(p['normalization']['train'])['sha256']==p['original_training']['train32_sha256']
 assets={n:ref(Path(p['model'])/n) for n in p['model_files']};assert {n:r['sha256'] for n,r in assets.items()}==p['model_files'];tok=BertWordPieceTokenizer(str(Path(p['model'])/'vocab.txt'),lowercase=True);native=AutoTokenizer.from_pretrained(p['model'],local_files_only=True,use_fast=True);hpo=HPO(p['normalization']['hpo']);rows=[];stats=[]
 for d in docs:
  text='\n\n'.join(x['text'] for x in d['full_text']);orig=tok.encode(text,add_special_tokens=False);actual=native(text,add_special_tokens=False,return_offsets_mapping=True);assert actual['input_ids']==orig.ids and [tuple(x) for x in actual['offset_mapping']]==orig.offsets
  r,s=geometry(d,tok,hpo,False,32);assert r['token_ids_sha256']==hashlib.sha256(json.dumps(orig.ids).encode()).hexdigest();assert 'bags' not in r and len(set(map(tuple,r['candidates'])))==len(r['candidates']);rows.append(r);stats.append({'pmc_id':r['pmc_id'],**s})
 f=packed(O/'held.json.gz',rows);save(O/'complete.json',{'status':'PASS_COMPLETE_PURE_F0_NATIVE_GEOMETRY','input':ref(a.input),'plan':ref(a.plan),'files':{'held':f},'documents':16,'patients':47,'train_ids':sorted(ti),'held_ids':sorted(hi),'tokens':sum(r['token_length'] for r in rows),'windows':sum(len(r['windows']) for r in rows),'candidates':sum(len(r['candidates']) for r in rows),'maximum_windows':max(len(r['windows']) for r in rows),'maximum_patient_candidates':max(len(r['patients'])*len(r['candidates']) for r in rows),'native_ids_offsets_exact':True,'all_original_tokens_covered':True,'training_or_gold':False,'by_document':stats,'seconds':time.monotonic()-t});print(json.dumps(json.loads((O/'complete.json').read_text())),flush=True)
if __name__=='__main__':main()
