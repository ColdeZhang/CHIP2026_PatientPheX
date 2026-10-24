"""F0-only adapter of frozen DocRE public-candidate and full-window helpers."""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import sys
sys.path.insert(0, str(Path(__file__).parent / 'vendor'))
from prepare import build
from encode import encode_document
from marker_encode import transform, contracts

def ref(path):
    p=Path(path);return {'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def write(p,obj):
    p=Path(p)
    with p.open('x') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('--mode',choices=['public','encode'],required=True);ap.add_argument('--output',required=True);ap.add_argument('--prepared');ap.add_argument('--model');a=ap.parse_args()
    plan=json.loads(Path(a.plan).read_text());out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if a.mode=='public':
        source=plan['input'];assert ref(source['path'])==source
        docs=[json.loads(x) for x in Path(source['path']).read_text().splitlines()];ids={str(d['pmc_id']) for d in docs}
        assert len(docs)==len(ids)==16 and sorted(ids)==plan['held_ids']
        assert all('entities' not in d and 'association' not in d for d in docs)
        candidates=defaultdict(lambda:defaultdict(dict));counts={}
        for source in plan['public_caches']:
            assert ref(source['path'])==source;count=0
            for line in Path(source['path']).read_text().splitlines():
                row=json.loads(line);pmc=str(row['pmc_id'])
                if pmc not in ids:continue
                concept=str(row['hpo_id']);assert re.fullmatch(r'HP:\d{7}',concept)
                key=(int(row['offset']),int(row['length']),row['text'])
                candidates[pmc][concept][key]={k:row[k] for k in ['offset','length','text']};count+=1
            counts[source['path']]=count
        rows=[build(d,candidates[str(d['pmc_id'])],training=False) for d in docs]
        assert all('labels' not in r for r in rows)
        keys=[tuple(k) for r in rows for k in r['keys']];assert len(set(keys))==len(keys)
        for r,d in zip(rows,docs):
            patients=[e['id'] for e in r['entities'] if e['kind']=='patient'];concepts=[e['id'] for e in r['entities'] if e['kind']=='phenotype']
            assert patients==[str(p['patient_id']) for p in d['patient']]
            assert r['keys']==[[str(d['pmc_id']),p,h] for p in patients for h in concepts]
        path=out/'prepared.json.gz';path.write_bytes(gzip.compress(json.dumps(rows,ensure_ascii=False).encode(),mtime=0))
        write(out/'preparation.json',{'status':'complete_all_F0_public_pairs_no_gold','plan':ref(a.plan),'input':plan['input'],'public_caches':plan['public_caches'],'prepared':ref(path),'documents':len(rows),'patients':sum(len(d['patient']) for d in docs),'pairs':len(keys),'physical_mentions':sum(r['unique_physical_mentions'] for r in rows),'source_rows':counts,'per_document':[{'pmc_id':r['pmc_id'],'pairs':len(r['pairs']),'patients':sum(e['kind']=='patient' for e in r['entities'])} for r in rows]})
    else:
        from transformers import AutoTokenizer
        import transformers
        for name,expected in plan['tokenizer_assets'].items():assert ref(Path(a.model)/name)['sha256']==expected
        tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True,use_fast=True);assert tok.is_fast
        assert tok.convert_tokens_to_ids('*')==14
        contract=contracts();rows=json.loads(gzip.decompress(Path(a.prepared).read_bytes()));assert len(rows)==16 and all('labels' not in r for r in rows)
        encoded=[encode_document(r,tok) for r in rows];marked=[transform(r,14) for r in encoded]
        for raw,e,m in zip(rows,encoded,marked):
            assert raw['pairs']==e['pairs']==m['pairs'] and raw['keys']==e['keys']==m['keys'];assert 'labels' not in m
        dest=out/'held.json.gz';dest.write_bytes(gzip.compress(json.dumps(marked).encode(),mtime=0))
        write(out/'encoding.json',{'status':'complete_full_tokens_all_pairs_unique_owner_markers','input':ref(a.prepared),'encoded':ref(dest),'tokenizer_class':type(tok).__name__,'transformers':transformers.__version__,'contracts':contract,'documents':len(marked),'patients':sum(sum(e['kind']=='patient' for e in r['entities']) for r in marked),'pairs':sum(len(r['pairs']) for r in marked),'windows':sum(len(r['windows']) for r in marked),'maximum_windows':max(len(r['windows']) for r in marked),'tokens':sum(r['token_length'] for r in marked),'physical_mentions':sum(len(r['ownership']) for r in marked),'per_document':[{'pmc_id':r['pmc_id'],'pairs':len(r['pairs']),'windows':len(r['windows']),'tokens':r['token_length']} for r in marked]})
if __name__=='__main__':main()
