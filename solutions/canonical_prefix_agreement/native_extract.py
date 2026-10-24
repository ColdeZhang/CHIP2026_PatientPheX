"""One bounded native-token replay of existing39 outputs; no model imports/calls."""
import argparse
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from prefix import trace
from contracts import run

def ref(p):
    p=Path(p).resolve();b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def checked(r):assert ref(r['path'])==r;return Path(r['path'])
def write(p,v):
    with Path(p).open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);a=ap.parse_args();start=time.monotonic();p=json.loads(Path(a.plan).read_text())
    assert os.environ['CUDA_VISIBLE_DEVICES']=='' and os.environ['OMP_NUM_THREADS']=='1'
    for r in p['sources'].values():checked(r)
    out=Path(p['output']);out.mkdir(parents=True,exist_ok=False);write(out/'started.json',dict(pid=os.getpid(),plan=ref(a.plan),no_NN=True))
    try:
        import tokenizers
        from tokenizers import Tokenizer
        assert tokenizers.__version__=='0.21.2' and 'torch' not in sys.modules
        spec=importlib.util.spec_from_file_location('_original_schema',checked(p['schema']));schema=importlib.util.module_from_spec(spec);spec.loader.exec_module(schema)
        contract=run(schema.CanonicalList);write(out/'contracts.json',contract)
        tk=Tokenizer.from_file(str(checked(p['tokenizer'])));names=json.loads(gzip.decompress(checked(p['names_gzip']).read_bytes()))
        assert hashlib.sha256(json.dumps(names,sort_keys=True,separators=(',',':')).encode()).hexdigest()==p['public_names_semantic_sha256']
        codec=schema.CanonicalList(tk,names,151645)
        oldlines=checked(p['canonical']).read_bytes().decode().splitlines(keepends=True);assert len(oldlines)==39
        lines=[];proofs=[];counts={'strict_unchanged':0,'caps':0,'original_relations':0,'prefix_relations':0}
        for index,line in enumerate(oldlines):
            row=json.loads(line);assert row['index']==index
            raw=json.loads(checked(p['raw_files'][f'{index:03d}-raw.json']).read_text());ids=json.loads(checked(p['raw_files'][f'{index:03d}-ids.json']).read_text())
            for key in ['row_id','pmc_id','target_patient_id','index','output_tokens','natural_eos','hit_token_limit']:
                assert raw[key]==ids[key]==row[key]
            assert len(ids['generated_ids'])==row['output_tokens']
            assert tk.decode(ids['generated_ids'],skip_special_tokens=False)==raw['raw_with_specials']
            result=trace(codec,ids['generated_ids'],raw['body'],row['natural_eos'])
            counts['original_relations']+=len(row['hpo_ids']);counts['prefix_relations']+=len(result['hpo_ids'])
            if row['strict_complete']:
                assert row['natural_eos'] and not row['hit_token_limit']
                assert schema.strict_parse(raw['body'],names,eos=True,capped=False)==row['hpo_ids']==result['hpo_ids']
                lines.append(line);counts['strict_unchanged']+=1
            else:
                assert row['hit_token_limit'] and not row['natural_eos'] and row['output_tokens']==512 and row['hpo_ids']==[]
                counts['caps']+=1;new={**row,'hpo_ids':result['hpo_ids'],'prefix_items_accepted':True,'original_strict_complete':False,
                    'interpretation':'New prefix-item policy; original full JSON remains incomplete and capped'}
                lines.append(json.dumps(new,ensure_ascii=False)+'\n')
            proofs.append(dict(index=index,row_id=row['row_id'],original_record=row,raw_ref=p['raw_files'][f'{index:03d}-raw.json'],ids_ref=p['raw_files'][f'{index:03d}-ids.json'],**result))
        assert counts['strict_unchanged']==27 and counts['caps']==12 and counts['original_relations']==55
        with (out/'prefix-canonical.jsonl').open('x') as f:f.write(''.join(lines));f.flush();os.fsync(f.fileno())
        with (out/'item-provenance.jsonl').open('x') as f:
            for r in proofs:f.write(json.dumps(r,ensure_ascii=False)+'\n')
            f.flush();os.fsync(f.fileno())
        write(out/'complete.json',dict(status='COMPLETE_ORIGINAL39_NATIVE_PREFIX_POLICY',rows=39,counts=counts,
            prefix=ref(out/'prefix-canonical.jsonl'),provenance=ref(out/'item-provenance.jsonl'),contracts=ref(out/'contracts.json'),
            tokenizer=ref(p['tokenizer']['path']),schema=p['schema'],sources=p['sources'],seconds=time.monotonic()-start,
            strict27_original_lines_byteexact=True,raw_ids_modified=False,NN_calls=0,model_imports=0,quality_calls=0))
    except BaseException as exc:write(out/'failure.json',dict(error=type(exc).__name__,message=str(exc)));raise

if __name__=='__main__':main()
