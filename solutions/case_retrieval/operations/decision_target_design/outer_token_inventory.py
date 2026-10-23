#!/usr/bin/env python3
"""Bounded CPU-only F0 slot inventory, reusing exact frozen input functions."""
from pathlib import Path
import argparse, ast, collections, copy, gzip, hashlib, json, os, re, time
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['TOKENIZERS_PARALLELISM']='false'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p):return [json.loads(line) for line in Path(p).read_text().splitlines() if line]
def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--model',required=True);a=p.parse_args();b=a.directory;started=time.monotonic()
    plan=json.loads((b/'plan.json').read_text());assert sha(__file__)==plan['source_sha256']
    for path,digest in plan['references'].items():assert sha(path)==digest
    import torch
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    assert not torch.cuda.is_initialized()
    tree=ast.parse(Path(plan['design_source']).read_text())
    main_fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    functions=[n for n in main_fn.body if isinstance(n,ast.FunctionDef) and n.name in ['template','slot_input']]
    assert len(functions)==2
    env=dict(tokenizer=tokenizer,copy=copy,json=json,hashlib=hashlib)
    exec(compile(ast.Module(body=functions,type_ignores=[]),plan['design_source'],'exec'),env)
    encode=env['slot_input']
    meta=rows(plan['F1_prompts']);known=json.loads(gzip.decompress(Path(plan['F1_encoded']).read_bytes()))
    assert [encode(row) for row in meta]==known
    outer=rows(plan['F0_prompts']);train=rows(plan['train32'])
    ids={str(x['pmc_id']) for x in outer};tid={str(x['pmc_id']) for x in train};mid={str(x['pmc_id']) for x in meta}
    assert len(ids)==16 and len(tid)==32 and len(mid)==16 and not ids&tid and not ids&mid and not tid&mid
    assert all('answer' not in row and 'targets' not in row and 'labels' not in row for row in outer)
    encoded=[encode(row) for row in outer];units=collections.Counter();compound=re.compile(r'HP:[0-9]{7}(?:;HP:[0-9]{7})+')
    non_atomic=[];expansions=0
    for row in outer:
        for cid,concept in row['code_map'].items():
            assert isinstance(concept,str)
            if not re.fullmatch(r'HP:[0-9]{7}',concept):non_atomic.append([row['row_id'],cid,concept])
            values=concept.split(';') if compound.fullmatch(concept) else [concept]
            expansions+=(len(values)-1)*len(row['patient_ids'])
            for pid in row['patient_ids']:
                for value in values:units[(str(row['pmc_id']),pid,value)]+=1
    conflicts=[dict(key=list(k),count=v) for k,v in units.items() if v!=1]
    lengths=[len(x['input_ids']) for x in encoded];pair_count=sum(len(x['keys']) for x in encoded)
    payload=gzip.compress(json.dumps(encoded,ensure_ascii=False,separators=(',',':')).encode(),mtime=0)
    (b/'F0-slot-inputs.json.gz').write_bytes(payload)
    report=dict(status='CPU_only_input_inventory_complete',rows=len(outer),docs=len(ids),CIDs=sum(len(x['code_map']) for x in outer),
        pairs=pair_count,patients=len({k[:2] for k in units}),literal_units=len(units),compound_expansion_pairs=expansions,
        duplicate_literal_conflicts=conflicts,one_to_one=not conflicts and not expansions,non_atomic_HPO=non_atomic,
        full_tokens=sum(lengths),max_tokens=max(lengths),over7168=sum(x>7168 for x in lengths),
        row_lengths=lengths,slot_position_count=pair_count,F1_exact_replay_rows=len(meta),
        source_sha256=sha(__file__),plan_sha256=sha(b/'plan.json'),encoded_sha256=sha(b/'F0-slot-inputs.json.gz'),
        training_docs=sorted(tid),F0_docs=sorted(ids),F1_docs=sorted(mid),disjoint=True,
        cost_seconds_token_linear=55.779866865836084*sum(lengths)/268028,
        cost_boundary='Prior F1 adapter80 actual55.7799s forward scaled by tokens, not actual F0 or loading estimate.',
        no_GPU_initialized=not torch.cuda.is_initialized(),no_F0_labels_or_quality=True,
        model_scope='binary32 is trained only F2+F3; old JSON SFT48 uses F1+F2+F3. Both exclude F0, but are different training sizes.',
        elapsed_seconds=time.monotonic()-started)
    assert report['elapsed_seconds']<60
    with (b/'summary.json').open('x') as f:f.write(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report),flush=True)
if __name__=='__main__':main()
