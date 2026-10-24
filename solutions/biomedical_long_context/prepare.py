"""Encode complete train32 and label-free F1, preserving original GP admission."""
import ast
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import re
from tokenizers import Tokenizer
from train_census import lexical, chunks, ref


def encode_docs(docs,tok,allowed,aliases,training):
    rows=[];contracts=[];accepted=set();aligned=set();excluded=[];unrepresentable=[]
    for doc in docs:
        pmc=str(doc['pmc_id']);text='\n\n'.join(p['text'] for p in doc['full_text'])
        if not training:assert 'entities' not in doc and 'association' not in doc
        enc=tok.encode(text,add_special_tokens=False);off=lexical(text,enc.offsets)
        maps=[];at=0;token_owner={};golds={}
        for pi,p in enumerate(doc['full_text']):
            t=p['text'];a=at;b=a+len(t);maps.append((a,b,p));at=b+2
            inds=[i for i,(s,e) in enumerate(off) if a<=s<e<=b]
            for i in inds:assert i not in token_owner;token_owner[i]=pi
            starts={off[i][0]:i for i in inds};ends={off[i][1]:i for i in inds}
            positive=[]
            for ent in doc.get('entities',[]) if training else []:
                eo=int(ent['offset']);ee=eo+int(ent['length']);base=int(p['offset'])
                if not base<=eo<ee<=base+len(t):continue
                if str(ent.get('note','')).upper()=='NO':continue
                ids={aliases.get(x.strip(),x.strip()) for x in str(ent['identifier']).split(';')}
                if str(ent['identifier'])!='-1' and not ids&allowed:
                    excluded.append((pmc,pi,eo,ee,str(ent['identifier'])));continue
                s=a+eo-base;e=s+int(ent['length']);assert text[s:e]==ent['text']
                key=(pmc,pi,eo,ee);accepted.add(key)
                if s in starts and e in ends:
                    i,j=starts[s],ends[e];assert i<=j
                    positive.append((i,j,key));aligned.add(key)
                else:unrepresentable.append({'key':key,'reason':'native_subtoken_endpoint','text':ent['text']})
            golds[pi]=positive
        assert len(token_owner)==sum(e>s for s,e in off),'Token crossing source passage'
        seen_tokens=set();seen_positive=set()
        for wi,(lo,hi) in enumerate(chunks(len(enc.ids),8190,512)):
            row={'row_id':pmc+':'+str(wi),'pmc_id':pmc,'window_index':wi,'body_range':[lo,hi],
                 'ids':[50281]+enc.ids[lo:hi]+[50282],'blocks':[]}
            for pi,(a,b,p) in enumerate(maps):
                inds=[i for i in range(lo,hi) if token_owner.get(i)==pi]
                if not inds:continue
                lookup={i:z for z,i in enumerate(inds)}
                block={'passage_index':pi,'passage_offset':int(p['offset']),
                       'indices':[i-lo+1 for i in inds],
                       'positions':inds,'offsets':[[off[i][0]-a,off[i][1]-a] for i in inds]}
                if training:
                    block['positive']=sorted({(lookup[i],lookup[j]) for i,j,k in golds[pi] if i in lookup and j in lookup})
                    seen_positive|={k for i,j,k in golds[pi] if i in lookup and j in lookup}
                row['blocks'].append(block)
            row['shape']={'encoder_tokens':len(row['ids']),
                          'local_square_cells':sum(len(b['indices'])**2 for b in row['blocks']),
                          'max_block_tokens':max(len(b['indices']) for b in row['blocks']),
                          'positive_spans':sum(len(b.get('positive',[])) for b in row['blocks'])}
            rows.append(row);seen_tokens.update(range(lo,hi))
        assert seen_tokens==set(range(len(enc.ids)))
        assert seen_positive=={k for vals in golds.values() for _,_,k in vals}
        def digest_offsets(x):return hashlib.sha256(json.dumps(x,separators=(',',':')).encode()).hexdigest()
        contracts.append({'pmc_id':pmc,'text':text,'ids':enc.ids,
                          'raw_offsets_sha256':digest_offsets(enc.offsets),
                          'lexical_offsets_sha256':digest_offsets(off),'body_tokens':len(enc.ids)})
    return rows,contracts,{'accepted_keys':sorted(accepted),'aligned_keys':sorted(aligned),
                           'excluded_nonphenotypic':excluded,'unrepresentable':unrepresentable}


def main():
    s=Path(__file__).resolve().parent;plan=json.loads((s/'candidate-plan.json').read_text())
    d=Path('/workspace/patientphex-runs/round2-20261023/research/biomedical-long-context-readiness-20261024')
    out=d/'probe-v1';out.mkdir(exist_ok=True)
    tokenizer=d/'public/tokenizer.txt';tok=Tokenizer.from_file(str(tokenizer));tok.no_truncation();tok.no_padding()
    hposrc=Path('/workspace/patientphex-worktrees/round2-sequence-crf/solutions/sequence_crf/run.py')
    tree=ast.parse(hposrc.read_text());fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='parse_hpo')
    ns={'Path':Path,'re':re};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(hposrc),'exec'),ns)
    _,allowed,aliases,version=ns['parse_hpo']('/workspace/patientphex-data/hp.obo')
    trpath=Path(plan['supervision']['train_source']['path']);vpath=Path(json.loads((d/'held-input-census.json').read_text())['input']['path'])
    tr=[json.loads(x) for x in trpath.read_text().splitlines()];va=[json.loads(x) for x in vpath.read_text().splitlines()]
    assert len(tr)==32 and len(va)==16 and not {x['pmc_id'] for x in tr}&{x['pmc_id'] for x in va}
    train,tc,target=encode_docs(tr,tok,allowed,aliases,True);held,hc,_=encode_docs(va,tok,allowed,aliases,False)
    assert len(target['accepted_keys'])==2266 and len(target['aligned_keys'])==2265
    assert len(target['excluded_nonphenotypic'])==8 and len(target['unrepresentable'])==1
    assert len(train)==32 and len(held)==17 and sum(len(x['patient']) for x in va)==39
    # Exact same positive admission as legacy GP, independently enumerated from
    # original coordinates (before any native tokenizer representation).
    old=set()
    for doc in tr:
        for pi,p in enumerate(doc['full_text']):
            for e in doc['entities']:
                ids={aliases.get(x.strip(),x.strip()) for x in str(e['identifier']).split(';')}
                if str(e.get('note','')).upper()=='NO' or not(str(e['identifier'])=='-1' or ids&allowed):continue
                start=int(e['offset']);end=start+int(e['length'])
                if int(p['offset'])<=start<end<=int(p['offset'])+len(p['text']):old.add((str(doc['pmc_id']),pi,start,end))
    assert old==set(map(tuple,target['accepted_keys']))
    def keysha(x):return hashlib.sha256(json.dumps(x,separators=(',',':')).encode()).hexdigest()
    pareto={name:max(range(len(train)),key=lambda i:(train[i]['shape'][key],train[i]['row_id']))
            for name,key in [('longest','encoder_tokens'),('largest_span_blocks','local_square_cells'),
                            ('longest_passage','max_block_tokens')]}
    data={'train':train,'held':held,'native_contracts':tc+hc,'target_coverage':target,'pareto':pareto}
    raw=json.dumps(data,separators=(',',':'),ensure_ascii=False).encode();p=out/'data.json.gz';p.write_bytes(gzip.compress(raw,mtime=0))
    summary={'status':'LOCAL_FULL_ENCODING_NO_MODEL_OR_HELD_GOLD','data':ref(p),'decoded_sha256':hashlib.sha256(raw).hexdigest(),
       'train':{'documents':32,'windows':32,'tokens':sum(len(x['ids']) for x in train),'max':max(len(x['ids']) for x in train),
        'accepted':2266,'aligned':2265,'unrepresentable':1,'old_GP_admitted_key_set_exact':True,'key_sha256':keysha(target['accepted_keys'])},
       'held':{'documents':16,'patients':39,'windows':17,'tokens':sum(len(x['ids']) for x in held),'max':max(len(x['ids']) for x in held)},
       'pareto':{k:{'index':i,'row_id':train[i]['row_id'],**train[i]['shape']} for k,i in pareto.items()},
       'shape_sums':{k:sum(x['shape'][k] for x in train) for k in ('encoder_tokens','local_square_cells','positive_spans')},
       'original_GP_admission_source':ref(hposrc),'sources':{'train':ref(trpath),'held':ref(vpath),'tokenizer':ref(tokenizer)},
       'full_token_union':True,'untouched_text_and_offsets':True,'no_held_gold':True,'tokenizer_runtime':__import__('tokenizers').__version__}
    (out/'local-preparation.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))


if __name__=='__main__':main()
