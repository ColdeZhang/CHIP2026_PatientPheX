"""F0 native full-window geometry; exact prior public candidates, no held labels."""
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from collections import Counter,defaultdict

S=Path(__file__).resolve().parent
R=Path('/workspace/patientphex-runs/round2-20261023')
D=R/'research/long-context-relation-consensus-v1/F0-v1'
OLD=Path('/workspace/patientphex-worktrees/research-biomedical-long-context/solutions/biomedical_long_context')
sys.path.insert(0,str(S.parent));from run import ref,save,publish


def main():
    from tokenizers import Tokenizer
    sys.path.insert(0,str(OLD))
    spec=importlib.util.spec_from_file_location('original_long_context_prepare',OLD/'prepare.py')
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    op=Path('/workspace/patientphex-worktrees/explore-relation-consensus/solutions/relation_consensus/docre_f0/plan.json')
    plan=json.loads(op.read_text());inp=Path(plan['input']['path']);assert ref(inp)==plan['input']
    docs=[json.loads(x)for x in inp.read_text().splitlines()]
    assert len(docs)==16 and sum(len(x['patient'])for x in docs)==47
    assert all('entities'not in x and 'association'not in x for x in docs)
    source=R/'stage8-relation-consensus/docre-f0/prepared/prepared.json.gz'
    assert ref(source)['sha256']==plan['prepared_sha256']
    prepared=json.loads(gzip.decompress(source.read_bytes()))
    assert len(prepared)==16 and all('labels'not in x for x in prepared)
    tokenizer=R/'research/biomedical-long-context-readiness-20261024/public/tokenizer.txt'
    tok=Tokenizer.from_file(str(tokenizer));tok.no_truncation();tok.no_padding()
    encoded,contracts,_=old.encode_docs(docs,tok,set(),{},False)
    by=defaultdict(list)
    for row in encoded:by[row['pmc_id']].append(row)
    cs={x['pmc_id']:x for x in contracts};raw={str(x['pmc_id']):x for x in docs}
    geometry=[];counts=Counter();same_owner=0
    for doc in prepared:
        pmc=doc['pmc_id'];native=cs[pmc];assert doc['text']==native['text']
        passages=[];at=0
        for p in raw[pmc]['full_text']:passages.append(at);at+=len(p['text'])+2
        offsets={};tokenpass={};windows=by[pmc];ranges=[]
        for row in windows:
            lo,hi=row['body_range'];ranges.append((lo,hi));assert row['ids'][1:-1]==native['ids'][lo:hi]
            for block in row['blocks']:
                pi=block['passage_index'];base=passages[pi]
                for i,(a,z)in zip(block['positions'],block['offsets']):
                    value=(base+a,base+z);assert offsets.get(i,value)==value
                    offsets[i]=value;tokenpass[i]=pi
        entities=[];physical=set()
        for ei,e in enumerate(doc['entities']):
            spans=[];pis=set()
            for a,z in e['spans']:
                indices=[i for i,(x,y)in offsets.items()if x<z and a<y];assert indices
                i,j=min(indices),max(indices)+1;assert len({tokenpass[t]for t in indices})==1
                pi=tokenpass[i];pis.add(pi)
                owners=[(min(i-lo,hi-j),-w,w)for w,(lo,hi)in enumerate(ranges)if lo<=i<j<=hi];assert owners
                owner=max(owners)[2];spans.append((i,j,owner,pi));physical.add((ei,a,z))
                counts['exact_mention_endpoints'if offsets[i][0]==a and offsets[j-1][1]==z else'overlap_mention_endpoints']+=1
            entities.append(dict(kind=e['kind'],id=e['id'],spans=spans,passages=sorted(pis)))
        assert len(physical)==doc['unique_physical_mentions']
        for p,h in doc['pairs']:
            same_owner+=bool({x[2]for x in entities[p]['spans']}&{x[2]for x in entities[h]['spans']})
        geometry.append(dict(pmc_id=pmc,pairs=doc['pairs'],keys=doc['keys'],entities=entities,
            windows=[dict(row_id=x['row_id'],body_range=x['body_range'],ids=x['ids'])for x in windows]))
    assert sum(len(x['pairs'])for x in geometry)==2195
    train=R/'stage8-document-relations/prepared/train.jsonl'
    train_rows=[json.loads(x)for x in train.read_text().splitlines()]
    assert len(train_rows)==32 and sum(len(x['pairs'])for x in train_rows)==4495
    assert not {x['pmc_id']for x in train_rows}&set(raw)
    out=D/'cpu';out.mkdir(parents=True)
    payload=dict(held=geometry,native_contracts=contracts)
    publish(out/'data.json.gz',gzip.compress(json.dumps(payload,separators=(',',':'),ensure_ascii=False).encode(),mtime=0))
    publish(out/'geometry.json.gz',gzip.compress(json.dumps(geometry,separators=(',',':')).encode(),mtime=0))
    summary=dict(status='LOCAL_COMPLETE_F0_NATIVE_GEOMETRY_NO_GOLD_MODEL',data=ref(out/'data.json.gz'),geometry=ref(out/'geometry.json.gz'),
        documents=16,patients=47,pairs=2195,physical_mentions=sum(len(e['spans'])for d in geometry for e in d['entities']),
        windows=len(encoded),body_tokens=sum(len(x['ids'])for x in contracts),encoder_tokens=sum(len(x['ids'])for x in encoded),
        max_encoder_tokens=max(len(x['ids'])for x in encoded),max_pairs=max(len(x['pairs'])for x in geometry),
        counts=dict(counts),pairs_with_same_selected_owner=same_owner,complete_text_and_tokens=True,cut=0,
        sources=dict(input=ref(inp),prepared=ref(source),prior_plan=ref(op),tokenizer=ref(tokenizer),native_encoder_source=ref(OLD/'prepare.py'),train_geometry_source=ref(train)),
        ancestor_check=dict(train_documents=32,held_documents=16,intersection=0,training_same_original_F2F3=True,
          old_shared_candidates_ancestry=plan['ancestry'],public_sources=plan['public_caches'],
          new_checkpoint='63db2cca3913ca77b2d29a3ea38faf459b81846ae1e287aec7e6e55a32b58bd0',
          note='New public BioClinical parent + exact old train32 candidate labels only; no Task1-trained warmstart or F0-derived resources. F0 inputs and all3 public caches inherited from prior pure preparation.'),
        source=ref(__file__),tokenizers=__import__('tokenizers').__version__,new_gold_reads=0,new_model_calls=0)
    save(out/'preparation.json',summary);print(json.dumps(summary))


if __name__=='__main__':main()
