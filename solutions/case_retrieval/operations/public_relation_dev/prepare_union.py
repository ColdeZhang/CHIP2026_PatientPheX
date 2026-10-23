#!/usr/bin/env python3
"""Freeze only input-derived public .95 concept differences for originaldev replication.
"""
from pathlib import Path
from collections import defaultdict
import argparse,hashlib,json,re,sys,datetime

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return dict(path=str(p),bytes=Path(p).stat().st_size,sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
def save(p,x):
    with Path(p).open('x') as f:f.write(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def save_rows(p,x):
    with Path(p).open('x') as f:f.write(''.join(json.dumps(v,ensure_ascii=False)+'\n' for v in x))
def main():
    a=argparse.ArgumentParser();a.add_argument('--directory',type=Path,required=True);args=a.parse_args();b=args.directory.resolve();out=b/'public-new-relations-dev'/'candidate-difference';out.mkdir();sys.path.insert(0,str(b));import run as base;import prepare_pair_rag as pp
    assert sha(b/'run.py')=='41ee61bce9a8cfb5ef01d73c644172fdff0e29fd6aca2cf5aa08a9c880b5f5d0' and sha(b/'prepare_pair_rag.py')=='16345e096b462ae9ba2fc974bb60d14ff15333b09009b4add1f0a0e1bb23ed97'
    R=Path('/workspace/patientphex-runs/round2-20261023');research=R/'research';inputpath=Path('/workspace/patientphex-data/day1/dev16.input.jsonl');oldpath=R/'sequence_crf/gliner-dev-transfer/assembled/full-lr.jsonl';hpopath=Path('/workspace/patientphex-data/hp.obo');input64path=inputpath
    roots={'pubmed':research/'public-phenotagger-dev16-cache/terminal/native-dev16-final-cache','bioformer':research/'public-phenotagger-bioformer-dev16-cache/terminal/native-dev16-final-cache'}
    docs=rows(inputpath);by={str(x['pmc_id']):x for x in docs};old={str(x['pmc_id']):x for x in rows(oldpath)};allinputs={str(x['pmc_id']):x for x in rows(input64path)};assert len(by)==16 and set(by)==set(old) and all(by[k]==allinputs[k] for k in by)
    assert all('entities' not in x and 'association' not in x for x in docs)
    terms,allowed,alt,_=base.parse_hpo(str(hpopath));oldkeys=set();groups={};public={};sources={n:ref(p) for n,p in dict(source=Path(__file__),input=inputpath,old_candidates=oldpath,input64=input64path,hpo=hpopath,prompt_source=b/'prepare_pair_rag.py').items()}
    def typed(e):
        if e['identifier']=='-1':return [('text',e['text'])]
        values=e['identifier'].split(';');assert all(re.fullmatch(r'HP:\d{7}',x) for x in values)
        return [('hpo',alt.get(x,x)) for x in values]
    for did,doc in old.items():
        for e in doc['entities']:
            for kind,value in typed(e):oldkeys.add((did,kind,value))
    for name,root in roots.items():
        meta=read(root/'manifest.json');path=root/'canonical-entities.jsonl';assert meta['fixed_policy']['threshold_strictly_greater_than']==.95 and meta['public_model_training_on_task'] is False
        assert meta['input_sha256']==sha(input64path);assert sha(path)==meta['outputs']['canonical-entities.jsonl']['sha256'];sources[name+'_manifest']=ref(root/'manifest.json');sources[name+'_canonical']=ref(path)
        selected=[]
        for row in rows(path):
            did=str(row['pmc_id'])
            if did not in by:continue
            doc=by[did];passage=doc['full_text'][row['passage_index']];start=int(row['offset'])-int(passage['offset']);end=start+int(row['length'])
            assert int(passage['offset'])==row['passage_offset'] and start==row['local_start'] and end==row['local_end'] and 0<=start<end<=len(passage['text'])
            assert passage['text'][start:end]==row['text'] and row['hpo_id'] in allowed and alt.get(row['raw_hpo_id'],row['raw_hpo_id'])==row['hpo_id']
            selected.append(row)
        public[name]=selected
    # Union is fixed mechanically before any gold/missing-key file is opened.
    union={}
    for name in roots:
        for x in public[name]:union.setdefault((str(x['pmc_id']),x['offset'],x['length'],x['hpo_id']),x)
    public['union']=list(union.values());statistics={};newsets={};artifacts={}
    for name,pool in public.items():
        allkeys={(str(x['pmc_id']),'hpo',x['hpo_id']) for x in pool};new=allkeys-oldkeys;newsets[name]=new;es=defaultdict(list)
        for x in pool:
            if (str(x['pmc_id']),'hpo',x['hpo_id']) not in new:continue
            es[str(x['pmc_id'])].append(dict(identifier=x['hpo_id'],type='Phenotype',offset=x['offset'],length=x['length'],text=x['text'],note=None))
        entities=[];prompts=[]
        for doc in docs:
            did=str(doc['pmc_id']);seen=set();unique=[]
            for e in sorted(es[did],key=lambda e:(e['offset'],e['length'],e['identifier'])):
                k=(e['offset'],e['length'],e['identifier'])
                if k not in seen:seen.add(k);unique.append(e)
            entities.append({'pmc_id':doc['pmc_id'],'entities':unique});concepts=pp.grouped_concepts(unique);ordered=sorted(concepts)
            for start in range(0,len(ordered),8):
                chosen=ordered[start:start+8];codes={f'C{i:03d}':v for i,v in enumerate(chosen)};items=[dict(code=cid,meaning=terms[concept]['names'][0],evidence=pp.evidence(doc,concepts[concept],500)) for cid,concept in codes.items()]
                content='Patient identities and global anchors:\n'+pp.patient_anchors(doc)+'\n\nCandidate phenotypes:\n'+json.dumps(items,ensure_ascii=False)
                prompts.append(dict(row_id=f'{did}:public_new_{name}:{start}',pmc_id=doc['pmc_id'],code_map=codes,patient_ids=[str(p['patient_id']) for p in doc['patient']],messages=[{'role':'system','content':pp.SYSTEM},{'role':'user','content':content}]))
        assert all((str(p['pmc_id']),'hpo',v) not in oldkeys for p in prompts for v in p['code_map'].values())
        save_rows(out/(name+'.new-entities.jsonl'),entities);save_rows(out/(name+'.prompts.jsonl'),prompts);newfile=out/(name+'.new-concepts.json');save(newfile,sorted(new))
        statistics[name]=dict(public_originaldev_entities=len(pool),public_originaldev_concepts=len(allkeys),overlap_old_concepts=len(allkeys&oldkeys),new_concepts=len(new),new_entity_occurrences=sum(len(x['entities']) for x in entities),new_documents=sum(bool(x['entities']) for x in entities),new_patient_concept_pairs=sum(len(p['code_map'])*len(p['patient_ids']) for p in prompts),prompts=len(prompts),prompt_characters=sum(sum(len(m['content']) for m in p['messages']) for p in prompts))
        artifacts[name]={kind:ref(out/(name+suffix)) for kind,suffix in [('entities','.new-entities.jsonl'),('prompts','.prompts.jsonl'),('concepts','.new-concepts.json')]}
    save(out/'input-only-freeze.json',dict(status='fixed_public_source_differences_frozen_no_gold_read',at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),sources=sources,statistics=statistics,old_entity_count=sum(len(x['entities']) for x in old.values()),old_document_concepts=len(oldkeys),old_text_concepts=sorted(k for k in oldkeys if k[1]=='text'),artifacts=artifacts,no_gold_candidate_addition=True))
    print(json.dumps({'freeze':ref(out/'input-only-freeze.json'),'statistics':statistics},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
