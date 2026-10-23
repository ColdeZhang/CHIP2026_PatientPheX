#!/usr/bin/env python3
"""CPU-only strict F2+F3 public-concept augmentation; preserve all old SFT bytes."""
import argparse, datetime, hashlib, json, re, shutil, sys
from collections import defaultdict
from pathlib import Path

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p): return dict(path=str(Path(p).resolve()), bytes=Path(p).stat().st_size, sha256=sha(p))
def read(p): return json.loads(Path(p).read_text())
def rows(p): return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
def save(p, x):
    with Path(p).open('x') as f: f.write(json.dumps(x, ensure_ascii=False, indent=2)+'\n')
def save_rows(p, x):
    with Path(p).open('x') as f: f.write(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in x))
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def verify(r):
    assert ref(r['path']) == r, r
    return Path(r['path'])
def main():
    p=argparse.ArgumentParser(); p.add_argument('--base',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    b=a.base.resolve(); out=a.output.resolve(); out.mkdir(); source=out/'source'; source.mkdir(); shutil.copy2(__file__,source/'prepare.py')
    oldm=read(b/'preparation-manifest.json'); assert sha(b/'preparation-manifest.json')=='356109e9b35d71e36c315b012831166419b4e007db061a289a04771abee69a47'
    ids=set(oldm['training_document_ids']); excluded=set(oldm['excluded_document_ids']); assert len(ids)==len(excluded)==32 and not ids&excluded
    for name in ['run.py','prepare_pair_rag.py']: shutil.copy2(verify(oldm['sources'][name]),source/name)
    sys.path.insert(0,str(source)); import run as base; import prepare_pair_rag as pp
    hpo=verify(oldm['sources']['hpo']); terms,allowed,alt,_=base.parse_hpo(str(hpo))
    research=b.parents[2]/'research'
    input64=research/'public-phenotagger-train64-cache/train64.input.jsonl'
    allinputs={str(d['pmc_id']):d for d in rows(input64)}; assert ids|excluded==set(allinputs)
    docs=[allinputs[k] for k in sorted(ids)]; assert all('entities' not in d and 'association' not in d for d in docs)
    save_rows(out/'train32.input.jsonl',docs); by={str(d['pmc_id']):d for d in docs}
    sources={n:ref(v) for n,v in dict(old_preparation=b/'preparation-manifest.json',input64=input64,hpo=hpo,source=source/'prepare.py',old_prompts=b/'prompts.jsonl').items()}
    oldprompts=rows(b/'prompts.jsonl'); assert sha(b/'prompts.jsonl')==oldm['unlabelled_requests_sha256']; assert len(oldprompts)==124
    oldkeys=set(); oldraw=set(); pooled=[]; seen=set()
    def typed(e):
        if e['identifier']=='-1': return [('text',e['text'])]
        values=e['identifier'].split(';'); assert all(re.fullmatch(r'HP:\d{7}',v) for v in values)
        return [('hpo',alt.get(v,v)) for v in values]
    for m in oldm['provenance']:
        gen=set(m['generator_ids']); target=set(m['target_ids']); assert not gen&target and gen|target==ids and not (gen|target)&excluded and not target&seen; seen|=target
        for n in ['candidates','assembly_manifest','semantic_manifest','CRF_manifest','target_input']: verify(m[n])
        assert all(d==by[str(d['pmc_id'])] for d in rows(m['target_input']['path']))
        for d in rows(m['candidates']['path']):
            did=str(d['pmc_id']); assert did in target; pooled.append(d)
            for e in d['entities']:
                for kind,v in typed(e): oldkeys.add((did,kind,v))
    assert seen==ids
    for row in oldprompts:
        assert str(row['pmc_id']) in ids and 'answer' not in row and [m['role'] for m in row['messages']]==['system','user']
        oldraw.update((str(row['pmc_id']),pid,c) for pid in row['patient_ids'] for c in row['code_map'].values())
    roots={'pubmed':research/'public-phenotagger-train64-cache/terminal/native-train64-cache','bioformer':research/'public-phenotagger-bioformer-train64-cache/terminal/native-train64-final-cache'}
    union={}; public_stats={}
    for name,root in roots.items():
        m=read(root/'manifest.json'); path=root/'canonical-entities.jsonl'
        assert m['public_model_training_on_task'] is False and m['fixed_policy']['threshold_strictly_greater_than']==.95 and m['input_sha256']==sha(input64)
        assert sha(path)==m['outputs']['canonical-entities.jsonl']['sha256']; sources[name+'_manifest']=ref(root/'manifest.json'); sources[name+'_canonical']=ref(path)
        selected=[]
        for row in rows(path):
            did=str(row['pmc_id'])
            if did not in ids: continue
            passage=by[did]['full_text'][row['passage_index']]; st=row['offset']-passage['offset']; en=st+row['length']
            assert passage['offset']==row['passage_offset'] and st==row['local_start'] and en==row['local_end'] and 0<=st<en<=len(passage['text']) and passage['text'][st:en]==row['text']
            assert row['hpo_id'] in allowed and alt.get(row['raw_hpo_id'],row['raw_hpo_id'])==row['hpo_id']; selected.append(row)
            union.setdefault((did,row['offset'],row['length'],row['hpo_id']),row)
        keys={(str(x['pmc_id']),'hpo',x['hpo_id']) for x in selected}
        public_stats[name]=dict(spans=len(selected),concepts=len(keys),new_concepts=len(keys-oldkeys))
    publickeys={(str(x['pmc_id']),'hpo',x['hpo_id']) for x in union.values()}; newkeys=publickeys-oldkeys
    es=defaultdict(list)
    for x in union.values():
        did=str(x['pmc_id'])
        if (did,'hpo',x['hpo_id']) in newkeys: es[did].append(dict(identifier=x['hpo_id'],type='Phenotype',offset=x['offset'],length=x['length'],text=x['text'],note=None))
    entities=[]; prompts=[]
    for d in docs:
        did=str(d['pmc_id']); unique=sorted(es[did],key=lambda e:(e['offset'],e['length'],e['identifier'])); entities.append(dict(pmc_id=d['pmc_id'],entities=unique)); concepts=pp.grouped_concepts(unique); ordered=sorted(concepts)
        for start in range(0,len(ordered),8):
            codes={f'C{i:03d}':v for i,v in enumerate(ordered[start:start+8])}; items=[dict(code=cid,meaning=terms[c]['names'][0],evidence=pp.evidence(d,concepts[c],500)) for cid,c in codes.items()]
            content='Patient identities and global anchors:\n'+pp.patient_anchors(d)+'\n\nCandidate phenotypes:\n'+json.dumps(items,ensure_ascii=False)
            prompts.append(dict(row_id=f'{did}:public_new_union:{start}',pmc_id=d['pmc_id'],code_map=codes,patient_ids=[str(p['patient_id']) for p in d['patient']],messages=[dict(role='system',content=pp.SYSTEM),dict(role='user',content=content)]))
    assert sum(len(r['code_map']) for r in prompts)==len(newkeys) and not {r['row_id'] for r in oldprompts}&{r['row_id'] for r in prompts}
    save_rows(out/'new.entities.jsonl',entities); save_rows(out/'new.prompts.jsonl',prompts); save(out/'new.concepts.json',sorted(newkeys))
    save(out/'input-only-freeze.json',dict(status='all_public_train32_new_candidates_frozen_before_loading_train_labels',at_utc=now(),sources=sources,training_ids=sorted(ids),excluded_ids=sorted(excluded),old_strict_candidate_provenance=oldm['provenance'],public_stats=public_stats,old_document_concepts=len(oldkeys),union_public_concepts=len(publickeys),new_concepts=len(newkeys),new_spans=sum(map(len,es.values())),new_prompts=len(prompts),artifacts={n:ref(out/n) for n in ['train32.input.jsonl','new.entities.jsonl','new.prompts.jsonl','new.concepts.json']},gate=None,no_F0_F1_derived_gate_or_training_resource=True))
    # Only now read train32 labels. They cannot alter candidate membership/prompt grouping.
    trainpath=verify(oldm['sources']['train32']); goldrows=rows(trainpath); goldby={str(d['pmc_id']):d for d in goldrows}; assert set(goldby)==ids
    assert all({k:v for k,v in goldby[did].items() if k not in ['entities','association']}==by[did] for did in ids)
    oldtrain=rows(b/'train.jsonl'); assert sha(b/'train.jsonl')=='cb7f67e16959e4ec1357823303915aa44cc070c6f527b4c74ae66e23712d84a8'
    assert [{k:v for k,v in r.items() if k!='answer'} for r in oldtrain]==oldprompts
    newtrain=[]; truth=set(); newpairs=set(); stats={}
    for did,d in goldby.items():
        g=base.gold_associations(d); truth.update((did,str(pid),c) for pid,values in g.items() for c in values)
    for group,promptrows in [('old',oldprompts),('new',prompts)]:
        pos=0; empty=0; pairs=set(); concepts=set()
        for r in promptrows:
            did=str(r['pmc_id']); g=base.gold_associations(goldby[did]); answer={cid:[pid for pid in r['patient_ids'] if c in g.get(pid,set())] for cid,c in r['code_map'].items()}; pcount=sum(map(len,answer.values())); pos+=pcount; empty+=int(pcount==0)
            pair={(did,pid,c) for pid in r['patient_ids'] for c in r['code_map'].values()}; selected={(did,pid,r['code_map'][cid]) for cid,pids in answer.items() for pid in pids}; assert selected==truth&pair; pairs|=pair; concepts.update((did,c) for c in r['code_map'].values())
            labelled=dict(r,answer=json.dumps(answer,ensure_ascii=False,separators=(',',':')))
            if group=='new': newtrain.append(labelled)
        stats[group]=dict(rows=len(promptrows),documents=len({str(r['pmc_id']) for r in promptrows}),concepts=len(concepts),pairs=len(pairs),positive=pos,negative=len(pairs)-pos,all_empty_rows=empty,covered_truth=len(pairs&truth))
        if group=='new': newpairs=pairs
    # Independently re-derive every old answer rather than changing it.
    for r in oldtrain:
        g=base.gold_associations(goldby[str(r['pmc_id'])]); assert json.loads(r['answer'])=={cid:[pid for pid in r['patient_ids'] if c in g.get(pid,set())] for cid,c in r['code_map'].items()}
    assert not oldraw&newpairs
    save_rows(out/'new.train.jsonl',newtrain)
    oldbytes=(b/'train.jsonl').read_bytes(); assert oldbytes.endswith(b'\n')
    with (out/'train.jsonl').open('xb') as f: f.write(oldbytes); f.write((out/'new.train.jsonl').read_bytes())
    with (out/'prompts.jsonl').open('xb') as f: f.write((b/'prompts.jsonl').read_bytes()); f.write((out/'new.prompts.jsonl').read_bytes())
    assert (out/'train.jsonl').read_bytes()[:len(oldbytes)]==oldbytes
    stats['combined']={k:stats['old'][k]+stats['new'][k] for k in ['rows','concepts','pairs','positive','negative','all_empty_rows','covered_truth']}; stats['combined']['documents']=32
    coverage=dict(convention='literal original gold_associations; only syntactically complete compound HPO split, no alt remap',truth=len(truth),old_covered=len(truth&oldraw),old_missing=len(truth-oldraw),newly_covered=len(truth&newpairs),remaining_missing=len(truth-(oldraw|newpairs)),oracle_candidate_upper_bound_not_model_quality=True)
    f1freeze=b/'public-candidate-coverage/input-only-freeze.json'; f1prompts=b/'public-new-relations/prompts.jsonl'; f1=rows(f1prompts)
    result=dict(status='CPU_training_data_complete_pending_actual_tokenizer_no_training',at_utc=now(),input_only_freeze=ref(out/'input-only-freeze.json'),old_train=ref(b/'train.jsonl'),train32_labels=ref(trainpath),old124_bytes_exact=True,old_answers_roundtrip_exact=True,new_answers_roundtrip_exact=True,stats=stats,coverage=coverage,labels='Same raw gold_associations convention as original meta32, no canonicalization',sources=sources,artifacts={n:ref(out/n) for n in ['train.jsonl','prompts.jsonl','new.train.jsonl','new.prompts.jsonl']},comparison_F1=dict(freeze=ref(f1freeze),prompts=ref(f1prompts),documents=16,prompts_count=len(f1),concepts=sum(len(r['code_map']) for r in f1),pairs=sum(len(r['code_map'])*len(r['patient_ids']) for r in f1),same_fixed_public95_union_new_concept_algorithm=True,training_source_different='train32 strict two16 pools; F1 old pool from32, no F1 labels used here'),gpu_started=False)
    save(out/'preparation-manifest.json',result); print(json.dumps(dict(manifest=ref(out/'preparation-manifest.json'),stats=stats,coverage=coverage),indent=2))
if __name__=='__main__': main()
