"""Pure-input proof that actual B patient legal space is disjoint from public-new."""
import argparse
from collections import defaultdict, deque
import hashlib
import json
from pathlib import Path
import re
from patient_first import transform, inverse

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return dict(path=str(p),bytes=Path(p).stat().st_size,sha256=sha(p))
def rows(p):return [json.loads(x)for x in Path(p).read_text().splitlines()]

def main():
    a=argparse.ArgumentParser()
    for n in ['normal','old-pool','public-spaces','hpo','output']:a.add_argument('--'+n,required=True)
    x=a.parse_args();assert sha(x.normal)=='3a8e0a211797cd1136e5b36b606ad20fc89c83f8598a24779466d1ae9f9c0a31';assert sha(x.old_pool)=='77322710caeb8d87beada5695b73d8c4c7a6f0512a7ad399b220482a6002a184';assert sha(x.public_spaces)=='241d27abbbd5c919e59dec3fe02a20f8366e9d931aa42cac229b1c711aca5e94';assert sha(x.hpo)=='a5092cbdf605f568403cf7380d9173014015692433b2cc631bc5c1b053876b1b'
    # Independent OBO block/graph traversal reproduces the original allowed-descendant alias policy.
    text=Path(x.hpo).read_text();assert 'data-version: hp/releases/2026-06-23' in text
    terms={}
    for block in text.split('[Term]\n')[1:]:
        block=block.split('\n[',1)[0];fields=defaultdict(list)
        for line in block.splitlines():
            if ': ' in line:k,v=line.split(': ',1);fields[k].append(v)
        ident=fields['id'][0];terms[ident]=dict(parents=[v.split()[0]for v in fields['is_a']],alt=fields['alt_id'],obsolete='true'in fields['is_obsolete'])
    children=defaultdict(set)
    for ident,t in terms.items():
        if not t['obsolete']:
            for p in t['parents']:children[p].add(ident)
    allowed={'HP:0000118'};q=deque(allowed)
    while q:
        for child in children[q.popleft()]-allowed:allowed.add(child);q.append(child)
    aliases={alt:ident for ident in allowed for alt in terms[ident]['alt']}
    old={str(d['pmc_id']):d for d in rows(x.old_pool)};normal=rows(x.normal);patient=[transform(r)for r in normal]
    assert all(inverse(p)==n for p,n in zip(patient,normal));assert len(patient)==420
    by=defaultdict(list)
    for p in patient:by[str(p['pmc_id'])].append(p)
    public=json.loads(Path(x.public_spaces).read_text());assert set(old)==set(by)==set(public['documents']) and len(old)==100
    summaries={};legal_pairs=0;text_concepts=0;hpo_concepts=0
    for did,doc in old.items():
        original=set();canonical=set();literals=set()
        for e in doc['entities']:
            if e['identifier']=='-1':original.add(e['text']);literals.add(e['text'])
            else:
                ids=e['identifier'].split(';');assert all(re.fullmatch(r'HP:\d{7}',v)for v in ids)
                original.update(ids);canonical.update(aliases.get(v,v)for v in ids)
        pmap={p['row_id']:p['code_map']for p in by[did]};s=public['documents'][did];assert pmap==s['normal_code_maps']
        actual={v for p in by[did]for v in p['code_map'].values()};assert actual==original
        assert set(s['old_hpo_canonical'])==canonical and set(s['old_text_concepts'])==literals
        pids={str(a['patient_id'])for a in doc['association']}
        assert all(set(p['patient_ids'])==pids for p in by[did])
        pub={v for m in s['public_new_code_maps'].values()for v in m.values()};assert pub==set(s['new_public_canonical_hpo'])
        assert all(v in allowed and aliases.get(v,v)==v for v in pub)
        as_hpo={aliases.get(v,v)for v in actual if re.fullmatch(r'HP:\d{7}',v)}
        assert not pub&(canonical|as_hpo|actual),did
        legal_pairs+=len(actual)*len(pids);text_concepts+=len(literals);hpo_concepts+=len(actual-literals)
        summaries[did]=dict(patient_concepts=len(actual),text_concepts=len(literals),public_new_concepts=len(pub),patients=len(pids),canonical_intersection=0,raw_string_intersection=0)
    raw=''.join(json.dumps(p,ensure_ascii=False)+'\n'for p in patient).encode();out=Path(x.output);out.mkdir(parents=True,exist_ok=False);(out/'expected-patient-prompts.jsonl').write_bytes(raw)
    report=dict(status='ALL100_ACTUAL_TRANSPOSED_PATIENT_LEGAL_SPACE_DISJOINT_FROM_PUBLIC_NEW',source=ref(__file__),references={k:ref(getattr(x,k))for k in ['normal','old_pool','public_spaces','hpo']},expected_patient_prompts=ref(out/'expected-patient-prompts.jsonl'),documents=100,patients=244,rows=len(patient),concepts=sum(v['patient_concepts']for v in summaries.values()),pairs=legal_pairs,text_concepts=text_concepts,hpo_concepts=hpo_concepts,all_inverse_exact=True,original_code_maps_all_exact=True,independent_HPO_descendant_alias_graph=True,per_document=summaries,scope='Legal input-space proof only; actual positive output membership is verified after complete raw inference',patient_output_membership='PENDING_ACTUAL_INFERENCE',no_gold=True,no_fit=True,no_NN=True,no_score=True)
    (out/'proof.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(proof=ref(out/'proof.json'),expected_prompts=ref(out/'expected-patient-prompts.jsonl'),concepts=report['concepts'],pairs=legal_pairs,text_concepts=text_concepts)))

if __name__=='__main__':main()
