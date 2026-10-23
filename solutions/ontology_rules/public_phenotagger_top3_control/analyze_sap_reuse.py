"""CPU resource counts only: old normalized phrase caches and public HPO aliases."""
import argparse
import ast
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import re
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with gzip.open(path, 'rt') as stream:
        return [json.loads(line) for line in stream]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--semantic-source',required=True)
    a=ap.parse_args();start=time.monotonic();root=Path(a.root)
    out=root/'ontology_rules/public-phenotagger-top3-sap-feasibility'
    inventory=read(out/'remote-inventory.json')
    semantic=Path(a.semantic_source)
    assert sha(semantic)=='b1005687bc1162319aa5c87ab52fb6f8ef96ed8ed8518ea8295af4454c60647a'
    tree=ast.parse(semantic.read_text())
    selected=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['norm','hpo_terms']]
    scope={'Path':Path,'re':re};exec(compile(ast.Module(body=selected,type_ignores=[]),str(semantic),'exec'),scope)
    norm,hpo_terms=scope['norm'],scope['hpo_terms']
    small=inventory['small_files'];signature='sapbert'+small['semantic-neural/model/config.json']['sha256']+small['semantic-neural/model/manifest.json']['sha256']
    model=small['semantic-neural/model/manifest.json']['content']
    assert model['revision']=='090663c3ae57bf35ffe4d0d468a2a88d03051a4d'
    remote={r['directory'].split('/ontology-round2/')[1]:r for r in inventory['roots']}
    cached_phrases=set();catalog=[]
    for work,info in remote.items():
        local=root/'ontology_rules'/work;manifest=read(local/'manifest.json')
        assert sha(local/'manifest.json')==info['manifest_sha256']
        assert manifest['semantic_source_sha256']==sha(semantic) and manifest['model_manifest']==model
        assert manifest['inputs']['hpo']==sha('/workspace/patientphex-data/hp.obo')
        candidate=local/'scored-candidates.jsonl.gz';assert sha(candidate)==manifest['output']['sha256']
        actual={f['name']:f for f in info['files']};files=[];seen=set()
        for document in rows(candidate):
            names=sorted({norm(c['text']) for c in document['candidates']})
            digest=hashlib.sha256((signature+'\0'+'\0'.join(names)).encode()).hexdigest()
            file=actual[digest+'.npy'];assert file['header']==dict(descr='<f4',fortran_order=False,shape=[len(names),768])
            assert file['bytes']==128+len(names)*768*4
            cached_phrases.update(names);seen.update(names)
            files.append(dict(pmc_id=document['pmc_id'],file=digest+'.npy',rows=len(names)))
        catalog.append(dict(work=work,manifest_sha256=sha(local/'manifest.json'),candidates_sha256=sha(candidate),query_caches=files,unique_query_phrases=len(seen),all_query_files_present=True,bank_rows=manifest['training']['semantic_bank'],historical_total_run_seconds=manifest['elapsed_seconds']))
    hpo=Path('/workspace/patientphex-data/hp.obo');terms,canonical=hpo_terms(hpo)
    public=defaultdict(set)
    for hid,term in terms.items():
        if hid=='HP:0000118':continue
        for label in [term['name']]+term['synonyms']:
            if len(label)>=3:public[hid].add(norm(label))
    aliases={s for names in public.values() for s in names}
    mapping=root/'ontology_rules/phenotagger-fold0/sap-consensus100/request/bank-labels.json'
    bank=read(mapping);assert len(bank['labels'])==42236 and bank['labels']==sorted(set(bank['labels']))
    bankkey=hashlib.sha256((signature+'\0'+'\0'.join(bank['labels'])).encode()).hexdigest()
    assert bankkey=='19ab05a4a614bbc869be5c1d2943e08962c4e82f968026bcb41aa259e617d242'
    assert aliases<=set(bank['labels'])
    all_pairs=set();new_pairs=set();new_phrases=set();all_phrases=set();kinds=Counter();folds=[]
    prepared=read(root/'ontology_rules/public-phenotagger-top3-control/prepared/manifest.json')
    old=read(root/'ontology_rules/public-phenotagger-top3-control/old-pools/manifest.json')
    for newfold,oldfold in zip(prepared['folds'],old['folds']):
        counts={}
        for role in ['train','input']:
            ns=rows(newfold[role]['path']);os=rows(oldfold[role]['path']);stat=Counter();phrases=set();pairs=set()
            for n,o in zip(ns,os):
                assert n['pmc_id']==o['pmc_id']
                for index,c in enumerate(n['candidates']):
                    phrase=norm(c['text']);hid=c['identifier'];all_phrases.add(phrase)
                    if re.fullmatch(r'HP:\d{7}',hid):all_pairs.add((phrase,hid))
                    else:kinds['unlinked_or_compound_rows']+=1
                    if index<len(o['candidates']):continue
                    assert hid in public
                    new_pairs.add((phrase,hid));new_phrases.add(phrase);phrases.add(phrase);pairs.add((phrase,hid))
                    stat['new_ID_rows']+=1
                    stat['phrase_in_known_query_cache_rows']+=phrase in cached_phrases
                    stat['phrase_in_query_or_public_bank_rows']+=phrase in cached_phrases or phrase in aliases
            counts[role]=dict(stat,unique_new_phrases=len(phrases),unique_new_phrase_ID_pairs=len(pairs),missing_query_or_public_bank_phrases=len(phrases-(cached_phrases|aliases)))
        folds.append(dict(fold=newfold['fold'],counts=counts))
    known=cached_phrases|aliases;missing=sorted(new_phrases-known)
    result=dict(source_sha256=sha(__file__),semantic_source_sha256=sha(semantic),remote_inventory_sha256=sha(out/'remote-inventory.json'),model_revision=model['revision'],model_manifest=model,encoder_signature=signature,
                representation='SapBERT CLS; FP16 model forward; float32 L2-normalized768 vectors; max_length32; norm lower()+whitespace squeeze; sorted unique query rows',
                cache_catalog=catalog,cache_count=len(catalog),known_unique_query_phrases=len(cached_phrases),public_terms=len(public),public_unique_alias_phrases=len(aliases),public_alias_memberships=sum(map(len,public.values())),all_public_aliases_in_checked_bank=True,public_bank=inventory['selected_bank'],bank_labels_sha256=sha(mapping),
                all_atomic_phrase_ID_pairs=len(all_pairs),all_unique_phrases=len(all_phrases),added_atomic_phrase_ID_pairs=len(new_pairs),added_unique_phrases=len(new_phrases),added_phrases_in_query_caches=len(new_phrases&cached_phrases),added_phrases_in_public_alias_bank=len(new_phrases&aliases),added_missing_known_phrases=len(missing),all_missing_known_phrases=len(all_phrases-known),added_pairs_missing_query_or_bank=sum(p not in known for p,h in new_pairs),new_embedding_bytes_float32=len(missing)*768*4,all_phrase_embedding_bytes_float32=len(all_phrases)*768*4,
                newpair_alias_dot_products=sum(len(public[hid]) for phrase,hid in new_pairs),newpair_alias_dot_dimensions=sum(len(public[hid])*768 for phrase,hid in new_pairs),folds=folds,other_identifier_rows=dict(kinds),
                scope='No gold/dev read, no new embeddings, cosine feature or fit. Query cache headers/exact content-derived filenames checked; numeric query contents not rehashed. One public-bank full SHA checked. Other cache classes may contain additional phrases; current missing count is an upper bound on needed new encodes relative to the verified query+public sources.',
                alias_policy='Old bank has one training-frequency best_id per surface and mixed public/train aliases. A new public-only feature must ignore old label_ids: rebuild public current-HPO name/EXACT all-ID mapping and aggregate actual cosines over these aliases; never transfer task-derived best_id across folds.',
                interpretation='Old similarity is nearest-bank maximum, including exact-span ID overrides, not guaranteed candidate-specific. Missing top-k/ID is not zero. Reuse only text-only frozen-encoder vectors with explicit content/source mapping. Compound/NIL handling must be specified before any feature trial.',seconds=time.monotonic()-start)
    (out/'reuse-analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['cache_count','known_unique_query_phrases','public_terms','public_unique_alias_phrases','added_atomic_phrase_ID_pairs','added_unique_phrases','added_phrases_in_query_caches','added_missing_known_phrases','all_missing_known_phrases','new_embedding_bytes_float32','newpair_alias_dot_dimensions','seconds']}))


if __name__=='__main__':main()
