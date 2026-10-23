"""Label-free reconstruction of frozen OOF pools and the uniform span whitelist."""
import argparse
from collections import Counter
import gzip
import hashlib
import importlib
import io
import json
from pathlib import Path
import sys
import time


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return [json.loads(x) for x in Path(path).read_text().splitlines()]
def checked(path,digest):assert sha(path)==digest,path
def ref(path):return dict(path=str(path),sha256=sha(path),bytes=Path(path).stat().st_size)


def save_rows(path,rows):
    with Path(path).open('xb') as raw:
        with gzip.GzipFile(fileobj=raw,filename='',mode='wb',mtime=0) as zipped:
            with io.TextIOWrapper(zipped,encoding='utf8') as f:
                for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);a=ap.parse_args();start=time.monotonic()
    plan=json.loads(Path(a.plan).read_text());out=Path(plan['old_pool_directory']);out.mkdir(exist_ok=False)
    for p,h in plan['sources'].items():checked(p,h)
    sys.path.insert(0,plan['legacy_source_directory']);sys.path.insert(1,plan['ontology_source'])
    bio=importlib.import_module('select_public_bioformer');multi=importlib.import_module('calibrate_multi')
    single=multi.single;pub=None;bm=None;pure_docs=None;folds=[];allspans=set();allrefs={}
    for spec in plan['folds']:
        checked(spec['old_fit_manifest']['path'],spec['old_fit_manifest']['sha256'])
        old=json.loads(Path(spec['old_fit_manifest']['path']).read_text());args=old['arguments']
        checked(args['reference'],old['old_reference_sha256']);reference=json.loads(Path(args['reference']).read_text());config=reference['config']
        assert config['threshold']==.4 and config['crf_confidence'] is None and not config['without_crf']
        for name,hashvalue in old['dependencies'].items():checked(Path(plan['ontology_source'])/(name+'.py'),hashvalue)
        for role,hashfield in [('public','public_sha256'),('public_manifest','public_manifest_sha256'),('bioformer','bioformer_sha256'),('bioformer_manifest','bioformer_manifest_sha256')]:
            checked(args[role],old[hashfield]);allrefs[args[role]]=old[hashfield]
        if pure_docs is None:
            pm=json.loads(Path(args['public_manifest']).read_text());checked(pm['input_path'],pm['input_sha256'])
            pure_docs=multi.keyed(pm['input_path']);assert len(pure_docs)==64
            assert all(set(d)=={'pmc_id','pmid','patient','full_text'} for d in pure_docs.values())
            allrefs[pm['input_path']]=pm['input_sha256']
            pub=bio.public_rows(args['public'],pure_docs);bm=bio.public_rows(args['bioformer'],pure_docs)
        checked(config['training_manifest'],reference['inputs']['training_manifest'])
        origins=json.loads(Path(config['training_manifest']).read_text());assert len(origins)==3
        records={'train':[],'input':[]};counts=Counter();inner_ids=[]

        def build(inputs,base_path,raw_path,crf_path,origin_index,role):
            docs=multi.keyed(inputs);bases=multi.keyed(base_path);raw=single.read_candidates(raw_path);crfs=multi.keyed(crf_path)
            assert len(docs)==16 and set(docs)==set(bases)==set(raw)==set(crfs)
            assert all(set(d)=={'pmc_id','pmid','patient','full_text'} and d==pure_docs[did] for did,d in docs.items())
            for did,doc in docs.items():
                base=multi.pool(doc,bases[did],raw[did],crfs[did])
                public=bio.merge_pool(base,pub[did]);cs=bio.merge_bioformer(public,bm[did])
                counts[role+'_base']+=len(base);counts[role+'_public']+=len(public);counts[role+'_full']+=len(cs)
                assert len(cs)==len({(c['offset'],c['length'],c['identifier']) for c in cs})
                allspans.update((str(did),c['offset'],c['length']) for c in cs)
                records[role].append(dict(pmc_id=did,input=doc,base=bases[did],candidates=cs,origin_index=origin_index))
            return list(docs)

        for j,(origin,prior) in enumerate(zip(origins,old['training_origins'])):
            # Reuse audited generator provenance without parsing any gold file.
            for role in ['input','base','candidates','candidate_manifest','crf','crf_manifest']:
                checked(origin[role],prior[role]);allrefs[origin[role]]=prior[role]
            cm=json.loads(Path(origin['candidate_manifest']).read_text());crm=json.loads(Path(origin['crf_manifest']).read_text())
            assert cm['inputs']['input']==prior['input'] and cm['inputs']['train']==prior['generator_train'] and cm['output']['sha256']==prior['candidates']
            assert crm['input_sha256']==prior['input'] and crm['train_sha256']==prior['generator_train'] and crm['output_sha256']==prior['crf']
            inner_ids.append(build(origin['input'],origin['base'],origin['candidates'],origin['crf'],j,'train'))
        for role in ['input','base','candidates','crf','crf_manifest']:
            checked(config[role],reference['inputs'][role]);allrefs[config[role]]=reference['inputs'][role]
        input_ids=build(config['input'],config['base'],config['candidates'],config['crf'],None,'input')
        train_ids=[x for ids in inner_ids for x in ids]
        assert len(train_ids)==len(set(train_ids))==48 and not set(train_ids)&set(input_ids)
        assert set(train_ids)|set(input_ids)==set(pure_docs)
        assert counts['train_base']==old['old_training_examples'] and counts['train_full']==old['training_examples']
        assert counts['train_public']==old['old_training_examples']+old['added_public_training_examples']
        assert counts['input_full']==old['old_input_candidates']+old['added_public_input_candidates']+old['added_bioformer_input_candidates']
        scores=Path(args['output_dir'])/'selector-scores.jsonl';checked(scores,old['outputs']['selector-scores.jsonl'])
        scored=read(scores);assert [r['pmc_id'] for r in scored]==input_ids
        for record,old_score in zip(records['input'],scored):
            assert record['candidates']==[{k:v for k,v in c.items() if k!='selector_probability'} for c in old_score['candidates']]
        target=out/('fold'+str(spec['fold']));target.mkdir()
        artifacts={}
        for role in records:
            path=target/(role+'.pool.jsonl.gz');save_rows(path,records[role]);artifacts[role]=ref(path)
        folds.append(dict(fold=spec['fold'],**artifacts,old_train_rows=counts['train_full'],old_input_rows=counts['input_full'],
            train_ids=train_ids,input_ids=input_ids,nested_input_ids=inner_ids,old_held_candidates_exact=True,
            old_fit_manifest=spec['old_fit_manifest'],provenance='Same audited three train32 candidate generators, no labels parsed in this preparation.'))
    whitelist=out/'all-old-spans.jsonl'
    whitelist.write_text(''.join(json.dumps(dict(pmc_id=did,offset=offset,length=length),separators=(',',':'))+'\n' for did,offset,length in sorted(allspans)))
    manifest=dict(status='label-free old pools reconstructed',source_sha256=sha(__file__),plan_sha256=sha(a.plan),
        folds=folds,span_whitelist=ref(whitelist),unique_spans=len(allspans),input_documents=len(pure_docs),actual_read_inputs=allrefs,
        no_gold_parsed=True,no_fit=True,no_GPU=True,seconds=time.monotonic()-start)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'fold_rows':[(x['old_train_rows'],x['old_input_rows']) for x in folds],
        'unique_spans':len(allspans),'whitelist':manifest['span_whitelist'],'seconds':manifest['seconds']}))


if __name__=='__main__':main()
