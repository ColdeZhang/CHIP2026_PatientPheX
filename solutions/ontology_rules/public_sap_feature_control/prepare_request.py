"""Export only existing normalized candidate text/IDs and cache row maps."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def rows(p):
    with gzip.open(p,'rt') as f:return [json.loads(x) for x in f]
def norm(s):return ' '.join(s.lower().split()).strip()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
    root=Path(a.root);out=Path(a.output);out.mkdir(exist_ok=False)
    origin=root/'ontology_rules/public-phenotagger-top3-control/prepared/manifest.json';prepared=read(origin)
    phrases=set();pairs=set()
    for fold in prepared['folds']:
        for role in ['train','input']:
            p=fold[role]['path'];assert sha(p)==fold[role]['sha256']
            for row in rows(p):
                for c in row['candidates']:phrase=norm(c['text']);phrases.add(phrase);pairs.add((phrase,c['identifier']))
    inventory=read(root/'ontology_rules/public-phenotagger-top3-sap-feasibility/remote-inventory.json')
    caches=[];refs={str(origin):sha(origin)}
    for remote in inventory['roots']:
        work=remote['directory'].split('/ontology-round2/')[1];m=root/'ontology_rules'/work/'manifest.json';assert sha(m)==remote['manifest_sha256']
        manifest=read(m);source=m.parent/'scored-candidates.jsonl.gz';assert sha(source)==manifest['output']['sha256'];refs[str(m)]=sha(m);refs[str(source)]=sha(source)
        for row in rows(source):
            names=sorted({norm(c['text']) for c in row['candidates']});sig='sapbert'+inventory['small_files']['semantic-neural/model/config.json']['sha256']+inventory['small_files']['semantic-neural/model/manifest.json']['sha256']
            key=hashlib.sha256((sig+'\0'+'\0'.join(names)).encode()).hexdigest()
            caches.append(dict(pmc_id=row['pmc_id'],path='ontology-round2/'+work+'/embeddings/'+key+'.npy',texts=names))
    mapping=root/'ontology_rules/phenotagger-fold0/sap-consensus100/request/bank-labels.json';refs[str(mapping)]=sha(mapping)
    request=dict(phrases=sorted(phrases),pairs=sorted(pairs),query_caches=caches,bank_labels=read(mapping)['labels'])
    assert len(phrases)==34094 and len(caches)==256
    with gzip.open(out/'request.json.gz','wt') as f:json.dump(request,f,ensure_ascii=False)
    (out/'request-manifest.json').write_text(json.dumps(dict(source_sha256=sha(__file__),sources=refs,prepared_manifest_sha256=sha(origin),request_sha256=sha(out/'request.json.gz'),phrases=len(phrases),pairs=len(pairs),query_caches=len(caches),no_gold=True),indent=2)+'\n')
    print(json.dumps(dict(phrases=len(phrases),pairs=len(pairs),bytes=(out/'request.json.gz').stat().st_size)))


if __name__=='__main__':main()
