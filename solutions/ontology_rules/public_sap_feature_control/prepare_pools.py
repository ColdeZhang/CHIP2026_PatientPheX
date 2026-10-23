"""Append only two public Sap fields, preserving exact candidate objects/order."""
import argparse
from collections import Counter
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import time


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def ref(path):return dict(path=str(path),sha256=sha(path),bytes=Path(path).stat().st_size)
def read(path):return json.loads(Path(path).read_text())
def rows(path):
    with gzip.open(path,'rt') as f:return [json.loads(x) for x in f]
def norm(s):return ' '.join(s.lower().split()).strip()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);a=ap.parse_args();start=time.monotonic();plan=read(a.plan)
    assert sha(__file__)==plan['source_sha256']
    for role in ['top3_prepared_manifest','lookup','feature_manifest']:
        assert sha(plan[role]['path'])==plan[role]['sha256']
    manifest=read(plan['feature_manifest']['path']);assert manifest['lookup']['sha256']==plan['lookup']['sha256']
    lookup={}
    for row in rows(plan['lookup']['path']):
        key=(row['text'],row['identifier']);assert key not in lookup
        observed,score=row['public_sap_observed'],row['public_sap_cosine']
        assert type(observed) is bool and math.isfinite(score) and -1.0001<=score<=1.0001
        assert observed or score==0
        lookup[key]=(score,observed)
    old=read(plan['top3_prepared_manifest']['path']);out=Path(plan['output_dir']);out.mkdir(exist_ok=False);folds=[];total=Counter()
    for fold in old['folds']:
        target=out/('fold'+str(fold['fold']));target.mkdir();newfold=dict(fold);counts={}
        for role in ['train','input']:
            assert sha(fold[role]['path'])==fold[role]['sha256'];documents=rows(fold[role]['path']);stat=Counter();path=target/(role+'.pool.jsonl.gz')
            with path.open('xb') as raw:
                with gzip.GzipFile(fileobj=raw,mode='wb',filename='',mtime=0) as zipped:
                    with io.TextIOWrapper(zipped,encoding='utf8') as stream:
                        for doc in documents:
                            candidates=[]
                            for c in doc['candidates']:
                                assert not {'public_sap_cosine','public_sap_observed'}&set(c)
                                cosine,observed=lookup[norm(c['text']),c['identifier']]
                                n=dict(c,public_sap_cosine=cosine,public_sap_observed=observed)
                                assert {k:v for k,v in n.items() if k not in ['public_sap_cosine','public_sap_observed']}==c
                                assert all(type(n[k]) is type(v) for k,v in c.items())
                                candidates.append(n);stat['rows']+=1;stat['observed']+=observed;stat['missing']+=not observed
                            stream.write(json.dumps(dict(doc,candidates=candidates),ensure_ascii=False)+'\n')
            assert stat['rows']==fold['new_'+role+'_rows'];newfold[role]=ref(path);counts[role]=dict(stat);total.update(stat)
        newfold['sap_counts']=counts;folds.append(newfold)
    result=dict(status='two public Sap fields appended; exact top3 keys/fields preserved',source_sha256=sha(__file__),plan_sha256=sha(a.plan),
          top3_prepared_manifest=plan['top3_prepared_manifest'],lookup=plan['lookup'],feature_manifest=plan['feature_manifest'],folds=folds,total=dict(total),
          new_features=['public_sap_cosine','public_sap_observed'],estimated_four_fit_infer_seconds=plan['estimated_four_fit_infer_seconds'],
          estimate='Prior actual4fit145.395s, two extra columns plus15s allowance; estimate is not measured runtime.',no_new_candidates=True,no_old_field_changes=True,no_gold_read=True,seconds=time.monotonic()-start)
    (out/'manifest.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(total=dict(total),seconds=result['seconds'])))


if __name__=='__main__':main()
