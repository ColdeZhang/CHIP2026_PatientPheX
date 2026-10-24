"""Pre-output mapping of all fixed strong spans to already-defined GP matrices."""
import gzip
import hashlib
import json
from pathlib import Path


def ref(p):
    b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}


def main():
    d=Path('/workspace/patientphex-runs/round2-20261023/research/biomedical-long-context-readiness-20261024')
    data=json.loads(gzip.decompress((d/'probe-v1/data.json.gz').read_bytes()))
    p=Path('/workspace/patientphex-runs/round2-20261023/research/lightgbm-selector-readiness/formal-control/fold1/lightgbm.nested.jsonl')
    assert ref(p)['sha256']=='8cb61ba94fab8de4223acafeece83ef5a243954312d008073a42e5ba80398d6f'
    records=[];lookup={};window_targets={r['row_id']:[] for r in data['held']}
    for doc in map(json.loads,p.read_text().splitlines()):
        pmc=str(doc['pmc_id']);rows=[r for r in data['held'] if r['pmc_id']==pmc]
        assert rows
        for ei,e in enumerate(doc['entities']):
            target_id=pmc+':'+str(ei);assert target_id not in lookup
            record={'target_id':target_id,'pmc_id':pmc,'entity_index':ei,'offset':int(e['offset']),'length':int(e['length']),
                    'text_sha256':hashlib.sha256(e['text'].encode()).hexdigest(),'representable':False,'reason':'native_unaligned_endpoint'}
            start=int(e['offset']);end=start+int(e['length']);starts_seen=False;ends_seen=False
            for row in rows:
                for bi,b in enumerate(row['blocks']):
                    ss={lo+b['passage_offset']:i for i,(lo,hi) in enumerate(b['offsets'])}
                    ee={hi+b['passage_offset']:i for i,(lo,hi) in enumerate(b['offsets'])}
                    starts_seen|=start in ss;ends_seen|=end in ee
                    if start in ss and end in ee and ss[start]<=ee[end]:
                        window_targets[row['row_id']].append({'target_id':target_id,'block_index':bi,'i':ss[start],'j':ee[end]})
                        record['representable']=True;record['reason']='OK'
            if not record['representable'] and starts_seen and ends_seen:record['reason']='no_common_window'
            lookup[target_id]=len(records);records.append(record)
    assert len(records)==1313 and len({x['pmc_id'] for x in records})==16
    obj={'schema':'bioclinical-fixed-baseline-target-map-v1','source_baseline':ref(p),'source_encoded':ref(d/'probe-v1/data.json.gz'),
         'records':records,'window_targets':window_targets,'total':1313,'representable':sum(x['representable'] for x in records),
         'new_gold_reads':0,'model_calls':0,'formula_not_executed':'root independently may delete exact representable span iff max_score<0; this owner only exports scores'}
    out=d/'formal-v1/baseline-targets.json';out.write_text(json.dumps(obj,separators=(',',':'))+'\n')
    print(json.dumps({'mapping':ref(out),'targets':len(records),'representable':obj['representable']}))


if __name__=='__main__':main()
