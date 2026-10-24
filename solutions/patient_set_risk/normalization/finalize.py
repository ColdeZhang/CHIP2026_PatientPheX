"""All-or-nothing public evidence coverage, then fixed concept-set rewards."""
import argparse
from pathlib import Path
from common import *

def finalize(preparation,output,query_output=None,query_plan=None):
    p=load(preparation);assert p['schema']=='scst-normalization-preparation-v1' and p['status']=='ALL8_PREPARED'
    for k in ('raw','queries','used_cache','source','common','original_sources'):verify(p[k])
    validate_responses(load(p['raw']['path']))
    _,_,_,policy,_,decide=original();queries=load(p['queries']['path']);quotes=[x['quote'] for x in queries]
    assert quotes==sorted(set(quotes)) and len(quotes)==p['query_count']
    cached=load(p['used_cache']['path']);scores={x['quote']:x for x in cached};query_evidence=None
    if quotes:
        assert query_output and query_plan,'Complete new public scores are required'
        qp=load(query_plan);out=Path(query_output);c=load(out/'complete.json')
        assert qp['schema']=='scst-train-only-public-queries-v1' and qp['preparation']['sha256']==ref(preparation)['sha256']
        assert qp['files'][qp['query_source']]['sha256']==ref(HERE/'query.py')['sha256']
        assert qp['assets_inherited']['sha256']==ref(HERE/'target-assets100.json')['sha256']
        assert qp['files'][qp['queries']]['sha256']==p['queries']['sha256']
        assert c['status']=='COMPLETE_SCST_PUBLIC_TRAIN_QUERIES' and c['query_count']==len(quotes) and c['public_bank_rows']==41099
        assert c['plan']['sha256']==ref(query_plan)['sha256'] and c['queries']['sha256']==p['queries']['sha256']
        assert c['policy']['sha256']==POLICY_SHA
        assert c['source']['sha256']==qp['files'][qp['query_source']]['sha256']
        assert c['no_bank_reencoding'] and c['no_fit'] and c['no_quality']
        for k in ('scores','vectors','counts','encoder'):verify(c[k],out/Path(c[k]['path']).name)
        enc=load(out/'encoder-runtime.json');assert enc['device']=='cuda' and enc['training'] is False
        assert all(enc[k]==v for k,v in qp['expected_encoder_runtime'].items())
        new=load(out/'query-scores.json');assert [x['quote'] for x in new]==quotes
        assert all(type(x['token_count']) is int and x['token_count']>0 for x in new)
        assert not set(scores)&set(quotes)
        scores.update({x['quote']:x for x in new});query_evidence=dict(complete=ref(out/'complete.json'),plan=ref(query_plan))
    else:
        assert query_output is None and query_plan is None,'Zero queries require zero model stage'
    records=[]
    for row in p['rows']:
        pred=set();decisions=[]
        for a in row['anchors']:
            q=a['anchor']['quote'];v=decide(a['strict'],scores.get(q),policy);decisions.append(v)
            if v['status']=='mapped':pred.update(v['ids'])
            elif v['status']=='nil':pred.add(q)
        gold=set(row['gold']);reward=f1(gold,pred) if row['valid'] else 0.
        assert math.isfinite(reward) and 0<=reward<=1
        records.append(dict(row_id=row['row_id'],arm=row['arm'],response_binding=row['response_binding'],valid=row['valid'],gold_count=len(gold),
                            predicted_count=len(pred),gold=sorted(gold),predicted=sorted(pred),reward=reward,
                            TP=len(gold&pred),FP=len(pred-gold),FN=len(gold-pred),normalization=decisions))
    assert len(records)==8
    result=dict(schema='scst-concept-set-rewards-v1',status='COMPLETE_ALL8_REWARDS',preparation=ref(preparation),
                source=ref(__file__),common=ref(HERE/'common.py'),raw_responses=p['raw'],original_raw_responses=p['original_raw'],query_evidence=query_evidence,rows=records,
                **supply_gate(records),no_fit=True,no_quality=True,no_held_gold=True)
    out=Path(output);out.mkdir(parents=True,exist_ok=False);save(out/'rewards.json',result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--preparation',required=True);p.add_argument('--output',required=True);p.add_argument('--query-output');p.add_argument('--query-plan');a=p.parse_args()
    r=finalize(a.preparation,a.output,a.query_output,a.query_plan);print(__import__('json').dumps({k:r[k] for k in ('status','supply_gate_pass','pairs')}))
