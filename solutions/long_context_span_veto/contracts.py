"""Only new max-score/representation contracts; reuse prior veto byte/object contracts."""
import argparse
import copy
import hashlib
import json
import time
import run


def record(index, text, values=None, reason='OK'):
    values = values or []
    return dict(target_id=f'p:{index}', pmc_id='p', entity_index=index, offset=index*10,
        length=len(text), text_sha256=hashlib.sha256(text.encode()).hexdigest(),
        representable=bool(values), reason=reason,
        window_scores=[dict(row_id=f'w{i}',block_index=0,i=1,j=2,score=v) for i,v in enumerate(values)],
        max_score=max(values) if values else None)


def fails(action):
    try: action()
    except ValueError: return
    raise AssertionError('Expected full endpoint failure')


def main(output):
    start=time.monotonic()
    rows=[record(0,'aa',[-2.,-.1]),record(1,'bb',[-2.,0.]),record(2,'cc',[-2.,.1]),
          record(3,'dd',reason='native_unaligned_endpoint'),record(4,'ee',reason='no_common_window')]
    base=[dict(pmc_id='p',entities=[dict(offset=r['offset'],length=2,text=t,identifier='HP:0000001;HP:0000002',note=None)
          for r,t in zip(rows,['aa','bb','cc','dd','ee'])],association=[{'patient_id':'P','phenotype':[]}])]
    scores=dict(schema='bioclinical-baseline-span-scores-v1',status='COMPLETE_ALL_BASELINE_TARGET_SCORES',records=rows)
    keys=run.negative_spans(base,scores,['w0','w1']); assert keys=={('p',0,2,'aa')}
    pred,removed=run.veto(base,keys)
    assert len(removed)==1 and pred[0]['entities']==base[0]['entities'][1:] and pred[0]['association']==base[0]['association']
    passed=['cross-window max uses every representation; all-negative deletes; zero and positive keep',
            'unaligned and uncovered keep; compound object follows same exact span rule']
    for value in [float('nan'),float('inf'),-float('inf')]:
        bad=copy.deepcopy(scores);bad['records'][2]['window_scores'][0]['score']=value
        fails(lambda:run.negative_spans(base,bad,['w0','w1']))
        bad=copy.deepcopy(scores);bad['records'][2]['max_score']=value
        fails(lambda:run.negative_spans(base,bad,['w0','w1']))
    passed.append('any nonfinite raw score or max fails even in a kept record')
    for change in [lambda s:s['records'][0].update(max_score=-2.),
                   lambda s:s['records'][0]['window_scores'][0].update(row_id='missing'),
                   lambda s:s['records'].pop(),
                   lambda s:s['records'][0]['window_scores'].append(s['records'][0]['window_scores'][0]),
                   lambda s:s['records'][0].update(representable=False)]:
        bad=copy.deepcopy(scores);change(bad);fails(lambda:run.negative_spans(base,bad,['w0','w1']))
    passed.append('wrong max window identity coverage duplicate representation or flag fails')
    target_map=dict(records=[{k:v for k,v in r.items() if k not in ('window_scores','max_score')} for r in rows],
                    window_targets={w:[] for w in ('w0','w1')})
    for row in rows:
        for r in row['window_scores']:
            target_map['window_targets'][r['row_id']].append(dict(target_id=row['target_id'],block_index=r['block_index'],i=r['i'],j=r['j']))
    run.validate_supply(scores,target_map,['w0','w1'])
    bad=copy.deepcopy(scores);bad['records'][0]['window_scores'].pop(0)
    fails(lambda:run.validate_supply(bad,target_map,['w0','w1']))
    passed[-1]+='; missing representation rejected against input-only target map'
    duplicate=copy.deepcopy(base);duplicate[0]['entities'].append(copy.deepcopy(base[0]['entities'][0]))
    dup_scores=copy.deepcopy(scores);r=copy.deepcopy(rows[0]);r.update(target_id='p:5',entity_index=5);dup_scores['records'].append(r)
    assert len(run.veto(duplicate,run.negative_spans(duplicate,dup_scores,['w0','w1']))[1])==2
    dup_scores['records'][-1]['window_scores'][0]['score']=-1.
    fails(lambda:run.negative_spans(duplicate,dup_scores,['w0','w1']))
    passed.append('all same-span entity objects delete uniformly; conflicting records fail')
    data=b'  original bytes\r\n';assert run.payload(data,base,[])==data
    run.put(output,dict(status='PASS_NEW_LONG_CONTEXT_SCORE_CONTRACTS',source=run.source_refs(),
        groups=passed,seconds=time.monotonic()-start,actual_outputs_gold_model_quality_calls=0,
        reused_prior_contracts='boundary_veto@f09bb7b7 object preservation, exact spans, exclusive writes, gold guard; not re-expanded'))
    print(json.dumps({'groups':len(passed),'output':output}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);main(parser.parse_args().output)
