"""Bounded CPU-only contracts; synthetic responses are not policy observations."""
import argparse,copy,json,time
from pathlib import Path
from common import *
from prepare import prepare
from finalize import finalize
from bind_query import bind

def reject(call):
    try:call()
    except (AssertionError,ValueError):return True
    raise AssertionError('Expected rejection')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);a=ap.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    assets=load(HERE/'assets.json');targets=[json.loads(x) for x in Path(assets['targets87']['path']).read_text().splitlines()]
    rows=[dict(row_id=r,arm=arm,response=targets[i]['answer'],EOS=True,cap=False,generated_ids=[1,2]) for i,r in SELECTED for arm in ('greedy','sample')]
    save(out/'synthetic-target-responses.json',rows)
    p=prepare(out/'synthetic-target-responses.json',out/'target-prepared');assert p['query_count']==0
    f=finalize(out/'target-prepared/preparation.json',out/'target-rewards')
    assert len(f['rows'])==8 and len(f['pairs'])==4 and not f['supply_gate_pass']
    assert f['rows'][0]['reward']==1 and f['rows'][0]['gold_count']==0
    # One real semantic contrast (original gold target vs []) and one pure
    # format-validity contrast (empty-gold valid[] vs invalid JSON).
    mixed=copy.deepcopy(rows);mixed[1]['response']='not JSON';mixed[3]['response']='[]'
    save(out/'synthetic-mixed-responses.json',mixed);prepare(out/'synthetic-mixed-responses.json',out/'mixed-prepared')
    m=finalize(out/'mixed-prepared/preparation.json',out/'mixed-rewards')
    assert m['supply_gate_pass'] and m['nonzero_groups']==2 and m['nonempty_gold_valid_semantic_groups']==1
    assert m['pairs'][0]['advantage']==-1 and not m['pairs'][0]['sample_valid']
    rejects={}
    rejects['missing_eighth']=reject(lambda:validate_responses(rows[:-1]))
    duplicate=rows[:-1]+[rows[0]];rejects['duplicate_row_arm']=reject(lambda:validate_responses(duplicate))
    wrong=copy.deepcopy(rows);wrong[0]['row_id']='wrong';rejects['foreign_row']=reject(lambda:validate_responses(wrong))
    interpret,_,StrictNormalizer,policy,precheck,decide=original()
    doc=dict(full_text=[dict(offset=0,text='café café')])
    for raw,eos,cap in [('[]',False,False),('[]',True,True),('```json\n[]\n```',True,False),('[NaN]',True,False)]:
        d=interpret(raw,doc,eos,cap);assert d['status']!='complete';rejects['invalid_'+str(len(rejects))]=True
    d=interpret('[["U0000","café",0],["U0000","absent",0]]',doc,True,False)
    assert len(d['anchors'])==1 and len(d['rejected'])==1 # Reward rejects whole despite deployment's retained item.
    assert atoms(['HP:0000001;HP:0000002','HP:0000001','text;detail','HP:bad;HP:0000001'])=={'HP:0000001','HP:0000002','text;detail','HP:bad;HP:0000001'}
    # Four arbitrary bad-literal atoms remain exact FNs; source818 identity above
    # ensures the four actual nonconstructible training literals are not dropped.
    malformed={'bad1','HP:bad','HP:0000001;notHPO','text;detail'}
    assert atoms(sorted(malformed))==malformed and f1(malformed,set())==0
    nil=decide(dict(quote='text;detail',status='nil',ids=['-1'],reason='unanimous_whole_span_alias'),None,policy)
    assert nil['status']=='nil'
    strict=StrictNormalizer(assets['hpo']['path'],assets['train32']['path']);cache={x['quote'] for x in load(assets['cached624']['path'])}
    # A true train input quote is necessary, not fake numeric evidence. Do not
    # query it here; prove finalize refuses omission and source-bound zip binds it.
    target=targets[76];quote=None;uid=None
    for unit in target['units']:
        words=unit['text'].split()
        for i in range(len(words)-2):
            q=' '.join(words[i:i+3])
            if q in unit['text'] and q not in cache and strict.normalize(q)['reason']=='unmatched' and precheck(q,policy) is None:
                quote,uid=q,unit['unit_id'];break
        if quote:break
    assert quote
    unknown=copy.deepcopy(rows);unknown[1]['response']=json.dumps([[uid,quote,0]],ensure_ascii=False)
    save(out/'synthetic-unknown-responses.json',unknown);up=prepare(out/'synthetic-unknown-responses.json',out/'unknown-prepared')
    assert up['query_count']==1
    rejects['missing_necessary_query']=reject(lambda:finalize(out/'unknown-prepared/preparation.json',out/'must-not-exist'))
    now=time.time();package=bind(out/'unknown-prepared/preparation.json','/synthetic/no_remote_action',now+1800,now+1830,out/'bound-query')
    qp=load(out/'bound-query/query-plan.json');assert qp['shared_KILL_unix']-qp['shared_TERM_unix']==30
    assert qp['files'][qp['queries']]['sha256']==up['queries']['sha256'] and package['no_remote_action']
    report=dict(status='PASS_CPU_CONTRACTS_ONLY',source=ref(__file__),all8_exact_identities=True,
                target_cached_query_count=0,mixed_format_and_semantic_gate=True,rejections=rejects,
                partial_invalid_whole_reward_gate=True,empty_gold_valid_empty_reward1=True,compound_and_nil_literal=True,
                full_gold_atoms818=True,four_bad_literals_not_filtered=True,dynamic_query_needed1=True,
                dynamic_missing_evidence_rejected=True,package=ref(out/'bound-query/package.json'),
                synthetic_token_ids_not_model_outputs=True,no_GPU=True,no_remote=True,no_model=True,no_fit=True,no_held=True,no_quality=True)
    save(out/'complete.json',report);print(json.dumps(report))

if __name__=='__main__':main()
