"""Recheck complete saved raw records and freeze the negative supply endpoint on CPU."""
import argparse, datetime, gzip, hashlib, importlib.util, json
from pathlib import Path

def ref(p):
    p=Path(p).resolve();b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}

def main():
    p=argparse.ArgumentParser();p.add_argument('--collection',required=True);p.add_argument('--semantic-review',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    c=Path(a.collection);payload=c/'payload';terminal=payload/'terminal';out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    s=importlib.util.spec_from_file_location('frozen_evidence_contract',payload/'evidence_contract.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    result=json.loads((terminal/'result.json').read_text());complete=json.loads((c/'complete.json').read_text());driver=json.loads((payload/'driver-complete.json').read_text());review=json.loads(Path(a.semantic_review).read_text());inv=json.loads((payload/'inventory.json').read_text())
    assert complete['members']==69 and len(inv['files'])==68 and driver['returncode']==0 and not driver['TERM_sent'] and not driver['KILL_sent']
    for name,expected in inv['files'].items():
        actual=ref(payload/name);assert {'bytes':actual['bytes'],'sha256':actual['sha256']}==expected,name
    originals=[json.loads((terminal/f'row{i}.raw.json').read_text()) for i in range(15)];bundle=[json.loads(x) for x in gzip.decompress((terminal/'responses.raw.jsonl.gz').read_bytes()).splitlines()];assert originals==bundle
    assert len({r['row_id'] for r in originals})==15 and {r['index'] for r in review['rows']}==set(range(15))
    rows=[]
    for i,r in enumerate(originals):
        parsed=json.loads((terminal/f'row{i}.parsed.json').read_text());assert r['native_EOS_observed']==(r['generated_ids'][-1]==151645);assert r['generated_tokens']==len(r['generated_ids'])
        assert r['stopped_at_token_cap']==(not r['native_EOS_observed'] and len(r['generated_ids'])>=3072);assert not r['stopped_at_wall_budget']
        try:m.final_json(r['final_text_excluding_terminal_EOS'],eos_observed=r['native_EOS_observed'],stopped_at_cap=r['stopped_at_token_cap'])
        except ValueError as e:assert parsed['status']=='MECHANICAL_REJECT' and parsed['error_type']==type(e).__name__ and parsed['error']==str(e)
        else:raise AssertionError('Unexpected mechanically accepted row; no automatic semantic decision')
        sr=next(x for x in review['rows'] if x['index']==i);assert sr['row_id']==r['row_id'] and sr['accepted_for_training'] is False and sr['category']=='MECHANICAL_REJECT'
        rows.append({'index':i,'row_id':r['row_id'],'tokens':len(r['generated_ids']),'EOS':r['native_EOS_observed'],'cap':r['stopped_at_token_cap'],'raw':ref(terminal/f'row{i}.raw.json'),'parsed':ref(terminal/f'row{i}.parsed.json')})
    x={'status':'CLOSED_NEGATIVE_FIXED15_EVIDENCE_SUPPLY_NOT_STUDENT_QUALITY','at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source':ref(__file__),'execution':ref(payload/'execution.json'),'collection':ref(c/'complete.json'),'native_result':ref(terminal/'result.json'),'driver':driver,'semantic_review':ref(a.semantic_review),'rows':rows,'completed':15,'EOS':sum(r['EOS'] for r in rows),'cap':sum(r['cap'] for r in rows),'generated_tokens':sum(r['tokens'] for r in rows),'mechanical_PASS':0,'accepted_attribution_chains':0,'accepted_cross_sentence_or_group_chains':0,'prelocked_gate':{'minimum_accepted':10,'minimum_cross_sentence_or_group':5,'passed':False},'native_program_seconds':result['program_seconds'],'peak_memory':result['peak_memory'],'no_parser_changes':True,'quality_calls':0,'student_training':False,'full814_teacher':False,'held_gold_reads':0,'review_is_blinded_or_independent':False}
    (out/'result.json').write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:x[k] for k in ('status','completed','EOS','cap','generated_tokens','accepted_attribution_chains')}))

if __name__=='__main__':main()
