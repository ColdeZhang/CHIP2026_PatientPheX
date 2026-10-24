"""New limited-codec boundaries plus exact old23 strict equivalence; no old suite."""
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

from final_parser import parse_final
from limited_codec import classify_final


def ref(path):
    raw=Path(path).read_bytes()
    return dict(path=str(path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())


def main():
    began=time.monotonic(); b=Path(__file__).resolve().parent
    plan=json.loads((b/'plan.json').read_text())
    for item in [plan['original23'],plan['prompts']]:
        actual=ref(item['path']);assert actual==item
    prompts={x['row_id']:x for x in map(json.loads,Path(plan['prompts']['path']).read_text().splitlines())}
    old=list(map(json.loads,Path(plan['original23']['path']).read_text().splitlines()))
    assert len(old)==len({x['row_id'] for x in old})==23
    cached=[]
    for record in old:
        prompt=prompts[record['row_id']]; strict=parse_final(record,prompt); limited=classify_final(record,prompt)
        expected=[[pid,cid] for pid in prompt['patient_ids'] for cid in prompt['code_map'] if pid in strict[cid]]
        assert limited['status']=='complete' and limited['selected_pairs']==expected
        assert limited['observed_key_subset']==prompt['patient_ids']
        cached.append(dict(row_id=record['row_id'],strict_valid=True,limited_complete=True,
                           selected_pair_count=len(expected),exact_pair_and_observed_equivalence=True))
    prompt=dict(row_id='synthetic:0',pmc_id='synthetic',patient_ids=['P1','P10'],
        code_map={'C000':'HP:0000001','C001':'HP:0000002'},messages=[
            dict(role='system',content='synthetic codec input'),dict(role='user',content='No clinical content')])
    def record(text):
        return dict(row_id=prompt['row_id'],pmc_id=prompt['pmc_id'],response=text,
                    generated_token_ids=[100,101,248046],output_tokens=3,
                    hit_token_limit=False,removed_demonstrations=0)
    positive=[
        ('complete_reordered_keys','{"P10":["C001"],"P1":[]}', 'complete',['P1','P10'],[['P10','C001']]),
        ('partial_positive_does_not_fill_other_patient','{"P1":["C000"]}', 'partial',['P1'],[['P1','C000']]),
        ('partial_explicit_empty_is_observed_not_missing','{"P10":[]}', 'partial',['P10'],[]),
        ('empty_object_abstains_all','{}','abstain',[],[])]
    checks=[]
    for name,text,status,observed,pairs in positive:
        out=classify_final(record(text),prompt)
        assert (out['status'],out['observed_key_subset'],out['selected_pairs'])==(status,observed,pairs)
        checks.append(dict(name=name,status=out['status'],reason=out['reason']))
    negative={
        'valid_prefix_unknown_PID_discards_all':'{"P1":["C000"],"P2":[]}',
        'escaped_duplicate_PID_discards_all':'{"P1":["C000"],"\\u00501":[]}',
        'valid_PID_then_duplicate_CID_discards_all':'{"P1":["C000"],"P10":["C001","C001"]}',
        'valid_PID_then_unknown_CID_discards_all':'{"P1":["C000"],"P10":["C999"]}',
        'partial_wrong_value_not_salvaged':'{"P1":["C000"],"P10":null}',
        'partial_nonstring_CID_not_salvaged':'{"P1":["C000"],"P10":[true]}',
        'nonfinite_constant_rejected':'{"P1":["C000"],"P10":[NaN]}',
        'overflow_numeric_rejected':'{"P1":["C000"],"P10":[1e999]}',
        'array_of_objects_not_salvaged':'[{"P1":["C000"]}]',
        'fragment_not_recovered':'prefix {"P1":["C000"]}',
        'second_JSON_not_ignored':'{"P1":["C000"]} {}'}
    for name,text in negative.items():
        out=classify_final(record(text),prompt)
        assert out['status']=='abstain' and not out['observed_key_subset'] and not out['selected_pairs']
        checks.append(dict(name=name,status=out['status'],reason=out['reason']))
    changes={
        'cap_even_valid_partial':dict(hit_token_limit=True),
        'missing_EOS_even_valid_partial':dict(generated_token_ids=[100,101,102]),
        'internal_EOS_even_valid_partial':dict(generated_token_ids=[248046,101,248046]),
        'wrong_actual_count':dict(output_tokens=2),
        'empty_token_record':dict(generated_token_ids=[],output_tokens=0),
        'boolean_token_ID':dict(generated_token_ids=[True,101,248046]),
        'wrong_row_identity':dict(row_id='another:0'),
        'removed_prompt':dict(removed_demonstrations=1),
        'annotation_disagrees_with_actual_EOS':dict(terminal_eos_id=248044),
        'annotation_denies_actual_EOS':dict(actual_last_token_eos=False)}
    for name,change in changes.items():
        rec=record('{"P1":["C000"]}');rec.update(change);out=classify_final(rec,prompt)
        assert out['status']=='abstain' and not out['observed_key_subset'] and not out['selected_pairs']
        checks.append(dict(name=name,status=out['status'],reason=out['reason']))
    annotated=record('{"P1":[]}');annotated.update(terminal_eos_id=248046,actual_last_token_eos=True)
    assert classify_final(annotated,prompt)==classify_final(record('{"P1":[]}'),prompt)
    checks.append(dict(name='truthful_optional_annotations_do_not_change_facts',status='partial'))
    wrong=copy.deepcopy(prompt);wrong['answer']='{}'
    try:classify_final(record('{}'),wrong)
    except ValueError:checks.append(dict(name='answer_in_prompt_is_caller_error',status='raised'))
    else:raise AssertionError('Gold/answer-bearing prompt accepted')
    report=dict(status='PASS_new_limited_codec_and_cached23_only',at_utc=datetime.now(timezone.utc).isoformat(),
        seconds=time.monotonic()-began,source=ref(b/'limited_codec.py'),checker=ref(__file__),
        plan=ref(b/'plan.json'),source_plan=ref(b/'source-plan.json'),references=[plan['original23'],plan['prompts'],*plan['strict_sources'].values()],
        cached23=dict(rows=23,all_original_strict_valid=True,all_limited_complete_and_equivalent=True,rows_checked=cached),
        new_boundary_checks=checks,new_boundary_check_count=len(checks),
        raw_original_unchanged=ref(plan['original23']['path'])==plan['original23'],
        zero_gold_model_generation_scoring=True,old25_14_15_suites_not_rerun=True,
        tokenizer_replay_not_rerun=True,limitations=['Synthetic tests validate codec, not text-to-token correspondence; actual raw persistence/codec integration belongs to owner runner','No complete new35 inference/58 assembly/quality tested here'])
    (b/'receipt.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],seconds=report['seconds'],cached23=True,new_boundaries=len(checks),receipt=ref(b/'receipt.json')),indent=2))


if __name__=='__main__':main()
