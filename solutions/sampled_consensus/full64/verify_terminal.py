"""CPU closure of the actual native128 artifacts before any prediction quality."""
import argparse
import hashlib
import json
from pathlib import Path
from assemble import reference,read_jsonl,save_json,jsha,f1
load_native_parser=f1.load_native_parser
from sampling import GENERATION,SEEDS,row_seed

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('--collected',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();plan=json.loads(Path(a.plan).read_text());c=Path(a.collected);native=c/'run-v1/native';complete=json.loads((native/'complete.json').read_text());driver=json.loads((c/'run-v1/driver-result.json').read_text());start=json.loads((c/'run-v1/driver-start.json').read_text());release=json.loads((c/'run-v1/released-gpu.json').read_text())
    assert complete['status']=='COMPLETE_FIXED128_FRESH_NATIVE_NONTHINKING_NO_QUALITY' and complete['fresh_rows']==128 and complete['cached_rows']==0
    assert complete['total_parameters']==8997081600 and complete['base_parameters']==8953803264
    assert driver['return_code']==0 and not driver['timed_out'] and driver['error'] is None and driver['elapsed_seconds']<=2100
    assert start['budget_seconds']==2100 and start['grace_seconds']==30 and abs(start['deadline_KILL_unix']-start['deadline_TERM_unix']-30)<1e-6
    assert not release['target_compute_rows'] and int(release['gpu'].split(',')[2])<=100
    expected=reference(a.plan)['sha256'];assert complete['plan_sha256']==driver['plan_sha256']==start['plan_sha256']==expected
    for path,expected in plan['files'].items():
        remote_root=plan['source'].rsplit('/',1)[0];relative=Path(path).relative_to(remote_root);assert reference(c/relative)['sha256']==expected
    actual=json.loads((native/'actual-generation.json').read_text());assert actual['kwargs']=={**GENERATION,'eos_token_id':plan['eos_ids'],'pad_token_id':plan['pad_token_id']};assert actual['presence_penalty']==0 and actual['presence_processor_added'] is False
    actual_inputs=json.loads((native/'actual-input-and-assets.json').read_text());assert actual_inputs['processing_original_indices'][0]==1 and len(set(actual_inputs['processing_original_indices']))==64 and actual['longest_first_per_seed_is_formal_no_probe']
    prompts=read_jsonl(plan['input_refs']['prompts']['path'])[1];audit=json.loads(Path(plan['input_refs']['audit']['path']).read_text());parser=load_native_parser(Path(__file__).parent/'vendor/probe_runner.py');summaries=[]
    for seed in SEEDS:
        sd=native/str(seed);raws=read_jsonl(sd/'responses.jsonl')[1];canon=read_jsonl(sd/'canonical.jsonl')[1];br={r['row_id']:r for r in raws};bc={r['row_id']:r for r in canon};assert len(raws)==len(br)==len(canon)==len(bc)==64 and set(br)==set(bc)=={r['row_id'] for r in prompts}
        summary=next(s for s in complete['seeds'] if s['seed']==seed);assert reference(sd/'responses.jsonl')['sha256']==summary['responses_sha256'] and reference(sd/'canonical.jsonl')['sha256']==summary['canonical_sha256']
        for i,(p,t) in enumerate(zip(prompts,audit)):
            r=br[p['row_id']];saved=json.loads((sd/'raw-responses'/f'{i:03d}.json').read_text());ids=json.loads((sd/'generated-ids'/f'{i:03d}.json').read_text());savedcanon=json.loads((sd/'canonical-records'/f'{i:03d}.json').read_text());assert r==saved and bc[p['row_id']]==savedcanon
            assert r['pmc_id']==p['pmc_id'] and r['seed']==seed and r['fresh'] is True and r['row_seed']==row_seed(seed,p['row_id'])
            for k in ['row_id','seed','row_seed','input_ids_sha256','generated_ids','configured_eos_ids','natural_eos','hit_token_limit']:assert r[k]==ids[k],k
            assert r['input_ids_sha256']==t['expanded']['input_ids_sha256'] and r['rendered_sha256']==t['expanded']['rendered_sha256'] and r['input_tokens']==t['expanded']['tokens'] and r['output_tokens']==len(r['generated_ids'])
            assert r['response']==r['native_response'] and r['raw_with_special_tokens']==r['native_raw_with_special_tokens']
            natural=bool(r['generated_ids'] and r['generated_ids'][-1] in plan['eos_ids']);hit=len(r['generated_ids'])>=1600 and not natural;assert natural==r['natural_eos'] and hit==r['hit_token_limit']
            parsed=None
            try:
                if not natural or hit or r['has_think_marker']:raise ValueError('Predeclared response abstention')
                parsed=parser(r['response'],p)
            except (ValueError,TypeError):pass
            assert savedcanon['strict_complete']==(parsed is not None) and savedcanon['candidate_patients']==(parsed if parsed is not None else {})
        assert sum(r['input_tokens'] for r in raws)==512091 and sum(r['output_tokens'] for r in raws)==summary['output_tokens']
        summaries.append({'seed':seed,'raws':reference(sd/'responses.jsonl'),'canonical':reference(sd/'canonical.jsonl'),'rows':64,'strict_complete':sum(r['strict_complete'] for r in canon),'output_tokens':sum(r['output_tokens'] for r in raws)})
    save_json(a.output,{'status':'PASS_ALL128_ORIGINAL_RAW_IDS_CANONICAL_INPUTS_AND_SOURCE_EXACT_NO_QUALITY','plan':reference(a.plan),'complete':reference(native/'complete.json'),'driver':reference(c/'run-v1/driver-result.json'),'release':reference(c/'run-v1/released-gpu.json'),'streams':summaries,'no_gold_read':True,'no_quality_calls':True})
    print(json.dumps({'status':'PASS_ALL128_NATIVE_TERMINAL','streams':summaries}),flush=True)
if __name__=='__main__':main()
