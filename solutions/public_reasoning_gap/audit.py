"""Bounded existing-source/response-metadata audit; no model, parser, fit or score."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path


def ref(path):
    p=Path(path);b=p.read_bytes()
    return dict(path=str(p.resolve()),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())


def checked(expected):
    p=Path(expected['path']);r=ref(p)
    assert r['bytes']==expected['bytes'] and r['sha256']==expected['sha256']
    return p


def read(path):return json.loads(Path(path).read_text())


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--runs',type=Path,required=True)
    ap.add_argument('--case-route',type=Path,required=True);ap.add_argument('--sequence-route',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    root=a.runs/'research/deepseek-r1-0528-qwen3-8b-feasibility';r=root/'full80-preparation'
    plan=read(r/'plan.json');terminal=read(r/'recovered/terminal/result.json');collection=read(r/'response-collection.json')
    source=r/'infer_full80.py';src=source.read_text()
    for name,sha in plan['sources'].items():assert ref(r/name)['sha256']==sha
    assert terminal['source']['sha256']==ref(source)['sha256']
    assert terminal['plan']['sha256']==ref(r/'plan.json')['sha256']
    assert 'PeftModel' not in src and 'load_adapter' not in src
    assert 'AutoModelForCausalLM.from_pretrained(modeldir' in src
    assert terminal['parameters']['count']==8190735360 and terminal['parameters']['tensors']==399
    actual=gzip.decompress(checked(collection['local']).read_bytes())
    assert hashlib.sha256(actual).hexdigest()==terminal['response_uncompressed_sha256']
    rows=[json.loads(line) for line in actual.splitlines()]
    assert len(rows)==80 and len({x['row_id'] for x in rows})==80
    # Read already-generated records to count boundaries only; no answer parsing,
    # semantic repair, tokenization, new generation, or relation-quality analysis.
    boundary=Counter()
    for row in rows:
        text=row['raw_generated_text'];counts=(text.count('<think>'),text.count('</think>'))
        assert counts==(row['think_open_count'],row['think_close_count'])
        assert text.lstrip().startswith('<think>')
        boundary[str(counts)]+=1
    assert boundary=={'(1, 1)':80}
    tc=read(root/'metadata/tokenizer_config.json')
    assert 'enable_thinking' not in tc['chat_template']
    assert "'<｜User｜>' + content + '<｜Assistant｜>'" in tc['chat_template']
    assert 'enable_thinking=False' not in src
    prompts_ref=read(r/'input-census-independent.json')['inputs']['frozen_prompts']
    prompts=[json.loads(line) for line in checked(prompts_ref).read_text().splitlines()]
    assert len(prompts)==80 and all([m['role'] for m in row['messages']]==['system','user'] for row in prompts)
    prior=read(a.case_route/'evidence/round2/qwen35-successor-readonly-review.json')
    old=read(checked(prior['evidence']['cpu-findings.json']))
    catalog=[]
    for entry in old['actual_qwen35_manifests']:
        d=read(checked(entry['file']));f=d['fingerprint'];assert f['thinking'] is False
        catalog.append(dict(file=entry['file'],thinking=False,adapter_present=f['configuration']['adapter'] is not None,
                            max_new_tokens=f['configuration']['max_new_tokens'],expected_rows=d['expected_rows']))
    assert len(catalog)==12
    files={n:ref(r/n) for n in ['plan.json','infer_full80.py','full80_contract.py','normalized_final.py','strict_final.py',
        'terminal-manifest.json','terminal-review.json','response-collection.json','quality/summary.json','RESULT.md']}
    files.update(R1_template=ref(root/'metadata/tokenizer_config.json'),R1_actual_terminal=ref(r/'recovered/terminal/result.json'),
                 R1_prompts=prompts_ref,R1_responses=collection['local'],
                 Qwen_nonthinking_source=ref(a.case_route/'qwen35_runner.py'),
                 Qwen_thinking4096_source=ref(a.case_route/'operations/thinking_sampling/runner.py'))
    for name in ['qwen35-successor-readonly-review.json','qwen35-thinking12-run-plan.json','qwen35-thinking12-result.json','thinking-sampling-plan.json','thinking-sampling-result.json']:
        files[name]=ref(a.case_route/'evidence/round2'/name)
    files['SFT_budget80_deferred']=ref(a.sequence_route/'experiments/thinking-budget-80-plan.md')
    files['SFT_budget_single_result']=ref(a.sequence_route/'evidence/round2/thinking-budget-final-results.json')
    result=dict(status='CLOSE_GENERIC_PUBLIC_REASONING_PROPOSAL_ALREADY_COMPLETED_IN_R1',source=ref(__file__),refs=files,
        R1=dict(public_model='deepseek-ai/DeepSeek-R1-0528-Qwen3-8B',MS_revision=plan['MS_revision'],parameters=8190735360,
                task_adapter_loaded=False,source_matches_actual_terminal=True,rows=80,fresh=78,same_configuration_reused=2,
                generation=terminal['generation_arguments'],native_seconds=terminal['program_seconds'],
                raw_think_boundary_counts=dict(boundary),output_tokens=sum(x['generated_tokens'] for x in rows),
                closed_think_and_valid_final=sum(x['normalized_valid'] for x in rows),
                capped=sum(x['stopped_at_token_cap'] for x in rows),
                native_EOS=sum(x['native_EOS_observed'] for x in rows),
                template_switch='native published template has no enable_thinking toggle; model actually emitted80 closed think segments',
                prompt_policy='Original system/user only, no assistant demonstrations; candidate HPO meanings+verbatim local evidence+global patient anchors. System requests final JSON without explanation/markdown.',
                final_policy='Separate closing-think boundary plus entire JSON; predeclared whole fence/int-P codec only, invalid/capped row abstains, missingCID not negative.'),
        Qwen35=dict(prior_actual_nonthinking_manifests=catalog,public_base_full_reasoning_F1_found=False,
                    caveat='Scoped evidence only. Public9B thinking full F1 not found; observed true1600/4096 and forced-final are task-SFT adapter diagnostics, not public-base quality controls.'),
        read_scope=dict(held_gold=0,training_targets=0,new_model_calls=0,new_tokenizer_calls=0,new_final_parser_calls=0,new_fit=0,new_quality=0,
                        weights_rehashed=False,existing_generated_records_boundary_counts_only=True),
        decision='Do not rename the completed R1 public-thinking+final-parser mechanism as untried. No new GPU/fit plan. Exact publicQwen9B variant is not proven tested but currently lacks train-only evidence for a distinct mechanism.',
        limits=['R1 result only covers its fixed candidate/context and4096 codec; does not disprove all reasoning approaches.',
                'Open evidence retrieval, different full-report input, budget forcing, or model swap are not claimed equivalent; any proposal needs its own substantive train-only gap evidence.',
                'Official second-team overview is motivation, not an executable prompt/model/budget specification.'])
    with a.output.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(dict(status=result['status'],R1=result['R1'],qwen_prior_nonthinking_count=len(catalog))))


if __name__=='__main__':main()
