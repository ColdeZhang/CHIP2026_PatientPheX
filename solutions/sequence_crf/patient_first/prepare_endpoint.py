"""CPU-only final envelope checks and synthetic dense output cost from inputs."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

from patient_first import compact, validate_input
from final_parser import parse_final


def ref(path):
    path = Path(path); raw = path.read_bytes()
    return dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--prompts', type=Path, required=True)
    parser.add_argument('--token-audit', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--template', type=Path, required=True)
    parser.add_argument('--cpu-libraries', required=True)
    args = parser.parse_args(); started = time.monotonic()
    assert ref(args.prompts)['sha256'] == '32b52a2d580024342f25a5d74388c61323f2cc3bbf2bdb0eea404b19a9d64b18'
    assert ref(args.token_audit)['sha256'] == '7fc89abd9075705853071502f0131c4b50ec8fe025154926a778ef74f68de3be'
    assert ref(args.tokenizer)['sha256'] == '5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42'
    assert ref(args.template)['sha256'] == 'a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715'
    sys.path.insert(0, args.cpu_libraries)
    import tokenizers
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    assert tokenizers.__version__ == '0.23.1'
    tok = tokenizers.Tokenizer.from_file(str(args.tokenizer))
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.globals['raise_exception'] = lambda value: (_ for _ in ()).throw(ValueError(value))
    template = env.from_string(args.template.read_text())
    prompts = [json.loads(line) for line in args.prompts.read_text().splitlines()]
    audit = {r['row_id']:r['patient_first'] for r in json.loads(args.token_audit.read_text())}
    dense = []
    for prompt in prompts:
        validate_input(prompt); assert 'answer' not in prompt
        # Synthetic maximal-density assignment is input-only cost material,
        # not a prediction, label, or claim about arbitrary whitespace output.
        answer = compact({pid:list(prompt['code_map']) for pid in prompt['patient_ids']})
        prefix = template.render(messages=prompt['messages'], tools=None, add_generation_prompt=True, enable_thinking=False)
        full = template.render(messages=prompt['messages']+[dict(role='assistant',content=answer)],
                               tools=None, add_generation_prompt=False, enable_thinking=False)
        prefix_ids = tok.encode(prefix, add_special_tokens=False).ids
        ids = tok.encode(full, add_special_tokens=False).ids
        assert ids[:len(prefix_ids)] == prefix_ids
        assert len(prefix_ids) == audit[prompt['row_id']]['input_tokens']
        suffix = ids[len(prefix_ids):]; assert suffix.count(248046) == 1
        through_eos = suffix[:suffix.index(248046)+1]
        assert tok.decode(through_eos[:-1], skip_special_tokens=False) == answer
        dense.append(dict(row_id=prompt['row_id'], input_tokens=len(prefix_ids),
                          dense_compact_response_tokens_including_EOS=len(through_eos)))
    assert len(dense) == 80
    # Only the new EOS/identity envelope is exercised; prior25 schema cases
    # remain frozen and are not repeated.
    prompt = prompts[0]
    response = compact({pid:[] for pid in prompt['patient_ids']})
    ids = tok.encode(response, add_special_tokens=False).ids+[248046]
    valid = dict(row_id=prompt['row_id'], pmc_id=prompt['pmc_id'], response=response,
                 generated_token_ids=ids, output_tokens=len(ids), hit_token_limit=False, removed_demonstrations=0)
    assert parse_final(valid, prompt) == {cid:[] for cid in prompt['code_map']}
    variants = {}
    for name in ['missing_EOS','embedded_EOS','wrong_identity','token_count','token_limit','prompt_removed']:
        item = copy.deepcopy(valid)
        if name == 'missing_EOS': item['generated_token_ids'][-1] = 198
        elif name == 'embedded_EOS': item['generated_token_ids'][0] = 248044
        elif name == 'wrong_identity': item['row_id'] += ':other'
        elif name == 'token_count': item['output_tokens'] += 1
        elif name == 'token_limit': item['hit_token_limit'] = True
        else: item['removed_demonstrations'] = 1
        try: parse_final(item, prompt)
        except ValueError: variants[name] = 'rejected'
        else: raise AssertionError('Envelope accepted '+name)
    report = dict(status='PASS_new_envelope_and_input_only_cost_not_quality',
         source=ref(__file__), prompts=ref(args.prompts), audit=ref(args.token_audit),
         tokenizer=ref(args.tokenizer), template=ref(args.template),
         rows=80, input_tokens=sum(r['input_tokens'] for r in dense),
         max_input_tokens=max(r['input_tokens'] for r in dense),
         max_dense_compact_response_tokens=max(r['dense_compact_response_tokens_including_EOS'] for r in dense),
         dense_examples=dense, envelope_valid_cases=1, envelope_rejections=variants,
         old25_schema_cases_not_repeated=True, training_or_F1_targets_read=False,
         generation_policy='Same old greedy nonthinking max_input14000/max_new1600; no budget sweep.',
         cost_limit='Dense compact values measure a reference shape, not a bound on arbitrary JSON whitespace or model output.',
         torch_imported='torch' in sys.modules, seconds=time.monotonic()-started)
    assert not report['torch_imported']
    with (args.directory/'endpoint-CPU-contract.json').open('x') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2); handle.write('\n')
    print(json.dumps({k:report[k] for k in ['status','rows','input_tokens','max_input_tokens',
          'max_dense_compact_response_tokens','seconds']}))


if __name__ == '__main__':
    main()
