"""Only stdlib/static/synthetic raw-first checks; never invoke load_inputs or inference."""
from pathlib import Path
import ast
import copy
import hashlib
import json
import tempfile
import time

from full58_contract import COMPLETE, parse_item, validate_plan
from raw_first import persist_fresh, ref


def main():
    start = time.monotonic()
    base = Path(__file__).resolve().parent
    d = base.parents[1]
    plan = json.loads((base / 'plan.template.json').read_text())
    for name, digest in plan['sources'].items():
        data = (base / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == digest
        compile(data, str(base / name), 'exec')
    assert plan['execution_ready'] is False
    try:
        validate_plan(plan)
    except AssertionError:
        pass
    else:
        raise AssertionError('Disabled template accepted')
    unbound = copy.deepcopy(plan)
    unbound['execution_ready'] = True
    try:
        validate_plan(unbound)
    except AssertionError:
        pass
    else:
        raise AssertionError('Missing authorization/budget/cache/ACK accepted')

    def part(text, begin, end):
        return text[text.index(begin):text.index(end, text.index(begin))]

    runner = (base / 'infer_full58.py').read_text()
    probe = (d / 'f0-preparation/probe-source/runner.py').read_text()
    f1 = (d / 'full80-preparation/full80_contract.py').read_text()
    contract = (base / 'full58_contract.py').read_text()
    assert part(runner, '        assets=json.loads(', "        rows,encoded,audits,cached=load_inputs(plan)") == part(
        probe, '        assets=json.loads(', '        for key,output_name in')
    assert part(runner, '        load_start=time.monotonic()', '        # Same per-row sampling;') == part(
        probe, '        load_start=time.monotonic()', '        # A fixed internal wall stop')
    assert part(contract, 'def parse_item(', '\ndef load_inputs(').strip() == part(
        f1, 'def parse_item(', '\ndef load_inputs(').strip()
    assert all(hashlib.sha256((base / name).read_bytes()).hexdigest() ==
               hashlib.sha256((d / 'full80-preparation' / name).read_bytes()).hexdigest()
               for name in ('strict_final.py', 'sparse_final.py', 'normalized_final.py'))
    assert '10800' not in runner and '10770' not in runner and 'range(80)' not in runner
    assert 'new_generation_calls_returned' in runner and 'complete_response_contract=False' in runner
    assert 'if stopper.wall_stop:raise TimeoutError' in runner
    assert not any(isinstance(node, ast.Break) for node in ast.walk(ast.parse(runner)))
    assert plan['evaluation_contract_template']['complete_status'] == COMPLETE
    assert plan['evaluation_contract_template']['wrapper_coverage_flag'] == 'fixed58_whole_response_contract'
    assert plan['evaluation_contract_template']['result_target_count_field'] == 'original58_target'

    prompt = {'row_id': 'synthetic:relations:0', 'pmc_id': 'synthetic',
              'code_map': {'C000': 'HP:0000001', 'C001': 'literal concept'},
              'patient_ids': ['P1', 'OII-1'], 'messages': []}
    original = {'row_id': prompt['row_id'], 'original_index': 0, 'fresh_NN': True,
                'seed': 20261024, 'generated_ids': [17, 151645], 'generated_tokens': 2,
                'native_EOS_observed': True, 'stopped_at_token_cap': False,
                'stopped_at_wall_budget': False, 'quality_evaluated': False,
                'input_prefix_exact': True, 'generated_length_within_cap': True}
    counts = {}
    with tempfile.TemporaryDirectory(prefix='raw-first-pure-fixtures-', dir=base) as folder:
        root = Path(folder)
        for case, body in [('valid_sparse_literal', '{"C001":["OII-1"]}'),
                           ('invalid_whole_row', '{"C000":["P1"],"C001":[999]}')]:
            output = root / case
            output.mkdir()
            item = persist_fresh(output, 0, original,
                                 lambda ids: body + ('<EOS>' if len(ids) == 2 else ''),
                                 parse_item, prompt)
            assert (output / 'row0.ids.json').exists() and (output / 'row0.raw.json').exists()
            assert item['raw_reference'] == ref(output / 'row0.raw.json')
            if case == 'valid_sparse_literal':
                assert item['normalized_evidence']['provided_mapping'] == {'C001': ['OII-1']}
                assert item['coverage']['abstained_CIDs'] == ['C000']
            else:
                assert item['normalized_valid'] is False and item['coverage']['provided_CIDs'] == []
                assert item['coverage']['invalid_whole_row_abstention'] is True
            counts[case] = True
        for case in ('decode_exception', 'parser_exception'):
            output = root / case
            output.mkdir()
            def explode(*args, **kwargs):
                raise RuntimeError('synthetic ' + case)
            decode = explode if case == 'decode_exception' else lambda ids: '{}'
            parser = explode if case == 'parser_exception' else parse_item
            try:
                persist_fresh(output, 0, original, decode, parser, prompt)
            except RuntimeError:
                pass
            else:
                raise AssertionError('Unknown error was swallowed')
            assert json.loads((output / 'row0.ids.json').read_text())['generated_ids'] == [17, 151645]
            assert not (output / 'row0.json').exists()
            if case == 'parser_exception':
                raw = ref(output / 'row0.raw.json')
                failure = json.loads((output / 'row0.parse-error.json').read_text())
                assert failure['raw_reference'] == raw
            else:
                assert (output / 'row0.decode-error.json').exists()
            counts[case] = True
    result = {'status': 'PASS_FULL58_SOURCE_ONLY_STATIC_AND_PURE_RAW_FIRST_FIXTURES',
              'sources': {name: ref(base / name) for name in plan['sources']},
              'template': ref(base / 'plan.template.json'), 'check_source': ref(__file__),
              'actual_fixture_seconds': time.monotonic() - start,
              'F0_A800_asset_copy_gate_byte_identical': True,
              'mature_native399_BF16finite_load_rotary_block_byte_identical': True,
              'three_parsers_and_parse_item_byte_identical': True,
              'fixtures': counts, 'null_authorization_budget_cache_ACK_rejected': True,
              'normal_invalid_is_complete_row_abstention_no_fragment': True,
              'unknown_parser_error_retains_full_IDs_and_raw': True,
              'decode_error_retains_full_IDs': True,
              'F1_or_probe_source_modified': False,
              'actual_load_inputs_native_tokenizer_model_generation_gold_or_quality_executed': False,
              'complete_full58_budget_cache_or_launch_authorized': False}
    with (base / 'source-check.json').open('x') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
