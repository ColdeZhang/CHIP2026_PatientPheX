"""Real input checks and tiny negative contracts, no tokenizer/model/MCP calls."""
from pathlib import Path
from types import SimpleNamespace
import argparse
import copy
import hashlib
import json
import sys
import tempfile
import ast

sys.dont_write_bytecode = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/workspace/patientphex-runs/round2-20261023'))
    parser.add_argument('--output', required=True, type=Path)
    opts = parser.parse_args()
    e = opts.root / 'research/medgemma-relation-control'
    d = e / 'fullcontext-preparation'
    p = opts.root / 'case_retrieval/qwen35-preparation/sft-meta1-fusion'
    here = Path(__file__).resolve().parent
    sys.path[:0] = [str(here), str(d), str(e), str(p)]
    import fullcontext_contract as contract
    import train_medgemma32 as training
    import medgemma_interface as interface
    args = SimpleNamespace(train=str(d/'train.fullcontext.jsonl'), original_train=str(p/'train.jsonl'),
                           train_input=str(p/'public-training-candidate-preparation/train32.input.jsonl'),
                           excluded_input=str(p/'excluded.input.jsonl'), data_manifest=str(p/'preparation-manifest.json'),
                           model_source_manifest=str(opts.root/'research/medical-small-model-readiness/assets/expected-files.json'),
                           fullcontext_manifest=str(d/'preparation.json'), cpu_reference=str(d/'actual-cpu/summary.json'),
                           cpu_row_audit=str(d/'actual-cpu/row-token-audit.json'))
    rows, train_ids, excluded, summary, audited = contract.verify_sources_and_data(args, training, interface)
    checks = ['actual124_fullcontext_restores_original124_with32train32excluded', 'frozen_CPU_summary_and_row_audit_bound']

    def reject(name, fn):
        try:
            fn()
        except (ValueError, AssertionError):
            checks.append(name)
        else:
            raise AssertionError('Expected rejection: ' + name)

    with tempfile.TemporaryDirectory(prefix='med-fullcontext-contract-') as temporary:
        changed = Path(temporary)/'changed.jsonl'
        altered = copy.deepcopy(rows)
        altered[0]['answer'] = '[]' if rows[0]['answer'] != '[]' else '{}'
        changed.write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in altered))
        modified = copy.copy(args)
        modified.train = str(changed)
        reject('modified_training_target_rejected_before_model', lambda: contract.verify_sources_and_data(modified, training, interface))
        bad_summary = Path(temporary)/'summary.json'
        bad = dict(summary, train_full_tokens=dict(summary['train_full_tokens'], max=8192))
        bad_summary.write_text(json.dumps(bad))
        modified = copy.copy(args)
        modified.cpu_reference = str(bad_summary)
        reject('modified_cpu_capacity_receipt_rejected', lambda: contract.verify_sources_and_data(modified, training, interface))

    toy = {'row_id':'toy', 'input_ids':[2,7,8,106,9], 'labels':[-100,-100,8,106,9],
           'prefix_tokens':2, 'supervised_tokens':3, 'eos_token_id':106, 'eos_supervised_position':3,
           'rendered_prefix_sha256':'prefix', 'rendered_full_sha256':'full'}
    expected = {'row_id':'toy', 'full_tokens':5, 'prefix_tokens':2, 'supervised_tokens':3,
                'token_ids_sha256':contract.construct.jsha(toy['input_ids']),
                'supervised_ids_sha256':contract.construct.jsha(toy['input_ids'][2:]),
                'rendered_prefix_sha256':'prefix', 'rendered_full_sha256':'full',
                'prefix_mask_eos_verified':True, 'original_answer_unchanged':True}
    contract.verify_one_encoding(toy, expected)
    checks.append('tiny_token_contract_positive_not_native_encoding')
    for name, field, value in [
            ('unmasked_prompt_rejected','labels',[2,-100,8,106,9]),
            ('same_length_changed_target_token_rejected','input_ids',[2,7,10,106,9]),
            ('wrong_native_stop_position_rejected','eos_supervised_position',2),
            ('changed_rendered_prompt_rejected','rendered_prefix_sha256','different')]:
        changed = copy.deepcopy(toy)
        changed[field] = value
        reject(name, lambda changed=changed: contract.verify_one_encoding(changed, expected))
    reject('shortened_row_collection_rejected', lambda: contract.verify_encoded([toy],[expected],summary))

    original = (e/'probe_longest.py').read_text()
    candidate = (here/'probe_fullcontext_longest.py').read_text()
    assert hashlib.sha256(original.encode()).hexdigest() == contract.ORIGINAL_PROBE_SHA
    segment = lambda source: source[source.index("  stage='native_load'"):source.index('\n except Exception as error:')]
    original_segment, new_segment = segment(original), segment(candidate)
    assert original_segment == new_segment
    checks.append('complete_native_load_update_cache_terminal_body_byte_identical')
    assert training.CONFIG['max_seq_length'] == 8192 and contract.PROBE_SEQUENCE_CAP == 14000
    assert max(audited,key=lambda row:row['full_tokens'])['row_id'] == contract.LONGEST_ROW_ID
    assert max(row['full_tokens'] for row in audited) == 12824
    checks.append('audited_longest12824_new14000_old8192_separate')
    for path in here.glob('*.py'):
        ast.parse(path.read_text(), filename=str(path))
    assert 'PeftModel.from_pretrained' not in candidate and 'load_adapter' not in candidate
    assert 'torch' not in sys.modules and 'transformers' not in sys.modules
    checks.append('no_adapter_resume_or_model_import_during_fixture')
    result = {'status':'CPU_source_and_input_contracts_passed', 'checks_passed':checks,
              'check_count':len(checks), 'actual_input_rows':len(rows), 'train_documents':len(train_ids),
              'excluded_input_documents':len(excluded), 'native_tokenizer_rerun':False,
              'asset_files_rehashed':False, 'GPU_or_MCP_or_quality':False,
              'source_sha256':contract.sha(__file__), 'probe_sha256':contract.sha(here/'probe_fullcontext_longest.py'),
              'contract_sha256':contract.sha(here/'fullcontext_contract.py'), 'CPU_reference_sha256':contract.CPU_SHA,
              'original_probe_sha256':contract.ORIGINAL_PROBE_SHA,
              'unchanged_numeric_body_bytes':len(new_segment.encode()),
              'unchanged_numeric_body_sha256':hashlib.sha256(new_segment.encode()).hexdigest()}
    with opts.output.open('x') as handle:
        handle.write(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'checks_passed':len(checks), 'receipt_sha256':contract.sha(opts.output)}))


if __name__ == '__main__':
    main()
