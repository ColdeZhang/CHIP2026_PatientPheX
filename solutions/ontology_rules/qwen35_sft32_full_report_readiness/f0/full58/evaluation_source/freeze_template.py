#!/usr/bin/env python3
"""Freeze actual probe + formal58 source/plan into a new DISABLED evaluation template.

This binds metadata and known input identities only; it does not run inference or quality.
"""
import argparse
import copy
from datetime import datetime, timezone
import importlib.util
from pathlib import Path
import sys
sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location('sft48_F0_evaluation_prepare', Path(__file__).with_name('evaluate_full58.py'))
e = importlib.util.module_from_spec(spec); spec.loader.exec_module(e)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pending-template', required=True)
    parser.add_argument('--full-plan', required=True)
    parser.add_argument('--inference-source', required=True)
    parser.add_argument('--inference-launcher', required=True)
    parser.add_argument('--probe-root', required=True)
    parser.add_argument('--output-template', required=True)
    args = parser.parse_args()
    pending_ref = e.ref(args.pending_template); template = e.read(args.pending_template)
    e.require(template['execution_ready'] is False and template['template_stage'] == 'pending_actual_probe_and_formal58_plan' and
              template['schema'] == e.SCHEMA and template['policy'] == e.POLICY and
              template['control_sha256'] == e.sha(e.__file__) and template['helpers_sha256'] == e.HELPERS_SHA,
              'Incompatible pending template')
    e.require(set(template['references']) == set(e.SHARED_SHA) | e.DYNAMIC_COMMON and
              all(template['references'][key] is None for key in e.DYNAMIC_COMMON) and
              all(item is None for item in template['inference_references'].values()) and
              all(item is None for item in template['actual_references'].values()), 'Pending actual fields were filled early')
    for key, digest in e.SHARED_SHA.items():
        e.require(template['references'][key]['sha256'] == digest, 'Fixed F0 identity differs')
        e.checked(template['references'][key])
    e.checked(template['gold_metadata_source'])  # JSON metadata only, never the referenced gold file.
    probe_root = Path(args.probe_root).resolve()
    e.require(Path(template['references']['probe_plan']['path']).parent.resolve() == probe_root, 'Wrong probe root')
    common = template['references']
    for key, name in {'probe_raw':'raw-response.json','probe_ids':'generated-ids.json',
                     'probe_format':'format-result.json','probe_summary':'summary.json',
                     'probe_assets':'actual-input-and-assets.json'}.items():
        common[key] = e.ref(probe_root / 'terminal' / name)
    probe_plan = e.read(common['probe_plan']['path']); probe = e.read(common['probe_summary']['path'])
    form = e.read(common['probe_format']['path']); raw = e.read(common['probe_raw']['path']); ids = e.read(common['probe_ids']['path'])
    prompt = e.rows(common['prompts']['path'])[30]; cpu = e.read(common['input_audit']['path'])[30]
    e.require(probe['status'] == 'PASS_NATIVE_LONGEST_STRICT_COMPLETE' and probe['index'] == 30 and probe['input_tokens'] == 16245 and
              form['strict_complete'] is True and form['error'] is None, 'Actual longest probe did not pass')
    canonical = {'row_id':raw['row_id'],'pmc_id':raw['pmc_id'],'candidate_patients':form['parsed'],
                 'strict_complete':True,'abstention':None,'provenance':'cached_probe'}
    e.canonical_gate(prompt, raw, canonical)
    e.require(raw['input_tokens'] == 16245 and ids['row_id'] == prompt['row_id'] and
              ids['input_ids_sha256'] == cpu['expanded']['input_ids_sha256'] and
              len(ids['generated_ids']) == raw['output_tokens'] and ids['natural_eos'] is True and
              ids['hit_token_limit'] is False and ids['generated_ids'][-1] in ids['configured_eos_ids'], 'Actual probe IDs differ')
    template['inference_references'] = {'inference_plan':e.ref(args.full_plan),
        'inference_source':e.ref(args.inference_source),'inference_launcher':e.ref(args.inference_launcher)}
    fixed = e.read(args.full_plan)
    e.require(fixed['expected_rows'] == 58 and fixed['fresh_rows'] == 57 and fixed['cached_original_indices'] == [30] and
              fixed['cached_probe_index'] == 30 and fixed['expected_input_tokens'] == 505090 and fixed['expected_input_max'] == 16245 and
              fixed['cap_seconds'] == 1200 and fixed['term_grace_seconds'] == 30 and fixed['input_limit'] == 16384 and
              fixed['generation_limit'] == 1600 and fixed['adapter_files'] == e.ADAPTER_FILES, 'Formal58 contract differs')
    for key in ('model','adapter','adapter_files','generation','gpu_uuid','python'):
        e.require(fixed[key] == probe_plan[key], 'Formal58/probe config differs: ' + key)
    mapping = {'infer_full58.py':template['inference_references']['inference_source'],
        'launch.py':template['inference_references']['inference_launcher'],'expanded58.jsonl':common['expanded_prompts'],
        'row-audit.json':common['input_audit'],'cache-probe-raw.json':common['probe_raw'],
        'cache-probe-ids.json':common['probe_ids'],'cache-probe-format.json':common['probe_format'],
        'cache-probe-summary.json':common['probe_summary']}
    for name, record in mapping.items():
        e.require(fixed['files'][name]['bytes'] == record['bytes'] and fixed['files'][name]['sha256'] == record['sha256'],
                  'Formal58 deployment differs: ' + name)
    keys = {'torch','transformers','peft','tokenizers','actual_tokenizer_class','matmul_TF32','cudnn_TF32'}
    assets = e.read(common['probe_assets']['path'])
    e.require(set(fixed['cached_probe_runtime']) == keys and fixed['cached_probe_runtime'] == {k:assets[k] for k in keys},
              'Full58 runtime differs from actual probe')
    template['template_stage'] = 'actual_probe_and_formal58_source_frozen_before_full_quality'
    template['prequality_binding'] = {'pending_template':pending_ref,'builder':e.ref(__file__),
        'at_utc':datetime.now(timezone.utc).isoformat(),'gold_read':False,'quality_calls':0,
        'actual_full58_outputs_bound':False}
    e.reference_gate(template, require_ready=False)
    e.checked(pending_ref)
    e.save(Path(args.output_template), template)
    print(e.ref(args.output_template))


if __name__ == '__main__': main()
