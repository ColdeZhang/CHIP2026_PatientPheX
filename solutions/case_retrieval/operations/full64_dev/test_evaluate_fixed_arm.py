"""Synthetic CPU checks; imports are inert and no real gold/runner is accessed."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('fixed_arm_under_test', HERE / 'evaluate_fixed_arm.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def training_fixture():
    files = {name: dict(bytes=10, sha256=str(i) * 64) for i, name in enumerate(m.ADAPTER_NAMES, 1)}
    model = {'config.json': dict(bytes=1, sha256='9' * 64)}
    data = dict(training_rows=265, training_documents=64, training_sha256='3' * 64,
                fresh_base_required=True, old_adapter_continuation_forbidden=True,
                all_target_documents_excluded_from_own_generators=True,
                training_document_ids=[str(i) for i in range(64)], excluded_document_ids=[str(i) for i in range(64, 80)],
                excluded_dev_input_sha256='4' * 64)
    plan = dict(formal=dict(rows=265, documents=64, updates=68, fresh_base=True, continue_probe_or_48_adapter=False),
                source_sha256={'train_qwen35_full64.py': '5' * 64}, data_sha256={'train.jsonl': '3' * 64, 'preparation-manifest.json': '6' * 64},
                excluded_sha256='4' * 64, source_manifest_sha256='7' * 64)
    fixed = dict(rank=16, alpha=32, dropout=.05, epochs=2, max_seq_length=6400, batch_size=1, gradient_accumulation=8,
                 seed=20261023, learning_rate=1e-4, quantization=False, packing=False,
                 targets=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'in_proj_qkv', 'in_proj_z', 'in_proj_b', 'in_proj_a',
                          'out_proj', 'gate_proj', 'up_proj', 'down_proj'])
    fp = dict(schema='patientphex-qwen35-full64-relation-lora-v1', source_sha256='5' * 64, train_sha256='3' * 64,
              full64_preparation_sha256='6' * 64, training_document_ids=data['training_document_ids'],
              excluded_document_ids=data['excluded_document_ids'], excluded_input_sha256='4' * 64,
              base_files=dict(all15_files_verified=True, actual_files=model, source_manifest_sha256='7' * 64),
              configuration=dict(fixed=fixed))
    checkpoint = dict(smoke_only=False, completed_epochs=2, global_step=68, fingerprint=fp,
                      files={'adapter/' + n: files[n]['sha256'] for n in files})
    actual = {'epoch-002': dict(completed_epochs=2, global_step=68, files=copy.deepcopy(checkpoint['files']))}
    complete = dict(smoke_only=False, epochs=2, global_step=68, source_sha256='5' * 64,
                    base_parameters=m.BASE_PARAMETERS, trainable_adapter_parameters=m.ADAPTER_PARAMETERS,
                    total_parameters=m.BASE_PARAMETERS + m.ADAPTER_PARAMETERS)
    manifest = dict(fingerprint=copy.deepcopy(fp), examples=265, max_tokens=6153, total_tokens=864937, supervised_tokens=15410,
                    steps_per_epoch=34, epochs=2, total_optimizer_steps=68, warmup_steps=4,
                    smoke_only=False, fresh_base_on_initial_run=True, validation_labels_used=False)
    transfer = dict(status='complete_full_sha', source_epoch=2, source_training_global_step=68,
                    source_files=copy.deepcopy(files), destination_files=copy.deepcopy(files))
    return [complete, checkpoint, actual, manifest, data, plan, files, transfer, model]


class FixedArmTests(unittest.TestCase):
    def test_reference_requires_real_bytes_and_rejects_placeholder(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'evidence.json'
            path.write_text('{}\n')
            ref = m.file_ref(path)
            self.assertEqual(m.check_ref(ref), path)
            path.write_text('{"changed":true}\n')
            with self.assertRaisesRegex(ValueError, 'Evidence bytes changed'):
                m.check_ref(ref)
            for invalid in [None, dict(path=str(path), bytes=1, sha256=None), dict(path=str(path), bytes=1, sha256='pending')]:
                with self.assertRaises(ValueError):
                    m.check_ref(invalid)

    def test_full64_binding_accepts_complete_and_rejects_old48_or_partial(self):
        m.validate_training(*training_fixture())
        for position, key, value in [(0, 'global_step', 52), (1, 'completed_epochs', 1), (3, 'examples', 204), (4, 'training_documents', 48)]:
            fixture = training_fixture()
            fixture[position][key] = value
            with self.subTest(position=position, key=key), self.assertRaises(ValueError):
                m.validate_training(*fixture)

    def test_heldout_overlap_and_copied_adapter_mismatch_fail(self):
        fixture = training_fixture()
        fixture[4]['excluded_document_ids'][0] = fixture[4]['training_document_ids'][0]
        fixture[1]['fingerprint']['excluded_document_ids'] = fixture[4]['excluded_document_ids']
        fixture[3]['fingerprint'] = copy.deepcopy(fixture[1]['fingerprint'])
        with self.assertRaisesRegex(ValueError, 'Training/dev overlap'):
            m.validate_training(*fixture)
        fixture = training_fixture()
        fixture[7]['destination_files']['adapter_model.safetensors']['sha256'] = '8' * 64
        with self.assertRaisesRegex(ValueError, 'Source/destination'):
            m.validate_training(*fixture)
        fixture = training_fixture()
        fixture[2]['epoch-002']['files']['adapter/adapter_model.safetensors'] = '8' * 64
        with self.assertRaisesRegex(ValueError, 'Checkpoint/actual/transfer'):
            m.validate_training(*fixture)

    def test_unbound_adapter_does_not_pass(self):
        fixture = training_fixture()
        fixture[6]['adapter_model.safetensors']['sha256'] = None
        with self.assertRaisesRegex(ValueError, 'Unbound adapter'):
            m.validate_training(*fixture)

    def test_fingerprint_rejects_runtime_drift_and_old_adapter_reuse(self):
        files = {n: dict(bytes=2, sha256=str(i) * 64) for i, n in enumerate(m.ADAPTER_NAMES, 1)}
        old = dict(source_sha256='a' * 64, prompts_sha256='b' * 64, source_manifest_sha256='c' * 64,
                   model_files={'synthetic': 1}, runtime={'gpu': 'A30', 'torch': 'frozen'}, precision='bf16', batch_size=1, thinking=False,
                   adapter_files={'adapter_config.json': '1' * 64, 'adapter_model.safetensors': '3' * 64},
                   configuration=dict(model='same-model', max_input_tokens=14000, max_new_tokens=1600,
                                      source_manifest='source', prompts='prompts', output='old.jsonl', adapter='old-adapter'))
        fp = copy.deepcopy(old)
        fp['configuration'].update(output='new/full64-adapter.responses.jsonl', adapter='new-adapter')
        fp['adapter_files'] = {n: files[n]['sha256'] for n in files}
        meta = dict(fingerprint=fp, base_parameters=m.BASE_PARAMETERS, total_parameters=m.BASE_PARAMETERS + m.ADAPTER_PARAMETERS,
                    expected_rows=64, eos_token_ids=[248044, 248046], pad_token_id=248044,
                    load_information=dict(missing_keys=[], unexpected_keys=[], mismatched_keys=[], error_msgs=[]))
        m.validate_fingerprint(meta, copy.deepcopy(fp), copy.deepcopy(fp), old, files)
        changed = copy.deepcopy(meta)
        changed['fingerprint']['runtime']['torch'] = 'other'
        with self.assertRaisesRegex(ValueError, 'same-A30 comparison field'):
            m.validate_fingerprint(changed, changed['fingerprint'], changed['fingerprint'], old, files)
        files['adapter_model.safetensors']['sha256'] = '3' * 64
        fp['adapter_files']['adapter_model.safetensors'] = '3' * 64
        with self.assertRaisesRegex(ValueError, 'Old48 adapter reused'):
            m.validate_fingerprint(meta, fp, fp, old, files)

    def test_response_audit_keeps_real_empty_and_falls_back_unknown_identity(self):
        prompts = [dict(row_id='doc:a', pmc_id='doc', patient_ids=['OII:3'], code_map={'C0': 'HP:1', 'C1': 'HP:2'})]
        response = dict(row_id='doc:a', pmc_id='doc', input_tokens=12, output_tokens=9,
                        removed_demonstrations=0, hit_token_limit=False,
                        response=json.dumps({'C0': [], 'C1': ['OII:3: II:3'], 'extra': []}))
        result = m.audit_responses(prompts, [response], {'doc:a': 12}, json.loads)
        self.assertEqual(result['counts']['valid_candidate_decisions'], 1)
        self.assertEqual(result['rows'][0]['fallback_codes'], ['C1'])
        self.assertEqual(result['rows'][0]['extra_codes'], ['extra'])
        response['hit_token_limit'] = True
        result = m.audit_responses(prompts, [response], {'doc:a': 12}, lambda _: self.fail('Hit-limit response must not parse'))
        self.assertEqual(result['counts']['unresolved_candidate_decisions'], 2)

    def test_duplicate_and_wrong_token_rows_fail_before_parser(self):
        prompts = [dict(row_id='a', pmc_id='doc', patient_ids=['P'], code_map={'C': 'HP:1'}),
                   dict(row_id='b', pmc_id='doc', patient_ids=['P'], code_map={'C': 'HP:1'})]
        raw = [dict(row_id='a', pmc_id='doc', input_tokens=10), dict(row_id='a', pmc_id='doc', input_tokens=10)]
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            m.audit_responses(prompts, raw, {'a': 10, 'b': 11}, lambda _: self.fail('No parsing before identity checks'))
        raw[1]['row_id'] = 'b'
        with self.assertRaisesRegex(ValueError, 'tokenizer'):
            m.audit_responses(prompts, raw, {'a': 10, 'b': 11}, lambda _: self.fail('No parsing before token checks'))

    def test_entity_or_patient_mutation_is_rejected(self):
        inp = [dict(pmc_id='doc', pmid='x', patient=[dict(patient_id='P')])]
        base = [dict(pmc_id='doc', entities=[dict(start=0, end=2, hpo_id='HP:1')])]
        pred = [dict(pmc_id='doc', pmid='x', entities=copy.deepcopy(base[0]['entities']), association=[dict(patient_id='P', phenotype=[])])]
        m.validate_prediction(pred, inp, base)
        pred[0]['entities'][0]['end'] = 3
        with self.assertRaisesRegex(ValueError, 'entity'):
            m.validate_prediction(pred, inp, base)
        pred[0]['entities'] = copy.deepcopy(base[0]['entities'])
        pred[0]['association'].append(dict(patient_id='P', phenotype=[]))
        with self.assertRaisesRegex(ValueError, 'patient'):
            m.validate_prediction(pred, inp, base)

    def test_no_freeze_blocks_before_any_gold_or_command_access(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(m, 'sha', side_effect=AssertionError('Must not hash anything before freeze exists')), \
                 patch.object(m, 'run_command', side_effect=AssertionError('Must not start evaluation')):
                with self.assertRaisesRegex(ValueError, 'gold access forbidden'):
                    m.evaluate_arm(Path(directory))

    def test_existing_freeze_blocks_merger_and_exclusive_writer(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'full64-predictions-freeze.json'
            m.write_new(path, {'synthetic': True})
            with patch.object(m, 'preflight', side_effect=AssertionError('Existing freeze must short-circuit')):
                with self.assertRaisesRegex(ValueError, 'already exists'):
                    m.freeze_arm(Path(directory))
            with self.assertRaises(FileExistsError):
                m.write_new(path, {'synthetic': False})
            self.assertEqual(json.loads(path.read_text()), {'synthetic': True})


if __name__ == '__main__':
    unittest.main()
