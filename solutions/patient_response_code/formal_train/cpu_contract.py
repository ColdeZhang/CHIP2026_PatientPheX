"""New binding and unchanged training/resume source checks; stdlib only.

Uses safe native receipts and byte hashes; no training answer parsing, tokenizer,
Torch, model, network, inference, or scoring. Prior numeric/resume tests are reused.
"""
from pathlib import Path
import argparse, ast, copy, hashlib, importlib.util, json, random, sys, time
sys.dont_write_bytecode = True
B = Path(__file__).resolve().parent


def ref(path):
    p = Path(path)
    return {'path': str(p), 'bytes': p.stat().st_size,
            'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}


def load_contract(path):
    spec = importlib.util.spec_from_file_location('response_runtime_contract', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def function(tree, name):
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True, type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    plan = json.loads(args.plan.read_text())
    checks = []

    def okay(name, value=True):
        if not value:
            raise AssertionError(name)
        checks.append({'name': name, 'pass': True})

    def reject(name, callback):
        try:
            callback()
        except (ValueError, KeyError, AssertionError, TimeoutError):
            okay(name)
            return
        raise AssertionError('Invalid fixture accepted: ' + name)

    for record in list(plan['source_files'].values()) + list(plan['data_arguments'].values()) + list(plan['reused_evidence'].values()):
        okay('exact source/data/evidence bytes: ' + record['path'], ref(record['path']) == record)
    old = ast.parse(Path(plan['adapted_trainer']['path']).read_text())
    current_text = (B / 'train.py').read_text()
    current = ast.parse(current_text)
    dump = lambda x: ast.dump(x, include_attributes=False)
    for name in ['semantic_configuration', 'model_files', 'validate_checkpoint', 'batch_plan',
                 'verify_fingerprint', 'validate_stop_epoch', 'epoch_outcome', 'collate']:
        okay('unchanged existing function: ' + name, dump(function(old, name)) == dump(function(current, name)))
    old_run, new_run = function(old, 'run'), function(current, 'run')
    epoch_loop = lambda root: next(n for n in ast.walk(root) if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == 'epoch')
    okay('entire existing indexed epoch loop AST unchanged', dump(epoch_loop(old_run)) == dump(epoch_loop(new_run)))
    restore = lambda root: next(n for n in ast.walk(root) if isinstance(n, ast.If) and ast.unparse(n.test) == 'checkpoint' and any(isinstance(t, ast.Call) and isinstance(t.func, ast.Attribute) and t.func.attr == 'load_state_dict' for t in ast.walk(n)))
    okay('complete existing Adam/scheduler/RNG restore AST unchanged', dump(restore(old_run)) == dump(restore(new_run)))
    old_runtime = Path(plan['adapted_runtime']['path']).read_text().replace(
        '/ontology-round2/qwen35-fulltext32-indexed-formal-v1/checkpoints/epoch-001',
        '/patient-response-code/normal2ep-v1/checkpoints/epoch-001')
    okay('runtime source identical except ACK persistent namespace', old_runtime == (B / 'runtime_contract.py').read_text())
    namespace = {'random': random}
    for name in ['batch_plan', 'native_training_rows', 'validate_response_metadata']:
        exec(compile(ast.Module(body=[function(current, name)], type_ignores=[]), str(B / 'train.py'), 'exec'), namespace)
    constants = {}
    for n in current.body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) and n.targets[0].id.endswith('_SHA'):
            constants[n.targets[0].id] = ast.literal_eval(n.value)
    namespace.update(constants)
    for epoch in [0, 1]:
        groups = namespace['batch_plan'](124, 1, 8, 20261023, epoch)
        okay('epoch%d retains 15 groups of8 plus tail4 and every row once' % epoch,
             [len(g) for g in groups] == [8] * 15 + [4] and
             sorted(i for g in groups for batch in g for i in batch) == list(range(124)))
    response = json.loads(Path(plan['data_arguments']['response_preparation']['path']).read_text())
    native = json.loads(Path(plan['data_arguments']['native_receipt']['path']).read_text())
    audit = json.loads(Path(plan['data_arguments']['native_row_audit']['path']).read_text())
    original = json.loads(Path(plan['data_arguments']['data_manifest']['path']).read_text())
    train_ids, excluded_ids = original['training_document_ids'], original['excluded_document_ids']
    validate = namespace['validate_response_metadata']
    okay('actual response preparation/native identity and all32 excluded source contract',
         validate(response, native, audit, train_ids, excluded_ids) and len(train_ids) == len(excluded_ids) == 32)
    rows = namespace['native_training_rows'](audit)
    okay('new8138 target tokens disclosed, original7170 not asserted unchanged',
         sum(r['supervised_tokens'] for r in rows) == 8138 and plan['original_supervised_tokens'] == 7170 and plan['CE_target_token_representation_changed'])
    for field, value in [('rows', 123), ('total_tokens', 1009181), ('max_tokens', 13549),
                         ('supervised_tokens', 7170), ('encoded_sha256', '0' * 64), ('proposed_512_rejection_cap', 13824)]:
        bad = copy.deepcopy(native)
        bad['groups']['train']['arms']['response_code'][field] = value
        reject('reject stale or changed native field ' + field,
               lambda bad=bad: validate(response, bad, audit, train_ids, excluded_ids))
    for field, value in [('all_original_user_UTF8_preserved', False), ('all_original_PID_and_CID_metadata_exact', False)]:
        bad = copy.deepcopy(response)
        bad['groups']['train'][field] = value
        reject('reject preparation failure ' + field,
               lambda bad=bad: validate(bad, native, audit, train_ids, excluded_ids))
    for name, change in [
        ('missing row', lambda r: r.pop()),
        ('changed index', lambda r: r[0].update(index=1)),
        ('changed mask', lambda r: r[0].update(prefix_mask_exact=False)),
        ('changed document', lambda r: r[0].update(pmc_id='unseen-document')),
    ]:
        bad = copy.deepcopy(rows)
        change(bad)
        other = [r for r in audit if not (r['group'] == 'train' and r['arm'] == 'response_code')]
        reject('reject native audit ' + name,
               lambda bad=bad: validate(response, native, other + bad, train_ids, excluded_ids))
    reject('reject held document contaminating training', lambda: validate(response, native, audit, train_ids, train_ids))
    # Runtime will encode real rows and compare both exact IDs and labels against
    # the prior native audit. This CPU increment never reconstructs those arrays.
    for literal in ["helpers.json_sha(row['input_ids'])!=actual['input_ids_sha256']",
                    "helpers.json_sha(row['labels'])!=actual['labels_sha256']",
                    "helpers.json_sha(encoded)!=ENCODED_SHA",
                    "args.absolute_deadline_unix-time.time()>3600",
                    "torch.__version__!='2.6.0+cu124'",
                    "importlib.metadata.version('tokenizers')!='0.23.1'",
                    "ack_receipt=contract.validate_export_ack"]:
        okay('runtime fixed binding ' + literal, literal in current_text)
    okay('ACK gate before model load and logical restore before training',
         current_text.index('ack_receipt=contract.validate_export_ack') < current_text.index("status('load_complete_text_base')") and
         current_text.index('restored_digest=contract.tree_digest(actual_state,torch)') < current_text.index('model.train();first_update_checked=False'))
    contract = load_contract(B / 'runtime_contract.py')
    reject('expired shared deadline rejected', lambda: contract.check_deadline(time.time() - 1))
    okay('no Torch/tokenizer/models imported', all(x not in sys.modules for x in ['torch', 'transformers', 'tokenizers']))
    result = {'status': 'PASS_STDLIB_RESPONSE_BINDINGS_AND_UNCHANGED_INDEXED_TRAINING_RESTORE_SOURCE',
              'seconds': time.monotonic() - started, 'checks_count': len(checks), 'checks': checks,
              'source': ref(B / 'train.py'), 'runtime_contract': ref(B / 'runtime_contract.py'),
              'cpu_source': ref(__file__), 'plan_sha256': ref(args.plan)['sha256'],
              'raw_training_answers_parsed': False, 'native_census_repeated': False,
              'Torch_model_tokenizer_loaded': False, 'remote_GPU_training_or_quality': False,
              'scope': 'New source/data binding and runtime namespace guards only. Actual A100 stress/full training/state restore remain future evidence; old exact logical restore and head numerical proofs reused, no new bitwise-equivalence claim.'}
    with (B / 'CPU-contract-result.json').open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(json.dumps({'status': result['status'], 'checks': len(checks), 'seconds': result['seconds']}))


if __name__ == '__main__':
    main()
