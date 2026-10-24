"""Synthetic CPU contracts only; deny gold reads and never invoke any stage."""
import ast
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
from common import G, S, read, ref, save, verify_sources


def denied_gold(event, args):
    if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
        name = os.fsdecode(args[0])
        if name.endswith('.gold.jsonl'):
            raise PermissionError('Synthetic contracts never open gold')


sys.addaudithook(denied_gold)
sys.path.insert(0, str(G.parents[1]))
sys.path.insert(0, str(G / 'quality'))
from admission import admit, remaining
from bind import validate_terminal
from whole_row import interpret
from postprocess import primary_keys
import support


def rejected(fn):
    try: fn()
    except (AssertionError, ValueError, PermissionError): return
    raise AssertionError('Expected fail-closed contract')


def main():
    verify_sources()
    evidence = S / 'evidence'; evidence.mkdir(exist_ok=True)
    doc = dict(full_text=[dict(offset=10, text='café café')])
    good = interpret('[["U0000","café",1]]', doc, True, False)
    assert good['status'] == 'complete' and good['anchors'][0]['offset'] == 15
    from solutions.grounded_generation.inference.decode import interpret as parent_interpret
    assert interpret is parent_interpret
    for raw in ['[["U0000","café",1],["U0000","cafe",0]]',
                '[["U0000","café",1],["U0000","café",1]]']:
        got = interpret(raw, doc, True, False)
        assert got==parent_interpret(raw,doc,True,False)
        assert got['status']=='complete' and len(got['anchors'])==1 and len(got['rejected'])==1
    invalid=interpret('[["U0000","café",true]]',doc,True,False)
    assert invalid['status']=='complete' and not invalid['anchors'] and len(invalid['rejected'])==1
    for raw in ['[NaN]', '[] trailing', '```json\n[]\n```']:
        got = interpret(raw, doc, True, False)
        assert got['status']=='abstain' and not got['anchors']
    for eos, cap in [(False, False), (False, True), (True, True)]:
        assert not interpret('[["U0000","café",1]]', doc, eos, cap)['anchors']
    assert interpret('[]', doc, True, False)['status'] == 'complete'
    p = dict(budget_seconds=1800, authorized_budget_seconds=1800, grace_seconds=30,
        admission=dict(schema='training-seed-F1-admission-v1', bound_at_unix=1000,
            launch_not_after_unix=1300, deadline_TERM_unix=2800, deadline_KILL_unix=2830))
    admit(p, 'synthetic', now=1299)
    rejected(lambda: admit(p, 'expired_launch_but_runtime_remains', now=1301))
    assert remaining(p, now=2700) == 100
    bad = copy.deepcopy(p); bad['admission']['deadline_TERM_unix'] += 1
    rejected(lambda: admit(bad, 'budget_extension', now=1200))
    plan = dict(trainer_sha256='synthetic-seed-source')
    with tempfile.TemporaryDirectory(prefix='synthetic-', dir=evidence) as name:
        root = Path(name); terminal = root / 'terminal'; output = terminal / 'output'
        output.mkdir(parents=True)
        def write(path, value): path.write_text(json.dumps(value))
        wrapper = dict(returncode=0, timed_out=False,
            all_epochs_persistent_and_readback_verified=True, durability={})
        complete = dict(epochs=2, global_step=22, rows=87, source_sha256=plan['trainer_sha256'],
            probe_state_used=False, both_epochs_file_directory_fsync_and_reopened_SHA_verified=True)
        write(output / 'training_complete.json', complete)
        wrapper['training_complete_sha256'] = ref(output / 'training_complete.json')['sha256']
        for epoch in (1, 2):
            directory = output / ('epoch-%03d' % epoch); directory.mkdir()
            checkpoint = dict(completed_epochs=epoch, global_step=11 * epoch, smoke_only=False,
                fingerprint=dict(source_sha256=plan['trainer_sha256']),
                files={'adapter/adapter_config.json':'synthetic','adapter/adapter_model.safetensors':'synthetic'})
            write(directory / 'checkpoint.json', checkpoint)
            ack = dict(completed_epochs=epoch, global_step=11 * epoch,
                all_members_reopened_and_SHA_verified=True, reopened_SHA_all_exact=True,
                file_directory_fsync=True, checkpoint_metadata_sha256=ref(directory / 'checkpoint.json')['sha256'])
            write(output / ('durability-epoch-%03d.json' % epoch), ack)
            wrapper['durability'][str(epoch)] = dict(receipt=ack)
        write(root / 'collection.json', dict(all_actual_SHA_verified=True, files={}))
        write(terminal / 'wrapper-complete.json', wrapper)
        validate_terminal(root, plan)
        bad_wrapper = copy.deepcopy(wrapper); del bad_wrapper['durability']['1']
        write(terminal / 'wrapper-complete.json', bad_wrapper)
        rejected(lambda: validate_terminal(root, plan))
        write(terminal / 'wrapper-complete.json', wrapper)
        rejected(lambda: validate_terminal(root, dict(trainer_sha256='wrong-seed')))
        checkpoint['global_step'] = 21
        write(output / 'epoch-002/checkpoint.json', checkpoint)
        rejected(lambda: validate_terminal(root, plan))
    assert primary_keys({'base'}, {'new', 'matched'}, {'matched', 'old'}) == {'base', 'matched'}
    assert primary_keys({'base'}, set(), {'old'}) == {'base'}
    old = read(G / 'quality/plan.json')
    spans, mask = support.byte_helpers(old['references']['byte_helper'])
    raw = '{"pmc_id":"D", "entities":[],"association":[{"patient_id":"P", "phenotype":["HP:0001250;HP:0001251", "literal;text"], "extra":7}]}\n'
    prior = json.loads(raw)['association'][0]['phenotype']
    assert support.replace_phenotypes(raw, {'P': prior}, spans, mask) == raw
    merged = support.replace_phenotypes(raw, {'P': prior + ['HP:0001252']}, spans, mask)
    assert mask(merged) == mask(raw) and json.loads(merged)['association'][0]['phenotype'][:2] == prior
    # The model.generate call is byte-for-byte AST-equivalent to the ordinary G parent.
    def generation(path):
        tree = ast.parse(path.read_text())
        return [ast.dump(n, include_attributes=False) for n in ast.walk(tree)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'generate']
    assert generation(S / 'run.py') == generation(G / 'inference/run.py')
    tree = ast.parse((S / 'postprocess.py').read_text())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'evaluate_report']
    assert len(calls) == 1
    assert not any(name in sys.modules for name in ('torch', 'transformers', 'peft', 'tokenizers'))
    result = dict(status='PASS_SYNTHETIC_CPU_ONLY', source=ref(__file__), source_manifest=ref(S / 'source-manifest.json'),
        same_parent_interpret_object=True,mixed_valid_invalid_items_preserve_parent_anchors=True,whole_row_only_cap_nonEOS_invalid_JSON=True, unicode_exact_parser=True, empty_array_valid=True,
        separate_TTL_and_absolute_1800_budget=True, missing_epoch_wrong_source_partial_step_rejected=True,
        ordinary_G_generate_AST_identical=True, fixed_intersection_and_B4_preservation=True,
        nonphenotype_bytes_and_compound_prefixes_exact=True, zero_add_byteexact=True,
        evaluator_call_sites=1, model_imports=0, gold_opens=0, actual_inference=0, normalization_calls=0,
        new_v2=0, old_v2=0, actual_adapter_binding=None)
    save(evidence / 'cpu-contract.json', result)
    print(json.dumps(ref(evidence / 'cpu-contract.json')))


if __name__ == '__main__': main()
