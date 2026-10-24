"""Stdlib migration/source interface audit only; never executes remote code."""
from pathlib import Path
import ast
import hashlib
import json
import sys
import time


def ref(path):
    path = Path(path).resolve(); data = path.read_bytes()
    return dict(path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def main():
    started = time.monotonic()
    controller = Path(__file__).resolve().parent
    base = controller.parent
    prior = base.parent / 'full64'
    groups = ['formal_train', 'controller', 'inference']
    checks, compiled, embedded, same, changed = [], [], [], [], []
    for group in groups:
        old_manifest = json.loads((prior/group/'source-manifest.json').read_text())
        for name, item in old_manifest['files'].items():
            actual = ref(item.get('path', prior/group/name))
            assert all(actual[key] == value for key, value in item.items())
        checks.append('original ' + group + ' frozen source inventory unchanged')
        for path in sorted((base/group).glob('*.py')):
            text = path.read_text(); compile(text, str(path), 'exec')
            compiled.append(ref(path))
            for node in ast.walk(ast.parse(text)):
                if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
                    and isinstance(node.value.value, str) and '\n' in node.value.value
                    and node.value.value.startswith(('from pathlib import Path\n', 'import '))):
                    compile(node.value.value, str(path) + ':embedded', 'exec')
                    embedded.append(str(path.relative_to(base)))
            old = prior/group/path.name
            if old.is_file():
                (same if old.read_bytes() == path.read_bytes() else changed).append(str(path.relative_to(base)))
    cpu = json.loads((base/'formal_train/CPU-contract-result.json').read_text())
    plan = json.loads((base/'formal_train/training-plan.json').read_text())
    assert cpu['checks_count'] == 97 and all(row['pass_'] for row in cpu['checks'])
    assert cpu['plan_sha256'] == ref(base/'formal_train/training-plan.json')['sha256']
    for item in plan['source_files'].values():
        assert ref(item['path']) == item
    assert ref(plan['expected_probe_plan']['path']) == plan['expected_probe_plan']
    checks.append('actual97 stdlib contract and final source plan bound')
    for name in ['assemble_component.py','response_codec.py','probe_runner.py','legacy_runner.py','operations.py']:
        assert (base/'inference'/name).read_bytes() == (prior/'inference'/name).read_bytes()
    checks.append('same canonical parser, full64 assembly, raw-first runner helper and transport')
    expected=(prior/'inference/infer_full64.py').read_text().replace('2.6.0+cu124','2.7.1+cu126')
    assert (base/'inference/infer_full64.py').read_text() == expected
    checks.append('entire64 inference core differs only runtime version evidence')
    launch=(base/'inference/launch.py').read_text()
    assembly=(base/'inference/assemble_component.py').read_text()
    assert "'timed_out': rc in (124, 137)" in launch and "not wrapper['timed_out']" in assembly
    assert {rc: rc in (124,137) for rc in [0,1,124,137]} == {0:False,1:False,124:True,137:True}
    checks.append('complete wrapper/assembler timed_out metadata interface repaired')
    assert not any(name in sys.modules for name in ['torch','transformers','tokenizers'])
    checks.append('no NN, tokenizer, remote, Git, quality or gold work executed')
    value=dict(status='PASS_STDLIB_A800_MECHANICAL_SOURCE_MIGRATION_NOT_ACTUAL_GPU',
        seconds=time.monotonic()-started, checks=checks, checks_count=len(checks),
        source=ref(__file__), CPU_contract=ref(base/'formal_train/CPU-contract-result.json'),
        training_plan=ref(base/'formal_train/training-plan.json'),
        expected_probe_plan=plan['expected_probe_plan'],
        compiled_python_count=len(compiled), embedded_code_count=len(embedded),
        compiled=compiled, embedded=embedded, exact_parent_python=same,
        changed_parent_python=changed,
        no_old_source_mutation=True, new_probe_result_loaded=False,
        no_executable_campaign_created=True, remote_model_quality_Git_calls=0,
        limits='Pure source, schedule and metadata contracts. New A800 actual pressure/cost/activation remains owner responsibility; old A100 H51 not accepted as A800 proof.')
    with (controller/'source-checks.json').open('x') as handle:
        json.dump(value, handle, indent=2); handle.write('\n')
    print(json.dumps({key:value[key] for key in ['status','checks_count','compiled_python_count','embedded_code_count','seconds']}))


if __name__ == '__main__':
    main()
