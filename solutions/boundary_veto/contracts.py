"""Small discriminating synthetic contracts. No real outputs, model, gold, or evaluator."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import run


def fixture(text='{"BOUNDARY":true}', *, eos=True, capped=False):
    mention = dict(row_id='p:3:4:mention-canonical', pmc_id='p', quote='pain', offset=3, length=4,
                   candidates=[dict(code='C01', identifier='HP:0000001', name='pain', score=1)])
    generated = [1] * (128 if capped else 4)
    if eos:
        generated[-1] = 151645
    raw = dict(row_id=mention['row_id'], index=0, input_tokens=2, generated_ids=generated,
               raw_text_with_special=text + ('<|im_end|>' if eos else ''), seconds=.1,
               max_new_tokens=128, eos_token_id=151645)
    try:
        parsed = run.strict_parse(text, mention['candidates'], eos=eos, capped=capped)
        rejection = None
    except (ValueError, TypeError) as error:
        parsed, rejection = None, str(error)
    result = dict(raw, text=text, eos=eos, capped=capped, parsed=parsed, rejection=rejection)
    encoded = dict(mention, input_ids=[5, 6])
    return [[mention], [result], [raw], [copy.deepcopy(result)], [encoded]]


def expect_error(action):
    try:
        action()
    except ValueError:
        return
    raise AssertionError('Expected strict contract failure')


def main(output):
    start = time.monotonic(); passed = []
    complete = fixture(); keys = run.boundary_keys(*complete)
    assert keys == {('p', 3, 4, 'pain')}; passed.append('complete exact BOUNDARY accepted')
    for value in ('{"NO":true}', '{"NIL":true}', '{"selected":["C01"],"missing":false}',
                  '{"selected":[],"missing":true}', '{"BOUNDARY":true,"NO":true}',
                  '{"BOUNDARY":true,"BOUNDARY":true}', '{"BOUNDARY":false}'):
        assert run.boundary_keys(*fixture(value)) == set()
    passed.append('NO NIL HPO missing-HPO and malformed boundary never delete')
    for eos, capped in ((False, False), (True, True), (False, True)):
        assert run.boundary_keys(*fixture(eos=eos, capped=capped)) == set()
    passed.append('missing EOS or cap never delete')
    for label, edit in (
        ('swapped row', lambda x: x[1][0].update(row_id='different')),
        ('wrong index', lambda x: x[1][0].update(index=1)),
        ('missing raw member', lambda x: x[2].clear()),
        ('parsed summary mismatch', lambda x: x[3][0].update(text='{"NO":true}')),
        ('raw token mismatch', lambda x: x[2][0].update(generated_ids=[1])),
        ('encoded length mismatch', lambda x: x[4][0].update(input_ids=[1])),
    ):
        bad = copy.deepcopy(complete); edit(bad)
        expect_error(lambda: run.boundary_keys(*bad)); passed.append(label + ' fails whole contract')
    bad = copy.deepcopy(complete)
    for collection in bad:
        collection.append(copy.deepcopy(collection[0]))
    expect_error(lambda: run.boundary_keys(*bad)); passed.append('duplicate row/span fails; no first-match selection')
    for field, value in (('eos', False), ('parsed', {'status':'NO','selected':[],'ids':[]}),
                         ('rejection', 'invented')):
        bad = copy.deepcopy(complete); bad[1][0][field] = value; bad[3][0][field] = value
        expect_error(lambda: run.boundary_keys(*bad))
    passed.append('internally altered EOS mode or rejection fails against raw/strict parser')
    entities = [dict(text='pain',offset=3,length=4,identifier='HP:0000001;HP:0000002',note=None),
                dict(text='pain',offset=3,length=4,identifier='-1',note='NO'),
                dict(text='pain',offset=30,length=4,identifier='HP:0000001',note=None),
                dict(text='painful',offset=3,length=7,identifier='HP:0000003',note=None),
                dict(text='ain',offset=4,length=3,identifier='HP:0000004',note=None)]
    base = [dict(pmc_id='p',entities=entities,patient=[{'patient_id':'P1'}],
                 association=[{'patient_id':'P1','HPO':['HP:0000001']}], full_text=['untouched'], extra={'x':[1,2]}),
            dict(pmc_id='other',entities=[copy.deepcopy(entities[0])],patient=[],association=[])]
    original = copy.deepcopy(base); pred, removed = run.veto(base, keys)
    assert base == original and pred[0]['entities'] == entities[2:] and pred[1] == base[1]
    assert [r['entity_index'] for r in removed] == [0,1]
    assert {k:v for k,v in pred[0].items() if k!='entities'} == {k:v for k,v in base[0].items() if k!='entities'}
    passed.append('compound and every same-span entity removed uniformly; input objects unmodified')
    passed.append('nested crossing repeated-text and other-document entities preserved unless exact match')
    passed.append('relations patients document extras and surviving entity order preserved')
    original_bytes = ('  '+json.dumps(base[0],ensure_ascii=False)+' \r\n\t'+json.dumps(base[1])+'\r\n').encode()
    unchanged, removed = run.veto(base, set())
    assert run.payload(original_bytes, unchanged, removed) == original_bytes
    passed.append('zero deletion preserves whitespace CRLF and every original byte')
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp)/'synthetic-gold.txt'; target.write_text('synthetic sentinel')
        program = ('import run,sys\nfrom pathlib import Path\nrun.block_gold(sys.argv[1])\n'
                   'try: Path(sys.argv[1]).read_bytes()\nexcept PermissionError: sys.exit(0)\n'
                   'sys.exit(1)\n')
        subprocess.run([sys.executable,'-c',program,str(target)],cwd=run.HERE,check=True)
        once = Path(tmp)/'intent.json'; run.put(once, {'one':1})
        try:
            run.put(once, {'two':2})
        except FileExistsError:
            pass
        else:
            raise AssertionError('Exclusive output was overwritten')
    passed.append('pre-freeze gold-open audit guard blocks synthetic sentinel')
    passed.append('exclusive writes prevent repeated intent or overwrite')
    result = dict(status='PASS_DIFFERENTIATING_SYNTHETIC_BOUNDARY_CONTRACTS',
        source=run.source_refs(), contracts=passed, seconds=time.monotonic()-start,
        no_actual_model_outputs_gold_model_forward_fit_or_quality=True)
    run.put(output,result); print(json.dumps({'passed':len(passed),'output':output}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output',required=True)
    main(parser.parse_args().output)
