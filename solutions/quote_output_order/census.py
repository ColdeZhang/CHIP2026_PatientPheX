"""Finite train32-only output permutation census; no model, GPU or evaluator."""
import argparse
import collections
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import signal
import statistics
import sys
import time


def ref(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    return dict(path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def digest(value):
    return hashlib.sha256(json.dumps(value).encode()).hexdigest()


def save(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write('\n'); handle.flush(); os.fsync(handle.fileno())
    return ref(path)


def stats(values):
    return dict(total=sum(values), maximum=max(values), minimum=min(values),
                median=statistics.median(values))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True, type=Path)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text()); start = time.monotonic()
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('CPU120')))
    signal.alarm(plan['cpu_seconds'])
    assert ref(__file__) == plan['source']
    for item in plan['references'].values():
        assert ref(item['path']) == item
    output = Path(plan['output']); output.mkdir(parents=True, exist_ok=False)
    save(output/'intent.json', dict(plan=ref(args.plan), no_model_no_fit_no_held=True))
    os.environ['CUDA_VISIBLE_DEVICES'] = ''; os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    sys.path.insert(0, plan['lightweight_libraries'])
    source = plan['references']['native_tokenizer']['path']
    spec = importlib.util.spec_from_file_location('inherited_native', source)
    native = importlib.util.module_from_spec(spec); spec.loader.exec_module(native)
    tokenizer = native.FrozenTokenizer(plan['references']['tokenizer']['path'], plan['references']['template']['path'])
    rows = [json.loads(line) for line in Path(plan['references']['train']['path']).read_text().splitlines()]
    old_audit = json.loads(Path(plan['references']['old_train_audit']['path']).read_text())
    assert len(rows) == len(old_audit) == 87
    assert len({row['pmc_id'] for row in rows}) == 32
    original = []; variants = [[], []]; audit = []
    for row, old in zip(rows, old_audit):
        assert row['row_id'] == old['row_id']
        encoded = native.encode(row, tokenizer)
        assert digest(encoded['input_ids']) == old['input_ids_sha256']
        assert digest(encoded['labels']) == old['labels_sha256']
        target = json.loads(row['answer'])
        keys = [json.dumps(item, ensure_ascii=False, separators=(',', ':')) for item in target]
        assert len(keys) == len(set(keys))
        n = len(keys)
        original.append(encoded)
        item = dict(row_id=row['row_id'], quotes=n, log10_possible_orders=math.lgamma(n+1)/math.log(10), views=[])
        for epoch in [0, 1]:
            seed = int(hashlib.sha256(f"{plan['seed']}|{epoch}|{row['row_id']}".encode()).hexdigest(), 16)
            order = list(range(n)); random.Random(seed).shuffle(order)
            new = dict(row, answer=json.dumps([target[index] for index in order], ensure_ascii=False, separators=(',', ':')))
            assert new['messages'] == row['messages'] and new['units'] == row['units']
            assert collections.Counter(json.dumps(t,ensure_ascii=False) for t in json.loads(new['answer'])) == collections.Counter(json.dumps(t,ensure_ascii=False) for t in target)
            value = native.encode(new, tokenizer); prefix = value['prefix_tokens']
            assert value['input_ids'][:prefix] == encoded['input_ids'][:prefix]
            assert value['labels'][:prefix] == [-100]*prefix
            assert value['labels'][prefix:] == value['input_ids'][prefix:]
            assert value['labels'].count(value['eos_token_id']) == 1
            variants[epoch].append(value)
            item['views'].append(dict(epoch=epoch+1, permutation=order, changed=new['answer'] != row['answer'],
                fixed_positions=sum(i == j for i,j in enumerate(order)),
                inversions=sum(order[a] > order[b] for a in range(n) for b in range(a+1,n)),
                full_tokens=len(value['input_ids']), supervised_tokens=value['supervised_tokens'],
                input_ids_sha256=digest(value['input_ids']), labels_sha256=digest(value['labels']),
                prefix_sha256=value['rendered_prefix_sha256']))
        audit.append(item)
    cardinalities = [item['quotes'] for item in audit]
    def tokenstats(values):
        return dict(full=stats([len(v['input_ids']) for v in values]),
                    supervision=stats([v['supervised_tokens'] for v in values]))
    baseline = tokenstats(original)
    report = dict(status='TRAIN32_CPU_READINESS_ONLY_NOT_TRAINING', plan=ref(args.plan), source=ref(__file__),
        references=plan['references'], rows=87, documents=32, quotes=sum(cardinalities),
        empty=sum(n == 0 for n in cardinalities), singleton=sum(n == 1 for n in cardinalities),
        multi_quote_rows=sum(n >= 2 for n in cardinalities), cardinality=stats(cardinalities),
        cardinality_histogram=dict(sorted(collections.Counter(cardinalities).items())),
        original_actual_ids_and_labels_exact=True, original=baseline,
        permutations=[dict(epoch=e+1, tokens=tokenstats(v), changed_rows=sum(a['views'][e]['changed'] for a in audit),
                           changed_multi_quote_rows=sum(a['quotes']>=2 and a['views'][e]['changed'] for a in audit),
                           fixed_item_positions=sum(a['views'][e]['fixed_positions'] for a in audit)) for e,v in enumerate(variants)],
        same_inputs_units_patients_triple_multisets=True, all_labels_prefix_masked_single_EOS=True,
        native_runtime_replay_required_if_activated=True, proposed_exposures=174, proposed_optimizer_calls=22,
        proposed_tail_per_epoch=7, no_held_input_no_gold_no_fit_no_model_no_quality=True,
        audit=save(output/'patient-order-audit.json',audit), seconds=time.monotonic()-start)
    print(json.dumps(save(output/'result.json',report)))


if __name__ == '__main__':
    main()
