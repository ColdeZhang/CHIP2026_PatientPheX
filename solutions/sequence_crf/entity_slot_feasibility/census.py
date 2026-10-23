#!/usr/bin/env python3
"""Bounded, label-independent entity-question inventory; never fit or score F1.

Uses tokenizers' actual fixed tokenizer.json. The plain system/user template
specialization is checked against all 124 existing AutoTokenizer slot records.
The only annotation read for targets is the explicitly supplied train32 file.
"""
import argparse
import ast
from collections import Counter, defaultdict
import copy
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
from tokenizers import Tokenizer, __version__ as tokenizers_version


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ref(path):
    path = Path(path)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))


def read(path):
    path = Path(path)
    content = gzip.decompress(path.read_bytes()) if path.suffix == '.gz' else path.read_bytes()
    return [json.loads(line) for line in content.splitlines() if line]


def write(path, value):
    path = Path(path)
    assert not path.exists(), path
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def gzwrite(path, rows):
    path = Path(path)
    assert not path.exists(), path
    path.write_bytes(gzip.compress((''.join(json.dumps(r, ensure_ascii=False) + '\n'
                                           for r in rows)).encode(), mtime=0))


def now():
    return datetime.now(timezone.utc).isoformat()


def ids(path):
    return {str(r['pmc_id']) for r in read(path)}


def render(messages):
    # Exact specialization of frozen native Jinja: no tools/images/history.
    assert [m['role'] for m in messages] == ['system', 'user']
    assert all(isinstance(m['content'], str) for m in messages)
    return ''.join('<|im_start|>' + m['role'] + '\n' + m['content'].strip()
                   + '<|im_end|>\n' for m in messages) + '<|im_start|>assistant\n<think>\n\n</think>\n\n'


def encode_slots(tokenizer, messages, questions):
    prefix = render(messages)
    text = prefix
    ends = []
    for question in questions:
        text += question
        ends.append(len(text))
    enc = tokenizer.encode(text, add_special_tokens=False)
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False).ids
    assert enc.ids[:len(prefix_ids)] == prefix_ids
    assert tokenizer.decode(enc.ids, skip_special_tokens=False) == text
    positions = []
    for end in ends:
        found = [i for i, (_, right) in enumerate(enc.offsets) if right == end]
        assert len(found) == 1
        i = found[0]
        assert i >= len(prefix_ids)
        assert i + 1 == len(enc.ids) or enc.offsets[i + 1][0] >= end
        positions.append(i)
    assert positions == sorted(set(positions)) and len(positions) == len(questions)
    return dict(input_ids=enc.ids, slot_positions=positions,
                prefix_tokens=len(prefix_ids),
                rendered_input_sha256=hashlib.sha256(text.encode()).hexdigest())


def ontology(path):
    terms, aliases = {}, {}
    current = None
    for line in Path(path).read_text().splitlines() + ['[Term]']:
        if line.startswith('['):
            if current and 'id' in current and not current.get('obsolete'):
                terms[current['id']] = current
                for alias in current['alt']:
                    aliases[alias] = current['id']
            current = {'alt': []} if line == '[Term]' else None
        elif current is not None:
            if line.startswith('id: '): current['id'] = line[4:]
            elif line.startswith('name: '): current['name'] = line[6:]
            elif line.startswith('def: '):
                current['definition'] = json.JSONDecoder().raw_decode(line[5:])[0]
            elif line.startswith('alt_id: '): current['alt'].append(line[8:])
            elif line == 'is_obsolete: true': current['obsolete'] = True
    return terms, aliases


def name_ids(identifier, terms, aliases):
    rows = []
    for h in sorted(set(identifier.split(';'))):
        if h == '-1':
            rows.append(dict(id=h, name='Unmapped phenotype', definition='A phenotype mention without a mapped HPO term.'))
        else:
            h = aliases.get(h, h)
            assert h in terms, h
            rows.append(dict(id=h, name=terms[h]['name'], definition=terms[h].get('definition', '')))
    return rows


SYSTEM = (
    'Assess each fixed candidate entity in the supplied biomedical passage independently. '
    'Class 1 means the exact candidate span is a phenotype mention and its proposed HPO '
    'meaning is correct in context; class 0 means wrong boundary, wrong meaning, non-phenotype, '
    'or explicitly negated/normal finding. General biomedical descriptions may contain valid '
    'phenotype mentions: no assignment to a particular patient is required. '
    'For a compound candidate all listed meanings must fit the span. Unmapped phenotype '
    'means a phenotype with no supplied HPO mapping. Do not invent spans or identifiers. '
    'Each question slot names one supplied candidate and contains no answer.'
)


def grouped(doc, candidates, terms, aliases):
    buckets = defaultdict(list)
    seen = set()
    for c in candidates:
        key = (c['offset'], c['length'], c['identifier'])
        assert key not in seen
        seen.add(key)
        locations = [(i, p) for i, p in enumerate(doc['full_text'])
                     if p['offset'] <= c['offset']
                     and c['offset'] + c['length'] <= p['offset'] + len(p['text'])]
        assert len(locations) == 1
        pi, p = locations[0]
        local = c['offset'] - p['offset']
        assert p['text'][local:local + c['length']] == c['text']
        buckets[(pi, local // 512)].append(c)
    records = []
    for (pi, bin_id), cs in sorted(buckets.items()):
        p = doc['full_text'][pi]
        cs.sort(key=lambda c: (c['offset'], c['length'], c['identifier']))
        for start in range(0, len(cs), 32):
            batch = cs[start:start + 32]
            left = max(0, min(c['offset'] for c in batch) - p['offset'] - 500)
            right = min(len(p['text']), max(c['offset'] + c['length'] for c in batch) - p['offset'] + 500)
            context = p['text'][left:right]
            origin = p['offset'] + left
            rows = []
            for index, c in enumerate(batch):
                local = c['offset'] - origin
                assert context[local:local + c['length']] == c['text']
                rows.append(dict(cid=f'C{index}', offset=c['offset'], length=c['length'],
                                 text=c['text'], meanings=name_ids(c['identifier'], terms, aliases)))
            user = dict(section=p.get('section_type'), passage_type=p.get('type'),
                        context_start=origin, context=context, candidates=rows)
            records.append(dict(row_id=f"{doc['pmc_id']}:{pi}:{bin_id}:{start // 32}",
                                pmc_id=str(doc['pmc_id']),
                                candidates=copy.deepcopy(batch),
                                messages=[dict(role='system', content=SYSTEM),
                                          dict(role='user', content=json.dumps(user, ensure_ascii=False))]))
    assert sum(len(r['candidates']) for r in records) == len(candidates)
    return records


def summary(records, encoded):
    lengths = sorted(len(r['input_ids']) for r in encoded)
    return dict(documents=len({r['pmc_id'] for r in records}), prompts=len(records),
                candidates=sum(len(r['candidates']) for r in records),
                tokens=sum(lengths), min_tokens=min(lengths), max_tokens=max(lengths),
                median_tokens=lengths[len(lengths)//2], p95_tokens=lengths[int(.95*(len(lengths)-1))],
                over6777=sum(n > 6777 for n in lengths), over14000=sum(n > 14000 for n in lengths),
                longest_row_id=encoded[max(range(len(encoded)), key=lambda i: len(encoded[i]['input_ids']))]['row_id'])


def main():
    ap = argparse.ArgumentParser()
    for key in ['ce-root', 'nested-manifest', 'tokenizer', 'slot-reference', 'hpo', 'output']:
        ap.add_argument('--' + key, type=Path, required=True)
    a = ap.parse_args()
    started = time.monotonic()
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('CPU census >120s')))
    signal.alarm(120)
    out = a.output
    plan = json.loads((out/'plan.json').read_text())
    assert plan['cpu_cap_seconds'] == 120 and not plan['training_authorized']
    dependencies = [ref(__file__), ref(out/'plan.json'), ref(a.hpo), ref(a.tokenizer/'tokenizer.json'),
                    ref(a.tokenizer/'chat_template.jinja'), ref(a.slot_reference), ref(a.nested_manifest)]
    tokenizer = Tokenizer.from_file(str(a.tokenizer/'tokenizer.json'))
    assert [tokenizer.encode(str(i), add_special_tokens=False).ids for i in [0, 1]] == [[15], [16]]
    reference = json.loads(gzip.decompress(a.slot_reference.read_bytes()))
    assert len(reference) == 124
    for row in reference:
        questions = [f'Question {cid} / {pid}:\n' for cid, pid in row['keys']]
        actual = encode_slots(tokenizer, row['messages'], questions)
        for key, value in actual.items(): assert value == row[key], (row['row_id'], key)
    # Extract the already-frozen pool function alone: no import of ML/evaluator.
    pool_path = out/'sources/calibrate.py'
    dependencies.append(ref(pool_path))
    tree = ast.parse(pool_path.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'pool')
    ns = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(pool_path), 'exec'), ns)
    pool = ns['pool']
    croot = a.ce_root
    train_path = croot/'fold1/train32.jsonl'
    train_ids, held_ids = ids(train_path), ids(croot/'fold1/input16.jsonl')
    assert len(train_ids) == 32 and len(held_ids) == 16 and not train_ids & held_ids
    specs = json.loads((croot/'fold1/training-origins.json').read_text())
    dependencies += [ref(train_path), ref(croot/'fold1/input16.jsonl'),
                     ref(croot/'fold1/training-origins.json')]
    terms, aliases = ontology(a.hpo)
    all_records = {'train': [], 'heldout': []}
    origins, candidate_inventory = [], {}
    seen_docs = set()
    for spec in specs:
        # Intentionally never open spec['gold']; inputs are annotation-free.
        paths = {k: croot/spec[k] for k in ['input', 'generator_train', 'base', 'candidates', 'candidate_manifest']}
        docs = read(paths['input'])
        ds, gs = {str(d['pmc_id']) for d in docs}, ids(paths['generator_train'])
        assert ds <= train_ids and gs <= train_ids and not ds & gs and not (ds | gs) & held_ids
        assert not seen_docs & ds
        seen_docs |= ds
        manifest = json.loads(paths['candidate_manifest'].read_text())
        assert manifest['inputs']['train'] == sha(paths['generator_train'])
        assert manifest['inputs']['input'] == sha(paths['input'])
        assert manifest['output']['sha256'] == sha(paths['candidates'])
        base = {str(d['pmc_id']): d for d in read(paths['base'])}
        raw = {str(d['pmc_id']): d['candidates'] for d in read(paths['candidates'])}
        assert set(base) == set(raw) == ds
        for d in docs:
            assert not {'entities', 'association'} & d.keys()
            cs = pool(d, base[str(d['pmc_id'])], raw[str(d['pmc_id'])])
            candidate_inventory[str(d['pmc_id'])] = len(cs)
            all_records['train'].extend(grouped(d, cs, terms, aliases))
        origins.append(dict(kind='train', generator_ids=sorted(gs), target_ids=sorted(ds),
                            source_paths={k: ref(p) for k, p in paths.items()},
                            future_F1_intersection=[]))
    assert seen_docs == train_ids
    spec = json.loads(a.nested_manifest.read_text())[0]
    paths = {k: Path(spec[k]) for k in ['input', 'generator_train', 'base', 'candidates', 'candidate_manifest']}
    assert ids(paths['input']) == held_ids and ids(paths['generator_train']) == train_ids
    m = json.loads(paths['candidate_manifest'].read_text())
    assert m['inputs']['train'] == sha(paths['generator_train'])
    assert m['inputs']['input'] == sha(paths['input'])
    assert m['output']['sha256'] == sha(paths['candidates'])
    base = {str(d['pmc_id']): d for d in read(paths['base'])}
    raw = {str(d['pmc_id']): d['candidates'] for d in read(paths['candidates'])}
    assert set(base) == set(raw) == held_ids
    for d in read(paths['input']):
        assert not {'entities', 'association'} & d.keys()
        cs = pool(d, base[str(d['pmc_id'])], raw[str(d['pmc_id'])])
        candidate_inventory[str(d['pmc_id'])] = len(cs)
        all_records['heldout'].extend(grouped(d, cs, terms, aliases))
    origins.append(dict(kind='heldout', generator_ids=sorted(train_ids), target_ids=sorted(held_ids),
                        source_paths={k: ref(p) for k,p in paths.items()}))
    encoded, stats = {}, {}
    for split, rows in all_records.items():
        encoded[split] = []
        for row in rows:
            record = dict(row_id=row['row_id'], pmc_id=row['pmc_id'],
                          keys=[[c['offset'], c['length'], c['identifier']] for c in row['candidates']],
                          **encode_slots(tokenizer, row['messages'],
                                         [f'Question C{i}:\n' for i in range(len(row['candidates']))]))
            encoded[split].append(record)
        stats[split] = summary(rows, encoded[split])
        gzwrite(out/f'{split}.prompts.jsonl.gz', rows)
        gzwrite(out/f'{split}.encoded.jsonl.gz', encoded[split])
    write(out/'input-only-freeze.json', dict(frozen_at_utc=now(), source=ref(__file__),
          origins=origins, train_ids=sorted(train_ids), heldout_ids=sorted(held_ids),
          policy=plan, candidate_counts_by_document=candidate_inventory, stats=stats,
          artifacts=[ref(out/f'{s}.{kind}.jsonl.gz') for s in all_records for kind in ['prompts', 'encoded']]))
    # Only now derive labels, always from train32, never F1 gold.
    train_docs = {str(d['pmc_id']): d for d in read(train_path)}
    labels, counts = [], Counter()
    per_doc = {}
    truths = {}
    for pmc, d in train_docs.items():
        truth = defaultdict(set)
        for entity in d['entities']:
            if str(entity.get('note', '')).upper() != 'NO':
                truth[(entity['offset'], entity['length'])].update(entity['identifier'].split(';'))
        truths[pmc] = truth
        per_doc[pmc] = Counter()
    for row, record in zip(all_records['train'], encoded['train']):
        targets = []
        for c in row['candidates']:
            gold = truths[row['pmc_id']].get((c['offset'], c['length']), set())
            target = int(bool(gold and (set(c['identifier'].split(';')) <= gold or '-1' in gold)))
            targets.append(target)
        labels.append(dict(row_id=row['row_id'], targets=targets))
        counts.update(positive=sum(targets), negative=len(targets)-sum(targets),
                      all_negative_rows=int(not any(targets)), mixed_rows=int(0 < sum(targets) < len(targets)))
        per_doc[row['pmc_id']].update(positive=sum(targets), negative=len(targets)-sum(targets))
        # Labels cannot enter slot input: encode takes only messages and questions.
        corrupted = dict(record, targets=[1-t for t in targets])
        assert {k: v for k, v in corrupted.items() if k != 'targets'} == record
    gzwrite(out/'train.targets.jsonl.gz', labels)
    report = dict(completed_at_utc=now(), seconds=time.monotonic()-started,
                  python=platform.python_version(), tokenizers=tokenizers_version,
                  train=stats['train'], heldout=stats['heldout'], train_labels=dict(counts),
                  train_labels_by_document=per_doc,
                  contracts=dict(actual_existing_AutoTokenizer_rows_exact=124,
                      actual_0_1_token_ids=[15, 16], all_candidates_used=True,
                      label_independent_inputs=True, heldout_gold_read=False,
                      originaldev_read=False, model_loaded=False, GPU_used=False,
                      inputs_frozen_before_target_derivation=True,
                      all_exact_span_and_slot_boundaries=True, no_input_truncation=True),
                  context_window='same passage, start bins512 chars, max32 candidates, ±500 chars around batch',
                  training_candidate_generator_size=16, heldout_candidate_generator_size=32,
                  limitation='same semantic+ontology source family, no CRF-only/public-source extras; candidate generator training size differs by nested design',
                  input_freeze=ref(out/'input-only-freeze.json'), dependencies=dependencies,
                  target_file=ref(out/'train.targets.jsonl.gz'))
    write(out/'census.json', report)
    signal.alarm(0)
    print(json.dumps({k: report[k] for k in ['seconds', 'train', 'heldout', 'train_labels', 'contracts']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
