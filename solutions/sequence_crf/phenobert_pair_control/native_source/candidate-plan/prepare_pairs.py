"""Label-free current top3-pool census using the original pure tokenizer only."""
import ast
import collections
import gzip
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import string
import time
import types
import unicodedata

B = Path(__file__).resolve().parent
R = B.parent.parent
S = R / 'research/phenobert-readiness'
I = R / 'research/public-phenotype-model-inventory'


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


def ref(path):
    p = Path(path)
    return {'path': str(p), 'sha256': sha(p), 'bytes': p.stat().st_size}


def write_gzip(path, rows):
    with open(path, 'wb') as out:
        with gzip.GzipFile(fileobj=out, mode='wb', mtime=0, filename='') as f:
            for row in rows:
                f.write((compact(row) + '\n').encode())


def distribution(values):
    x = sorted(values)
    return {'count': len(x), 'min': x[0], 'mean': sum(x) / len(x),
            **{f'p{q}': x[max(0, math.ceil(len(x) * q / 100) - 1)]
               for q in (50, 90, 95, 99)}, 'max': x[-1]}


def ontology(path):
    terms = {}
    item = None
    for line in Path(path).read_text().splitlines() + ['[END]']:
        if line.startswith('['):
            if item and 'id' in item:
                terms[item['id']] = item
            item = {'parents': []} if line == '[Term]' else None
        elif item is not None:
            if line.startswith('id: '): item['id'] = line[4:]
            elif line.startswith('name: '): item['name'] = line[6:]
            elif line.startswith('is_a: '): item['parents'].append(line[6:].split()[0])
            elif line == 'is_obsolete: true': item['obsolete'] = True
    children = collections.defaultdict(set)
    for hid, term in terms.items():
        for parent in term['parents']: children[parent].add(hid)
    pending = ['HP:0000118']
    branch = set()
    while pending:
        current = pending.pop()
        if current in branch: continue
        branch.add(current)
        pending.extend(children[current])
    names = {hid: terms[hid]['name'] for hid in branch
             if not terms[hid].get('obsolete') and 'name' in terms[hid]}
    return names


def pure_tokenizer():
    source = (S / 'phenobert__utils__fastNLP__modules__encoder__bert.py').read_text()
    wanted = {'WordpieceTokenizer', 'BasicTokenizer', 'BertTokenizer', 'whitespace_tokenize',
              'load_vocab', '_is_whitespace', '_is_control', '_is_punctuation'}
    nodes = [n for n in ast.parse(source).body
             if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in wanted]
    assert len(nodes) == len(wanted)
    ns = {'os': os, 'collections': collections, 'unicodedata': unicodedata,
          'logger': logging.getLogger('original-pure-tokenizer')}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'fixed_original_tokenizer', 'exec'), ns)
    tokenizer = ns['BertTokenizer'](str(S / 'biobert-vocab.txt'))
    assert len(tokenizer.vocab) == 28996
    source = (I / 'phenobert-pair-train.txt').read_text()
    wanted = {'addWordPiece', 'addWords', 'processNum', 'processItem', 'addSeqlen'}
    nodes = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    assert len(nodes) == len(wanted)
    funcs = {'punctuation': string.punctuation,
             'embed': types.SimpleNamespace(model=types.SimpleNamespace(tokenzier=tokenizer))}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'fixed_original_pair_transforms', 'exec'), funcs)
    return tokenizer, funcs


def main():
    started = time.monotonic()
    plan_path = B / 'plan.json'
    plan = json.loads(plan_path.read_text())
    for record in plan['sources'].values():
        assert sha(record['path']) == record['sha256'], record['path']
    manifest = json.loads(Path(plan['sources']['pools']['path']).read_text())
    names = ontology(plan['sources']['ontology']['path'])
    assert len(names) == 19120
    inputs = {}
    with open(plan['sources']['train64_input']['path']) as f:
        for line in f:
            d = json.loads(line)
            assert set(d) == {'pmc_id', 'pmid', 'patient', 'full_text'}
            inputs[str(d['pmc_id'])] = d
    assert len(inputs) == 64
    pair_counts = collections.Counter()
    excluded = collections.Counter()
    doc_ids = set()
    rows = compounds = atomic_rows = 0
    unique_candidates = set()
    pool_files = []
    for fold in manifest['folds']:
        for split in ('train', 'input'):
            record = fold[split]
            assert sha(record['path']) == record['sha256']
            pool_files.append(record)
            with gzip.open(record['path'], 'rt') as f:
                for line in f:
                    d = json.loads(line)
                    pmc = str(d['pmc_id'])
                    doc_ids.add(pmc)
                    # Only the input and candidate fields are used, never d['base'].
                    assert d['input'] == inputs[pmc]
                    passages = inputs[pmc]['full_text']
                    for c in d['candidates']:
                        rows += 1
                        offset, length, surface = int(c['offset']), int(c['length']), c['text']
                        matching = [p for p in passages if int(p['offset']) <= offset
                                    and offset + length <= int(p['offset']) + len(p['text'])]
                        assert len(matching) == 1
                        p = matching[0]
                        assert p['text'][offset-int(p['offset']):offset-int(p['offset'])+length] == surface
                        atoms = list(dict.fromkeys(c['identifier'].split(';')))
                        compounds += len(atoms) > 1
                        unique_candidates.add((pmc, offset, length, c['identifier']))
                        for hid in atoms:
                            atomic_rows += 1
                            if hid not in names:
                                excluded[hid] += 1
                                continue
                            pair_counts[(surface, hid)] += 1
    assert rows == manifest['total']['new_rows'] == 747178
    assert doc_ids == set(inputs)
    tokenizer, funcs = pure_tokenizer()
    representations = {}
    atom_rows = []
    invalid_separator = []
    for surface, hid in sorted(pair_counts):
        name = names[hid]
        if '::' in surface or '::' in name:
            invalid_separator.append({'text': surface, 'hpo_id': hid, 'name': name})
            continue
        x = {'raw_words': surface + '::' + name}
        x['p_words'] = funcs['addWords'](x)
        x['t_words'] = funcs['addWordPiece'](x)
        wp = funcs['processItem'](x)
        wn = funcs['processNum'](x)
        seq = funcs['addSeqlen'](x)
        assert len(wn) == seq and sum(wn) == len(wp) and tokenizer.vocab['[SEP]'] in wp
        representation = {'word_pieces': wp, 'word_nums': wn, 'seq_len': seq}
        key = hashlib.sha256(compact(representation).encode()).hexdigest()
        if key in representations:
            assert representations[key]['input'] == representation
        else:
            representations[key] = {'pair_key': key, 'input': representation,
                                    'example': {'text': surface, 'hpo_id': hid, 'name': name}}
        atom_rows.append({'text': surface, 'hpo_id': hid, 'name': name,
                         'pair_key': key, 'pool_occurrences': pair_counts[(surface, hid)]})
    assert not invalid_separator, invalid_separator[:5]
    model_rows = [representations[k] for k in sorted(representations)]
    lengths = [len(x['input']['word_pieces']) + 2 for x in model_rows]
    assert max(lengths) <= 512, 'Original auto_truncate=False: stop rather than truncate.'
    write_gzip(B / 'atomic-request.jsonl.gz', atom_rows)
    write_gzip(B / 'model-inputs.jsonl.gz', model_rows)
    # Deterministic evenly spaced length ranks, including the actual shortest and longest.
    sorted_models = sorted(model_rows, key=lambda x: (len(x['input']['word_pieces']), x['pair_key']))
    n = min(512, len(sorted_models))
    pilot = [sorted_models[(i * (len(sorted_models)-1)) // (n-1)] for i in range(n)]
    assert len({x['pair_key'] for x in pilot}) == n
    write_gzip(B / 'pilot-inputs.jsonl.gz', pilot)
    result = {
        'status': 'PASS_LABEL_FREE_POOL_AND_EXACT_NATIVE_TOKENIZER_CENSUS_ONLY',
        'plan': ref(plan_path), 'source': ref(__file__), 'pool_files': pool_files,
        'input_documents': len(inputs), 'original_surface_slice_mismatches': 0,
        'candidate_rows_repeated_across_nested_and_held': rows,
        'unique_document_span_identifier_candidates': len(unique_candidates),
        'compound_candidate_rows': compounds, 'atomic_candidate_rows': atomic_rows,
        'missing_or_invalid_atomic_ids': dict(excluded),
        'unique_raw_surface_atomic_id_pairs': len(pair_counts),
        'unique_original_native_input_tuples': len(model_rows),
        'unique_current_atomic_ids': len({x['hpo_id'] for x in atom_rows}),
        'unique_original_surfaces': len({x['text'] for x in atom_rows}),
        'native_wordpiece_length_including_cls_and_final_sep': distribution(lengths),
        'native_word_count': distribution([x['input']['seq_len'] for x in model_rows]),
        'raw_surface_character_count': distribution([len(x['text']) for x in atom_rows]),
        'preferred_name_character_count': distribution([len(x['name']) for x in atom_rows]),
        'length_greater_than_512': sum(n > 512 for n in lengths),
        'reserved_separator_conflicts': len(invalid_separator),
        'normalization_or_wordpiece_deduplicated_pairs': len(atom_rows)-len(model_rows),
        'artifacts': {key: ref(B / file) for key, file in [
            ('atomic_request','atomic-request.jsonl.gz'), ('model_inputs','model-inputs.jsonl.gz'),
            ('pilot_inputs','pilot-inputs.jsonl.gz')]},
        'pilot_sampling': {'rows': n, 'rule': 'evenly spaced sorted native token length ranks, including min and max; ties pair_key',
                           'no_labels_or_quality_selection': True},
        'no_checkpoint_loaded': True, 'no_neural_forward': True, 'no_gold_used': True,
        'no_dev_input': True, 'no_new_candidates_or_labels': True,
        'seconds': time.monotonic()-started}
    (B / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in {'pool_files','artifacts'}}, ensure_ascii=False))


if __name__ == '__main__':
    main()
