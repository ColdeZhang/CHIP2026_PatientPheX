#!/usr/bin/env python3
"""Conservative Task2-only relations from public original article tables.

Policy defined on the first two TABLE-bearing train48 articles, before fold0
evaluation. Table text never becomes a Task1 entity or changes provided offsets.
"""
import argparse
from collections import defaultdict, Counter
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import run as base
import table_structure as structure

POSITIVE = {'+', 'yes', 'y', 'present', 'positive'}
NONPOSITIVE = {'', '-', 'no', 'n', 'absent', 'negative', 'normal', 'unremarkable',
               'n/a', 'na', 'nr', 'not available', 'not reported', 'unknown', '?'}
CLINICAL = re.compile(r'clinical|phenotyp|features|characteristics', re.I)
NEGATED = re.compile(r'\b(?:no|not|without|absent|normal|unremarkable)\b', re.I)


def abbreviations(table):
    result = {}
    for foot in table['footnotes']:
        text = re.sub(r'^Abbreviations:\s*', '', foot['text'], flags=re.I)
        for item in text.split(';'):
            match = re.fullmatch(r'\s*([A-Z][A-Z0-9/]{1,9}),\s*([^;]+?)\.?\s*', item)
            if match:
                result[match[1]] = match[2]
    return result


def aliases(train, hpo):
    terms, allowed, alternate, version = base.parse_hpo(hpo)
    bank = defaultdict(set)
    for hp in sorted(allowed):
        for name in terms[hp]['names']:
            if len(base.norm(name)) >= 4:
                bank[base.norm(name)].add((hp,))
    for doc in train:
        for entity in doc.get('entities', []):
            if str(entity.get('note', '')).upper() == 'NO':
                continue
            keys = tuple(sorted({alternate.get(x.strip(), x.strip())
                                 for x in str(entity.get('identifier', '')).split(';')
                                 if alternate.get(x.strip(), x.strip()) in allowed}))
            if keys and len(base.norm(entity.get('text', ''))) >= 4:
                bank[base.norm(entity['text'])].add(keys)
    return {name: next(iter(ids)) for name, ids in bank.items() if len(ids) == 1}


def map_phrase(text, bank, short):
    # Abbreviation expansions come solely from the same source table footnotes.
    text = re.sub(r'\b[A-Z][A-Z0-9/]{1,9}\b', lambda m: short.get(m.group(), m.group()), text)
    if NEGATED.search(text):
        return set()
    result = set()
    for phrase in re.split(r'\s*[,;]\s*|\s+and\s+', text):
        key = base.norm(phrase)
        # Exact names only: no numerical interpretation or fuzzy similarity.
        result.update(bank.get(key, ()))
    return result


def extract(doc, tables, bank):
    evidence, rejected = [], Counter()
    for table in tables:
        if not CLINICAL.search(table['caption']):
            rejected['nonclinical_caption'] += 1
            continue
        if table['geometry_issues']:
            rejected['geometry_warning'] += 1
            continue
        mapping = structure.match(table, doc['patient'], 'mentions_plus_official_o_id')
        cells = {c['id']: c for c in table['cells']}
        short = abbreviations(table)
        columns = [m for m in mapping['accepted'] if m['axis'] == 'column']
        patient_columns = {m['index'] for m in columns}
        for match in columns:
            pid = match['matches'][0]['patient_id']
            col = match['index']
            for row in table['grid_cell_ids']:
                cell = cells.get(row[col])
                if not cell or cell['row_group'] == 'thead' or cell['colspan'] != 1:
                    continue
                left = [cells[cid] for i, cid in enumerate(row[:col])
                        if cid and i not in patient_columns]
                # The first nonempty non-patient cell is the row label; no shifting
                # missing cells or borrowing the previous row is permitted.
                header = next((c for c in left if c['identity_text']), None)
                if header is None or header['id'] == cell['id']:
                    continue
                value = structure.norm(cell['identity_text'])
                if value in NONPOSITIVE:
                    continue
                text = header['identity_text'] if value in POSITIVE else cell['identity_text']
                for hp in sorted(map_phrase(text, bank, short)):
                    evidence.append({'pmc_id': str(doc['pmc_id']), 'patient_id': pid,
                                     'identifier': hp, 'table_id': table['table_wrap_id'],
                                     'header_cell': header['id'], 'value_cell': cell['id'],
                                     'row_label': header['identity_text'],
                                     'value': cell['identity_text'], 'mapped_text': text,
                                     'patient_match': match})
    return evidence, dict(rejected)


def main():
    p = argparse.ArgumentParser()
    for key in ['train', 'input', 'hpo', 'predictions', 'tables-dir', 'output-dir']:
        p.add_argument('--' + key, required=True)
    args = p.parse_args()
    train, docs = base.read_jsonl(args.train), base.read_jsonl(args.input)
    assert all('entities' not in d and 'association' not in d for d in docs)
    assert not {str(d['pmc_id']) for d in train} & {str(d['pmc_id']) for d in docs}
    bank = aliases(train, args.hpo)
    pred = {str(d['pmc_id']): d for d in base.read_jsonl(args.predictions)}
    result, evidence, source_files, summary = [], [], {}, Counter()
    for doc in docs:
        pmc = str(doc['pmc_id'])
        path = Path(args.tables_dir) / f'PMC{pmc}.tables.json'
        found = []
        if path.exists():
            record = json.loads(path.read_text())
            assert str(record['pmc_id']) == pmc
            found, rejected = extract(doc, record['tables'], bank)
            summary.update(rejected)
            source_files[str(path)] = base.sha(path)
            for item in found:
                item['source_sha256'] = record['source_sha256']
            evidence.extend(found)
        old = pred[pmc]
        vals = {a['patient_id']: set(a['phenotype']) for a in old['association']}
        before = sum(map(len, vals.values()))
        for item in found:
            vals[item['patient_id']].add(item['identifier'])
        summary['new_relations'] += sum(map(len, vals.values())) - before
        result.append(dict(old, association=[{'patient_id': x['patient_id'],
                       'phenotype': sorted(vals[x['patient_id']])} for x in doc['patient']]))
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, values in [('pred', result), ('evidence', evidence)]:
        (out / f'{name}.jsonl').write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in values))
    manifest = {'configuration': vars(args), 'policy': 'unique exact table patient identity; clinical caption; exact HPO/train aliases; explicit positives/text; additive Task2 only',
                'source_sha256': base.sha(__file__), 'parser_sha256': base.sha(structure.__file__),
                'base_sha256': base.sha(base.__file__), 'files': {k: base.sha(getattr(args, k)) for k in ['train', 'input', 'hpo', 'predictions']},
                'table_files': source_files, 'stats': dict(summary), 'evidence_items': len(evidence),
                'output_sha256': base.sha(out / 'pred.jsonl'), 'input_annotations_used': False,
                'task1_entities_unchanged': all(d['entities'] == pred[str(d['pmc_id'])]['entities'] for d in result)}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest))


if __name__ == '__main__':
    main()
