#!/usr/bin/env python3
"""Input-only exact old normal/reverse/public-new preparation; no NN or fit.

Requires a bound plan. Frozen source functions are extracted by AST, preserving
their code, without importing unused training/ML modules from the old scripts.
"""
from pathlib import Path
from collections import defaultdict
import argparse
import ast
import copy
import datetime
import hashlib
import json
import os
import re
import signal
import sys
import time

PINS = {
    'base_source': '41ee61bce9a8cfb5ef01d73c644172fdff0e29fd6aca2cf5aa08a9c880b5f5d0',
    'prompt_source': '16345e096b462ae9ba2fc974bb60d14ff15333b09009b4add1f0a0e1bb23ed97',
    'reverse_source': '990ccc23329e1848d68d86ac3e78cb5a5cd47c6ab76bc3fd93df472ccf5c96c4',
    'public_source': 'fbb3290008ee8fc85615139fbfa67b3e41531a1132027880433401d3ef3ba7e5',
    'hpo': 'a5092cbdf605f568403cf7380d9173014015692433b2cc631bc5c1b053876b1b',
    'input_manifest': '2c1690660c58337235e9e11d96633109ac3ac3ffd62fa5e0600980c4b8a2ba21',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ref(path):
    p = Path(path)
    return {'path': str(p.resolve()), 'bytes': p.stat().st_size, 'sha256': sha(p)}


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def save(path, value, jsonl=False):
    with Path(path).open('x', encoding='utf-8') as handle:
        if jsonl:
            for row in value:
                handle.write(json.dumps(row, ensure_ascii=False) + '\n')
        else:
            handle.write(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
        handle.flush()
        os.fsync(handle.fileno())


def frozen_functions(path, names, scope):
    """Execute only named unchanged function definitions from SHA-checked source."""
    source = Path(path).read_text()
    tree = ast.parse(source)
    selected = [node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in selected} == set(names)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), scope)


def frozen_literal(path, name):
    for node in ast.parse(Path(path).read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError('Missing frozen constant: ' + name)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    started = time.monotonic()
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('CPU preparation >60s')))
    signal.alarm(60)
    plan = read(args.plan)
    assert plan['status'] == 'BOUND_PUBLIC_B_INPUTS_NO_QUALITY'
    assert plan['source_sha256'] == sha(__file__)
    references = plan['references']
    required = set(PINS) | {'input', 'input_manifest', 'old_candidates', 'old_candidate_manifest',
                            'pubmed_manifest', 'pubmed_canonical',
                            'bioformer_manifest', 'bioformer_canonical'}
    assert set(references) == required
    for name, entry in references.items():
        assert entry is not None and ref(entry['path']) == entry, name
        if name in PINS:
            assert entry['sha256'] == PINS[name], name
    assert plan['documents'] == 100 and plan['patients'] == 244
    input_path = references['input']['path']
    input_contract = read(references['input_manifest']['path'])
    assert input_contract['status'] == 'frozen_official_B_input_only'
    assert input_contract['input'] == references['input'] and input_contract['hpo'] == references['hpo']
    assert input_contract['document_ids_unique_and_disjoint_from_train80'] is True
    docs = rows(input_path)
    by = {str(d['pmc_id']): d for d in docs}
    old_meta = read(references['old_candidate_manifest']['path'])
    assert old_meta['input_sha256'] == references['input']['sha256']
    assert old_meta['fit_called'] is False and old_meta['train_or_gold_read'] is False
    assert old_meta['policy']['relation_threshold'] == .55
    assert old_meta['stats']['documents'] == 100 and old_meta['stats']['patients'] == 244
    assert old_meta['outputs']['full-lr.jsonl']['sha256'] == references['old_candidates']['sha256']
    assert old_meta['outputs']['full-lr.jsonl']['bytes'] == references['old_candidates']['bytes']
    old_rows = rows(references['old_candidates']['path'])
    old = {str(d['pmc_id']): d for d in old_rows}
    assert len(docs) == len(by) == len(old_rows) == len(old) == plan['documents']
    assert set(by) == set(old)
    assert sum(len(d.get('patient', [])) for d in docs) == plan['patients']
    assert all('entities' not in d and 'association' not in d for d in docs)
    assert all(set(d) <= set(input_contract['allowed_fields']) for d in docs)
    assert sum(len(d['full_text']) for d in docs) == input_contract['passages']
    assert all(isinstance(d['entities'], list) and isinstance(d['association'], list) for d in old_rows)
    # No training functions/classes/imports are loaded from the frozen base source.
    scope = dict(Path=Path, re=re, json=json, defaultdict=defaultdict, copy=copy)
    frozen_functions(references['base_source']['path'], ['parse_hpo', 'association_keys'], scope)
    frozen_functions(references['prompt_source']['path'],
                     ['grouped_concepts', 'mark_patients', 'patient_anchors', 'evidence'], scope)
    scope['DIVIDER'] = frozen_literal(references['reverse_source']['path'], 'DIVIDER')
    frozen_functions(references['reverse_source']['path'], ['parts', 'replace_top_code', 'reverse'], scope)
    system = frozen_literal(references['prompt_source']['path'], 'SYSTEM')
    terms, allowed, alt, version = scope['parse_hpo'](references['hpo']['path'])
    oldkeys, spaces = set(), {}
    for did, d in old.items():
        atoms, raw = set(), set()
        text_concepts = set()
        for entity in d['entities']:
            identifier = entity['identifier']
            assert isinstance(identifier, str)
            raw.add(identifier)
            if identifier == '-1':
                text_concepts.add(entity['text'])
                oldkeys.add((did, 'text', entity['text']))
            else:
                parts = identifier.split(';')
                assert all(re.fullmatch(r'HP:\d{7}', h) for h in parts)
                for atom in parts:
                    atoms.add(atom)
                    oldkeys.add((did, 'hpo', alt.get(atom, atom)))
        spaces[did] = dict(old_raw_identifiers=sorted(raw), old_atomic_hpo=sorted(atoms),
                          old_hpo_canonical=sorted({alt.get(a, a) for a in atoms}),
                          old_text_concepts=sorted(text_concepts),
                          observed_old_alt_mapping={a: alt[a] for a in sorted(atoms) if a in alt})

    def make_prompts(entities_by_doc, public=False):
        result = []
        for doc in docs:
            did = str(doc['pmc_id'])
            concepts = scope['grouped_concepts'](entities_by_doc[did])
            ordered = sorted(concepts)
            anchors = scope['patient_anchors'](doc)
            for start in range(0, len(ordered), 8):
                selected = ordered[start:start + 8]
                codes = {f'C{i:03d}': concept for i, concept in enumerate(selected)}
                items = []
                for code, concept in codes.items():
                    # Same original normal fallback / public required HPO meaning.
                    meaning = terms[concept]['names'][0] if public else terms.get(concept, {}).get('names', [concept])[0]
                    items.append(dict(code=code, meaning=meaning,
                                      evidence=scope['evidence'](doc, concepts[concept], 500)))
                content = 'Patient identities and global anchors:\n' + anchors + scope['DIVIDER'] + json.dumps(items, ensure_ascii=False)
                prefix = 'public_new_union' if public else 'relations'
                result.append(dict(row_id=f'{did}:{prefix}:{start}', pmc_id=doc['pmc_id'], code_map=codes,
                                   patient_ids=[str(p['patient_id']) for p in doc.get('patient', [])],
                                   messages=[dict(role='system', content=system), dict(role='user', content=content)]))
        return result

    normal = make_prompts({did: d['entities'] for did, d in old.items()})
    reverse = []
    reverse_audit = []
    for row in normal:
        inverted, mapping = scope['reverse'](row)
        back, inverse = scope['reverse'](inverted)
        assert back == row and all(inverse[value] == key for key, value in mapping.items())
        reverse.append(inverted)
        reverse_audit.append(dict(row_id=row['row_id'], old_to_new=mapping, new_to_old=inverse,
                                  inverse_all_fields_and_user_byteexact=True))
    public, observed_alt = {}, {}
    for name in ('pubmed', 'bioformer'):
        meta = read(references[name + '_manifest']['path'])
        path = references[name + '_canonical']['path']
        assert meta['fixed_policy']['threshold_strictly_greater_than'] == .95
        assert meta['public_model_training_on_task'] is False
        assert meta['input_sha256'] == references['input']['sha256']
        assert meta['outputs']['canonical-entities.jsonl']['sha256'] == sha(path)
        selected = []
        mappings = set()
        for row in rows(path):
            did = str(row['pmc_id'])
            assert did in by
            passage = by[did]['full_text'][row['passage_index']]
            start = int(row['offset']) - int(passage['offset'])
            end = start + int(row['length'])
            assert int(passage['offset']) == row['passage_offset']
            assert start == row['local_start'] and end == row['local_end']
            assert 0 <= start < end <= len(passage['text'])
            assert passage['text'][start:end] == row['text']
            assert row['hpo_id'] in allowed and alt.get(row['raw_hpo_id'], row['raw_hpo_id']) == row['hpo_id']
            selected.append(row)
            mappings.add((row['raw_hpo_id'], row['hpo_id']))
        public[name] = selected
        observed_alt[name] = [list(item) for item in sorted(mappings)]
    union = {}
    for name in ('pubmed', 'bioformer'):
        for row in public[name]:
            union.setdefault((str(row['pmc_id']), row['offset'], row['length'], row['hpo_id']), row)
    allkeys = {(str(r['pmc_id']), 'hpo', r['hpo_id']) for r in union.values()}
    newkeys = allkeys - oldkeys
    new_entities = defaultdict(list)
    for row in union.values():
        did = str(row['pmc_id'])
        if (did, 'hpo', row['hpo_id']) in newkeys:
            new_entities[did].append(dict(identifier=row['hpo_id'], type='Phenotype', offset=row['offset'],
                                         length=row['length'], text=row['text'], note=None))
    entity_rows = []
    for doc in docs:
        did = str(doc['pmc_id'])
        # Union already deduplicated exact occurrence; keep original output order.
        new_entities[did].sort(key=lambda e: (e['offset'], e['length'], e['identifier']))
        entity_rows.append(dict(pmc_id=doc['pmc_id'], entities=new_entities[did]))
    newprompts = make_prompts(new_entities, public=True)
    assert {(str(p['pmc_id']), 'hpo', value) for p in newprompts for value in p['code_map'].values()} == newkeys
    assert not (newkeys & oldkeys)
    for did in by:
        s = spaces[did]
        for name in ('pubmed', 'bioformer'):
            s[name + '_canonical_hpo'] = sorted({r['hpo_id'] for r in public[name] if str(r['pmc_id']) == did})
        s['new_public_canonical_hpo'] = sorted(v for d, kind, v in newkeys if d == did)
        s['normal_code_maps'] = {r['row_id']: r['code_map'] for r in normal if str(r['pmc_id']) == did}
        s['reverse_code_maps'] = {r['row_id']: r['code_map'] for r in reverse if str(r['pmc_id']) == did}
        s['public_new_code_maps'] = {r['row_id']: r['code_map'] for r in newprompts if str(r['pmc_id']) == did}
        assert not set(s['new_public_canonical_hpo']) & set(s['old_hpo_canonical'])
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    for name, records in [('normal', normal), ('reverse', reverse), ('public-new', newprompts)]:
        save(out / (name + '.prompts.jsonl'), records, jsonl=True)
    save(out / 'public-new.entities.jsonl', entity_rows, jsonl=True)
    save(out / 'reverse-inverse.json', reverse_audit)
    space = dict(status='OLD_VS_PUBLIC_NEW_CANONICAL_DISJOINT_ALL_B_DOCUMENTS',
                 hpo_version=version, hpo=references['hpo'],
                 policy='Original -1 text keys remain typed text; HPO compounds split then official alt_id mapped; public canonical HPO minus ALL old candidate concepts, before inference/gating.',
                 source_public_policy=references['public_source'], references=references,
                 public_observed_raw_to_canonical=observed_alt, documents=spaces,
                 patient_space_independent_verification='PENDING_SEQUENCE_OWNER; this manifest proves old-vs-public space only')
    save(out / 'candidate-space.json', space)
    artifacts = {p.name: ref(p) for p in sorted(out.iterdir()) if p.is_file()}
    counts = {name: dict(prompts=len(rs), concepts=sum(len(r['code_map']) for r in rs),
                        patient_concept_pairs=sum(len(r['code_map']) * len(r['patient_ids']) for r in rs))
              for name, rs in [('normal', normal), ('reverse', reverse), ('public-new', newprompts)]}
    save(out / 'preparation.json', dict(status='PREPARED_INPUT_ONLY_NOT_INFERRED',
         at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), source=ref(__file__), plan=ref(args.plan),
         references=references, documents=len(docs), patients=plan['patients'],
         old_entity_count=sum(len(d['entities']) for d in old.values()), counts=counts,
         public_union_occurrences=len(union), public_union_concepts=len(allkeys),
         excluded_old_document_concepts=len(allkeys & oldkeys), new_public_concepts=len(newkeys),
         old_vs_public_canonical_intersection=0, artifacts=artifacts,
         no_fit=True, no_neural_inference=True, no_gold_read=True, no_quality_evaluation=True,
         cpu_seconds=time.monotonic() - started))
    fd = os.open(out, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    print(json.dumps(ref(out / 'preparation.json')))


if __name__ == '__main__':
    main()
