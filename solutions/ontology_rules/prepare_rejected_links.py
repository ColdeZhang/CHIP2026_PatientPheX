#!/usr/bin/env python3
"""Prepare a fixed contextual linker control from frozen GLiNER/Sap rejections.

CPU only. Never fits a classifier, encodes a new vector, or reads evaluation gold.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def select(args):
    import run as ontology
    docs = {d['pmc_id']: d for d in rows(args.input)}
    bases = {d['pmc_id']: d for d in rows(args.base)}
    candidates = rows(args.candidates)
    train = rows(args.train)
    assert len(train) == 48 and not set(docs) & {d['pmc_id'] for d in train}
    assert set(docs) == set(bases) == {r['pmc_id'] for r in candidates}
    assert all('entities' not in d and 'association' not in d for d in docs.values())
    source = json.loads(Path(args.candidate_manifest).read_text())
    for name in ('input', 'train', 'hpo'):
        assert sha(getattr(args, name)) == source['inputs'][name]
    assert sha(args.candidates) == source['output']['sha256']
    counts = Counter()
    requests = []
    for row in candidates:
        doc, base = docs[row['pmc_id']], bases[row['pmc_id']]
        for c in sorted(row['candidates'], key=lambda c: (-c['score'], -c['length'], c['offset'])):
            counts['all'] += 1
            if c['score'] <= .5:
                counts['not_original_gliner_threshold'] += 1
                continue
            if c['probability'] >= .65 and c['similarity'] >= .92:
                counts['sap_already_passes'] += 1
                continue
            if any(c['offset'] < e['offset'] + e['length'] and e['offset'] < c['offset'] + c['length'] for e in base['entities']):
                counts['overlap_frozen_base'] += 1
                continue
            matches = [p for p in doc['full_text'] if p['offset'] <= c['offset'] and c['offset'] + c['length'] <= p['offset'] + len(p['text'])]
            assert len(matches) == 1
            passage = matches[0]
            start = c['offset'] - passage['offset']
            end = start + c['length']
            assert passage['text'][start:end] == c['text']
            if ontology.is_negated(passage['text'], start, end):
                counts['negated'] += 1
                continue
            left, right = max(0, start - 240), min(len(passage['text']), end + 240)
            requests.append({'row_id': f'L{len(requests):04d}', 'pmc_id': row['pmc_id'],
                             'candidate': c, 'context': passage['text'][left:right],
                             'context_offset': passage['offset'] + left,
                             'mention_start': start - left, 'mention_end': end - left,
                             'section': passage.get('section_type'),
                             'sap_rejection': {'probability': c['probability'] < .65,
                                               'similarity': c['similarity'] < .92}})
            counts['pending'] += 1
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'requests.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in requests))
    save(out / 'selection.json', {'source_sha256': sha(__file__), 'counts': dict(counts),
         'files': {name: {'path': str(Path(getattr(args, name)).resolve()), 'sha256': sha(getattr(args, name))}
                   for name in ('train', 'input', 'base', 'candidates', 'candidate_manifest', 'hpo')},
         'documents_with_requests': len({r['pmc_id'] for r in requests}),
         'request_sha256': sha(out / 'requests.jsonl'), 'negation_source_sha256': sha(ontology.__file__),
         'new_overlap_policy': 'Resolve only accepted choices by frozen GLiNER score,length,offset order.',
         'gold_read': False})
    print(json.dumps(counts))


def rank(args):
    import numpy as np
    started = time.monotonic()
    spec = importlib.util.spec_from_file_location('frozen_semantic', args.semantic_source)
    sem = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sem)
    reconstructed = json.loads(Path(args.reconstruction_manifest).read_text())
    assert sha(args.semantic_source) == reconstructed['semantic_source_sha256']
    assert sha(args.bank) == reconstructed['bank_cache']['sha256']
    mapping = json.loads(Path(args.bank_labels).read_text())
    labels, ids = mapping['labels'], mapping['label_ids']
    assert hashlib.sha256(json.dumps(labels, ensure_ascii=False).encode()).hexdigest() == reconstructed['bank_cache']['ordered_texts_sha256']
    assert len(labels) == len(ids) == 42236
    bank = np.load(args.bank, allow_pickle=False)
    assert bank.shape == (len(labels), 768)
    candidates = {r['pmc_id']: r['candidates'] for r in rows(args.all_candidates)}
    requests = rows(args.requests)
    cache, cache_evidence, output = {}, {}, []
    terms, _ = sem.hpo_terms(args.hpo)
    allowed = {hid for hid in terms if hid != 'HP:0000118' and re.fullmatch(r'HP:\d{7}', hid)}
    max_similarity_difference = 0.0
    for request in requests:
        pmc = request['pmc_id']
        if pmc not in cache:
            names = sorted({sem.norm(c['text']) for c in candidates[pmc]})
            key = hashlib.sha256((reconstructed['encoder_signature'] + '\0' + '\0'.join(names)).encode()).hexdigest()
            path = Path(args.embedding_dir) / (key + '.npy')
            assert path.exists(), 'Cache miss forbidden: ' + str(path)
            vectors = np.load(path, allow_pickle=False)
            assert vectors.shape == (len(names), 768)
            cache[pmc] = (vectors, {name: i for i, name in enumerate(names)})
            cache_evidence[pmc] = {'path': str(path), 'sha256': sha(path), 'rows': len(names), 'cache_key': key}
        vectors, lookup = cache[pmc]
        vector = vectors[lookup[sem.norm(request['candidate']['text'])]]
        similarities = vector @ bank.T
        max_similarity_difference = max(max_similarity_difference, abs(float(similarities.max()) - request['candidate']['similarity']))
        order = np.lexsort((np.arange(len(bank)), -similarities))
        selected, seen = [], set()
        for i in order:
            hid = ids[i]
            if hid not in allowed or hid in seen:
                continue
            seen.add(hid)
            selected.append({'identifier': hid, 'similarity': float(similarities[i]),
                             'bank_index': int(i), 'alias': labels[i]})
            if len(selected) == 5:
                break
        assert len(selected) == 5
        output.append({'row_id': request['row_id'], 'pmc_id': pmc,
                       'text': request['candidate']['text'], 'candidates': selected})
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    save(out / 'top5.json', output)
    save(out / 'ranking.json', {'source_sha256': sha(__file__), 'files':
         {name: {'path': str(Path(getattr(args, name)).resolve()), 'sha256': sha(getattr(args, name))}
          for name in ('semantic_source', 'bank', 'bank_labels', 'reconstruction_manifest', 'all_candidates', 'requests', 'hpo')},
         'cache_files': cache_evidence, 'top5_sha256': sha(out / 'top5.json'),
         'requests': len(requests), 'unmapped_or_compound_bank_rows_excluded': sum(hid not in allowed for hid in ids),
         'max_abs_nearest_similarity_difference_vs_old_gpu': max_similarity_difference,
         'policy': 'CPU FP32 cosine, stable row-index ties, max alias score per distinct allowed atomic HPO; no exact-surface override.',
         'new_embeddings': 0, 'classifier_fit': False, 'gold_read': False,
         'elapsed_seconds': time.monotonic() - started})
    print(json.dumps({'requests': len(requests), 'seconds': time.monotonic() - started,
                      'max_abs_cosine_difference': max_similarity_difference}))


LINK_SYSTEM = '''Read the highlighted original phrase in its actual context. Decide whether the phrase itself expresses a phenotypic abnormality, then select one semantically equivalent candidate.
Return only a JSON object {"choice":"C1"}, where choice is C1, C2, C3, C4, C5, or ABSTAIN.
Preserve anatomy, direction, severity and specificity. A related disorder or a broader/narrower finding is not automatically equivalent. Do not infer an unmentioned phenotype from a disease, gene, test or treatment. A normal anatomical structure alone is not a phenotype. Reject a negated finding or a phrase that does not itself express a phenotype.
Use ABSTAIN whenever the phrase is not a positive phenotype, the meaning is uncertain, or none of the five candidates is equivalent. Missing a match in five choices does not prove that the ontology has no corresponding concept. Do not reject a phenotype solely because it refers to a relative or appears in general medical prose. This task does not decide patient ownership.
Choose exactly one listed code. Never output an HPO identifier, list, explanation, or markdown.'''


def prompts(args):
    import run as ontology
    requests = rows(args.requests)
    rankings = {r['row_id']: r for r in json.loads(Path(args.rankings).read_text())}
    terms, allowed, _, _ = ontology.load_hpo(args.hpo)
    definitions = {}
    current = None
    for line in Path(args.hpo).read_text().splitlines():
        if line.startswith('id: '):
            current = line[4:]
        elif line.startswith('def: ') and current:
            match = re.match(r'def: "((?:[^"\\]|\\.)*)"', line)
            if match:
                definitions[current] = match.group(1).replace('\\"', '"').replace('\\n', ' ')
    result, counts = [], Counter()
    for request in requests:
        ranking = rankings[request['row_id']]
        assert ranking['pmc_id'] == request['pmc_id'] and ranking['text'] == request['candidate']['text']
        choices, mapping = [], {}
        for i, c in enumerate(ranking['candidates'], 1):
            hid = c['identifier']
            assert hid in allowed and hid != ontology.ROOT
            definition = definitions.get(hid, '')
            counts['definition_missing'] += not bool(definition)
            counts['definition_truncated_at480'] += len(definition) > 480
            code = f'C{i}'
            choices.append({'code': code, 'name': terms[hid]['name'], 'definition': definition[:480]})
            mapping[code] = hid
        context = request['context']
        a, b = request['mention_start'], request['mention_end']
        assert context[a:b] == request['candidate']['text']
        highlighted = context[:a] + '<mention>' + context[a:b] + '</mention>' + context[b:]
        user = {'phrase': request['candidate']['text'], 'context': highlighted, 'candidates': choices}
        result.append({'row_id': request['row_id'], 'pmc_id': request['pmc_id'],
                       'messages': [{'role': 'system', 'content': LINK_SYSTEM},
                                    {'role': 'user', 'content': json.dumps(user, ensure_ascii=False)}],
                       'code_to_hpo': mapping, 'candidate': request['candidate']})
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'prompts.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in result))
    save(out / 'prompt-manifest.json', {'source_sha256': sha(__file__),
         'inputs': {name: sha(getattr(args, name)) for name in ('requests', 'rankings', 'hpo')},
         'prompts_sha256': sha(out / 'prompts.jsonl'), 'requests': len(result),
         'counts': dict(counts), 'gold_read': False, 'choice_codes': ['C1', 'C2', 'C3', 'C4', 'C5', 'ABSTAIN'],
         'semantics': {'ABSTAIN': 'No new entity/association, whether nonphenotype, negated, uncertain, or absent from top5. Not evidence for gold NIL; this fixed control intentionally cannot recover new unmapped entities.'}})
    print(json.dumps({'requests': len(result), 'counts': counts}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name, function, required in [
        ('select', select, ('train', 'input', 'base', 'candidates', 'candidate-manifest', 'hpo')),
        ('rank', rank, ('requests', 'all-candidates', 'semantic-source', 'reconstruction-manifest', 'bank', 'bank-labels', 'embedding-dir', 'hpo')),
        ('prompts', prompts, ('requests', 'rankings', 'hpo')),
    ]:
        command = sub.add_parser(name)
        for argument in (*required, 'output-dir'):
            command.add_argument('--' + argument, required=True)
        command.set_defaults(function=function)
    args = parser.parse_args()
    args.function(args)
