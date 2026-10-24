"""Fold-local frozen SapBERT normalization for the complete declared GP pool."""
import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import sys
import time
import torch
from common import read_jsonl, write_json, write_jsonl, ref


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m; spec.loader.exec_module(m); return m


def main():
    ap = argparse.ArgumentParser()
    for name in ('train', 'input', 'hpo', 'span-pool', 'sapbert', 'legacy-source', 'semantic-source', 'output'):
        ap.add_argument('--' + name, required=True)
    a = ap.parse_args(); started = time.monotonic(); out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    base = module(a.legacy_source, 'gp_legacy_linker'); sem = module(a.semantic_source, 'gp_semantic_encoder')
    train = read_jsonl(a.train); inputs = read_jsonl(a.input); pool = read_jsonl(a.span_pool)
    assert len(train) == 32 and len(inputs) == 16
    assert not {str(x['pmc_id']) for x in train} & {str(x['pmc_id']) for x in inputs}
    assert all('entities' not in d and 'association' not in d for d in inputs)
    docs = {str(d['pmc_id']): d for d in inputs}; linker = base.Linker(train, a.hpo)
    train_surfaces = {base.norm(e['text']) for d in train for e in d.get('entities', [])
                      if str(e.get('note', '')).upper() != 'NO' and linker.canonical(e.get('identifier', '-1'))}
    for row in pool:
        passage = docs[row['pmc_id']]['full_text'][row['passage_index']]
        text = passage['text']; lo, hi = row['start'], row['end']
        assert 0 <= lo < hi <= len(text)
        surface = text[lo:hi]; row['text'] = surface; row['surface_key'] = base.norm(surface)
        row['offset'] = int(passage.get('offset', 0)) + lo; row['length'] = hi-lo
        left = lo == 0 or not (text[lo-1].isalnum() and text[lo].isalnum())
        right = hi == len(text) or not (text[hi-1].isalnum() and text[hi].isalnum())
        row['valid_word_boundary'] = bool(left and right or row['surface_key'] in train_surfaces)
        row['valid_surface'] = bool(surface.strip() == surface and any(c.isalpha() for c in surface))
    names = linker.names; ids = [linker.aliases[n].most_common(1)[0][0] for n in names]
    encoder = sem.Encoder(a.sapbert, 'sapbert', 'cuda', 256, out/'embeddings', 1)
    bank = encoder.encode(names, 'train32-hpo-alias-bank')
    surfaces = sorted({r['surface_key'] for r in pool if r['surface_key'] not in linker.aliases})
    vectors = encoder.encode(surfaces, 'gp-all-generated-pool')
    indexes, similarities = encoder.nearest(vectors, bank)
    semantic = {s: (ids[int(i)], float(v), names[int(i)]) for s, i, v in zip(surfaces, indexes, similarities)}
    lengths = encoder.tokenizer(surfaces, truncation=False, add_special_tokens=True)['input_ids'] if surfaces else []
    truncated = sum(len(x) > 32 for x in lengths)
    counts = Counter(); selected = {str(d['pmc_id']): [] for d in inputs}
    for row in pool:
        key = row.pop('surface_key'); char_id, char_score = linker.link(row['text'])
        if key in linker.aliases:
            identifier = char_id; similarity = 1.0; matched = key; origin = 'fold_local_exact_alias'
        else:
            sem_id, similarity, matched = semantic[key]
            identifier = sem_id if similarity >= .65 else char_id
            origin = 'SapBERT_cosine_ge_0.65' if similarity >= .65 else 'legacy_char_or_NIL_fallback'
        row.update(identifier=identifier, normalization_source=origin, sap_similarity=similarity,
                   matched_alias=matched, char_identifier=char_id, char_score=char_score,
                   gp_decision=row['gp_logit'] > 0.0 and row['valid_word_boundary'] and row['valid_surface'])
        counts[origin] += 1
        if row['gp_decision']:
            e = {k: row[k] for k in ('identifier', 'offset', 'length', 'text')}
            e.update(type='Phenotype', note=None); selected[row['pmc_id']].append(e)
    write_jsonl(out/'normalized-pool.jsonl', pool)
    entities = [{'pmc_id': d['pmc_id'], 'entities': sorted(selected[str(d['pmc_id'])],
                 key=lambda e: (e['offset'], e['length'], e['identifier']))} for d in inputs]
    write_jsonl(out/'entities.jsonl', entities)
    result = {'status': 'COMPLETE_ALL_DECLARED_CANDIDATES_NORMALIZED', 'pool_candidates': len(pool),
              'accepted_entities': sum(len(x['entities']) for x in entities), 'documents': len(inputs),
              'sources': {k: ref(getattr(a, k)) for k in ('train', 'input', 'hpo', 'span_pool', 'legacy_source', 'semantic_source')},
              'normalization': {'exact_fold_local_alias_first': True, 'sap_threshold': .65, 'sap_max_length': 32,
                                'char_fallback_unchanged': True, 'novel_surfaces_over_32_tokens': truncated,
                                'all_candidate_scores_preserved': True, 'candidate_floor': -2.0,
                                'decision_gp_logit_strictly_greater_than': 0.0, 'postscore_word_boundary_policy': 'alphanumeric boundary or observed positive train32 surface'},
              'normalization_counts': dict(counts), 'normalized_pool': ref(out/'normalized-pool.jsonl'),
              'entities': ref(out/'entities.jsonl'), 'seconds': time.monotonic()-started, 'held_gold_read': False}
    write_json(out/'complete.json', result); print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
