#!/usr/bin/env python3
"""Build ontology-classification rows from designated HPO and train-only text.

This is a transparent PyTorch-oriented adaptation of PhenoTagger's recipe.
No external phenotype annotation corpus or pretrained PhenoTagger head is used.
"""
import argparse
from collections import Counter, defaultdict
import gzip
import io
import json
from pathlib import Path
import random
import re
import time
import nltk
from nltk.stem import WordNetLemmatizer
import run as baseline

DISALLOWED_EDGE = {'IN', 'DT', 'CC', 'O', 'MD', 'EX', 'POS', 'WDT', 'WP', 'WP$', 'WRB', 'TO', 'PRP$'}
VERBS = {'is', 'are', 'was', 'were', 'had', 'have', 'has', 'be', 'been', 'also'}
PUNCT = re.compile(r'[.!;:?()\[\]{}]')


def tokens(text):
    return nltk.word_tokenize(text.strip().lower().replace('-', ' - ').replace('/', ' / '), preserve_line=True)


def forms(text, lemmatizer):
    words = tokens(text)
    tagged = nltk.pos_tag(words)
    pos = lambda tag: 'a' if tag.startswith('J') else 'v' if tag.startswith('V') else 'r' if tag.startswith('R') or tag == 'IN' else 'n'
    return ' '.join(words), ' '.join(lemmatizer.lemmatize(word, pos(tag)) for word, tag in tagged)


def ontology_pairs(path, lemmatizer):
    terms, allowed, alt, version = baseline.load_hpo(path)
    allowed -= {baseline.ROOT}
    raw = []; hid = None; skipped = Counter()
    for line in Path(path).read_text().splitlines():
        if line.startswith('id: '): hid = line[4:]
        if hid not in allowed: continue
        if line.startswith('name: '):
            raw.append((hid, line[6:], 'name'))
        elif line.startswith('synonym: '):
            match = re.match(r'synonym: "((?:[^"\\]|\\.)*)"\s+(\w+)', line)
            if not match: continue
            word = match[1].replace('\\"', '"')
            if word.isupper() and len(word) < 10:
                skipped['short_uppercase_synonyms'] += 1
                continue
            raw.append((hid, word, match[2]))
    pairs = set(); scopes = Counter(); normalized = {}
    for hid, word, scope in raw:
        if word not in normalized: normalized[word] = forms(word, lemmatizer)
        scopes[scope] += 1
        for value in normalized[word]:
            if value: pairs.add((value, hid))
    by_surface = defaultdict(set)
    for surface, hid in pairs: by_surface[surface].add(hid)
    return sorted(pairs), sorted(allowed), by_surface, {
        'version': version, 'scopes_before_deduplication': dict(scopes),
        'skipped': dict(skipped), 'raw_pairs': len(raw), 'lemma_surface_pairs': len(pairs),
        'unique_positive_surfaces': len(by_surface),
        'ambiguous_surfaces': sum(len(v) > 1 for v in by_surface.values()),
        'maximum_labels_per_surface': max(map(len, by_surface.values())),
        'conflict_policy': 'Keep each distinct surface-label pair once; never arbitrarily choose the first label.'}


def candidate_negatives(doc, protected, target, rng):
    candidates = []; stats = Counter()
    forbidden = [(e['offset'], e['offset'] + e['length']) for e in doc.get('entities', [])]
    for passage in doc['full_text']:
        if str(passage.get('section_type', '')).upper() in baseline.STOP_SECTIONS: continue
        text = passage['text']; start = passage['offset']; words = list(re.finditer(r'\S+', text))
        for i, word in enumerate(words):
            for j in range(i, min(len(words), i + 9)):
                a, b = word.start(), words[j].end(); raw = text[a:b]
                if len(raw) > 200 or PUNCT.search(raw): continue
                if any(start + a < end and begin < start + b for begin, end in forbidden):
                    stats['excluded_gold_overlap'] += 1; continue
                candidates.append((start + a, b - a, raw))
    rng.shuffle(candidates); selected = []; seen = set()
    for offset, length, raw in candidates:
        normalized = ' '.join(tokens(raw))
        if not normalized or normalized in protected or normalized in seen: continue
        seen.add(normalized); tagged = nltk.pos_tag(normalized.split())
        if tagged[0][1] in DISALLOWED_EDGE | {'PRP'} or tagged[-1][1] in DISALLOWED_EDGE: continue
        if tagged[0][0] in VERBS | {',', '-', '/'} or tagged[-1][0] in VERBS | {',', '-', '/'}: continue
        selected.append({'text': normalized, 'label': 'HP:None', 'source': 'train_text_weak_negative',
                         'pmc_id': doc['pmc_id'], 'offset': offset, 'length': length, 'raw_text': raw})
        if len(selected) == target: break
    stats.update(candidates_before_pos=len(candidates), selected=len(selected), requested=target)
    return selected, dict(stats)


def main():
    ap = argparse.ArgumentParser()
    for name in ['hpo', 'train', 'nltk-data', 'output-dir']: ap.add_argument('--' + name, required=True)
    ap.add_argument('--seed', type=int, default=20261023)
    a = ap.parse_args(); started = time.monotonic(); out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    nltk.data.path.insert(0, a.nltk_data); lemmatizer = WordNetLemmatizer()
    pairs, concepts, by_surface, ontology_stats = ontology_pairs(a.hpo, lemmatizer)
    labels = ['HP:None'] + concepts; assert len(labels) == len(set(labels))
    train = baseline.read_jsonl(a.train); assert len({d['pmc_id'] for d in train}) == len(train)
    protected = set(by_surface)
    for doc in train:
        for e in doc.get('entities', []): protected.update(forms(e['text'], lemmatizer))
    rows = [{'text': surface, 'label': hid, 'source': 'hpo_name_synonym_lemma'} for surface, hid in pairs]
    rng = random.Random(a.seed); per_doc = []; negative_pool = []
    for i, doc in enumerate(train):
        # Short documents can exhaust eligible text. Oversupply a bounded pool
        # per paper, then sample the final 1:1 ratio globally without repetition.
        target = 2 * (len(pairs) // len(train) + (i < len(pairs) % len(train)))
        negatives, stats = candidate_negatives(doc, protected, target, rng)
        negative_pool.extend(negatives); per_doc.append({'pmc_id': doc['pmc_id'], **stats})
        print(json.dumps({'event': 'negative_document', 'index': i, 'rows': len(negatives)}), flush=True)
    assert len(negative_pool) >= len(pairs), 'Not enough eligible train-only negatives'
    selected_negatives = rng.sample(negative_pool, len(pairs))
    rows.extend(selected_negatives)
    final_per_doc = Counter(row['pmc_id'] for row in selected_negatives)
    label_set = set(labels)
    assert all(row['label'] in label_set for row in rows)
    assert len(rows) == 2 * len(pairs)
    path = out / 'rows.jsonl.gz'
    with path.open('wb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding='utf-8') as f:
                for row in rows: f.write(json.dumps(row, ensure_ascii=False) + '\n')
    (out / 'labels.json').write_text(json.dumps(labels, indent=2) + '\n')
    manifest = {'config': vars(a), 'source_sha256': baseline.sha256(__file__),
        'baseline_source_sha256': baseline.sha256(baseline.__file__), 'hpo_sha256': baseline.sha256(a.hpo),
        'train_sha256': baseline.sha256(a.train), 'train_ids': sorted(d['pmc_id'] for d in train),
        'ontology': ontology_stats, 'classes_with_reject': len(labels), 'rows': len(rows),
        'negative_rows': len(pairs), 'negative_pool': len(negative_pool), 'per_document': per_doc,
        'final_negatives_per_document': dict(final_per_doc),
        'sampling': 'Up to twice equal-document quota eligible distinct surfaces per paper, then global sample without replacement to 1:1; short papers need not fill quota.',
        'negative_annotation_policy': 'Weak negatives only; avoid every annotated span including NO/-1 and all known phenotype surfaces. No competition gold entities added as positive training rows in this pilot.',
        'public_language_resources': json.loads((Path(a.nltk_data) / 'manifest.json').read_text()),
        'nltk_version': nltk.__version__, 'seconds': time.monotonic() - started,
        'outputs': {p.name: baseline.sha256(p) for p in [path, out / 'labels.json']}}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'event': 'finished', 'rows': len(rows), 'classes': len(labels), 'ontology': ontology_stats, 'seconds': manifest['seconds']}))


if __name__ == '__main__': main()
