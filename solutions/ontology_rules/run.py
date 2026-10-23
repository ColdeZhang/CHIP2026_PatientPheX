#!/usr/bin/env python3
"""Deterministic HPO gazetteer extraction and patient discourse attribution.

Only --train labels contribute aliases, ambiguity priors, or lexical filtering.
Input labels are deliberately ignored. All coordinates come from passage offsets.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time

ROOT = 'HP:0000118'
TOKEN = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?|\d+(?:\.\d+)?")
GAP = re.compile(r'[\s\-/]*\Z')
NEG_PRE = re.compile(r'\b(?:no|without|denies?|denied|absence\s+of|negative\s+for|free\s+of|neither|not\s+(?:have|show|exhibit|demonstrate|develop|associated\s+with))\b', re.I)
NEG_POST = re.compile(r'^\W*(?:(?:was|were|is|are)\s+)?(?:absent\b|not\s+(?:seen|observed|present|found|detected)\b|excluded\b|ruled\s+out\b)', re.I)
PSEUDO = re.compile(r'\b(?:not\s+only|no\s+(?:change|further|increase)|not\s+(?:excluded|ruled\s+out|uncommon))\b', re.I)
TERMINATE = re.compile(r'\b(?:but|however|except|whereas|although|nevertheless)\b', re.I)
FAMILY = re.compile(r'\b(?:mother|father|sister|brother|parents?|siblings?|grandmother|grandfather|aunt|uncle|cousin|daughter|son)\b', re.I)
CLINICAL = re.compile(r'\b(?:he|she|his|her|proband|patient|case|boy|girl|infant|child|examination|presented|developed|suffered|diagnosed|revealed|showed|observed|exhibited|at birth|follow.up)\b', re.I)
GENERAL = re.compile(r'\b(?:patients?\s+with|typically|commonly|usually|characterized\s+by|associated\s+with|has\s+been\s+reported|have\s+been\s+reported)\b', re.I)
STOP_SECTIONS = {'REF', 'REFERENCES', 'REFLIST', 'AUTH_CONT', 'COMP_INT', 'ACK_FUND', 'ACKNOWLEDGMENTS', 'SUPPL'}
GENERAL_SECTIONS = {'INTRO', 'INTRODUCTION', 'DISCUSS', 'DISCUSSION', 'BACKGROUND', 'CONCL', 'CONCLUSIONS'}


def read_jsonl(path):
    with open(path, encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalize(word):
    word = word.lower().replace('’', "'")
    if len(word) > 4 and word.endswith('ies'):
        return word[:-3] + 'y'
    if len(word) > 4 and word.endswith('s') and not word.endswith(('ss', 'us', 'is', 'ous')):
        return word[:-1]
    return word


def token_key(text):
    return tuple(normalize(m.group()) for m in TOKEN.finditer(text))


def identifier_parts(value):
    return tuple(x.strip() for x in str(value).split(';') if x.strip())


def load_hpo(path):
    terms = {}
    current = None
    version = None
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if line.startswith('data-version: '):
            version = line.split(': ', 1)[1]
        if line == '[Term]':
            current = {'parents': [], 'synonyms': [], 'alt': [], 'obsolete': False}
        elif line.startswith('['):
            current = None
        elif current is not None:
            if line.startswith('id: '):
                current['id'] = line[4:]
                terms[current['id']] = current
            elif line.startswith('name: '): current['name'] = line[6:]
            elif line.startswith('is_a: '): current['parents'].append(line[6:].split()[0])
            elif line.startswith('alt_id: '): current['alt'].append(line[8:])
            elif line.startswith('is_obsolete: true'): current['obsolete'] = True
            elif line.startswith('synonym: '):
                match = re.match(r'synonym: "((?:[^"\\]|\\.)*)"\s+EXACT\b', line)
                if match: current['synonyms'].append(match.group(1).replace('\\"', '"'))
    if version != 'hp/releases/2026-06-23':
        raise ValueError(f'Expected official HPO 2026-06-23; found {version}')
    children = defaultdict(list)
    for hid, term in terms.items():
        for parent in term['parents']: children[parent].append(hid)
    allowed = set()
    stack = [ROOT]
    while stack:
        hid = stack.pop()
        if hid in allowed: continue
        allowed.add(hid)
        stack.extend(children[hid])
    allowed = {hid for hid in allowed if not terms[hid]['obsolete']}
    aliases = {a: hid for hid in allowed for a in terms[hid]['alt']}
    return terms, allowed, aliases, version


@dataclass(frozen=True)
class Mention:
    start: int
    end: int
    key: tuple
    ids: tuple
    training_count: int


class Gazetteer:
    def __init__(self, hpo_path, training):
        terms, self.allowed, self.alt_ids, self.version = load_hpo(hpo_path)
        self.options = defaultdict(Counter)
        self.training_aliases = set()
        self.surface_examples = {}
        for hid in sorted(self.allowed):
            term = terms[hid]
            for surface in [term.get('name', '')] + term['synonyms']:
                key = token_key(surface)
                if key and len(key) <= 16 and len(''.join(key)) >= 3:
                    self.options[key][(hid,)] += 0
                    self.surface_examples.setdefault(key, surface)
        ignored_labels = Counter()
        for doc in training:
            for entity in doc.get('entities', []):
                key = token_key(entity.get('text', ''))
                ids = []
                for hid in identifier_parts(entity.get('identifier', '')):
                    hid = self.alt_ids.get(hid, hid)
                    if hid in self.allowed or hid == '-1': ids.append(hid)
                    else: ignored_labels[hid] += 1
                ids = tuple(sorted(set(ids)))
                if key and ids and len(key) <= 16 and len(''.join(key)) >= 2:
                    self.options[key][ids] += 1
                    self.training_aliases.add(key)
                    self.surface_examples.setdefault(key, entity['text'])
        self.ignored_training_labels = dict(ignored_labels)
        self.blocked = set()
        self._build()
        occurrences = Counter()
        annotated = Counter()
        for doc in training:
            gold = {(int(e['offset']), int(e['offset']) + int(e['length'])) for e in doc.get('entities', [])}
            for passage in doc.get('full_text', []):
                text = passage.get('text', '')
                base = int(passage.get('offset', 0))
                for mention in self.scan(text, apply_negation=False):
                    occurrences[mention.key] += 1
                    if (base + mention.start, base + mention.end) in gold:
                        annotated[mention.key] += 1
        # Corpus-specific noise learned only from training text and its labels.
        self.blocked = {key for key, n in occurrences.items() if n >= 5 and annotated[key] == 0 and key not in self.training_aliases}
        self._build()
        self.stats = {'allowed_hpo_concepts': len(self.allowed), 'normalized_terms': len(self.options), 'training_aliases': len(self.training_aliases), 'blocked_unannotated_terms': len(self.blocked), 'ignored_training_labels': self.ignored_training_labels}

    def _build(self):
        self.trie = {}
        for key, options in self.options.items():
            if key in self.blocked: continue
            # Most frequent annotated mapping; stable lexical tie break for unseen names.
            ids, count = min(options.items(), key=lambda x: (-x[1], len(x[0]), x[0]))
            node = self.trie
            for token in key: node = node.setdefault(token, {})
            node[None] = (key, ids, count)

    def scan(self, text, apply_negation=True):
        tokens = list(TOKEN.finditer(text))
        matches = []
        for i, token in enumerate(tokens):
            node = self.trie
            for j in range(i, min(i + 16, len(tokens))):
                if j > i and not GAP.fullmatch(text[tokens[j - 1].end():tokens[j].start()]): break
                key = normalize(tokens[j].group())
                if key not in node: break
                node = node[key]
                if None not in node: continue
                normalized, ids, count = node[None]
                start, end = token.start(), tokens[j].end()
                surface = text[start:end]
                letters = ''.join(normalized)
                if len(normalized) == 1 and len(letters) <= 3:
                    # Short abbreviations require actual uppercase text; never match a word interior.
                    if not surface.isupper() or len(letters) < 2: continue
                matches.append(Mention(start, end, normalized, ids, count))
        selected = []
        last_end = -1
        for mention in sorted(matches, key=lambda m: (m.start, -(m.end - m.start), -m.training_count)):
            if mention.start < last_end: continue
            last_end = mention.end
            if apply_negation and is_negated(text, mention.start, mention.end): continue
            selected.append(mention)
        return selected


def clause_bounds(text, start, end):
    left = 0
    right = len(text)
    for match in re.finditer(r'[.!?;\n]|\b(?:but|however|except|whereas|although|nevertheless)\b', text, re.I):
        if match.end() <= start: left = match.end()
        elif match.start() >= end:
            right = match.start()
            break
    return left, right


def is_negated(text, start, end):
    left, right = clause_bounds(text, start, end)
    before = text[left:start]
    after = text[end:right]
    before = PSEUDO.sub('', before)
    triggers = list(NEG_PRE.finditer(before))
    if triggers:
        tail = before[triggers[-1].end():]
        if len(TOKEN.findall(tail)) <= 16 and not re.search(r'\b(?:developed|presented|but|however)\b', tail, re.I):
            return True
    return bool(NEG_POST.search(after[:90]))


def patient_anchors(doc):
    anchors = []
    aliases = defaultdict(set)
    for patient in doc.get('patient', []):
        pid = patient['patient_id']
        for mention in patient.get('mention', []):
            start = int(mention.get('offset', -1))
            length = int(mention.get('length', len(mention.get('text', ''))))
            if start >= 0 and length > 0: anchors.append((start, start + length, pid))
            surface = mention.get('text', '').strip()
            if len(surface) >= 4 and not re.fullmatch(r'(?:he|she|his|her|the patient|patient|proband|child|boy|girl)', surface, re.I):
                aliases[surface.lower()].add(pid)
    for surface, pids in aliases.items():
        if len(pids) != 1: continue
        pid = next(iter(pids))
        pattern = re.compile(r'(?<!\w)' + re.escape(surface) + r'(?!\w)', re.I)
        for passage in doc.get('full_text', []):
            base = int(passage.get('offset', 0))
            for match in pattern.finditer(passage.get('text', '')):
                anchors.append((base + match.start(), base + match.end(), pid))
    return sorted(set(anchors))


def assign_patients(doc, passage, mention, anchors, active):
    patients = [p['patient_id'] for p in doc.get('patient', [])]
    if not patients: return set()
    text = passage['text']
    base = int(passage.get('offset', 0))
    start, end = base + mention.start, base + mention.end
    left, right = clause_bounds(text, mention.start, mention.end)
    clause = text[left:right]
    clause_anchors = [a for a in anchors if base + left <= a[0] < base + right]
    passage_anchors = [a for a in anchors if base <= a[0] < base + len(text)]
    if clause_anchors:
        pids = {a[2] for a in clause_anchors}
        if len(pids) == 1: return pids
        first, last = clause_anchors[0][0] - base, clause_anchors[-1][1] - base
        if start >= base + last and re.search(r'\b(?:and|both|all)\b', text[first:last], re.I): return pids
        preceding = [a for a in clause_anchors if a[0] <= start]
        return {(preceding[-1] if preceding else min(clause_anchors, key=lambda a: abs(a[0] - start)))[2]}
    # A relative who was not one of the given anchors is not automatically the proband.
    if FAMILY.search(clause) and re.search(r'\b(?:had|has|with|showed|presented|developed|affected|diagnosed|suffered)\b', clause, re.I):
        return set()
    if passage_anchors:
        pids = {a[2] for a in passage_anchors}
        if len(pids) == 1: return pids
        preceding = [a for a in passage_anchors if a[0] <= start]
        return {(preceding[-1] if preceding else passage_anchors[0])[2]}
    section = str(passage.get('section_type', '')).upper()
    if section in GENERAL_SECTIONS and not CLINICAL.search(clause): return set()
    if GENERAL.search(clause) and not re.search(r'\b(?:our|this|the)\s+(?:patient|case)|\b(?:he|she|his|her)\b', clause, re.I): return set()
    if active and (CLINICAL.search(text) or section not in GENERAL_SECTIONS): return {active}
    if len(patients) == 1 and CLINICAL.search(text): return {patients[0]}
    return set()


def predict(doc, gazetteer):
    anchors = patient_anchors(doc)
    associations = {p['patient_id']: set() for p in doc.get('patient', [])}
    entities = []
    active = None
    previous_section = None
    for passage in sorted(doc.get('full_text', []), key=lambda p: int(p.get('offset', 0))):
        text = passage.get('text', '')
        base = int(passage.get('offset', 0))
        section = str(passage.get('section_type', '')).upper()
        if section in STOP_SECTIONS: continue
        if section != previous_section and section in GENERAL_SECTIONS: active = None
        local_anchors = [a for a in anchors if base <= a[0] < base + len(text)]
        for mention in gazetteer.scan(text):
            surface = text[mention.start:mention.end]
            identifier = ';'.join(mention.ids)
            entity = {'identifier': identifier, 'type': 'Phenotype', 'offset': base + mention.start, 'length': mention.end - mention.start, 'text': surface, 'note': None}
            entities.append(entity)
            for pid in assign_patients(doc, passage, mention, anchors, active):
                for hid in mention.ids:
                    associations[pid].add(surface if hid == '-1' else hid)
        if local_anchors: active = local_anchors[-1][2]
        previous_section = section
    unique = {(e['offset'], e['length'], e['identifier']): e for e in entities}
    return {'pmc_id': doc['pmc_id'], 'pmid': doc.get('pmid'), 'entities': [unique[k] for k in sorted(unique)], 'association': [{'patient_id': pid, 'phenotype': sorted(ids)} for pid, ids in associations.items()]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', required=True)
    parser.add_argument('--input', required=True)
    parser.add_argument('--hpo', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--work-dir', required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    train = read_jsonl(args.train)
    inputs = read_jsonl(args.input)
    train_ids = {d['pmc_id'] for d in train}
    input_ids = [d['pmc_id'] for d in inputs]
    if len(set(input_ids)) != len(input_ids): raise ValueError('Duplicate input document IDs')
    overlap = sorted(train_ids & set(input_ids))
    if overlap: raise ValueError('Train and inference document IDs overlap; use disjoint dev or test input')
    model = Gazetteer(args.hpo, train)
    fitted = time.perf_counter()
    predictions = [predict(doc, model) for doc in inputs]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', encoding='utf-8') as handle:
        for prediction in predictions: handle.write(json.dumps(prediction, ensure_ascii=False) + '\n')
    run = Path(args.work_dir)
    run.mkdir(parents=True, exist_ok=True)
    stats = {'recorded_at_utc': datetime.now(timezone.utc).isoformat(), 'method': 'ontology_rules_v1', 'hpo_version': model.version, 'training_documents': len(train), 'input_documents': len(inputs), 'entities': sum(len(d['entities']) for d in predictions), 'patient_phenotype_links': sum(len(a['phenotype']) for d in predictions for a in d['association']), 'fit_seconds': fitted - started, 'predict_seconds': time.perf_counter() - fitted, 'lexicon': model.stats, 'inputs': {name: {'path': str(Path(getattr(args, name)).resolve()), 'sha256': sha256(getattr(args, name))} for name in ['train', 'input', 'hpo']}, 'output': {'path': str(output.resolve()), 'bytes': output.stat().st_size, 'sha256': sha256(output)}, 'source_sha256': sha256(__file__)}
    (run / 'manifest.json').write_text(json.dumps(stats, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: stats[k] for k in ['method', 'training_documents', 'input_documents', 'entities', 'patient_phenotype_links', 'fit_seconds', 'predict_seconds']}, ensure_ascii=False))


if __name__ == '__main__': main()
