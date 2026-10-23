"""Public phrase -> preferred-HPO-name scores; no model, labels, or fitting.

The three added columns are an arithmetic mean over unique atomic IDs.  They
are features, not a probability that an entire compound annotation is correct.
All-zero denotes an unencoded NIL/out-of-branch candidate; real softmax rows
must sum to one.  Missing scores for a valid pair are errors, never zeros.
"""
import collections
import gzip
import hashlib
import json
import math
from pathlib import Path

FEATURES = ['phenobert_not_match', 'phenobert_related', 'phenobert_match']
CLASSES = ['Not Match', 'Related', 'Match']
WEIGHT_SHA = '353fc58889b2e59ddb9bf845c037283247268812ff3959cead118f35db90bea8'
HPO_SHA = 'a5092cbdf605f568403cf7380d9173014015692433b2cc631bc5c1b053876b1b'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def checked(ref):
    path = Path(ref['path'])
    if not path.is_absolute() or sha(path) != ref['sha256']:
        raise ValueError(('artifact_identity', str(path)))
    if 'bytes' in ref and path.stat().st_size != ref['bytes']:
        raise ValueError(('artifact_size', str(path)))
    return path


def read_rows(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            yield json.loads(line)


def preferred_names(path):
    """Active designated phenotype branch, including its named root."""
    terms, item = {}, None
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
    pending, branch = ['HP:0000118'], set()
    while pending:
        current = pending.pop()
        if current in branch: continue
        branch.add(current)
        pending.extend(children[current])
    return {hid: terms[hid]['name'] for hid in branch
            if not terms[hid].get('obsolete') and 'name' in terms[hid]}


def probabilities(values):
    if not isinstance(values, list) or len(values) != 3:
        raise ValueError('three ordered class probabilities required')
    if any(type(x) not in (int, float) or not math.isfinite(x) or not 0 <= x <= 1
           for x in values):
        raise ValueError('finite numeric probabilities in [0,1] required')
    if not math.isclose(math.fsum(values), 1., rel_tol=0., abs_tol=2e-6):
        raise ValueError('real model probabilities must sum to one')
    return tuple(float(x) for x in values)


def atomic_ids(identifier, names):
    if not isinstance(identifier, str) or not identifier:
        raise ValueError('nonempty literal identifier required')
    atoms = sorted(set(identifier.split(';')))
    if any(not atom for atom in atoms):
        raise ValueError('empty compound atom')
    valid = [atom for atom in atoms if atom in names]
    if valid and len(valid) != len(atoms):
        raise ValueError(('mixed valid/unencoded compound', identifier))
    return valid


class PairFeatures:
    def __init__(self, names, requests, scores):
        self.names = names
        self.by_atomic_pair = {}
        self.model_scores = {}
        self.counts = collections.Counter()
        for row in scores:
            pair_key = row['pair_key']
            if not isinstance(pair_key, str) or len(pair_key) != 64:
                raise ValueError('invalid native input key')
            if pair_key in self.model_scores:
                raise ValueError('duplicate model output key')
            self.model_scores[pair_key] = probabilities(row['probabilities'])
        used = set()
        for row in requests:
            text, hid, name, pair_key = [row[k] for k in ('text', 'hpo_id', 'name', 'pair_key')]
            if not isinstance(text, str) or not text or self.names.get(hid) != name:
                raise ValueError('request must retain raw surface and current preferred name')
            key = (text, hid)
            if key in self.by_atomic_pair:
                raise ValueError('duplicate raw surface / atomic ID request')
            if pair_key not in self.model_scores:
                raise ValueError('valid requested pair has no model score')
            used.add(pair_key)
            self.by_atomic_pair[key] = self.model_scores[pair_key]
        if used != self.model_scores.keys():
            raise ValueError('unrequested model outputs')

    def features(self, candidate):
        if set(FEATURES) & candidate.keys():
            raise ValueError('pair columns already present: cannot overwrite')
        text = candidate['text']
        if not isinstance(text, str) or not text:
            raise ValueError('nonempty unmodified candidate surface required')
        atoms = atomic_ids(candidate['identifier'], self.names)
        self.counts['candidate_visits'] += 1
        if not atoms:
            self.counts['unencoded_visits'] += 1
            values = (0., 0., 0.)
        else:
            rows = [self.by_atomic_pair[text, atom] for atom in atoms]
            values = tuple(math.fsum(row[i] for row in rows) / len(rows) for i in range(3))
            self.counts['encoded_visits'] += 1
            self.counts['compound_visits'] += len(atoms) > 1
        return dict(zip(FEATURES, values))


def load(config):
    """Bind one already computed public cache.  This function never forwards."""
    if config['class_order'] != CLASSES or config['weight_sha256'] != WEIGHT_SHA:
        raise ValueError('wrong public model/class ordering')
    if config['direction'] != 'original_span_text -> current_preferred_name':
        raise ValueError('pair direction must not be reversed')
    if config['compound'] != 'arithmetic_mean_over_unique_atomic_ids':
        raise ValueError('unexpected compound aggregation')
    if config['unencoded'] != 'three_zeros_no_model_request':
        raise ValueError('unexpected unencoded policy')
    ontology = checked(config['hpo'])
    if config['hpo']['sha256'] != HPO_SHA:
        raise ValueError('wrong designated ontology')
    names = preferred_names(ontology)
    if len(names) != 19120:
        raise ValueError('current active phenotype branch mismatch')
    manifest_path = checked(config['cache_manifest'])
    manifest = json.loads(manifest_path.read_text())
    for key in ('class_order', 'weight_sha256', 'direction'):
        if manifest[key] != config[key]: raise ValueError(('cache_contract', key))
    if manifest['public_model_task_fit'] is not False:
        raise ValueError('task-fitted weights are not this public feature control')
    for key in ('atomic_request', 'scores', 'hpo'):
        if manifest[key] != config[key]: raise ValueError(('cache_artifact', key))
    return PairFeatures(names, read_rows(checked(config['atomic_request'])),
                        read_rows(checked(config['scores'])))
