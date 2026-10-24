"""Map the immutable NCR head to current HPO using the frozen public policy.

Only explicit main() reads files. No model, examples, gold, or task data is read.
The OBO parser and mapping precedence follow the SHA-bound public mapper;
the NCR negative class is the literal string 'None', not PhenoTagger 'HP:None'.
"""
import argparse
from collections import Counter, defaultdict, deque
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import time


HPO_SHA256 = "a5092cbdf605f568403cf7380d9173014015692433b2cc631bc5c1b053876b1b"
HPO_BYTES = 11222341
COLUMNS_SHA256 = "cb8f61996d9f155b7e2c033cd5443d9cdb854bbed625d1bbb3b5d5ac60a84fc9"
COLUMNS_BYTES = 185423
COLUMNS_ORDER_SHA256 = "bb2edeb6af490408a31580d73f8959a34a2738884b1f79ab6328fdf4cd68e030"
ROOT_ID = "HP:0000118"
UPSTREAM_POLICY = {
    "reference_path": "/workspace/patientphex-runs/round2-20261023/research/public-phenotagger-reuse/build_label_map.py",
    "bytes": 2740,
    "sha256": "1b4a60a8a965a4704a3a78e52fa56e628fc855cf0781ec7b9c32b5cb3cd57e03",
    "use": "Copied OBO parsing, branch traversal and mapping precedence; not imported or executed at runtime.",
    "only_head_policy_difference": "NCR literal None replaces PhenoTagger HP:None; no columns filtered or reordered."
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def ref(path, data=None):
    path = Path(path).resolve()
    if data is None:
        data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": sha256(data)}


def parse_ontology(text):
    """Same frozen OBO parser/branch traversal; count checked by main()."""
    terms = {}
    cur = None
    for line in text.splitlines():
        if line == '[Term]':
            cur = {'parents': [], 'alts': [], 'replaced_by': []}
        elif line.startswith('['):
            cur = None
        elif cur is not None:
            if line.startswith('id: '):
                cur['id'] = line[4:].strip()
                terms[cur['id']] = cur
            elif line.startswith('is_a: '):
                cur['parents'].append(line[6:].split()[0])
            elif line.startswith('alt_id: '):
                cur['alts'].append(line[8:].strip())
            elif line == 'is_obsolete: true':
                cur['obsolete'] = True
            elif line.startswith('replaced_by: '):
                cur['replaced_by'].append(line[13:].split()[0])
    children = defaultdict(list)
    for k, v in terms.items():
        for par in v['parents']:
            children[par].append(k)
    seen = set()
    q = deque(['HP:0000118'])
    while q:
        k = q.popleft()
        if k in seen:
            continue
        seen.add(k)
        q.extend(children[k])
    allowed = {k for k in seen if not terms[k].get('obsolete')}
    alts = {a: k for k, v in terms.items() if not v.get('obsolete') for a in v['alts']}
    return terms, allowed, alts


def map_label(label, terms, allowed, alts):
    """Return (current_label or None, reason), without recursive guessing."""
    target = None
    why = 'outside_current_branch_or_unresolved'
    if label == 'None':
        why = 'none_class'
    elif label in allowed:
        target = label
        why = 'same_valid_current_id'
    elif alts.get(label) in allowed:
        target = alts[label]
        why = 'official_alt_id'
    elif (label in terms and terms[label].get('obsolete') and
          len(terms[label]['replaced_by']) == 1 and
          terms[label]['replaced_by'][0] in allowed):
        target = terms[label]['replaced_by'][0]
        why = 'single_official_replaced_by'
    return target, why


def publish_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    data = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')
    with temporary.open('xb') as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.link(str(temporary), str(path))
    temporary.unlink()
    return ref(path, data)


def build(columns_path, hpo_path, output_path):
    started = time.monotonic()
    columns_path = Path(columns_path).resolve()
    hpo_path = Path(hpo_path).resolve()
    output_path = Path(output_path).resolve()
    summary_path = output_path.with_name(output_path.name + '.summary.json')
    for path in (output_path, summary_path):
        require(path not in (columns_path, hpo_path, Path(__file__).resolve()),
                'Output would replace an input/source')
        require(not path.exists() and not path.with_name(path.name + '.tmp').exists(),
                'Refusing to overwrite existing output or temporary file: ' + str(path))
    columns_bytes = columns_path.read_bytes()
    hpo_bytes = hpo_path.read_bytes()
    require(len(columns_bytes) == COLUMNS_BYTES and sha256(columns_bytes) == COLUMNS_SHA256,
            'NCR columns file differs from frozen preparation output')
    require(len(hpo_bytes) == HPO_BYTES and sha256(hpo_bytes) == HPO_SHA256,
            'HPO bytes differ from the frozen 2026-06-23 ontology')
    head = json.loads(columns_bytes)
    require(isinstance(head, dict) and set(head) == {'labels', 'root_id'}, 'Unexpected columns schema')
    labels = head['labels']
    require(isinstance(labels, list) and len(labels) == 14261 and
            all(isinstance(label, str) for label in labels) and
            len(set(labels)) == 14261 and labels[-1] == 'None' and
            all(re.fullmatch(r'HP:[0-9]{7}', label) for label in labels[:-1]),
            'Expected 14260 distinct raw NCR HPO concepts and final literal None')
    require(head['root_id'] == ROOT_ID and ROOT_ID in labels[:-1], 'NCR root changed')
    order_sha = sha256('\n'.join(labels).encode('utf-8'))
    require(order_sha == COLUMNS_ORDER_SHA256, 'Original NCR column order changed')
    terms, allowed, alts = parse_ontology(hpo_bytes.decode('utf-8'))
    require(len(allowed) == 19120, 'Current allowed HPO branch must contain exactly 19120 terms')
    rows = []
    for column, label in enumerate(labels):
        target, why = map_label(label, terms, allowed, alts)
        rows.append({'column': column, 'original_label': label,
                     'current_label': target, 'reason': why})
    require([row['original_label'] for row in rows] == labels and
            [row['column'] for row in rows] == list(range(14261)),
            'Mapping changed raw model columns')
    require(rows[-1] == {'column': 14260, 'original_label': 'None',
                         'current_label': None, 'reason': 'none_class'},
            'NCR None column handling changed')
    counts = dict(Counter(row['reason'] for row in rows))
    refs = {'source': ref(__file__), 'columns': ref(columns_path, columns_bytes),
            'hpo': ref(hpo_path, hpo_bytes)}
    summary = {
        'status': 'complete_ontology_only_original_NCR_head_mapping',
        'scope': 'Public ontology and immutable NCR head only; apply after native classification/decoding. No head filtering, reordering or renormalization before top1.',
        'refs': refs, 'upstream_policy_reference': UPSTREAM_POLICY,
        'columns': len(rows), 'HPO_classes': len(rows) - 1, 'None_index': 14260,
        'root_id': head['root_id'], 'original_order_sha256': order_sha,
        'HPO_sha256': HPO_SHA256, 'columns_file_sha256': COLUMNS_SHA256,
        'current_valid_branch_count': len(allowed), 'counts': counts,
        'mapped_current_ids': len({row['current_label'] for row in rows if row['current_label']}),
        'raw_columns_unchanged': True, 'model_or_NN_run': False,
        'task_examples_or_gold_read': False, 'fit_or_quality_run': False,
        'finished_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'mapping_seconds_before_serialization': time.monotonic() - started
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mapping_ref = publish_json(output_path, dict(summary, mapping=rows))
    summary['mapping_file'] = mapping_ref
    summary['total_seconds_before_summary_write'] = time.monotonic() - started
    summary_ref = publish_json(summary_path, summary)
    print(json.dumps({'status': summary['status'], 'mapping_file': mapping_ref,
                      'summary_file': summary_ref, 'counts': counts, 'columns': len(rows)}))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--columns', required=True)
    parser.add_argument('--hpo', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    build(args.columns, args.hpo, args.output)


if __name__ == '__main__':
    main()
