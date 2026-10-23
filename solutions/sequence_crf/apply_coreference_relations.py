#!/usr/bin/env python3
"""Freeze three rules controls: original, full alias expansion, generated offsets only."""
import argparse
import copy
import json
from pathlib import Path
import run as base
import combine


def check_expansion(original, expanded):
    assert original['pmc_id'] == expanded['pmc_id']
    assert len(original['patient']) == len(expanded['patient'])
    restored = copy.deepcopy(expanded)
    for old, new, reset in zip(original['patient'], expanded['patient'], restored['patient']):
        assert old['patient_id'] == new['patient_id']
        assert new['mention'][:len(old['mention'])] == old['mention']
        reset['mention'] = old['mention']
    assert restored == original, 'Only appending patient mention spans is allowed'


def offset_anchors(original, expanded):
    anchors = set(combine.ontology.patient_anchors(original))
    for old, new in zip(original['patient'], expanded['patient']):
        for mention in new['mention'][len(old['mention']):]:
            start = int(mention['offset'])
            end = start + int(mention['length'])
            assert any(int(p['offset']) <= start < end <= int(p['offset'])+len(p['text'])
                       and p['text'][start-int(p['offset']):end-int(p['offset'])] == mention['text']
                       for p in original['full_text'])
            anchors.add((start, end, old['patient_id']))
    return sorted(anchors)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['original-input', 'expanded-input', 'entities', 'output-dir']:
        p.add_argument('--'+name, required=True)
    args = p.parse_args()
    original = base.read_jsonl(args.original_input)
    expanded = {str(d['pmc_id']): d for d in base.read_jsonl(args.expanded_input)}
    candidates = {str(d['pmc_id']): d for d in base.read_jsonl(args.entities)}
    assert {str(d['pmc_id']) for d in original} == set(expanded) == set(candidates)
    assert all('entities' not in d and 'association' not in d for d in original+list(expanded.values()))
    results = {name: [] for name in ['original', 'expanded-aliases', 'expanded-offsets']}
    counts = {name: 0 for name in ['original_anchors', 'expanded_alias_anchors', 'expanded_offset_anchors']}
    for old in original:
        pmc = str(old['pmc_id'])
        new = expanded[pmc]
        check_expansion(old, new)
        given = combine.ontology.patient_anchors(old)
        broadcast = combine.ontology.patient_anchors(new)
        grounded = offset_anchors(old, new)
        for key, anchors in zip(counts, [given, broadcast, grounded]):
            counts[key] += len(anchors)
        for name, doc, anchors in [('original', old, given), ('expanded-aliases', new, broadcast),
                                   ('expanded-offsets', new, grounded)]:
            prediction = copy.deepcopy(candidates[pmc])
            prediction['association'] = combine.assign(doc, prediction['entities'], anchors=anchors)
            results[name].append(prediction)
    directory = Path(args.output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for name, rows in results.items():
        path = directory / (name+'.pred.jsonl')
        path.write_text(''.join(json.dumps(d, ensure_ascii=False)+'\n' for d in rows))
        outputs[name] = {'path': str(path), 'sha256': base.sha(path)}
    manifest = {'configuration': vars(args), 'anchor_counts': counts, 'outputs': outputs,
                'files': {key: base.sha(getattr(args, key)) for key in ['original_input', 'expanded_input', 'entities']},
                'sources': {'driver': base.sha(__file__), 'combine': base.sha(combine.__file__),
                            'ontology': base.sha(combine.ontology.__file__), 'base': base.sha(base.__file__)},
                'scope': 'no labels read; fixed entities and identical association function; original aliases unchanged; generated offset-only versus existing alias expansion compared separately'}
    (directory/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest))


if __name__ == '__main__':
    main()
