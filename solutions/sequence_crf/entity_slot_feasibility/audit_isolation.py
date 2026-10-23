#!/usr/bin/env python3
"""Document-ID-only source-chain check; annotation fields are never parsed."""
from pathlib import Path
import argparse
import hashlib
import json
import re


def ref(path):
    path = Path(path)
    return dict(path=str(path), bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def pmc_ids(path):
    # Unlike json.loads(document), this never constructs entity/association fields.
    found = []
    for line in Path(path).read_text().splitlines():
        if not line.strip(): continue
        match = re.search(r'"pmc_id"\s*:\s*"([^"]+)"', line)
        assert match
        found.append(match.group(1))
    assert len(found) == len(set(found))
    return set(found)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    c = a.root/'ontology_rules/ce-oof-control'
    held = c/'fold1/input16.jsonl'
    held_ids = pmc_ids(held)
    assert len(held_ids) == 16
    unsafe = []
    for fold in [2, 3]:
        fitpath = a.root/f'sequence_crf/public-bioformer-selector-control/fold{fold}/fitted/manifest.json'
        fit = json.loads(fitpath.read_text())
        reference = Path(fit['arguments']['reference'])
        assert ref(reference)['sha256'] == fit['old_reference_sha256']
        old = json.loads(reference.read_text())
        train, inp = Path(old['config']['train']), Path(old['config']['input'])
        assert ref(train)['sha256'] == old['inputs']['train']
        assert ref(inp)['sha256'] == old['inputs']['input']
        ts, ins = pmc_ids(train), pmc_ids(inp)
        assert len(ts) == 48 and len(ins) == 16 and not ts & ins
        unsafe.append(dict(kind='old fourfold HGB', target_fold=fold,
            fit_manifest=ref(fitpath), reference=ref(reference), train=ref(train), input=ref(inp),
            train_ids=sorted(ts), own_target_intersection=[],
            new_F1_intersection=sorted(ts & held_ids), count=len(ts & held_ids)))
    nestedpath = a.root/'ontology_rules/nested-crf/confidence-calibration-manifest.json'
    for fold, spec in enumerate(json.loads(nestedpath.read_text()), 1):
        if fold == 1: continue
        train, inp = Path(spec['generator_train']), Path(spec['input'])
        ts, ins = pmc_ids(train), pmc_ids(inp)
        assert len(ts) == 32 and len(ins) == 16 and not ts & ins
        unsafe.append(dict(kind='old outer0 nested candidate generator', target_fold=fold,
            reference=ref(nestedpath), train=ref(train), input=ref(inp),
            train_ids=sorted(ts), own_target_intersection=[],
            new_F1_intersection=sorted(ts & held_ids), count=len(ts & held_ids)))
    safe = []
    for spec in json.loads((c/'fold1/training-origins.json').read_text()):
        train, inp = c/spec['generator_train'], c/spec['input']
        ts, ins = pmc_ids(train), pmc_ids(inp)
        assert len(ts) == len(ins) == 16 and not ts & ins
        assert not (ts | ins) & held_ids
        safe.append(dict(train=ref(train), input=ref(inp), manifest=ref(c/spec['candidate_manifest']),
            train_ids=sorted(ts), input_ids=sorted(ins), own_target_intersection=[],
            new_F1_intersection=[], count=0))
    output = dict(source=ref(__file__), new_heldout=ref(held), new_heldout_ids=sorted(held_ids),
        risk_examples=unsafe, replacement_chains=safe,
        interpretation='Preparation-stage risk prevented: old sources are valid for their original targets; they must not be reused as unseen-F1 resources for a new F2+F3 training task.',
        no_annotation_fields_parsed=True, no_model_or_quality_run=True)
    assert all(row['count'] == 16 for row in unsafe)
    assert not a.output.exists()
    a.output.write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps(dict(output=ref(a.output), risk_counts=[x['count'] for x in unsafe],
                          replacement_counts=[x['count'] for x in safe])))


if __name__ == '__main__': main()
