#!/usr/bin/env python3
"""Apply the saved corrected LR to supplied entities, optionally add raw Sap.

Use without --sap-candidates after GLiNER entity augmentation, to retain the
original full patient fallback associations without fitting anything.
"""
import argparse
import json
from pathlib import Path
import time
from load_predict import (SEQUENCE_SHA, ONTOLOGY_SHA, augment, load_module,
                          load_pipeline, read_jsonl, ref, sha, write_jsonl)


def main():
    ap = argparse.ArgumentParser()
    for name in ['input', 'input-sha256', 'base', 'base-sha256', 'pipeline', 'crf-model',
                 'sequence-source', 'ontology-source', 'output']:
        ap.add_argument('--'+name, required=True)
    ap.add_argument('--sap-candidates')
    ap.add_argument('--sap-candidates-sha256')
    args = ap.parse_args()
    started = time.monotonic()
    assert sha(args.input) == args.input_sha256 and sha(args.base) == args.base_sha256
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    assert not out.exists()
    sequence = load_module('frozen_sequence', args.sequence_source, SEQUENCE_SHA)
    ontology = load_module('frozen_ontology', args.ontology_source, ONTOLOGY_SHA)
    _, _, vectorizer, relation, _, _ = load_pipeline(sequence, args.pipeline, args.crf_model)
    docs = read_jsonl(args.input)
    assert all('entities' not in d and 'association' not in d for d in docs)
    bases = {str(d['pmc_id']): d for d in read_jsonl(args.base)}
    assert set(bases) == {str(d['pmc_id']) for d in docs}
    raw = {}
    if args.sap_candidates:
        assert sha(args.sap_candidates) == args.sap_candidates_sha256
        raw = {str(d['pmc_id']): d['candidates'] for d in read_jsonl(args.sap_candidates)}
        assert set(raw) == set(bases)
    result = [augment(sequence, ontology, d, bases[str(d['pmc_id'])], raw.get(str(d['pmc_id']), []),
                      vectorizer, relation) for d in docs]
    write_jsonl(out, result)
    manifest = {'config': vars(args), 'source': ref(__file__), 'implementation': ref(Path(__file__).with_name('load_predict.py')),
        'inputs': {k: ref(getattr(args, k)) for k in ['input', 'base', 'pipeline', 'crf_model', 'sap_candidates'] if getattr(args, k)},
        'dependencies': {k: ref(getattr(args, k)) for k in ['sequence_source', 'ontology_source']},
        'output': ref(out), 'documents': len(result), 'entities': sum(len(d['entities']) for d in result),
        'patient_relations': sum(len(a['phenotype']) for d in result for a in d['association']),
        'relation_threshold': .55, 'train_or_gold_read': False, 'fit_called': False, 'seconds': time.monotonic()-started}
    out.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest['output']), flush=True)


if __name__ == '__main__':
    main()
