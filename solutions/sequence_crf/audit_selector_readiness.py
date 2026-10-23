#!/usr/bin/env python3
"""Read-only source/cache audit for one proposed public-source selector control.

No fitting, inference, annotations, or quality evaluation. Existing aggregate
reports are reused; mutable source bytes are copied to immutable external files.
"""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ontology-source', required=True)
    p.add_argument('--runs', required=True)
    p.add_argument('--output-dir', required=True)
    a = p.parse_args()
    src, runs, out = map(Path, [a.ontology_source, a.runs, a.output_dir])
    out.mkdir(parents=True, exist_ok=True)
    assert not (out/'report.json').exists()
    research = runs/'research/public-phenotagger-reuse'
    paths = [src/n for n in ['calibrate.py', 'calibrate_multi.py', 'crf_confidence.py',
                            'context_calibrate.py', 'ce_oof_control.py']]
    paths += [src/'evidence/round2'/n for n in
              ['multi-fourfold-strategy-freeze.json', 'ce-oof-results.json']]
    paths += [research/'native-pipeline'/n for n in
              ['run_native_entities.py', 'full-inference-plan.json', 'gpu-native-smoke.json']]
    paths += [research/'review.md']
    sources = []
    for i, path in enumerate(paths):
        content = path.read_bytes()
        saved = out/'source-snapshots'/f'{i:02d}-{path.name}'
        saved.parent.mkdir(exist_ok=True)
        saved.write_bytes(content)
        sources.append(dict(path=str(path), snapshot=str(saved), sha256=sha(saved), bytes=len(content)))
    strategy = json.loads((src/'evidence/round2/multi-fourfold-strategy-freeze.json').read_text())
    ce = json.loads((src/'evidence/round2/ce-oof-results.json').read_text())
    histories = []
    for mode in strategy['results']:
        folds = []
        for item in mode['folds']:
            path = Path(item['prediction']).with_suffix('.manifest.json')
            m = json.loads(path.read_text())
            assert sha(path) == item['model_manifest_sha256']
            specs_path = Path(m['config']['training_manifest'])
            specs = json.loads(specs_path.read_text())
            origins = []
            for spec in specs:
                cm = json.loads(Path(spec['candidate_manifest']).read_text())
                crf = json.loads(Path(spec['crf_manifest']).read_text())
                assert cm['inputs']['train'] == sha(spec['generator_train'])
                assert cm['inputs']['input'] == sha(spec['input'])
                assert cm['output']['sha256'] == sha(spec['candidates'])
                assert crf['train_sha256'] == sha(spec['generator_train'])
                assert crf['input_sha256'] == sha(spec['input'])
                assert crf['output_sha256'] == sha(spec['crf'])
                origins.append(dict(candidate_manifest_sha256=sha(spec['candidate_manifest']),
                    crf_manifest_sha256=sha(spec['crf_manifest']),
                    spec_keys=sorted(spec), crf_confidence_available='crf_confidence' in spec))
            folds.append(dict(fold=item['fold'], manifest=str(path), manifest_sha256=sha(path),
                training_manifest=str(specs_path), training_manifest_sha256=sha(specs_path),
                training_documents=m['training_documents'], training_examples=m['training_examples'],
                positive=m['training_positive'], features=m['features'], actual_seconds=m['seconds'],
                source_sha256=m['source_sha256'], origins=origins))
        histories.append(dict(mode=mode['mode'], pooled_metrics=mode['pooled_metrics'], folds=folds))
    native = json.loads((research/'native-pipeline/full-inference-plan.json').read_text())
    result = dict(status='read_only_readiness; no new model or prediction',
        source_sha256=sha(__file__), source_snapshots=sources,
        existing_selector=dict(selected_mode=strategy['selected'], threshold=strategy['threshold'],
            iterations=strategy['hgb_iterations'], early_stopping=strategy['early_stopping'],
            selection_rule=strategy['selection_rule'], histories=histories),
        CE_previous_control={k: ce[k] for k in ['scope', 'campaign_seconds', 'base', 'CE', 'delta', 'count_deltas']},
        public_model=dict(repo=native['model_repo'], revision=native['revision'],
            weight_sha256=native['weight_sha256'], task_examples=native['train_examples_used_by_public_model_experiment'],
            policy=native['policy'], current_input_sha256=native['input_sha256'],
            current_counts=native['counts_expected'], current_budget_basis=native['budget_basis']),
        annotation_records_parsed=False, training_input_files_hashed_only=True,
        new_training=False, new_inference=False,
        note='Only metadata, source and previously completed aggregate metrics; no new gold evaluation.')
    (out/'report.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(report=str(out/'report.json'), sha256=sha(out/'report.json'),
                          source_files=len(sources), fold_configurations=sum(len(x['folds']) for x in histories))))


if __name__ == '__main__':
    main()
