"""One fixed-policy CPU fold migration from a frozen input-only native cache."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import assemble_public_phenotagger as assembly
import run as baseline


def now():
    return datetime.now(timezone.utc).isoformat()


def index(rows):
    result = {str(d['pmc_id']): d for d in rows}
    assert len(rows) == len(result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--gold', required=True)
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    for entry in plan['inputs'].values():
        assert assembly.sha(entry['path']) == entry['sha256'], entry['path']
    for name, path in [('entry', __file__), ('assembly', assembly.__file__), ('baseline', baseline.__file__)]:
        assert assembly.sha(path) == plan['sources'][name]
    paths = {k: Path(v['path']) for k, v in plan['inputs'].items()}
    previous = json.loads(paths['previous_assembly_plan'].read_text())
    assert plan['policy'] == previous['policy']
    assert plan['sources']['assembly'] == previous['assembly_sources']['assembler']
    assert plan['sources']['baseline'] == previous['assembly_sources']['baseline']
    cache_manifest = json.loads(paths['cache_manifest'].read_text())
    # Explicit frozen JSON paths bind the cache provider's schema without
    # relabelling its train64 input hash as a direct outer-fold inference.
    for check in plan['cache_contract']:
        value = cache_manifest
        for part in check['keys']:
            value = value[part]
        assert value == check['expected'], check['keys']
    docs = assembly.read(paths['input'])
    target = index(docs)
    complete_input = index(assembly.read(paths['cache_input']))
    assert len(target) == 16 and len(complete_input) == 64
    assert set(target) <= set(complete_input)
    assert all(set(d) <= {'pmc_id', 'pmid', 'patient', 'full_text'} for d in complete_input.values())
    assert all(complete_input[key] == doc for key, doc in target.items())
    previous_docs = index(assembly.read(paths['previous_input']))
    assert set(target).isdisjoint(previous_docs), 'Migration must use the distinct planned fold'
    bases = index(assembly.read(paths['base']))
    assert set(bases) == set(target)
    rows = []
    bydoc = defaultdict(list)
    cache_rows = 0
    for line in paths['cache_entities'].read_text().splitlines(keepends=True):
        row = json.loads(line)
        key = str(row['pmc_id'])
        assert key in complete_input
        cache_rows += 1
        if key in target:
            rows.append(line)
            bydoc[key].append(row)
    out = Path(plan['output_dir'])
    out.mkdir(parents=True, exist_ok=False)
    (out / 'canonical-subset.jsonl').write_text(''.join(rows))
    projection = {'created_at_utc': now(), 'gold_read': False, 'neural_inference_performed': False,
        'operation': 'exact original cache JSONL lines selected by input pmc_id; input documents equal byte-decoded JSON',
        'parent_cache_manifest_sha256': assembly.sha(paths['cache_manifest']),
        'parent_input_sha256': assembly.sha(paths['cache_input']),
        'parent_entities_sha256': assembly.sha(paths['cache_entities']),
        'target_input_sha256': assembly.sha(paths['input']), 'target_document_ids': sorted(target),
        'parent_rows': cache_rows, 'selected_rows': len(rows), 'documents': len(target),
        'subset_sha256': assembly.sha(out / 'canonical-subset.jsonl')}
    assembly.save(out / 'projection-manifest.json', projection)
    allowed = baseline.load_hpo(paths['hpo'])[1]
    output = []
    counts = Counter()
    for doc in docs:
        pred, count = assembly.assemble_document(doc, bases[str(doc['pmc_id'])], bydoc[str(doc['pmc_id'])], allowed)
        output.append(pred)
        counts.update(count)
    assert counts['native_candidates'] == len(rows)
    (out / 'prediction.jsonl').write_text(''.join(json.dumps(d, ensure_ascii=False) + '\n' for d in output))
    manifest = {'created_at_utc': now(), 'gold_read': False, 'plan_sha256': assembly.sha(args.plan),
        'sources': plan['sources'], 'policy': plan['policy'], 'counts': dict(counts),
        'all_base_entities_preserved': True, 'all_other_fields_equal': True,
        'projection_manifest_sha256': assembly.sha(out / 'projection-manifest.json'),
        'base_sha256': assembly.sha(paths['base']), 'prediction_sha256': assembly.sha(out / 'prediction.jsonl')}
    assembly.save(out / 'assembly-manifest.json', manifest)
    assembly.save(out / 'prediction-freeze.json', {'frozen_at_utc': now(), 'gold_read': False,
        'prediction_sha256': manifest['prediction_sha256'], 'manifest_sha256': assembly.sha(out / 'assembly-manifest.json'),
        'plan_sha256': assembly.sha(args.plan), 'base_sha256': manifest['base_sha256']})
    jobs = []
    def run(stage, argv):
        started = time.monotonic()
        job = {'stage': stage, 'argv': argv, 'started_at_utc': now()}
        with (out / (stage + '.stdout')).open('x') as stdout, (out / (stage + '.stderr')).open('x') as stderr:
            process = subprocess.run(argv, stdout=stdout, stderr=stderr, timeout=120,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1'))
        job.update(ended_at_utc=now(), seconds=time.monotonic()-started, return_code=process.returncode)
        jobs.append(job)
        (out / 'jobs.json').write_text(json.dumps(jobs, indent=2) + '\n')
        assert process.returncode == 0, stage
    run('format', [sys.executable, str(paths['validator']), '--input', str(paths['input']),
        '--pred', str(out/'prediction.jsonl'), '--hpo', str(paths['hpo']), '--output', str(out/'validation.json')])
    run('evaluate', [sys.executable, str(paths['evaluator']), '--gold', args.gold, '--pred', str(out/'prediction.jsonl'),
        '--body-scope', 'all', '--no-id-doc', 'text', '--output', str(out/'metrics.json')])
    old = json.loads(paths['baseline_metrics'].read_text())
    new = json.loads((out/'metrics.json').read_text())
    valid = json.loads((out/'validation.json').read_text())
    assert old['gold_sha256'] == new['gold_sha256']
    assert old['prediction_sha256'] == manifest['base_sha256']
    assert old['metrics']['Score'] == plan['baseline_score']
    assert old['configuration'] == new['configuration']
    assert old['evaluator_version'] == new['evaluator_version'] == 'patientphex-public-approx-v2'
    assert old['metrics']['micF1'] == new['metrics']['micF1'] and old['metrics']['macF1'] == new['metrics']['macF1']
    assert old['counts']['relation'] == new['counts']['relation']
    result = {'complete': True, 'ended_at_utc': now(), 'plan_sha256': assembly.sha(args.plan),
        'prediction_sha256': manifest['prediction_sha256'], 'counts': new['counts'],
        'baseline_metrics': old['metrics'], 'metrics': new['metrics'],
        'delta': {k: new['metrics'][k]-old['metrics'][k] for k in new['metrics']},
        'count_delta': {kind: {k: new['counts'][kind][k]-old['counts'][kind][k] for k in ['tp','fp','fn']}
            for kind in new['counts']}, 'assembly_counts': dict(counts),
        'format': {'errors': valid['error_count'], 'warnings': valid['warning_count']},
        'artifacts': {p.name: {'sha256': assembly.sha(p), 'bytes': p.stat().st_size}
            for p in out.iterdir() if p.is_file() and p.suffix not in {'.stdout','.stderr'}}}
    assembly.save(out / 'summary.json', result)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
