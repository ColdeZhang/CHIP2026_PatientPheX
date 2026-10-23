"""Single CPU assemble/freeze/validate/evaluate endpoint, with immutable inputs."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from assemble_public_phenotagger import sha, save


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--gold', required=True)
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    for group in ['local_inputs', 'sources', 'native_terminal']:
        for entry in plan[group].values():
            assert sha(entry['path']) == entry['sha256'], entry['path']
    assert sha(__file__) == plan['sources']['endpoint']['sha256']
    root = Path(plan['artifacts']['root'])
    terminal = root / 'cpu-terminal'
    assert not (root / 'assembled').exists()
    terminal.mkdir(exist_ok=False)
    jobs = []
    def run(stage, argv):
        started = time.monotonic()
        item = {'stage': stage, 'argv': argv, 'started_at_utc': now()}
        with (terminal / (stage + '.stdout')).open('x') as out, (terminal / (stage + '.stderr')).open('x') as err:
            p = subprocess.run(argv, stdout=out, stderr=err, timeout=120,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1'))
        item.update(ended_at_utc=now(), seconds=time.monotonic()-started, return_code=p.returncode)
        jobs.append(item)
        (terminal / 'jobs.json').write_text(json.dumps(jobs, indent=2) + '\n')
        assert p.returncode == 0, stage
    out = root / 'assembled'
    run('assemble', [sys.executable, str(Path(__file__).with_name('assemble_public_phenotagger.py')),
        '--plan', args.plan, '--entities', plan['native_terminal']['entities']['path'],
        '--native-manifest', plan['native_terminal']['manifest']['path'], '--output-dir', str(out)])
    freeze = json.loads((out / 'prediction-freeze.json').read_text())
    assert freeze['gold_read'] is False
    assert freeze['prediction_sha256'] == sha(out / 'prediction.jsonl')
    assert freeze['manifest_sha256'] == sha(out / 'manifest.json')
    assert freeze['base_sha256'] == plan['local_inputs']['base']['sha256']
    save(terminal / 'before-evaluation-freeze.json', {'verified_at_utc': now(), 'gold_read': False,
         'prediction_freeze_sha256': sha(out / 'prediction-freeze.json'),
         'prediction_sha256': freeze['prediction_sha256'], 'assembly_plan_sha256': sha(args.plan)})
    run('format', [sys.executable, plan['sources']['format_validator']['path'],
        '--input', plan['local_inputs']['input']['path'], '--pred', str(out / 'prediction.jsonl'),
        '--hpo', plan['local_inputs']['hpo']['path'], '--output', str(out / 'validation.json')])
    run('evaluate', [sys.executable, plan['sources']['evaluator']['path'], '--gold', args.gold,
        '--pred', str(out / 'prediction.jsonl'), '--body-scope', 'all', '--no-id-doc', 'text',
        '--output', str(out / 'metrics.json')])
    base_entry = plan['comparison']['baseline_metrics']
    assert sha(base_entry['path']) == base_entry['sha256']
    base = json.loads(Path(base_entry['path']).read_text())
    new = json.loads((out / 'metrics.json').read_text())
    validation = json.loads((out / 'validation.json').read_text())
    assert base['prediction_sha256'] == plan['local_inputs']['base']['sha256']
    assert base['gold_sha256'] == new['gold_sha256']
    assert base['evaluator_version'] == new['evaluator_version'] == 'patientphex-public-approx-v2'
    assert base['configuration'] == new['configuration']
    assert new['metrics']['micF1'] == base['metrics']['micF1']
    assert new['metrics']['macF1'] == base['metrics']['macF1']
    assert new['counts']['relation'] == base['counts']['relation']
    result = {'complete': True, 'ended_at_utc': now(), 'plan_sha256': sha(args.plan),
        'prediction_sha256': sha(out / 'prediction.jsonl'), 'freeze_sha256': sha(out / 'prediction-freeze.json'),
        'baseline_metrics': base['metrics'], 'metrics': new['metrics'],
        'delta': {k: new['metrics'][k]-base['metrics'][k] for k in new['metrics']},
        'counts': new['counts'], 'count_delta': {kind: {k: new['counts'][kind][k]-base['counts'][kind][k]
            for k in ['tp', 'fp', 'fn']} for kind in new['counts']},
        'format': {'errors': validation['error_count'], 'warnings': validation['warning_count']},
        'artifacts': {str(p.relative_to(root)): {'sha256': sha(p), 'bytes': p.stat().st_size}
            for p in [out/'prediction.jsonl', out/'manifest.json', out/'prediction-freeze.json',
                      out/'validation.json', out/'metrics.json', terminal/'before-evaluation-freeze.json', terminal/'jobs.json']}}
    save(terminal / 'summary.json', result)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
