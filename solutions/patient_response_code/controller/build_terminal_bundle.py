"""Build one private terminal bundle; clinical responses stay outside Git.

Only explicitly selected original artifacts enter the bundle. The source bridge
capability, transport intents, held gold, and model/state arrays are excluded.
"""
from pathlib import Path
import argparse, hashlib, json, os, tarfile


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--controller', type=Path, required=True)
    p.add_argument('--inference', type=Path, required=True)
    p.add_argument('--probe', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    C, I, P, D = a.controller, a.inference, a.probe, a.output
    S = Path(__file__).resolve().parents[1]
    assert json.loads((C/'campaign-complete.json').read_text())['status'] == 'TWO_COMPLETE_EPOCHS_32_STEPS_DURABLE_ACKS_VERIFIED'
    assert json.loads((I/'quality/result.json').read_text())['status'] == 'COMPLETE_FIXED_RESPONSE_CODE_F1'
    assert json.loads((I/'terminal-manifest.json').read_text())['status'] == 'ACTUAL_UNIQUE_FULL80_TERMINAL_COLLECTED'
    D.mkdir(exist_ok=False)
    paths = {}

    def add(name, path):
        path = Path(path)
        assert path.is_file() and not path.is_symlink() and name not in paths
        assert not Path(name).is_absolute() and '..' not in Path(name).parts
        assert path.name != 'access-token' and path.suffix not in {'.pt', '.safetensors', '.bin'}
        paths[name] = path

    for path in S.rglob('*'):
        if path.is_file() and path.suffix in {'.py', '.md', '.json', '.diff'} and '__pycache__' not in path.parts:
            add('source/'+path.relative_to(S).as_posix(), path)
    for n in ['controller-plan.json', 'campaign.json', 'campaign-complete.json',
              'entry.json', 'bootstrap-entry.json', 'target-deployment.json',
              'local-watch-intent.json', 'local-watch-events.jsonl', 'local-watch-failure.json']:
        add('controller/'+n, C/n)
    for epoch in [1, 2]:
        for suffix in ['job', 'start-intent', 'state', 'export-ack', 'export-verification', 'source-ack-copy']:
            n = f'epoch-{epoch:03d}-{suffix}.json'
            add('controller/'+n, C/n)
    # Independent monitor recovery receipts are local metadata, never capability-bearing.
    for path in (C/'watch-recovery-1').iterdir():
        if path.is_file() and path.suffix in {'.json', '.jsonl'}:
            add('controller/watch-recovery-1/'+path.name, path)
    for n in ['terminal.zip', 'manifest.json', 'checks.json']:
        add('training-terminal/'+n, C/'training-terminal'/n)
    for n in ['plan.json', 'entry.json', 'deployment-manifest.json', 'start-intent.json',
              'job.json', 'terminal-manifest.json', 'terminal.zip']:
        add('inference/'+n, I/n)
    for path in (I/'quality').rglob('*'):
        if path.is_file():
            add('quality/'+path.relative_to(I/'quality').as_posix(), path)
    # Probe collection names are intentionally supplied via its immutable manifest.
    probe_collection = json.loads((P/'collection.json').read_text())
    assert probe_collection['all_actual_SHA_verified']
    add('probe/collection.json', P/'collection.json')
    add('probe/probe-output.zip', probe_collection['archive']['path'])
    add('final-review-entry.json', C.parent/'final-review-entry.json')
    refs = {n: dict(bytes=q.stat().st_size, sha256=hashlib.sha256(q.read_bytes()).hexdigest()) for n, q in paths.items()}
    archive = D/'terminal.tar.gz'
    with tarfile.open(archive, 'x:gz') as t:
        for n, q in sorted(paths.items()):
            t.add(q, arcname=n, recursive=False)
    with archive.open('rb') as f:
        os.fsync(f.fileno())
    with tarfile.open(archive) as t:
        assert set(t.getnames()) == set(refs)
        for n, r in refs.items():
            b = t.extractfile(n).read()
            assert len(b) == r['bytes'] and hashlib.sha256(b).hexdigest() == r['sha256']
    manifest = dict(archive=dict(path=str(archive), bytes=archive.stat().st_size,
                                sha256=hashlib.sha256(archive.read_bytes()).hexdigest()),
                    members=refs, private_clinical_responses=True, no_held_gold=True,
                    no_model_optimizer_arrays=True, capability_excluded=True,
                    full_checkpoint_storage='Both complete epoch states are separately .100 durable with exact ACKs')
    with (D/'archive-manifest.json').open('x') as f:
        json.dump(manifest, f, indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
    print(json.dumps(dict(archive=manifest['archive'], members=len(refs))))


if __name__ == '__main__':
    main()
