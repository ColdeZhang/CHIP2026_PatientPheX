"""Freeze an already-observed complete two-HTTP cleanup; no remote/signals."""
import argparse
import hashlib
import json
import os
from pathlib import Path

B = Path('/workspace/patientphex-runs/round2-20261023/research/grounded-generation-full64-v1')


def ref(p):
    p = Path(p).resolve(); data = p.read_bytes()
    return dict(path=str(p), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--observation', type=Path, required=True)
    a = ap.parse_args(); directory = B / 'export-guardian-v1'
    assert a.observation.resolve().parent == directory
    actual = json.loads(a.observation.read_text())
    campaign = json.loads((B / 'formal-v1/campaign-complete.json').read_text())
    assert campaign['status'] == 'TWO_COMPLETE_EPOCHS_DURABLE_ACKS_VERIFIED'
    assert actual['complete.json']['closed_epochs'] == [1, 2]
    assert actual['complete.json']['no_training_process_touched']
    assert actual['complete.json']['no_model_or_export_restarted']
    source = Path(__file__).parent / 'export_guardian.py'
    assert actual['started.json']['source_sha256'] == ref(source)['sha256']
    assert actual['started.json']['GPU_hidden'] and actual['started.json']['models_or_HTTP_start_or_export_calls'] == 0
    for epoch in (1, 2):
        closed = actual['epoch-%03d-closed.json' % epoch]
        if closed.get('status') == 'already_exited_no_signal':
            continue
        identity = actual['epoch-%03d-identity.json' % epoch]['child']
        child = closed['child']
        for key in ['pid', 'start_ticks', 'uid', 'pgid', 'sid', 'cmdline']:
            assert child[key] == identity[key]
        assert child['uid'] == 1045 and child['pid'] == child['pgid'] == child['sid']
        assert Path(child['cmdline'][2]).name == 'export_checkpoint_server.py'
        assert closed['signals'] in [[], ['TERM'], ['TERM', 'KILL']]
        assert closed['exit_observed'] and (closed['after'] is None or closed['after']['state'] in ['Z', 'X'])
    result = dict(schema='grounded-two-owned-HTTP-closure-actual-v1', observation=ref(a.observation),
                  source=ref(source), collector_source=ref(__file__),
                  job=ref(directory / 'job.json'), campaign_complete=ref(B / 'formal-v1/campaign-complete.json'),
                  actual=actual, both_original_HTTP_children_exited=True,
                  collection_only_no_remote_or_signals=True)
    path = directory / 'terminal-collection.json'
    with path.open('x') as f:
        json.dump(result, f, indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
    current = directory
    while True:
        fd = os.open(current, os.O_RDONLY | os.O_DIRECTORY); os.fsync(fd); os.close(fd)
        if current.parent == current: break
        current = current.parent
    assert json.loads(path.read_text()) == result
    print(json.dumps(ref(path)))


if __name__ == '__main__':
    main()
