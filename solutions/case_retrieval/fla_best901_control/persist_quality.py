"""Persist the ended fixed original901 four-flow FLA quality supplement; inherited NN/state stay in place."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys
import tarfile

MOUNT = '5a9955da-5240-41cc-9e52-05d3d8da8a3c'
REMOTE = 'research/fla-best901-originaldev-20261024/full234-v1/quality-evidence-v1'



def ref(path):
    path = Path(path).resolve(); data = path.read_bytes()
    return dict(path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def save(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, indent=2); handle.write('\n'); handle.flush(); os.fsync(handle.fileno())


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--review', type=Path, required=True); parser.add_argument('--review-sha256', required=True)
    parser.add_argument('--quality-dir', type=Path, required=True);parser.add_argument('--execution-dir', type=Path, required=True)
    parser.add_argument('--quality-plan', type=Path, required=True);parser.add_argument('--summary-sha256', required=True)
    args = parser.parse_args(); run = args.run.resolve(); out = run / 'quality-persistent-supplement'
    quality=args.quality_dir.resolve();execution=args.execution_dir.resolve()
    assert ref(quality / 'summary.json')['sha256'] == args.summary_sha256
    assert ref(args.review)['sha256'] == args.review_sha256
    terminal = json.loads((execution / 'terminal.json').read_bytes())
    assert terminal['status'] == 'COMPLETE_SINGLE_PIPELINE_QUALITY' and terminal['exit_code'] == 0 and not terminal['timeout_triggered']
    out.mkdir(exist_ok=False); source = Path(__file__).resolve().parent
    files = [p for d in (quality,execution) for p in d.rglob('*') if p.is_file()]
    files += [args.quality_plan.resolve(), run / 'collection.json', run / 'terminal-artifacts.manifest.json', args.review.resolve()]
    files += [source / name for name in ('gpu/collect_terminal.py', 'persist_quality.py', 'README.md',
        'quality-result-pointer.json', 'QUALITY-RESULT.md')]
    files += sorted((source/'quality').glob('*.py')) + [source/'quality/template.json',source/'quality/contract.json',source/'quality/static-check.json',source/'assembly/contract.json',source/'assembly/plan.json',source/'assembly/assemble.py',source/'assembly/reproduction-pointer.json']
    members = {p.relative_to('/workspace').as_posix(): p for p in files}
    assert len(members) == len(files) and all(p.is_file() and not p.is_symlink() for p in files)
    archive = out / ('FLA-original901-quality-'+args.summary_sha256[:8]+'.tar.gz')
    with tarfile.open(archive, 'w:gz') as pack:
        for name, path in sorted(members.items()): pack.add(path, arcname=name, recursive=False)
    with archive.open('rb') as stream: os.fsync(stream.fileno())
    inherited = json.loads((run / 'collection.json').read_bytes())
    manifest = dict(scope='Quality/source/review supplement only. Original fresh234 four-flow archive and frozen full64 native901, models and CPU assets inherited without duplication.',
        archive=ref(archive), members=[dict(path=name, bytes=ref(path)['bytes'], sha256=ref(path)['sha256'])
                                     for name, path in sorted(members.items())],
        review=ref(args.review), summary=ref(quality / 'summary.json'), inherited_original_NN_archive=inherited['archive'],
        inherited_original_NN_manifest=inherited['manifest'], mount_id=MOUNT, remote_directory=REMOTE,
        remote_archive=REMOTE + '/' + archive.name, remote_manifest=REMOTE + '/manifest.json', source=ref(__file__))
    save(out / 'manifest.json', manifest); metadata = ref(out / 'manifest.json')
    save(out / 'intent.json', dict(archive=manifest['archive'], manifest=metadata, remote_directory=REMOTE))
    sys.path.insert(0, '/workspace/CHIP2026_PatientPheX/scripts')
    from mcp_client import Client
    client = Client(); client.initialize()
    def call(name, label, **arguments):
        result = client.request('tools/call', dict(name=name, arguments=dict(mount_id=MOUNT, **arguments)))
        save(out / (label + '.json'), result); assert not result.get('isError')
        return result.get('structuredContent') or json.loads(result['content'][0]['text'])
    assert manifest['archive']['bytes'] < 200000
    call('write_file', 'upload-archive', path=manifest['remote_archive'], encoding='base64', create_parents=True,
         content=base64.b64encode(archive.read_bytes()).decode())
    call('write_file', 'upload-manifest', path=manifest['remote_manifest'], encoding='utf8', create_parents=True,
         content=(out / 'manifest.json').read_text())
    code = r'''
import pathlib,json,hashlib,tarfile,os
p=pathlib.Path('/home/zhangyuheng/projects/CHIP2026_PatientPheX')/REMOTE
mp=p/'manifest.json';assert mp.stat().st_size==MBYTES and hashlib.sha256(mp.read_bytes()).hexdigest()==MSHA
m=json.loads(mp.read_bytes());a=p/NAME;assert a.stat().st_size==m['archive']['bytes'] and hashlib.sha256(a.read_bytes()).hexdigest()==m['archive']['sha256']
with tarfile.open(a,'r:gz') as archive:
 actual=archive.getmembers();expected={row['path']:row for row in m['members']}
 assert len(actual)==len({row.name for row in actual})==len(expected) and {row.name for row in actual}==set(expected)
 for row in actual:
  assert row.isfile() and not pathlib.Path(row.name).is_absolute() and '..' not in pathlib.Path(row.name).parts
  data=archive.extractfile(row).read();r=expected[row.name];assert len(data)==r['bytes'] and hashlib.sha256(data).hexdigest()==r['sha256']
for q in (a,mp):
 with q.open('rb') as f:os.fsync(f.fileno())
for q in [p]+list(p.parents):
 fd=os.open(str(q),os.O_RDONLY);os.fsync(fd);os.close(fd)
assert hashlib.sha256(a.read_bytes()).hexdigest()==m['archive']['sha256']
print(json.dumps(dict(status='FLA_ORIGINAL901_QUALITY_SUPPLEMENT_ALL_MEMBER_SHA_FSYNC_REOPEN_PASS',archive_path=str(a),archive_sha256=m['archive']['sha256'],manifest_path=str(mp),manifest_sha256=MSHA,members=len(expected),no_model_or_quality_execution=True)))
'''
    for key, value in [('REMOTE', repr(REMOTE)), ('MBYTES', str(metadata['bytes'])), ('MSHA', repr(metadata['sha256'])), ('NAME', repr(archive.name))]:
        code = code.replace(key, value)
    result = call('run_command', 'server-verification', command='python3 -c ' + shlex.quote(code), timeout_seconds=30)
    assert result['exit_code'] == 0
    save(out / 'persistent-receipt.json', json.loads(result['stdout']))
    print(json.dumps(dict(manifest=metadata, archive=manifest['archive'], receipt=ref(out / 'persistent-receipt.json'), remote_manifest=manifest['remote_manifest'])))


if __name__ == '__main__':
    main()
