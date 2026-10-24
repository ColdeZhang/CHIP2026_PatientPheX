"""Persist the minimal accepted3049 supplement via small MCP files; never restore."""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys
import tarfile


def ref(path):
    path = Path(path).resolve(); data = path.read_bytes()
    return dict(path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def checked(value):
    path = Path(value['path'])
    assert not path.is_symlink() and ref(path) == value
    return path


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def main():
    parser = argparse.ArgumentParser()
    for name in ['bundle-plan', 'reproduction-directory', 'reproduction-terminal',
                 'reproduction-pointer', 'output-directory']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.bundle_plan.read_text())
    assert plan['server'] == '10.5.2.100'
    assert plan['remote_directory'] == 'research/public-new-lgb-support-control-20261024/accepted3049-recovery-v1'
    assert plan['mount_id'] == '5a9955da-5240-41cc-9e52-05d3d8da8a3c'
    reproduction_path = args.reproduction_directory / 'reproduction.json'
    reproduction = json.loads(reproduction_path.read_text())
    terminal = json.loads(args.reproduction_terminal.read_text())
    assert reproduction['status'] == 'PASS_COMPLETE16_LITERAL_BYTEEXACT_ACCEPTED3049_NO_MODEL_GOLD_OR_SCORE'
    assert reproduction['complete_bytes_equal'] and not reproduction['gold_read']
    assert all(reproduction[key] == 0 for key in ['fit_calls', 'model_predict_calls', 'NN_calls', 'quality_calls'])
    assert reproduction['prediction']['sha256'] == '3049d9fe20ea84aeed448ab12e80bd0d8cf351cdf087c06dec6375103969571f'
    assert terminal['returncode'] == 0 and terminal['status'] == 'complete' and terminal['executions'] == 1
    assert terminal['reproduction'] == ref(reproduction_path)
    files = {}
    for item in plan['members']:
        path = checked(item['reference']); member = item['archive_member']
        assert member == path.relative_to('/workspace').as_posix() and member not in files
        files[member] = path
    assert len(files) == plan['member_count_now']
    for path in [reproduction_path, args.reproduction_terminal, args.reproduction_pointer,
                 Path(__file__), args.bundle_plan]:
        path = path.resolve(); assert path.is_file() and not path.is_symlink()
        member = path.relative_to('/workspace').as_posix(); assert member not in files
        files[member] = path
    review = next(path for path in files.values() if path.name == 'review.json')
    assert ref(review)['sha256'] == plan['review_sha256']
    coverage = json.loads(checked(plan['prior_coverage']).read_text())
    unchanged = {(row['reference']['sha256'], row['reference']['bytes']) for row in coverage['coverage']}
    assert not any((ref(path)['sha256'], ref(path)['bytes']) in unchanged for path in files.values())
    output = args.output_directory.resolve(); output.mkdir(parents=True, exist_ok=False)
    stem = 'accepted3049-cache-supplement-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    archive = output / (stem + '.tar.gz'); manifest_path = output / (stem + '.manifest.json')
    members = [dict(path=name, bytes=path.stat().st_size, sha256=ref(path)['sha256'])
               for name, path in sorted(files.items())]
    with tarfile.open(archive, 'w:gz') as handle:
        for name, path in sorted(files.items()):
            handle.add(path, arcname=name, recursive=False)
    with tarfile.open(archive, 'r:gz') as handle:
        assert handle.getnames() == [member['path'] for member in members]
        for member in members:
            data = handle.extractfile(member['path']).read()
            assert len(data) == member['bytes'] and hashlib.sha256(data).hexdigest() == member['sha256']
    archive_ref = ref(archive)
    assert archive_ref['bytes'] <= 512 * 1024, 'Only small MCP payload upload authorized by this fixed entry'
    remote = plan['remote_directory']
    manifest = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), server=plan['server'],
                    mount_id=plan['mount_id'], remote_directory=remote,
                    remote_absolute_directory='/home/zhangyuheng/projects/CHIP2026_PatientPheX/' + remote,
                    remote_manifest=remote + '/' + manifest_path.name,
                    files=[dict(filename=archive.name, remote_path=remote + '/' + archive.name,
                                bytes=archive_ref['bytes'], sha256=archive_ref['sha256'],
                                local_path=str(archive), kind='accepted3049_minimal_cache_recovery_supplement')],
                    members=members, member_count=len(members), member_bytes=sum(x['bytes'] for x in members),
                    previous_chains=coverage['manifests'], prior_coverage=plan['prior_coverage'],
                    accepted_prediction_sha256=reproduction['prediction']['sha256'],
                    accepted_Score=0.7542169692819372, accepted_review_sha256=plan['review_sha256'],
                    reproduction=ref(reproduction_path), process_terminal=ref(args.reproduction_terminal),
                    generator=ref(__file__), bundle_plan=ref(args.bundle_plan),
                    scope=plan['scope'], no_restore_performed=True, restore_owner='root',
                    fetch_manifest_tool=dict(name='read_file', arguments=dict(mount_id=plan['mount_id'],
                                                                             path=remote + '/' + manifest_path.name)))
    save(manifest_path, manifest)
    manifest_ref = ref(manifest_path); assert manifest_ref['bytes'] <= 512 * 1024
    for path in [archive, manifest_path]:
        with path.open('rb') as stream:
            os.fsync(stream.fileno())
    save(output / 'intent.json', dict(archive=archive_ref, manifest=manifest_ref,
                                     remote_directory=remote, member_count=len(members), no_restore=True))
    sys.path.insert(0, '/workspace/CHIP2026_PatientPheX/scripts')
    from mcp_client import Client
    client = Client(); client.initialize()
    def call(name, **arguments):
        answer = client.request('tools/call', dict(name=name, arguments=dict(mount_id=plan['mount_id'], **arguments)))
        if answer.get('isError'):
            raise RuntimeError('MCP persistence request failed; preserve local intent')
        if answer.get('structuredContent') is not None:
            return answer['structuredContent']
        return json.loads('\n'.join(x['text'] for x in answer.get('content', []) if x.get('type') == 'text'))
    staged = []
    for path in [archive, manifest_path]:
        spec = ref(path); stage = remote + '/.staged-' + spec['sha256'] + '-' + path.name
        call('write_file', path=stage, content=base64.b64encode(path.read_bytes()).decode('ascii'),
             encoding='base64', create_parents=True)
        staged.append(dict(stage=stage, final=remote + '/' + path.name, bytes=spec['bytes'], sha256=spec['sha256']))
    code = '''from pathlib import Path
import hashlib,json,os,tarfile,time
specs=SPECIFICATIONS
for item in specs:
 staged=Path(item['stage']);final=Path(item['final'])
 assert not staged.is_symlink() and not final.is_symlink()
 assert staged.stat().st_size==item['bytes'] and hashlib.sha256(staged.read_bytes()).hexdigest()==item['sha256']
 with staged.open('rb') as stream:os.fsync(stream.fileno())
 try:os.link(staged,final)
 except FileExistsError:
  assert final.stat().st_size==item['bytes'] and hashlib.sha256(final.read_bytes()).hexdigest()==item['sha256']
 staged.unlink()
 with final.open('rb') as stream:os.fsync(stream.fileno())
archive,manifest=map(lambda x:Path(x['final']),specs)
m=json.loads(manifest.read_text());expected={item['path']:item for item in m['members']}
with tarfile.open(archive,'r:gz') as handle:
 assert len(handle.getmembers())==len(expected) and set(handle.getnames())==set(expected)
 for member in handle.getmembers():
  relative=Path(member.name);assert member.isfile() and not relative.is_absolute() and '..' not in relative.parts
  data=handle.extractfile(member).read();item=expected[member.name]
  assert len(data)==item['bytes'] and hashlib.sha256(data).hexdigest()==item['sha256']
for parent in [archive.parent]+list(archive.parent.parents):
 fd=os.open(parent,os.O_RDONLY)
 try:os.fsync(fd)
 finally:os.close(fd)
print(json.dumps(dict(status='ARCHIVE_MANIFEST_ALL_MEMBER_SHA_VERIFIED_FILES_AND_PARENT_CHAIN_FSYNCED',server_epoch=time.time(),member_count=len(expected),archive_sha256=specs[0]['sha256'],manifest_sha256=specs[1]['sha256'],archive_path=str(archive),manifest_path=str(manifest),restore_performed=False)))
'''.replace('SPECIFICATIONS', repr(staged))
    answer = call('run_command', command='python3 -c ' + shlex.quote(code), cwd='', timeout_seconds=30)
    assert answer['exit_code'] == 0, 'Remote verification failed; preserve local intent'
    receipt = json.loads(answer['stdout'])
    assert receipt['member_count'] == len(members) and receipt['archive_sha256'] == archive_ref['sha256']
    assert receipt['manifest_sha256'] == manifest_ref['sha256'] and not receipt['restore_performed']
    save(output / 'actual-persistent.json', receipt)
    print(json.dumps(dict(manifest=manifest_ref, archive=archive_ref, remote_manifest=manifest['remote_manifest'],
                          member_count=len(members), receipt=ref(output / 'actual-persistent.json'))))


if __name__ == '__main__':
    main()
