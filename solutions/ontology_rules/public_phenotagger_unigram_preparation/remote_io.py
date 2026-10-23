"""Transport only the small frozen CPU diagnostic and its terminal evidence."""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import shlex
import sys
import zipfile


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['deploy-start', 'status', 'collect'])
    p.add_argument('--root', required=True)
    p.add_argument('--client-dir', required=True)
    a = p.parse_args()
    root, sources = Path(a.root), Path(__file__).parent
    plan = json.loads((root / 'plan.json').read_text())
    runtime = plan['runtime']
    sys.path.insert(0, a.client_dir)
    from mcp_client import Client
    client = Client()
    client.initialize()

    def call(name, arguments):
        result = client.request('tools/call', {'name': name, 'arguments': {'mount_id': runtime['mount_id'], **arguments}})
        assert not result.get('isError'), result
        return result.get('structuredContent') or json.loads(result['content'][0]['text'])

    def command(code):
        result = call('run_command', dict(command=shlex.quote(runtime['python']) + ' -c ' + shlex.quote(code),
                      cwd=runtime['cwd'], env=runtime['environment'], timeout_seconds=30))
        assert result['exit_code'] == 0, result
        return json.loads(result['stdout'])

    if a.action == 'deploy-start':
        assert not (root / 'job.json').exists()
        names = {'plan.json': root / 'plan.json', 
                 'census.py': sources / 'census.py', 'run_cpu.py': sources / 'run_cpu.py'}
        assert sha(names['census.py'].read_bytes()) == plan['source_sha256']
        assert sha(names['run_cpu.py'].read_bytes()) == plan['runner_sha256']
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
            for name, path in names.items():
                archive.writestr(name, path.read_bytes())
        payload = buffer.getvalue()
        for index, offset in enumerate(range(0, len(payload), 196608)):
            call('write_file', dict(path=runtime['cwd'] + f'/upload/part-{index:03d}',
                 content=base64.b64encode(payload[offset:offset + 196608]).decode(), encoding='base64', create_parents=True))
        code = "from pathlib import Path;import io,zipfile,json,hashlib;assert not Path('plan.json').exists();parts=sorted(Path('upload').glob('part-*'));b=b''.join(x.read_bytes() for x in parts);assert hashlib.sha256(b).hexdigest()==" + repr(sha(payload)) + ";z=zipfile.ZipFile(io.BytesIO(b));z.extractall('.');print(json.dumps({n:hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in z.namelist()}));[x.unlink() for x in parts]"
        actual = command(code)
        assert actual == {name: sha(path.read_bytes()) for name, path in names.items()}
        (root / 'deployment.json').write_text(json.dumps(dict(files=actual, zip_bytes=len(payload)), indent=2) + '\n')
        argv = shlex.quote(runtime['python']) + ' run_cpu.py'
        job = call('start_process', dict(command=argv, cwd=runtime['cwd'], env=runtime['environment']))
        (root / 'job.json').write_text(json.dumps(dict(command=argv, plan_sha256=sha((root / 'plan.json').read_bytes()), result=job), indent=2) + '\n')
        print(json.dumps(job))
    elif a.action == 'status':
        print(json.dumps(command("from pathlib import Path;import json;print(json.dumps({'state':json.loads(Path('run-state.json').read_text()) if Path('run-state.json').exists() else None,'stderr':Path('stderr.log').read_text()[-3000:] if Path('stderr.log').exists() else None}))")))
    else:
        paths = ['run-state.json', 'stdout.log', 'stderr.log', 'output/summary.json', 'output/executed_unigram_function.py', 'output/input-only-unigram-coordinates.jsonl.gz', 'output/input-only-unique-tokens.jsonl.gz']
        code = "from pathlib import Path;import json,zipfile,hashlib;paths=" + repr(paths) + ";s=json.loads(Path('run-state.json').read_text());assert s['status']=='complete';z=zipfile.ZipFile('terminal.zip','w',zipfile.ZIP_DEFLATED);[z.write(n,n) for n in paths];z.close();p=Path('terminal.zip');print(json.dumps({'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'files':{n:{'sha256':hashlib.sha256(Path(n).read_bytes()).hexdigest(),'bytes':Path(n).stat().st_size} for n in paths}}))"
        index = command(code)
        chunks = []
        for offset in range(0, index['bytes'], 262144):
            result = call('read_file', dict(path=runtime['cwd'] + '/terminal.zip', offset=offset, limit=262144))
            assert result['encoding'] == 'base64'
            chunks.append(base64.b64decode(result['content']))
        payload = b''.join(chunks)
        assert len(payload) == index['bytes'] and sha(payload) == index['sha256']
        target = root / 'terminal'
        target.mkdir(exist_ok=False)
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            assert set(archive.namelist()) == set(paths)
            for name in paths:
                value = archive.read(name)
                assert sha(value) == index['files'][name]['sha256'] and len(value) == index['files'][name]['bytes']
                dest = target / name
                dest.parent.mkdir(exist_ok=True, parents=True)
                dest.write_bytes(value)
        (root / 'transfer.json').write_text(json.dumps(index, indent=2) + '\n')
        print(json.dumps(index))


if __name__ == '__main__':
    main()
