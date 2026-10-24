#!/usr/bin/env python3
"""Deploy small source/train-only payload; recover saved CPU output without reruns."""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import sys
import zipfile


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['deploy', 'recover'])
    ap.add_argument('--directory', required=True, type=Path)
    ap.add_argument('--mcp-client-dir', default='/workspace/patientphex-tools')
    a = ap.parse_args(); b = a.directory
    resource = json.loads((b/'resource.json').read_text())
    sys.path.insert(0, a.mcp_client_dir)
    from mcp_client import Client
    client = Client(); client.initialize()
    def save(name, value):
        with (b/name).open('x') as f:
            json.dump(value, f, indent=2); f.write('\n')
            f.flush(); os.fsync(f.fileno())
        descriptor = os.open(str(b), os.O_RDONLY | os.O_DIRECTORY)
        try: os.fsync(descriptor)
        finally: os.close(descriptor)
    def call(name, **args):
        result = client.request('tools/call', dict(name=name, arguments=dict(mount_id=resource['mount'], **args)))
        assert not result.get('isError'), result
        return result['structuredContent']
    def run(code, cwd=None, timeout=30):
        return call('run_command', command='python3 -c '+shlex.quote(code),
                    cwd=cwd or resource['remote'], timeout_seconds=timeout)
    if a.mode == 'deploy':
        assert not (b/'cpu-deployment-intent.json').exists(), 'Inspect existing remote artifacts, never blindly repeat deploy'
        payload = json.loads((b/'local-payload.json').read_text()); buffer = io.BytesIO(); members = {}
        with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as z:
            for name, ref in payload.items():
                assert Path(name).name == name
                raw = Path(ref['path']).read_bytes(); assert len(raw) == ref['bytes'] and digest(raw) == ref['sha256']
                z.writestr(name, raw); members[name] = {k:ref[k] for k in ['bytes','sha256']}
        archive = buffer.getvalue(); (b/'cpu-deployment.zip').write_bytes(archive)
        save('cpu-deployment-intent.json', dict(at_utc=datetime.now(timezone.utc).isoformat(), archive_sha256=digest(archive),
             archive_bytes=len(archive), members=members, remote=resource['remote'], no_GPU=True))
        code = 'from pathlib import Path;import shutil,json;p=Path('+repr(resource['remote'])+');assert not p.exists();free=shutil.disk_usage(".").free;assert free>=64*1024*1024;p.mkdir(parents=True);print(json.dumps(dict(free_disk_bytes=free,no_GPU=True)))'
        result = run(code, cwd='.'); assert result['exit_code'] == 0, result; save('actual-deployment-space.json', result)
        parts = []
        for i, offset in enumerate(range(0, len(archive), 256*1024)):
            name = f'upload.part{i:03d}'; parts.append(name)
            call('write_file', path=resource['remote']+'/'+name, encoding='base64',
                 content=base64.b64encode(archive[offset:offset+256*1024]).decode())
        code = 'from pathlib import Path;import hashlib,io,zipfile,json,py_compile;raw=b"".join(Path(x).read_bytes() for x in '+repr(parts)+');assert hashlib.sha256(raw).hexdigest()=='+repr(digest(archive))+';z=zipfile.ZipFile(io.BytesIO(raw));expected='+repr(members)+';assert set(z.namelist())==set(expected);[Path(n).write_bytes(z.read(n)) for n in expected];assert all(hashlib.sha256(Path(n).read_bytes()).hexdigest()==r["sha256"] for n,r in expected.items());[py_compile.compile(n,doraise=True) for n in expected if n.endswith(".py")];[Path(n).unlink() for n in '+repr(parts)+'];print(json.dumps(dict(files=len(expected),source_SHA_and_compile=True,no_GPU=True)))'
        result = run(code); assert result['exit_code'] == 0, result; save('cpu-source-deployment.json', result)
        # Two independent fresh CPU processes: native runtime inspection, then
        # tokenizer-only data contract. Both persist receipts before printing.
        for label, args in [('runtime', ['runtime_check.py']), ('data', ['data_contract.py','--plan','data-plan.json',
                                             '--model',resource['model'],'--output','cpu-native'])]:
            command = shlex.join(['/usr/bin/timeout','--signal=TERM','--kill-after=10s','50s',resource['python'],'-u']+args)
            result = call('run_command', command=command, cwd=resource['remote'], env=resource['CPU_env'], timeout_seconds=60)
            save('actual-'+label+'-CPU-execution.json', result); assert result['exit_code'] == 0, result
        print(json.dumps(dict(CPU_completed=True, next='recover', no_GPU=True)))
    else:
        assert not (b/'cpu-native-collection.json').exists()
        names = ['runtime-cpu.json','cpu-native/cpu-receipt.json','cpu-native/selected-encoded.json']
        code = 'from pathlib import Path;import json,hashlib,base64;names='+repr(names)+';print(json.dumps({n:dict(bytes=Path(n).stat().st_size,sha256=hashlib.sha256(Path(n).read_bytes()).hexdigest(),base64=base64.b64encode(Path(n).read_bytes()).decode()) for n in names if Path(n).exists()}))'
        result = run(code); assert result['exit_code'] == 0, result; items = json.loads(result['stdout'])
        if set(items) != set(names):
            print(json.dumps(dict(CPU_complete=False, present=list(items), no_GPU=True))); return
        dest = b/'cpu-native'; dest.mkdir(exist_ok=False); refs = {}
        for name, item in items.items():
            raw = base64.b64decode(item['base64']); assert len(raw) == item['bytes'] and digest(raw) == item['sha256']
            p = dest/Path(name).name; p.write_bytes(raw); refs[name] = {k:item[k] for k in ['bytes','sha256']}
        # Preserve original CPU files, copy exact new input/receipt into the
        # future probe's flat source directory for its frozen manifest.
        code = "from pathlib import Path;import hashlib,json;names=['cpu-receipt.json','selected-encoded.json'];[(Path(n).write_bytes((Path('cpu-native')/n).read_bytes())) for n in names if not Path(n).exists()];assert all(Path(n).read_bytes()==(Path('cpu-native')/n).read_bytes() for n in names);print(json.dumps(dict(flat_copies_exact=True,no_GPU=True)))"
        result = run(code); assert result['exit_code'] == 0, result
        save('cpu-native-collection.json', dict(at_utc=datetime.now(timezone.utc).isoformat(), files=refs,
             original_CPU_outputs_preserved=True, flat_copy_verification=result, no_GPU=True))
        print(json.dumps(dict(CPU_complete=True, files=refs, no_GPU=True)))


if __name__ == '__main__':
    main()
