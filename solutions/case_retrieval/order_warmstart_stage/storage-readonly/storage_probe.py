"""One bounded 1 MiB A800 -> .100 transport check; no GPU/model access."""
from pathlib import Path
import hashlib, json, shlex, sys, time, datetime

ROOT = Path(__file__).parent
sys.path.insert(0, '/workspace/CHIP2026_PatientPheX/scripts')
from mcp_client import Client

SRC = '69b991ee-183f-4add-af1a-579af92479c9'
DST = '5a9955da-5240-41cc-9e52-05d3d8da8a3c'
REL = 'case-retrieval/qwen35-preparation/order-warmstart-storage-probe-20261024'
SHM = '/dev/shm/patientphex-order-storage-probe-1045-20261024'
payload = (b'patientphex-order-checkpoint-transfer-v1\n' * 30000)[:1048576]
digest = hashlib.sha256(payload).hexdigest()


def call(mount, name, args):
    c = Client(); c.initialize()
    result = c.request('tools/call', {'name': name, 'arguments': {'mount_id': mount, **args}})
    assert not result.get('isError'), 'MCP tool failed'
    return result.get('structuredContent') or json.loads(result['content'][0]['text'])


def save(name, value):
    with (ROOT/name).open('x') as f:
        json.dump(value, f, indent=2); f.write('\n')


def main():
    assert not (ROOT/'storage-probe-job.json').exists(), 'Recover existing probe rather than duplicate'
    server = '''from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import os,hashlib,json
p=Path(SHM);assert not p.exists();p.mkdir(mode=0o700);assert p.stat().st_uid==os.getuid()==1045
b=(b'patientphex-order-checkpoint-transfer-v1\\n'*30000)[:1048576];assert hashlib.sha256(b).hexdigest()==DIGEST
f=p/'probe.bin';f.write_bytes(b)
class Handler(BaseHTTPRequestHandler):
 def log_message(self,*args):pass
 def do_GET(self):
  if self.path!='/probe.bin':self.send_error(404);return
  self.send_response(200);self.send_header('Content-Length',str(len(b)));self.end_headers();self.wfile.write(b)
s=ThreadingHTTPServer(('0.0.0.0',0),Handler)
print(json.dumps({'ready':True,'port':s.server_address[1],'bytes':len(b),'sha256':DIGEST,'source':str(f)}),flush=True)
s.serve_forever()
'''.replace('SHM', repr(SHM)).replace('DIGEST', repr(digest))
    save('storage-probe-plan.json', {'scope': __doc__, 'source_mount': SRC, 'target_mount': DST,
        'payload_bytes': len(payload), 'payload_sha256': digest, 'server_hard_seconds': 120,
        'target_command_hard_seconds': 30, 'server_sha256': hashlib.sha256(server.encode()).hexdigest(),
        'driver_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'no_model_weights_or_GPU': True})
    check = call(SRC, 'run_command', {'command': 'python3 -c '+shlex.quote('from pathlib import Path;assert not Path('+repr(REL)+').exists();assert not Path('+repr(SHM)+').exists()'), 'cwd': '.', 'timeout_seconds': 10})
    assert check['exit_code'] == 0
    call(SRC, 'write_file', {'path': REL+'/server.py', 'content': server, 'encoding': 'utf8', 'create_parents': True})
    job = call(SRC, 'start_process', {'command': 'timeout --signal=TERM --kill-after=5s 120s python3 -u server.py', 'cwd': REL})
    save('storage-probe-job.json', job)
    try:
        ready = None
        for _ in range(8):
            response = call(SRC, 'read_process_output', {'process_id': job['id'], 'stream': 'stdout', 'cursor': 0, 'limit': 2000})
            for line in response.get('content','').splitlines():
                if line.startswith('{'): ready = json.loads(line)
            if ready: break
            time.sleep(.5)
        assert ready and ready['sha256'] == digest and ready['bytes'] == len(payload)
        save('storage-probe-ready.json', ready)
        url = 'http://10.5.4.7:'+str(ready['port'])+'/probe.bin'
        code = '''from pathlib import Path
import urllib.request,hashlib,json,time,datetime,os
start=time.monotonic();root=Path(REL);assert not root.exists();root.mkdir(parents=True)
target=root/'probe.bin';assert not target.exists()
with urllib.request.urlopen(URL,timeout=20) as r:
 assert r.status==200 and int(r.headers['Content-Length'])==1048576
 data=r.read()
assert len(data)==1048576 and hashlib.sha256(data).hexdigest()==DIGEST
with target.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
assert hashlib.sha256(target.read_bytes()).hexdigest()==DIGEST
print(json.dumps({'status':'actual_persistent_write_and_full_SHA_pass','bytes':len(data),'sha256':DIGEST,'seconds':time.monotonic()-start,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'target':str(target),'uid':os.getuid()}))
'''.replace('REL', repr(REL)).replace('URL', repr(url)).replace('DIGEST', repr(digest))
        save('storage-probe-sources.json', {'server': server, 'consumer': code})
        result = call(DST, 'run_command', {'command': 'python3 -c '+shlex.quote(code), 'cwd': '.', 'timeout_seconds': 30})
        save('storage-probe-transfer.json', result)
        assert result['exit_code'] == 0
        value = json.loads(result['stdout']); assert value['sha256'] == digest
        print(json.dumps(value), flush=True)
    finally:
        save('storage-probe-server-stopped.json', call(SRC, 'stop_process', {'process_id': job['id']}))


if __name__ == '__main__':
    main()
