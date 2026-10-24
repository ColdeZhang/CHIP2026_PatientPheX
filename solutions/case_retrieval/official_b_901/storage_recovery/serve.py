"""Read only three fixed old checkpoint files over one private HTTP window."""
from http.server import HTTPServer, BaseHTTPRequestHandler
import hashlib
import json
import os
from pathlib import Path
import time

ROOT=Path('/home/zhangyuheng/projects/CHIP2026_PatientPheX/case-retrieval/qwen35-preparation/full64-relation-cpu-preparation/formal2ep/epoch-002')
FILES={'training_state.pt':('38b9877a280c7f300878f93d9f362803f75b011a6cb5fd26eb6bf78c3fd66c88',346667053),
       'checkpoint.json':('f217bdc5bd191fe4319c38c99b46ef7b8d1f505d5f89d8c4d2e21a9425511e87',10774),
       'adapter/adapter_config.json':('c4a478a7dd1aa65613279cdd4250da344ca16df8605de4249f445a1de2e8f2e2',1287)}


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    prefix='/'+os.environ['RECOVERY_PREFIX']+'/'
    served=set()
    for n,(expected,size) in FILES.items():
        p=ROOT/n;assert p.is_file() and not p.is_symlink() and p.stat().st_uid==os.getuid()
        assert p.stat().st_size==size and digest(p)==expected
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.client_address[0]!='10.5.2.100' or not self.path.startswith(prefix):self.send_error(403);return
            name=self.path[len(prefix):]
            if name=='ready':
                self.send_response(200);self.send_header('Content-Length','5');self.end_headers();self.wfile.write(b'READY');return
            if name not in FILES:self.send_error(404);return
            p=ROOT/name;self.connection.settimeout(30)
            self.send_response(200);self.send_header('Content-Length',str(p.stat().st_size));self.end_headers()
            with p.open('rb') as f:
                for b in iter(lambda:f.read(4*1024*1024),b''):self.wfile.write(b)
            self.wfile.flush();served.add(name)
        def log_message(self,*args):pass
    server=HTTPServer(('10.5.4.7',51963),Handler);server.timeout=.25;deadline=time.monotonic()+50
    try:
        while len(served)<len(FILES) and time.monotonic()<deadline:server.handle_request()
    finally:server.server_close()
    print(json.dumps(dict(status='READ_ONLY_SERVER_CLOSED',served=sorted(served),expected=len(FILES),source_preserved=True)),flush=True)


if __name__=='__main__':main()
