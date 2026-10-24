"""One immutable patient archive, one private peer, authenticated and TTL bounded."""
import hashlib
import http.server
import json
import os
from pathlib import Path
import signal
import threading
import time

ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT.parent / 'complete-collection/patient-complete.zip'
SHA = '1df686ceaaf69e55913198482494533ede264dad2effa4561dd4ed4da9593e36'
BODY = ARCHIVE.read_bytes()
assert len(BODY) == 897210 and hashlib.sha256(BODY).hexdigest() == SHA
TOKEN = os.environ['PATIENT_TRANSFER_TOKEN']
PORT = int(os.environ['PORT'])
TTL = 180


def save(name, value):
    with (ROOT / name).open('x') as f:
        json.dump(value, f, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    fd = os.open(ROOT, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        if (self.path != '/frozen.zip' or self.client_address[0] != '10.5.2.100'
                or self.headers.get('Authorization') != 'Bearer ' + TOKEN):
            self.send_error(403)
            return
        self.send_response(200)
        self.send_header('Content-Type', 'application/zip')
        self.send_header('Content-Length', str(len(BODY)))
        self.end_headers()
        self.wfile.write(BODY)


server = http.server.ThreadingHTTPServer(('10.5.3.12', PORT), Handler)
server.daemon_threads = True
save('started.json', dict(pid=os.getpid(), port=PORT, ttl_seconds=TTL,
     started_unix=time.time(), bind_ip='10.5.3.12', allowed_peer='10.5.2.100',
     archive_sha256=SHA, archive_bytes=len(BODY)))


def stop(*args):
    threading.Thread(target=server.shutdown, daemon=True).start()


signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)
timer = threading.Timer(TTL, stop)
timer.daemon = True
timer.start()
print(json.dumps(dict(ready=True, pid=os.getpid(), port=PORT, ttl_seconds=TTL)), flush=True)
try:
    server.serve_forever()
finally:
    timer.cancel()
    server.server_close()
    save('ended.json', dict(ended_unix=time.time(), pid=os.getpid(), archive_sha256=SHA))
