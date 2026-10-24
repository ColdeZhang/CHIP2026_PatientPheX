#!/usr/bin/env python3
"""Bounded, two-file-only transfer of the frozen old32 adapter; no model loads."""
import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import time
import urllib.request

FILES = {
    "adapter_config.json": "96564441328e6759568e60c49ea87db432b9ec45cfd57001ef81640d4ff9c4f9",
    "adapter_model.safetensors": "4f4815b6c6c245c43dc976406942f43cecc7250266a0ec91ee6e1807fe63bfff",
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["serve", "fetch"])
    p.add_argument("--directory", type=Path, required=True)
    p.add_argument("--source-url")
    p.add_argument("--bind", default="10.5.4.7")
    a = p.parse_args(); root = a.directory.resolve(); started = time.monotonic()
    if a.mode == "serve":
        for name, expected in FILES.items():
            assert digest(root / name) == expected
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def serve(self, body):
                if self.path not in {"/" + name for name in FILES}:
                    self.send_error(404); return
                path = root / self.path[1:]; size = path.stat().st_size
                start, end, status = 0, size - 1, 200
                if self.headers.get("Range"):
                    m = re.fullmatch(r"bytes=(\d+)-(\d*)", self.headers["Range"])
                    if not m:
                        self.send_error(416); return
                    start = int(m[1]); end = int(m[2]) if m[2] else size - 1
                    if not 0 <= start <= end < size:
                        self.send_error(416); return
                    status = 206
                self.send_response(status); self.send_header("Content-Length", str(end-start+1))
                self.send_header("Content-Type", "application/octet-stream")
                if status == 206:
                    self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                self.end_headers()
                if body:
                    with path.open("rb") as f:
                        f.seek(start); left = end-start+1
                        while left:
                            data = f.read(min(1024*1024, left))
                            self.wfile.write(data); left -= len(data)
            def do_GET(self):
                self.serve(True)
            def do_HEAD(self):
                self.serve(False)
        server = ThreadingHTTPServer((a.bind, 0), Handler)
        print(json.dumps(dict(ready=True, port=server.server_address[1], source=str(root), files=FILES)), flush=True)
        server.serve_forever()
    else:
        assert a.source_url and a.source_url.startswith("http://10.5.4.7:")
        root.mkdir(parents=True, exist_ok=True); actual = {}
        for name, expected in FILES.items():
            path = root/name; part = root/(name + ".part")
            if not path.exists():
                offset = part.stat().st_size if part.exists() else 0
                req = urllib.request.Request(a.source_url + "/" + name, headers={"Range": f"bytes={offset}-"} if offset else {})
                with urllib.request.urlopen(req, timeout=45) as response:
                    assert response.status == (206 if offset else 200)
                    with part.open("ab" if offset else "wb") as f:
                        for block in iter(lambda: response.read(1024*1024), b""):
                            f.write(block)
                assert digest(part) == expected
                part.replace(path)
            assert digest(path) == expected
            actual[name] = dict(bytes=path.stat().st_size, sha256=expected)
        report = dict(status="complete_actual_full_SHA_old32_adapter_only", files=actual,
                      elapsed_seconds=time.monotonic()-started, source_url=a.source_url,
                      destination=str(root), no_model_load=True, no_warm_adapter_or_optimizer=True)
        with (root/"transfer-receipt.json").open("x") as f:
            f.write(json.dumps(report, indent=2)+"\n")
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
