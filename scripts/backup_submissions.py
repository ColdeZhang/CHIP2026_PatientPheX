#!/usr/bin/env python3
"""Back up accepted PatientPheX uploads through MCP, without replacing differing files."""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import shlex
import sys
import tempfile

from mcp_client import Client

MOUNT = "5a9955da-5240-41cc-9e52-05d3d8da8a3c"
REMOTE = "day1/coordinator-backup/submissions"
GIT_REMOTE = "day1/coordinator-backup/git"
REMOTE_ABSOLUTE = "/home/zhangyuheng/projects/CHIP2026_PatientPheX/" + REMOTE
LOCAL = Path("/workspace/patientphex-runs/day1-20261023/submissions")
CHUNK = 512 * 1024


def sha(data):
    return hashlib.sha256(data).hexdigest()


def safe_path(path, directory=REMOTE):
    p = PurePosixPath(path)
    if directory not in (REMOTE, GIT_REMOTE) or p.is_absolute() or ".." in p.parts or not str(p).startswith(directory + "/"):
        raise ValueError("PathOutsideBackupDirectory")
    return str(p)


class Remote:
    def __init__(self, directory=REMOTE):
        if directory not in (REMOTE, GIT_REMOTE):
            raise ValueError("UnexpectedBackupDirectory")
        self.directory = directory
        self.client = Client()
        self.client.initialize()

    def call(self, name, **arguments):
        response = self.client.request("tools/call", {"name": name, "arguments": {"mount_id": MOUNT, **arguments}})
        if response.get("isError"):
            raise RuntimeError("RemoteToolError")
        result = response.get("structuredContent")
        if result is None:
            blocks = [c["text"] for c in response.get("content", []) if c.get("type") == "text"]
            result = json.loads("\n".join(blocks))
        return result

    def python(self, code):
        result = self.call("run_command", command="python3 -c " + shlex.quote(code), cwd="", timeout_seconds=30)
        if result.get("exit_code") != 0:
            raise RuntimeError("RemoteCommandError")
        return json.loads(result["stdout"])

    def states(self, paths):
        paths = [safe_path(p, self.directory) for p in paths]
        return self.python("from pathlib import Path; import hashlib,json\n"
                           "result={}\n"
                           f"for name in {paths!r}:\n"
                           " p=Path(name)\n"
                           " if p.is_symlink(): raise ValueError('SymlinkNotAllowed')\n"
                           " result[name]={'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} if p.exists() else None\n"
                           "print(json.dumps(result))")

    def put(self, filename, data):
        if Path(filename).name != filename:
            raise ValueError("FilenameMustBeBasename")
        path = safe_path(self.directory + "/" + filename, self.directory)
        digest = sha(data)
        expected = {"bytes": len(data), "sha256": digest}
        current = self.states([path])[path]
        if current:
            if current != expected:
                raise FileExistsError("RemoteFileHasDifferentHash")
            return {"filename": filename, "remote_path": path, **expected, "action": "skipped_identical"}
        pieces = [(safe_path(f"{self.directory}/.chunks/{digest}/{i:06d}", self.directory), data[start:start+CHUNK])
                  for i, start in enumerate(range(0, len(data), CHUNK))]
        states = self.states([p for p, _ in pieces]) if pieces else {}
        specifications = []
        for name, part in pieces:
            chunk_state = {"bytes": len(part), "sha256": sha(part)}
            if states[name] is not None and states[name] != chunk_state:
                raise FileExistsError("RemoteChunkHasDifferentHash")
            if states[name] is None:
                self.call("write_file", path=name, content=base64.b64encode(part).decode("ascii"),
                          encoding="base64", create_parents=True)
            specifications.append({"path": name, **chunk_state})
        code = f'''from pathlib import Path
import hashlib,json,os,tempfile
target=Path({path!r})
target.parent.mkdir(parents=True,exist_ok=True)
expected={expected!r}
def state(p):
 return {{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}}
if target.is_symlink(): raise ValueError('SymlinkNotAllowed')
if target.exists():
 if state(target)!=expected: raise FileExistsError('DifferentFinalHash')
else:
 fd,temp=tempfile.mkstemp(prefix='.assembling-',dir=target.parent)
 try:
  with os.fdopen(fd,'wb') as output:
   for spec in {specifications!r}:
    p=Path(spec['path'])
    if p.is_symlink() or state(p)!={{'bytes':spec['bytes'],'sha256':spec['sha256']}}: raise ValueError('ChunkVerificationFailed')
    output.write(p.read_bytes())
  if state(Path(temp))!=expected: raise ValueError('FinalVerificationFailed')
  try: os.link(temp,target)
  except FileExistsError:
   if state(target)!=expected: raise
 finally:
  Path(temp).unlink()
print(json.dumps(state(target)))
'''
        if self.python(code) != expected:
            raise ValueError("RemoteFinalVerificationFailed")
        return {"filename": filename, "remote_path": path, **expected, "action": "uploaded_verified"}


def accepted_files(directory):
    result = []
    for receipt in sorted(directory.glob("*-receipt.json")):
        data = receipt.read_bytes()
        record = json.loads(data)
        if record.get("status") != "accepted":
            continue
        prediction = Path(record["file"])
        if prediction.resolve().parent != directory.resolve() or prediction.name != record["filename"]:
            raise ValueError("ReceiptPredictionPathMismatch")
        contents = prediction.read_bytes()
        if sha(contents) != record["sha256"] or len(contents) != record["bytes"]:
            raise ValueError("ReceiptPredictionHashMismatch")
        for file, payload, kind in [(prediction, contents, "prediction"), (receipt, data, "acceptance_receipt")]:
            result.append((file, payload, {"route": record["route"], "kind": kind}))
    return result


def backup(args):
    files = accepted_files(args.source)
    if not files:
        raise ValueError("NoAcceptedReceipts")
    remote = Remote()
    rows = []
    for path, data, metadata in files:
        row = {**remote.put(path.name, data), **metadata, "local_path": str(path)}
        rows.append(row)
        print(json.dumps({k: row[k] for k in ("filename", "sha256", "bytes", "action")}), flush=True)
    created = datetime.now(timezone.utc)
    tag = created.strftime("%Y%m%dT%H%M%S%fZ")
    filename = "backup-manifest-" + tag + ".json"
    local_manifest = args.source.parent / filename
    manifest = {"created_at_utc": created.isoformat(), "server": "10.5.2.100", "mount_id": MOUNT,
                "remote_directory": REMOTE, "remote_absolute_directory": REMOTE_ABSOLUTE,
                "remote_manifest": REMOTE + "/" + filename, "chunk_bytes": CHUNK,
                "credentials": "MCP_URL and MCP_TOKEN environment variables; values never saved",
                "files": rows, "prediction_count": sum(r["kind"] == "prediction" for r in rows),
                "backup_command": "python /workspace/patientphex-tools/backup_submissions.py backup",
                "restore_command": "python /workspace/patientphex-tools/backup_submissions.py restore --manifest "
                                   + shlex.quote(str(local_manifest)) + " --destination /workspace/patientphex-restored/submissions",
                "fetch_manifest_tool": {"name": "read_file", "arguments": {"mount_id": MOUNT, "path": REMOTE + "/" + filename}},
                "tool_sha256": sha(Path(__file__).read_bytes())}
    encoded = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode()
    with local_manifest.open("xb") as out:
        out.write(encoded)
    remote.put(filename, encoded)
    print(json.dumps({"manifest": str(local_manifest), "sha256": sha(encoded),
                      "remote_manifest": manifest["remote_manifest"], "verified_files": len(rows)}), flush=True)


def restore(args):
    manifest = json.loads(args.manifest.read_text())
    if manifest["mount_id"] != MOUNT or manifest["remote_directory"] not in (REMOTE, GIT_REMOTE):
        raise ValueError("UnexpectedBackupLocation")
    remote = Remote(manifest["remote_directory"])
    args.destination.mkdir(parents=True, exist_ok=True)
    for row in manifest["files"]:
        name = row["filename"]
        if Path(name).name != name:
            raise ValueError("FilenameMustBeBasename")
        destination = args.destination / name
        if destination.exists():
            if sha(destination.read_bytes()) != row["sha256"]:
                raise FileExistsError("LocalFileHasDifferentHash")
            print(json.dumps({"filename": name, "action": "skipped_identical"}), flush=True)
            continue
        path = safe_path(row["remote_path"], manifest["remote_directory"])
        if remote.states([path])[path] != {"bytes": row["bytes"], "sha256": row["sha256"]}:
            raise ValueError("RemoteBackupChanged")
        with tempfile.NamedTemporaryFile(dir=args.destination, prefix=".restoring-", delete=False) as output:
            pending = Path(output.name)
            try:
                for offset in range(0, row["bytes"], CHUNK):
                    result = remote.call("read_file", path=path, offset=offset, limit=min(CHUNK, row["bytes"] - offset))
                    content = result["content"]
                    part = base64.b64decode(content) if result.get("encoding") == "base64" else content.encode("utf-8")
                    output.write(part)
                output.flush()
                if sha(pending.read_bytes()) != row["sha256"]:
                    raise ValueError("RestoredHashMismatch")
                # Hard link publishes without replacing a concurrently created file.
                import os
                os.link(pending, destination)
            finally:
                pending.unlink()
        print(json.dumps({"filename": name, "sha256": row["sha256"], "action": "restored_verified"}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("backup")
    b.add_argument("--source", type=Path, default=LOCAL)
    r = sub.add_parser("restore")
    r.add_argument("--manifest", type=Path, required=True)
    r.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    try:
        (backup if args.command == "backup" else restore)(args)
    except Exception as exc:
        # Never print HTTP request/header/token values or raw tool errors.
        print(json.dumps({"error_type": type(exc).__name__}), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
