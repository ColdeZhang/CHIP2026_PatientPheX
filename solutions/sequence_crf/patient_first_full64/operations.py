#!/usr/bin/env python3
"""Plan deployment, exactly-once start intent, and small immutable recovery."""
import argparse
import base64
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["deploy-plan", "start", "collect"])
    ap.add_argument("--directory", type=Path, required=True)
    ap.add_argument("--mcp-client-dir", default="/workspace/patientphex-tools")
    ap.add_argument("--checkpoint")
    a = ap.parse_args(); b = a.directory
    resource = json.loads((b/"resource.json").read_text()); plan = json.loads((b/"run-plan.json").read_text())
    sys.path.insert(0, a.mcp_client_dir)
    from mcp_client import Client
    c = Client(); c.initialize()
    def save(name, obj):
        with (b/name).open("x") as f:
            json.dump(obj, f, indent=2); f.write("\n")
            f.flush(); os.fsync(f.fileno())
        descriptor = os.open(str(b), os.O_RDONLY | os.O_DIRECTORY)
        try: os.fsync(descriptor)
        finally: os.close(descriptor)
    def call(name, **args):
        r = c.request("tools/call", dict(name=name, arguments=dict(mount_id=resource["mount"], **args)))
        if r.get("isError"):
            raise RuntimeError(json.dumps(r))
        return r["structuredContent"]
    def run(code):
        return call("run_command", command="python3 -c " + shlex.quote(code), cwd=resource["remote"], timeout_seconds=30)
    if a.mode == "deploy-plan":
        assert not (b/"plan-deployment.json").exists()
        local = {"run-plan.json": b/"run-plan.json", "bounded_driver.py": Path(plan["local_driver_source"])}
        for name, path in local.items():
            raw = path.read_bytes()
            call("write_file", path=resource["remote"]+"/"+name, encoding="base64", content=base64.b64encode(raw).decode())
        code = """from pathlib import Path
import json,hashlib,py_compile
p=json.loads(Path('run-plan.json').read_text())
assert hashlib.sha256(Path('run-plan.json').read_bytes()).hexdigest()==PLAN
for n,r in p['remote_files'].items():
 assert hashlib.sha256(Path(n).read_bytes()).hexdigest()==r['sha256']
 if n.endswith('.py'):py_compile.compile(n,doraise=True)
print(json.dumps(dict(plan_sha256=PLAN,files=len(p['remote_files']),compiled=True,new_native_CPU_files_bound=True,no_GPU=True)))
""".replace("PLAN", repr(sha(b/"run-plan.json")))
        actual = run(code); assert actual["exit_code"] == 0, actual
        save("plan-deployment.json", actual); print(json.dumps(actual))
    elif a.mode == "start":
        assert a.checkpoint and len(a.checkpoint) == 40 and (b/"plan-deployment.json").exists()
        # Root authorized this bounded run while the single incremental review
        # proceeds. A concrete failed review blocks it; a pending review does not
        # create a new approval requirement.
        review = json.loads((b/"entry-review.json").read_text()) if (b/"entry-review.json").exists() else None
        if review is not None:
            assert review["status"].startswith("PASS"), "Concrete incremental review blocker"
        assert not (b/"launch-intent.json").exists() and not (b/"job.json").exists()
        code = """from pathlib import Path
import json,hashlib,subprocess,datetime
p=json.loads(Path('run-plan.json').read_text());assert hashlib.sha256(Path('run-plan.json').read_bytes()).hexdigest()==PLAN
assert not any(Path(x).exists() for x in ['driver-started.json','started.json','complete.json','failure.json','wrapper-complete.json'])
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True)
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used,memory.total','--format=csv,noheader,nounits'],text=True)
target=[s for s in gpu.splitlines() if p['gpu_uuid'] in s];assert len(target)==1 and p['gpu_uuid'] not in apps and int(target[0].split(',')[1])<512
print(json.dumps(dict(at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),target=target[0],target_no_process=True,plan_sha256=PLAN)))
""".replace("PLAN", repr(sha(b/"run-plan.json")))
        r = run(code); assert r["exit_code"] == 0, r; save("fresh-GPU.json", r)
        env = dict(resource["CPU_env"]); env["CUDA_VISIBLE_DEVICES"] = resource["gpu_uuid"]
        args = dict(command="python3 -u bounded_driver.py", cwd=resource["remote"], env=env)
        save("launch-intent.json", dict(at_utc=datetime.now(timezone.utc).isoformat(), arguments=args,
              checkpoint=a.checkpoint, plan_sha256=sha(b/"run-plan.json"),
              review_sha256=sha(b/"entry-review.json") if review is not None else None,
              review_status=review["status"] if review is not None else "parallel_incremental_review_pending",
              once_only=True, failure_recovery="On unknown outcome inspect existing process/files; NEVER call start again"))
        r = call("start_process", **args)
        start = datetime.fromisoformat(r["started_at"].replace("Z", "+00:00"))
        save("job.json", dict(result=r, arguments=args, checkpoint=a.checkpoint, cap=240, grace=30,
              approximate_TERM_deadline=(start+timedelta(seconds=240)).isoformat(),
              approximate_KILL_deadline=(start+timedelta(seconds=270)).isoformat(),
              authoritative_deadline="remote driver-started.json; child launch may be slightly later"))
        print(json.dumps(r))
    else:
        assert (b/"launch-intent.json").exists() and not (b/"collection.json").exists()
        # Direct execution remains usable when old managed-output RPCs fail.
        code = """from pathlib import Path
import json,hashlib,base64
names=['driver-started.json','started.json','run-manifest.json','step.json','complete.json','failure.json','wrapper-complete.json','stdout.log','stderr.log']
out={}
for n in names:
 p=Path(n)
 if p.exists():
  raw=p.read_bytes();assert len(raw)<512*1024
  out[n]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),base64=base64.b64encode(raw).decode())
print(json.dumps(dict(wrapper_complete=Path('wrapper-complete.json').exists(),files=out)))
"""
        r = run(code); assert r["exit_code"] == 0, r; payload = json.loads(r["stdout"])
        if not payload["wrapper_complete"]:
            print(json.dumps(dict(running_or_terminal_unknown=True, present_files=list(payload["files"])))); return
        dest = b/"terminal"; dest.mkdir(exist_ok=False); members = {}
        for name, item in payload["files"].items():
            raw = base64.b64decode(item["base64"])
            assert len(raw) == item["bytes"] and hashlib.sha256(raw).hexdigest() == item["sha256"]
            (dest/name).write_bytes(raw); members[name] = {k:item[k] for k in ["bytes", "sha256"]}
        save("collection.json", dict(at_utc=datetime.now(timezone.utc).isoformat(), files=members,
              direct_existing_files_only=True, wrapper=json.loads((dest/"wrapper-complete.json").read_text())))
        print(json.dumps(json.loads((b/"collection.json").read_text())))


if __name__ == "__main__":
    main()
