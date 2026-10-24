#!/usr/bin/env python3
"""Persist the actual timeout return code even if managed MCP output fails."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time


def main():
    plan = json.loads(Path("run-plan.json").read_text())
    assert plan["budget_seconds"] == 180 and plan["grace_seconds"] == 30
    assert not any(Path(x).exists() for x in ["driver-started.json", "wrapper-complete.json", "started.json"])
    for name, ref in plan["remote_files"].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == ref["sha256"]
    command = ["/usr/bin/timeout", "--signal=TERM", "--kill-after=30s", "180s", plan["python"], "-u", "runner.py"]
    start = time.time()
    with Path("driver-started.json").open("x") as f:
        json.dump(dict(at_utc=datetime.now(timezone.utc).isoformat(), command=command,
                       plan_sha256=hashlib.sha256(Path("run-plan.json").read_bytes()).hexdigest(),
                       deadline_TERM_unix=start+180, deadline_KILL_unix=start+210), f, indent=2)
    with Path("stdout.log").open("xb") as out, Path("stderr.log").open("xb") as err:
        done = subprocess.run(command, stdout=out, stderr=err, check=False)
    report = dict(at_utc=datetime.now(timezone.utc).isoformat(), returncode=done.returncode,
                  elapsed_seconds=time.time()-start, cap_seconds=180, grace_seconds=30,
                  complete_exists=Path("complete.json").exists(), failure_exists=Path("failure.json").exists())
    try:
        apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
                                        "--format=csv,noheader,nounits"], text=True, timeout=10)
        memory = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,memory.used,memory.total",
                                          "--format=csv,noheader,nounits"], text=True, timeout=10)
        report["gpu_release"] = dict(at_utc=datetime.now(timezone.utc).isoformat(), target=plan["gpu_uuid"],
            target_has_no_process=plan["gpu_uuid"] not in apps,
            target_memory=[line for line in memory.splitlines() if plan["gpu_uuid"] in line])
    except Exception as exc:
        report["gpu_release_unknown"] = type(exc).__name__
    with Path("wrapper-complete.json").open("x") as f:
        json.dump(report, f, indent=2); f.write("\n")
    print(json.dumps(report), flush=True)
    raise SystemExit(done.returncode)


if __name__ == "__main__":
    main()
