"""One 120+15 second query job; no restart and no replacement bank encoding."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time


def save(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def gpu_processes(uuid):
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
        "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True, timeout=10)
    return [line for line in result.stdout.splitlines() if line.split(",")[0].strip() == uuid]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    if plan["budget_seconds"] != 120 or plan["grace_seconds"] != 15 or not plan["authorized"]:
        raise ValueError("Wrong authorized budget")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != plan["gpu_uuid"]:
        raise ValueError("Wrong GPU UUID")
    started = time.monotonic()
    utc = time.time()
    save("driver-started.json", {"pid": os.getpid(), "at_utc": datetime.now(timezone.utc).isoformat(),
        "deadline_TERM_unix": utc + 120, "deadline_KILL_unix": utc + 135, "no_automatic_retry": True})
    active = gpu_processes(plan["gpu_uuid"])
    if active:
        save("preflight-failed.json", {"target_gpu_processes": active, "no_model_started": True})
        raise RuntimeError("Allocated GPU not actually released")
    free = shutil.disk_usage(Path.cwd()).free
    save("resource-check.json", {"target_gpu_processes": active, "free_disk_bytes": free})
    if free < 300_000_000:
        raise RuntimeError("Insufficient disk for persistent public subset and query evidence")
    timed_out = False
    with open("stdout.log", "xb") as out, open("stderr.log", "xb") as err:
        process = subprocess.Popen(plan["argv"], stdout=out, stderr=err, start_new_session=True)
        save("child-started.json", {"pid": process.pid, "argv": plan["argv"]})
        try:
            rc = process.wait(timeout=max(.001, started + 120 - time.monotonic()))
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                rc = process.wait(timeout=max(.001, started + 135 - time.monotonic()))
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                rc = process.wait()
        for handle in [out, err]:
            handle.flush()
            os.fsync(handle.fileno())
    result = {"returncode": rc, "timed_out": timed_out, "seconds": time.monotonic() - started,
              "deadline_TERM_unix": utc + 120, "deadline_KILL_unix": utc + 135,
              "child_pid": process.pid, "no_automatic_retry": True}
    complete = Path(plan["output_dir"]) / "complete.json"
    if rc == 0 and not timed_out:
        try:
            value = json.loads(complete.read_text())
            assert value["status"] == "COMPLETE_FIXED_PUBLIC_TRAIN_QUERIES"
            assert value["query_count"] == plan["query_count"] == 624
            assert value["public_bank_rows"] == 41099 and value["no_fit"] and value["no_bank_reencoding"]
            result["complete"] = value
        except Exception as error:
            result["endpoint_error"] = {"type": type(error).__name__, "message": str(error)}
    result["target_gpu_processes_after_child"] = gpu_processes(plan["gpu_uuid"])
    save("wrapper-complete.json", result)
    print(json.dumps(result), flush=True)
    raise SystemExit(rc or int(timed_out or "endpoint_error" in result))


if __name__ == "__main__":
    main()
