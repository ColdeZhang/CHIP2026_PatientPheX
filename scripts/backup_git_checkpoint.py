#!/usr/bin/env python3
"""Create a self-contained main/day1 Git bundle and verify its independent MCP backup."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import sys

from backup_submissions import Remote, GIT_REMOTE, MOUNT, CHUNK, sha


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], stderr=subprocess.PIPE, text=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("/workspace/CHIP2026_PatientPheX"))
    parser.add_argument("--output-dir", type=Path, default=Path("/workspace/patientphex-runs/day1-20261023/git-backups"))
    args = parser.parse_args()
    try:
        names = git(args.repo, "for-each-ref", "--format=%(refname)", "refs/heads/main", "refs/heads/day1").splitlines()
        if "refs/heads/main" not in names or not all(n == "refs/heads/main" or n.startswith("refs/heads/day1/") for n in names):
            raise ValueError("UnexpectedGitReferences")
        created = datetime.now(timezone.utc)
        tag = created.strftime("%Y%m%dT%H%M%S%fZ")
        main_head = git(args.repo, "rev-parse", "refs/heads/main").strip()
        args.output_dir.mkdir(parents=True, exist_ok=True)
        bundle = args.output_dir / ("patientphex-main-day1-" + main_head[:12] + "-" + tag + ".bundle")
        if bundle.exists():
            raise FileExistsError("BundleAlreadyExists")
        git(args.repo, "bundle", "create", str(bundle), *names)
        git(args.repo, "bundle", "verify", str(bundle))
        heads = dict((name, commit) for commit, name in (line.split() for line in git(args.repo, "bundle", "list-heads", str(bundle)).splitlines()))
        contents = bundle.read_bytes()
        print(json.dumps({"bundle": str(bundle), "bytes": len(contents), "sha256": sha(contents), "refs": heads}), flush=True)
        row = Remote(GIT_REMOTE).put(bundle.name, contents)
        filename = bundle.stem + ".manifest.json"
        manifest_path = args.output_dir / filename
        manifest = {"created_at_utc": created.isoformat(), "server": "10.5.2.100", "mount_id": MOUNT,
                    "remote_directory": GIT_REMOTE, "remote_manifest": GIT_REMOTE + "/" + filename,
                    "chunk_bytes": CHUNK, "files": [dict(row, local_path=str(bundle), kind="git_bundle")],
                    "refs": heads, "self_contained": True, "bundle_verify": "passed",
                    "restore_command": "python /workspace/patientphex-tools/backup_submissions.py restore --manifest "
                        + shlex.quote(str(manifest_path)) + " --destination /workspace/patientphex-restored/git",
                    "clone_command": "git clone -b main /workspace/patientphex-restored/git/" + bundle.name
                        + " /workspace/CHIP2026_PatientPheX-restored",
                    "restore_all_branches_command": "git -C /workspace/CHIP2026_PatientPheX-restored fetch "
                        + "/workspace/patientphex-restored/git/" + bundle.name + " 'refs/heads/day1/*:refs/heads/day1/*'",
                    "fetch_manifest_tool": {"name": "read_file", "arguments": {"mount_id": MOUNT, "path": GIT_REMOTE + "/" + filename}},
                    "notes": "Contains committed main and day1 branches plus complete reachable history; excludes uncommitted files and external model/data artifacts.",
                    "tool_sha256": sha(Path(__file__).read_bytes())}
        data = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode()
        with manifest_path.open("xb") as out:
            out.write(data)
        Remote(GIT_REMOTE).put(filename, data)
        print(json.dumps({"manifest": str(manifest_path), "manifest_sha256": sha(data), "remote_bundle": row["remote_path"], "verified": True}), flush=True)
    except Exception as exc:
        print(json.dumps({"error_type": type(exc).__name__}), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
