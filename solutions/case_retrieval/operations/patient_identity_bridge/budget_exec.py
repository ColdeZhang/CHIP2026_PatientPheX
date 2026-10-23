#!/usr/bin/env python3
"""Enforce a server-clock shared deadline and a per-process maximum."""
import argparse
import json
import math
import os
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--deadline', type=float, required=True)
parser.add_argument('--max-seconds', type=int, required=True)
parser.add_argument('command', nargs=argparse.REMAINDER)
args = parser.parse_args()
argv = args.command[1:] if args.command[:1] == ['--'] else args.command
now = time.time()
seconds = min(args.max_seconds, math.floor(args.deadline - now))
assert argv and 0 < seconds <= 900, 'Shared training budget exhausted; no training started'
print(json.dumps(dict(stage='budget_bound', server_epoch=now, shared_deadline=args.deadline,
                      effective_timeout_seconds=seconds, kill_grace_seconds=30)), flush=True)
os.execv('/usr/bin/timeout', ['/usr/bin/timeout', '--signal=TERM', '--kill-after=30s', str(seconds), *argv])
