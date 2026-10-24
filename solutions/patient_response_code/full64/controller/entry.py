"""Explicit stage dispatcher; invocation alone never advances another stage.

All stages preserve their original exclusive intents and UNKNOWN handling.
No automatic model restart, repeated mutation or budget reset is added here.
"""
from pathlib import Path
import argparse, subprocess, sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare', 'deploy', 'preflight', 'start', 'watch',
        'inspect', 'start-server', 'start-fetch', 'collect-export', 'collect-training',
        'recover-watch', 'bootstrap-target', 'bootstrap-prepare', 'bootstrap-start', 'bind-inference'])
    p.add_argument('--controller', required=True, type=Path)
    p.add_argument('--epoch', type=int, choices=[1, 2], default=1)
    p.add_argument('--probe-terminal', type=Path); p.add_argument('--cost-decision', type=Path)
    p.add_argument('--recovery'); p.add_argument('--not-before', type=float)
    p.add_argument('--inference-directory', type=Path)
    a = p.parse_args(); S = Path(__file__).resolve().parent; C = str(a.controller)
    if a.action == 'prepare':
        if not a.probe_terminal or not a.cost_decision: p.error('Actual probe terminal and cost decision required')
        args = [str(S/'prepare.py'), '--directory', C, '--probe-terminal', str(a.probe_terminal), '--cost-decision', str(a.cost_decision)]
    elif a.action == 'deploy': args = [str(S/'deploy.py'), '--directory', C]
    elif a.action == 'watch': args = [str(S/'launch_watch.py'), '--directory', C]
    elif a.action == 'recover-watch':
        if not a.recovery or a.not_before is None: p.error('Recovery name and read-only backoff time required')
        args = [str(S/'launch_state_recovery.py'), '--controller', C, '--recovery', a.recovery, '--not-before', str(a.not_before)]
    elif a.action.startswith('bootstrap-'):
        action = {'bootstrap-target': 'target-stage', 'bootstrap-prepare': 'prepare', 'bootstrap-start': 'start'}[a.action]
        args = [str(S/'bootstrap.py'), action, '--controller', C]
    elif a.action == 'collect-training': args = [str(S/'collect_training_terminal.py'), '--controller', C]
    elif a.action == 'bind-inference':
        if not a.inference_directory: p.error('New inference directory required')
        args = [str(S.parent/'inference/bind_after_training.py'), '--controller', C, '--directory', str(a.inference_directory)]
    else:
        action = 'start-epoch' if a.action == 'start' else a.action
        args = [str(S/'operations.py'), action, '--directory', C, '--epoch', str(a.epoch)]
    raise SystemExit(subprocess.call([sys.executable] + args))


if __name__ == '__main__': main()
