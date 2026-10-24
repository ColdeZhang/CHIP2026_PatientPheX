"""Reuse the verified private backup transport under a new full64 namespace.

Only call after finalize.py has produced the one frozen terminal archive.
The original F0 source is read unchanged; no model or quality entry exists.
"""
from pathlib import Path
import hashlib
import importlib.util

source = Path(__file__).resolve().parents[1] / 'f0/store_terminal.py'
assert hashlib.sha256(source.read_bytes()).hexdigest() == 'ab7151fb16a4d2765d89d338941fa1046d1648fc1555c71d3710d793826f8528'
spec = importlib.util.spec_from_file_location('original_private_store', source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.REMOTE = 'research/patient-response-code/backups/full64-a800-terminal-v1'
module.ABS = '/home/zhangyuheng/projects/CHIP2026_PatientPheX/' + module.REMOTE
if __name__ == '__main__':
    module.main()
