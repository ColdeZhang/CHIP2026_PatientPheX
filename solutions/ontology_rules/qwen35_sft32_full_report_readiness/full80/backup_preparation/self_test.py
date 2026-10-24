#!/usr/bin/env python3
"""Synthetic archive fidelity and fail-closed selection; no real experiment reads."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time

sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location('terminal_packager_synthetic', Path(__file__).with_name('package.py'))
p = importlib.util.module_from_spec(spec); spec.loader.exec_module(p)


def reject(action):
    try:
        action()
    except (ValueError, FileNotFoundError):
        return
    raise AssertionError('Bad synthetic fixture accepted')


def main():
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix='terminal-backup-synthetic-') as temporary:
        root = Path(temporary)
        a, b = root / 'a.bin', root / 'b.json'
        a.write_bytes(b'\x00\xff\r\nutf8:' + '中文'.encode()); b.write_text('{"literal":"NIL;X"}\n')
        one, data1 = p.record(a, 'full80/raw.bin'); two, data2 = p.record(b, 'validation/review.json')
        manifest = p.encoded({'members': [one, two]})
        payload = p.archive_payload([one, two], [data1, data2], manifest)
        archive = root / 'good.tar.gz'; archive.write_bytes(payload)
        p.require(p.verify_archive(archive, p.digest(manifest)) == 2, 'Roundtrip member count differs')
        p.require(payload == p.archive_payload([one, two], [data1, data2], manifest), 'Same snapshot is nondeterministic')
        reject(lambda: p.verify_archive(archive, '0' * 64))
        bad = p.archive_payload([one, two], [b'changed', data2], manifest)
        corrupted = root / 'bad.tar.gz'; corrupted.write_bytes(bad)
        reject(lambda: p.verify_archive(corrupted, p.digest(manifest)))
        reject(lambda: p.archive_payload([one, one], [data1, data1], manifest))
        reject(lambda: p.archive_payload([dict(one, archive_path='../escape')], [data1], manifest))
        reject(lambda: p.record(root / 'missing.json', 'full80/missing.json'))
        symlink = root / 'link'; symlink.symlink_to(a)
        reject(lambda: p.file_bytes(symlink))
        policy = {'schema': 'sft32-full-report-terminal-fixed-selection-v1',
                  'review_required_exact_at_execution': True, 'required': {'probe': [], 'full80': []}}
        probe, full = root / 'probe-v1', root / 'full80-v1'; probe.mkdir(); full.mkdir()
        reject(lambda: p.select_files(probe, full, policy, root / 'pending-review.json', '0' * 64))
        reject(lambda: p.select_files(probe, full, policy, b, '0' * 64))
        selected = p.select_files(probe, full, policy, b, p.digest(b.read_bytes()))
        p.require(len(selected) == 241, 'Fixed 240 rows + exact review selection differs')
        reject(lambda: p.record(*selected[0]))
        p.require(not (full / 'persistent-terminal-backup').exists(), 'Missing terminal created an archive directory')
    elapsed = time.monotonic() - started
    p.require(elapsed < 60, 'Synthetic budget exceeded')
    print(json.dumps({'status': 'PASS_SYNTHETIC_SHA_ALL_MEMBERS_FIXED_SELECTION_NO_OUTPUT_ON_MISSING',
        'elapsed_seconds': elapsed, 'real_terminal_or_gold_read': False, 'quality_calls': 0,
        'remote_operations': 0, 'actual_terminal_archive_created': False,
        'package_sha256': p.digest(Path(p.__file__).read_bytes()),
        'self_test_sha256': p.digest(Path(__file__).read_bytes())}, indent=2))


if __name__ == '__main__':
    main()
