#!/usr/bin/env python3
"""Local-only fixed terminal backup; no model/evaluator/MCP imports or networking."""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import stat
import tarfile

MAX_MEMBER_BYTES = 1_048_576
MAX_TOTAL_BYTES = 16_777_216
MAX_ARCHIVE_BYTES = 1_048_576
POLICY_SHA = 'bfe2847ef345d0fd787d7c0cbb902cdfd3af7f1976f357f2fa79e167f827bbc5'
PLAN_SHA = 'ed429817ce6f5c28125515f88766e04197ae6a0519145e9c457e7230a2792466'
PROBE_PLAN_SHA = 'b21fb7f92e1621926ac51eae2497f6b0c519df155081dbf744090ca7e20d1734'
ENDPOINTS = ('diagnostic-fulltext-alone', 'primary-oldSFT48-union-fulltext')
ARCHIVE_NAME = 'qwen35-sft48-F0-full-report-terminal-v1.tar.gz'


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def unique(items):
    result = {}
    for key, value in items:
        require(key not in result, 'Duplicate JSON key: ' + key)
        result[key] = value
    return result


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=unique)


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()


def file_bytes(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and stat.S_ISREG(path.lstat().st_mode),
            'Required regular nonsymlink file missing: ' + str(path))
    require(path.stat().st_size <= MAX_MEMBER_BYTES, 'Oversized selected member: ' + str(path))
    return path.read_bytes()


def record(path, archive_path):
    path = Path(path).resolve()
    data = file_bytes(path)
    return {'path': str(path), 'bytes': len(data), 'sha256': digest(data), 'archive_path': archive_path}, data


def require_identity(path, reference):
    data = file_bytes(path)
    require(len(data) == reference['bytes'] and digest(data) == reference['sha256'],
            'Terminal identity changed: ' + str(path))


def valid_member_name(name):
    path = PurePosixPath(name)
    return (name and not path.is_absolute() and '..' not in path.parts and '.' not in path.parts
            and '\\' not in name and not name.endswith('/'))


def select_files(probe, full, policy, review, review_sha256):
    require(policy['schema'] == 'sft48-F0-full-report-terminal-fixed-selection-v1' and
            policy['review_required_exact_at_execution'] is True, 'Wrong fixed selection policy')
    selected = []
    for root_name, root in [('probe', probe), ('full58', full)]:
        for name in policy['required'][root_name]:
            require(valid_member_name(name), 'Unsafe relative member name')
            source = root / name
            require(not source.is_symlink() and source.resolve().is_relative_to(root), 'Path escapes selected root')
            selected.append((source, root_name + '/' + name))
    # Fixed 174 paths, not a glob. Incomplete rows cannot silently disappear.
    for folder in ('generated-ids', 'raw-responses', 'canonical-records'):
        for index in range(58):
            name = 'terminal/%s/%03d.json' % (folder, index)
            source = full / name
            require(not source.is_symlink() and source.resolve().is_relative_to(full), 'Row path escapes full58 root')
            selected.append((source, 'full58/' + name))
    review = Path(review)
    require(review.name.endswith('.json') and not review.is_symlink(), 'Require exact JSON review, not a tool dump')
    review_data = file_bytes(review)
    require(len(review_sha256) == 64 and digest(review_data) == review_sha256, 'Review missing or SHA mismatch')
    require(type(json.loads(review_data, object_pairs_hook=unique)) is dict, 'Review must be a JSON object')
    selected.append((review, 'validation/final-review.json'))
    return selected


def terminal_gate(probe, full):
    """Read only already-completed metadata; never import or execute quality code."""
    require(digest(file_bytes(probe / 'plan.json')) == PROBE_PLAN_SHA and
            digest(file_bytes(full / 'plan.json')) == PLAN_SHA, 'Wrong fixed probe/full58 run')
    p = load(probe / 'terminal/summary.json'); pw = load(probe / 'terminal/wrapper-complete.json')
    require(p['status'] == 'PASS_NATIVE_LONGEST_STRICT_COMPLETE' and p['index'] == 30 and pw['returncode'] == 0,
            'Probe has no successful actual terminal')
    complete = load(full / 'terminal/complete.json')
    require(complete == load(full / 'terminal/summary.json') and
            complete['status'] == 'COMPLETE_FIXED58_SFT48_FULL_REPORT_NO_QUALITY' and
            complete['expected_rows'] == 58 and complete['fresh_rows'] == 57 and complete['cached_rows'] == 1 and
            complete['cached_original_indices'] == [30] and complete['quality_calls'] == 0 and complete['gold_read'] is False,
            'Inference full58 not complete')
    require(complete['plan_sha256'] == PLAN_SHA and
            complete['responses_sha256'] == digest(file_bytes(full / 'terminal/responses.jsonl')) and
            complete['canonical_sha256'] == digest(file_bytes(full / 'terminal/canonical.jsonl')),
            'Complete response/canonical identities differ')
    wrapper = load(full / 'terminal/wrapper-complete.json'); release = load(full / 'terminal/gpu-release.json')
    plan = load(full / 'plan.json')
    require(wrapper['returncode'] == 0 and wrapper['plan_sha256'] == PLAN_SHA and
            wrapper['cap_seconds'] == 1200 and wrapper['term_grace_seconds'] == 30,
            'Full80 wrapper not complete')
    require(release['target_compute_rows'] == [] and len(release['target_csv']) == 1 and
            release['target_csv'][0].split(',')[1].strip() == plan['gpu_uuid'], 'Owned GPU release missing')
    freeze_path = full / 'quality/paired-predictions-freeze.json'; freeze = load(freeze_path)
    require(freeze['status'] == 'both_predeclared_complete_full_predictions_frozen_before_gold' and
            set(freeze['endpoints']) == set(ENDPOINTS) and freeze['gold_read'] is False and
            freeze['new_quality_calls'] == 0 and freeze['documents'] == 16 and freeze['entities'] == 947,
            'Both complete predictions lack the shared prequality freeze')
    summary = load(full / 'quality/summary.json')
    require(summary['status'] == 'complete_predeclared_SFT48_full_report_pair_F0_development' and
            summary['new_v2_calls'] == 2 and summary['old_v2_calls'] == 0 and
            summary['automatic_migration'] is False and set(summary['endpoints']) == set(ENDPOINTS),
            'Two-arm terminal evaluation is incomplete')
    require_identity(freeze_path, summary['paired_freeze'])
    require_identity(full / 'evaluation-bound.json', freeze['plan'])
    expected_shards = {str((full / 'terminal' / folder / ('%03d.json' % index)).resolve())
                       for folder in ('generated-ids', 'raw-responses', 'canonical-records') for index in range(58)}
    require(len(freeze['raw_shards']) == 174 and {r['path'] for r in freeze['raw_shards']} == expected_shards,
            'Common freeze does not bind every one of the 174 exact raw shards')
    for item in freeze['raw_shards']:
        require_identity(item['path'], item)
    for name in ENDPOINTS:
        require_identity(full / 'quality' / name / 'pred.jsonl', freeze['endpoints'][name]['prediction'])
        require_identity(full / 'quality' / name / 'metrics.json', summary['endpoints'][name]['metrics_file'])
        require_identity(full / 'quality' / name / 'validation.json', summary['endpoints'][name]['format_file'])
    # Check all fixed-source identities, without opening any referenced large input/model/gold file.
    for name in ('infer_full58.py', 'launch.py', 'probe_runner.py', 'legacy_runner.py', 'model-source-manifest.json', 'decision.json', 'row-audit.json'):
        require_identity(full / name, plan['files'][name])
    require(complete['source_sha256'] == digest(file_bytes(full / 'infer_full58.py')), 'Actual runner differs')
    require(wrapper['source_sha256'] == digest(file_bytes(full / 'launch.py')), 'Actual launcher differs')
    return {'fixed_probe_plan_sha256': PROBE_PLAN_SHA, 'fixed_full58_plan_sha256': PLAN_SHA,
            'full58_complete_sha256': digest(file_bytes(full / 'terminal/complete.json')),
            'quality_summary_sha256': digest(file_bytes(full / 'quality/summary.json')),
            'paired_freeze_sha256': digest(file_bytes(freeze_path)),
            'new_quality_calls_performed_by_packager': 0, 'gold_read_by_packager': False}


def archive_payload(members, blobs, manifest_bytes):
    require(len(members) == len(blobs) and len({r['archive_path'] for r in members}) == len(members),
            'Duplicate or incomplete member selection')
    require(sum(len(b) for b in blobs) + len(manifest_bytes) <= MAX_TOTAL_BYTES, 'Terminal package too large')
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode='w', format=tarfile.USTAR_FORMAT) as archive:
        for name, data in [('members.json', manifest_bytes)] + [(r['archive_path'], b) for r, b in zip(members, blobs)]:
            require(valid_member_name(name), 'Unsafe archive member')
            info = tarfile.TarInfo(name); info.size = len(data); info.mode = 0o600
            info.uid = info.gid = 0; info.mtime = 0; info.uname = info.gname = ''
            archive.addfile(info, io.BytesIO(data))
    compressed = gzip.compress(output.getvalue(), compresslevel=9, mtime=0)
    require(len(compressed) <= MAX_ARCHIVE_BYTES, 'Compressed terminal archive exceeds fixed 1 MiB scope')
    return compressed


def verify_archive(path, manifest_sha256):
    """No extraction; reject unsafe/duplicate/link members and verify every byte."""
    with tarfile.open(path, 'r:gz') as archive:
        entries = archive.getmembers()
        require(all(e.isfile() and not e.issym() and not e.islnk() and valid_member_name(e.name) for e in entries),
                'Unsafe archive member')
        require(len({e.name for e in entries}) == len(entries), 'Duplicate archive names')
        data = archive.extractfile('members.json').read()
        require(digest(data) == manifest_sha256, 'Embedded member manifest SHA differs')
        manifest = json.loads(data, object_pairs_hook=unique)
        expected = {r['archive_path']: r for r in manifest['members']}
        require(len(expected) == len(manifest['members']) and
                set(e.name for e in entries) == set(expected) | {'members.json'}, 'Archive member set differs')
        for name, record in expected.items():
            data = archive.extractfile(name).read()
            require(len(data) == record['bytes'] and digest(data) == record['sha256'], 'Restored member SHA differs: ' + name)
    return len(expected)


def exclusive_bytes(path, data):
    with Path(path).open('xb') as stream:
        stream.write(data); stream.flush(); os.fsync(stream.fileno())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe-root', required=True)
    parser.add_argument('--full-root', required=True)
    parser.add_argument('--review', required=True)
    parser.add_argument('--review-sha256', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    probe, full = Path(args.probe_root).resolve(), Path(args.full_root).resolve()
    require(probe.name == 'probe-v1' and full.name == 'full58-v1' and probe.parent == full.parent,
            'Require the paired probe-v1 and full58-v1 roots')
    prep = Path(__file__).resolve().parent; policy_path = prep / 'selection-policy.json'
    require(digest(file_bytes(policy_path)) == POLICY_SHA, 'Fixed selection policy bytes changed')
    policy = load(policy_path)
    selected = select_files(probe, full, policy, args.review, args.review_sha256)
    selected.extend((prep / name, 'backup-preparation/' + name) for name in
                    ('package.py', 'README.md', 'selection-policy.json', 'self_test.py', 'self-test-receipt.json'))
    # Gather all required real files first. No output is created on missing terminal/review.
    members, blobs = [], []
    for source, name in selected:
        item, data = record(source, name); members.append(item); blobs.append(data)
    status = terminal_gate(probe, full)
    manifest = {'status': 'SFT48_FULL_REPORT_SELECTED_COMPLETE_TERMINAL_MEMBERS',
        'at_utc': datetime.now(timezone.utc).isoformat(), 'members': members, 'terminal_checks': status,
        'selection_policy_sha256': digest(file_bytes(policy_path)), 'review_sha256': args.review_sha256,
        'review_included_as_exact_supplied_bytes_without_reinterpreting_verdict': True,
        'scope': 'Selected probe/full58 evidence, not a full corpus or reproducibility/model bundle.',
        'excluded': policy['excluded'], 'remote_write_performed': False}
    manifest_bytes = encoded(manifest); payload = archive_payload(members, blobs, manifest_bytes)
    # Inputs must still equal the frozen snapshot just before the exclusive final write.
    for item in members:
        require_identity(item['path'], item)
    out = Path(args.output_dir).resolve()
    require(not out.exists(), 'Backup output already exists; inspect existing intent/receipt instead of repackaging')
    out.mkdir(parents=True, exist_ok=False, mode=0o700)
    exclusive_bytes(out / 'members.json', manifest_bytes)
    archive_path = out / ARCHIVE_NAME; exclusive_bytes(archive_path, payload)
    count = verify_archive(archive_path, digest(manifest_bytes))
    receipt = {'status': 'LOCAL_ARCHIVE_FSYNC_ALL_MEMBERS_VERIFIED_NO_REMOTE_WRITE',
        'archive_path': str(archive_path), 'archive_bytes': len(payload), 'archive_sha256': digest(payload),
        'members_manifest_sha256': digest(manifest_bytes), 'member_count': count,
        'all_member_size_SHA_exact': True, 'gold_read': False, 'quality_calls': 0,
        'remote_write_performed': False, 'review_sha256': args.review_sha256,
        'package_source_sha256': digest(file_bytes(__file__))}
    exclusive_bytes(out / 'local-archive.json', encoded(receipt))
    fd = os.open(str(out), os.O_RDONLY); os.fsync(fd); os.close(fd)
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
