"""Incremental F0 endpoint backup; reference the restored F1 chain, copy no weights."""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import time
import zipfile

S = Path(__file__).resolve().parents[1]
B = Path('/workspace/patientphex-runs/round2-20261023/research/grounded-generation-v1')
TRANSPORT = Path('/workspace/patientphex-worktrees/round2-sequence-crf/solutions/sequence_crf/patient_balanced_loss/endpoint_supplement.py')
assert hashlib.sha256(TRANSPORT.read_bytes()).hexdigest() == 'e40d61d31a948c0426aa4960402f60371c68f615657bd59fe7740c0ad768bcde'
spec = importlib.util.spec_from_file_location('_grounded_F0_samebyte_transport', TRANSPORT)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
m.MOUNT = '5a9955da-5240-41cc-9e52-05d3d8da8a3c'
m.TARGET = 'research/grounded-generation-v1/F0-endpoint-supplement-20261024'
m.CHUNK = 131072
m.REMOTE_VERIFY = m.REMOTE_VERIFY.replace('patient-balanced-alpha05-endpoint-persistent-v1', 'grounded-F0-endpoint-persistent-v1').replace('original_raw_repackaged=False', 'original_raw_archives_copied_byteexact=True')


def pack(out, review_path, review_sha):
    review_path = Path(review_path).resolve()
    assert m.ref(review_path)['sha256'] == review_sha
    entries = {}
    for path in sorted((S / 'f0').rglob('*')):
        if path.is_file() and path.suffix in {'.py', '.json', '.md', '.diff'} and '__pycache__' not in path.parts:
            entries['source/f0/' + path.relative_to(S / 'f0').as_posix()] = path
    for path in [Path(__file__), S / 'f0/result.json', S / 'experiments/GG-20261024-003.md', S / 'persistence/F1-receipt.json']:
        entries['source/' + path.relative_to(S).as_posix()] = path
    for name in ['F0-preparation-v1', 'F0-quotes-v1', 'F0-normalization-preparation', 'F0-normalized-v1', 'F0-pure-relations-v1', 'strong-support-union-F0-v1']:
        for path in sorted((B / name).rglob('*')):
            if path.is_file(): entries['runs/' + path.relative_to(B).as_posix()] = path
    phases = {}
    for name in ['F0-native-cpu-v1', 'infer-F0-v1', 'normalization-held-F0-v1']:
        collection = json.loads((B / name / 'collection.json').read_text()); phases[name] = collection
        for suffix in ['collection.json', 'package.json', 'collection-transfer/archive-binding.json']:
            entries['runs/' + name + '/' + suffix] = B / name / suffix
        archive = Path(collection['archive']['path']); assert m.ref(archive) == collection['archive']
        entries['runs/' + name + '/original-' + archive.name] = archive
    for name in ['F0-copy-v1/copy-complete.json', 'F0-final-review-entry.json']:
        entries['runs/' + name] = B / name
    plan = json.loads((S / 'f0/quality-plan.json').read_text())
    for name in ['baseline', 'baseline_metrics', 'strong_result_pointer']:
        entries['fixed-prior/' + name + Path(plan['references'][name]['path']).suffix] = Path(plan['references'][name]['path'])
    entries['review/review.json'] = review_path
    for name in ['snapshot-manifest.json', 'evidence-manifest.json']:
        path = review_path.parent / name
        if path.is_file(): entries['review/' + name] = path
    entries['transport/original_endpoint_supplement.py'] = TRANSPORT
    previous = json.loads((S / 'persistence/F1-receipt.json').read_text())
    locators = dict(
        restored_F1_chain=previous,
        F1_actual_restore=m.ref(B.parents[1] / 'grounded-qwen-F1-supplements-restore-20261024/complete.json'),
        F0_inference_cache=json.loads((B / 'F0-copy-v1/copy-complete.json').read_text()),
        original_phase_archives={name: dict(mount=v['mount'], archive=v['remote_archive']) for name, v in phases.items()},
        original_training_full_state_remains_on_A100=True,
        model_bank_and_old_F1_resources_not_duplicated=True)
    payload = {}
    secrets = [v.encode() for k, v in os.environ.items() if len(v) > 8 and any(s in k.upper() for s in ['TOKEN', 'PASSWORD', 'PASSWD', 'COOKIE', 'SECRET'])]
    for name, path in entries.items():
        m.safe(name); assert path.is_file() and not path.is_symlink() and not path.name.endswith('.gold.jsonl')
        data = path.read_bytes(); assert not any(secret in data for secret in secrets)
        payload[name] = data
    manifest = dict(schema='grounded-F0-complete-endpoint-members-v1',
        members={name: dict(source_path=str(entries[name].resolve()), bytes=len(data), sha256=m.digest(data)) for name, data in payload.items()},
        external_resources=locators,
        exclusions=['training/held gold bodies', 'large model/adapter/optimizer/RNG/public bank', 'old F1 package contents', 'conditional full64 or canonical-name future work', 'credentials and transport chatter'],
        original_three_archives_copied_byteexact=True, no_NN_no_score_no_decode=True)
    manifest_bytes = m.encoded(manifest); assert len(manifest_bytes) <= m.CHUNK
    m.publish(out / 'members-manifest.json', manifest_bytes)
    archive = out / 'endpoint-supplement.zip'
    with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for name, value in sorted(dict(payload, **{'members-manifest.json': manifest_bytes}).items()):
            info = zipfile.ZipInfo(name, (2026, 10, 24, 0, 0, 0)); info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (stat.S_IFREG | 0o600) << 16
            z.writestr(info, value, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    with archive.open('rb') as f: os.fsync(f.fileno())
    m.sync(out)
    with zipfile.ZipFile(archive) as z:
        assert set(z.namelist()) == set(payload) | {'members-manifest.json'}
        assert z.read('members-manifest.json') == manifest_bytes
        for name, value in payload.items(): assert z.read(name) == value
    raw = archive.read_bytes(); chunks = []
    for offset in range(0, len(raw), 65536):
        piece = raw[offset:offset + 65536]
        chunks.append(dict(name=f'chunks/{offset:012d}.bin', offset=offset, bytes=len(piece), sha256=m.digest(piece)))
    package = dict(archive=m.ref(archive), manifest=m.ref(out / 'members-manifest.json'), members=len(payload),
        unpacked_bytes=sum(map(len, payload.values())), target_mount=m.MOUNT, target=m.TARGET, chunks=chunks,
        source=m.ref(__file__), transport_source=m.ref(TRANSPORT), external_archives=[], external_resources=locators,
        package_readback_exact=True, all_original_sources_SHA_in_manifest=True, review=m.ref(review_path))
    m.publish(out / 'package.json', m.encoded(package))
    print(json.dumps(dict(package=m.ref(out / 'package.json'), archive=m.ref(archive), members=len(payload), chunks=len(chunks))))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('stage', choices=['pack', 'deploy'])
    ap.add_argument('--review'); ap.add_argument('--review-sha256'); a = ap.parse_args(); os.umask(0o077)
    out = B / 'endpoint-F0-supplement-v1'; out.mkdir(exist_ok=True)
    lock = (out / 'owner.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if a.stage == 'pack':
        assert a.review and a.review_sha256; pack(out, a.review, a.review_sha256)
    else:
        for p in (out / 'deployment').glob('*.backoff.json'):
            assert time.time() >= json.loads(p.read_text())['retry_not_before_unix'] + 5
        m.deploy(B, out, Path('/workspace/patientphex-tools'))
