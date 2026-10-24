"""Back up the complete full64/dev36 supply; no gold, evaluator or model calls.

Original immutable terminal archives are copied byte-for-byte. Both complete
training states remain at their independently verified persistent ACK locators.
"""
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
R = Path('/workspace/patientphex-runs/round2-20261023')
B = R / 'research/grounded-generation-full64-v1'
READY = R / 'research/grounded-generation-full64-readiness-v1'
TRANSPORT = Path('/workspace/patientphex-worktrees/round2-sequence-crf/solutions/sequence_crf/patient_balanced_loss/endpoint_supplement.py')
assert hashlib.sha256(TRANSPORT.read_bytes()).hexdigest() == 'e40d61d31a948c0426aa4960402f60371c68f615657bd59fe7740c0ad768bcde'
spec = importlib.util.spec_from_file_location('_grounded_full64_samebyte_transport', TRANSPORT)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
m.MOUNT = '5a9955da-5240-41cc-9e52-05d3d8da8a3c'
m.TARGET = 'research/grounded-generation-full64-v1/dev36-endpoint-supplement-20261024'
m.CHUNK = 131072
m.REMOTE_VERIFY = m.REMOTE_VERIFY.replace('patient-balanced-alpha05-endpoint-persistent-v1', 'grounded-full64-dev36-endpoint-persistent-v1').replace('original_raw_repackaged=False', 'original_raw_archives_copied_byteexact=True')


def pack(out, review_path, review_sha):
    review_path = Path(review_path).resolve()
    assert m.ref(review_path)['sha256'] == review_sha
    freeze = json.loads((B / 'dev36-pure-relations-v1/freeze.json').read_text())
    assert freeze['all36_complete'] and freeze['no_gold_no_score'] and freeze['nonrelation_bytes_exact']
    campaign = json.loads((B / 'formal-v1/campaign-complete.json').read_text())
    assert campaign['status'] == 'TWO_COMPLETE_EPOCHS_DURABLE_ACKS_VERIFIED'
    entries = {}
    for dirname in ['full64_runtime', 'training_shared', 'full64_inference']:
        for path in sorted((S / dirname).rglob('*')):
            if path.is_file() and path.suffix in {'.py', '.json', '.md', '.diff'} and '__pycache__' not in path.parts:
                entries['source/' + path.relative_to(S).as_posix()] = path
    for relative in ['schema.py', 'native_tokenizer.py', 'token_census.py',
                     'inference/raw_store.py', 'inference/decode.py',
                     'normalization/strict.py', 'normalization/semantic.py',
                     'normalization/public_bank.py', 'normalization/policy_v1.json',
                     'normalization/held_query.py', 'normalization/held_driver.py',
                     'quality/metadata_recovery/support.py',
                     'experiments/GG-20261024-004.md', 'persistence/F0-receipt.json']:
        entries['source/' + relative] = S / relative
    entries['source/persistence/full64_supplement.py'] = Path(__file__)
    for name in ['dev36-quotes-v1', 'dev36-normalization-preparation',
                 'dev36-normalized-v1', 'dev36-pure-relations-v1']:
        for path in sorted((B / name).rglob('*')):
            if path.is_file(): entries['runs/' + path.relative_to(B).as_posix()] = path
    phases = {}
    names = ['native-cpu-v1', 'probe-v1', 'formal-v1', 'infer-dev36-v1']
    norm = json.loads((B / 'dev36-normalization-preparation/manifest.json').read_text())
    if norm['necessary_public_query_count']:
        names.append('normalization-dev36-v1')
    for name in names:
        directory = B / name
        collection = json.loads((directory / 'collection.json').read_text())
        assert collection['all_actual_SHA_verified']
        phases[name] = collection
        for suffix in ['collection.json', 'collection-transfer/archive-binding.json']:
            entries['runs/' + name + '/' + suffix] = directory / suffix
        package = 'collection-package.json' if name == 'formal-v1' else 'package.json'
        entries['runs/' + name + '/' + package] = directory / package
        archive = Path(collection['archive']['path']); assert m.ref(archive) == collection['archive']
        entries['runs/' + name + '/original-' + archive.name] = archive
    assert phases['formal-v1']['formal_reported_success']
    assert phases['infer-dev36-v1']['inference_reported_success']
    for name in ['campaign.json', 'campaign-complete.json', 'controller-plan.json',
                 'epoch-001-export-ack.json', 'epoch-002-export-ack.json',
                 'epoch-001-source-ack-copy.json', 'epoch-002-source-ack-copy.json',
                 'local-watch-complete.json']:
        entries['runs/formal-v1/' + name] = B / 'formal-v1' / name
    for name in ['deployment.json', 'start-intent.json', 'job.json', 'terminal-collection.json']:
        entries['runs/export-guardian-v1/' + name] = B / 'export-guardian-v1' / name
    for name in ['targets-v2/manifest.json', 'targets-v2/normalizer-train64-manifest.json',
                 'token-census-v1/manifest.json']:
        entries['readiness/' + name] = READY / name
    entries['runs/dev36-final-review-entry.json'] = B / 'dev36-final-review-entry.json'
    entries['review/review.json'] = review_path
    for name in ['snapshot-manifest.json', 'evidence-manifest.json']:
        path = review_path.parent / name
        if path.is_file(): entries['review/' + name] = path
    entries['transport/original_endpoint_supplement.py'] = TRANSPORT
    locators = dict(
        both_complete_persistent_checkpoints=phases['formal-v1']['actual_external_checkpoints'],
        actual_epoch_ACKs={str(e): json.loads((B / 'formal-v1' / ('epoch-%03d-export-ack.json' % e)).read_text()) for e in (1, 2)},
        original_phase_archives={name: dict(mount=v['mount'], archive=v['remote_archive']) for name, v in phases.items()},
        prior_restored_chain=json.loads((S / 'persistence/F0-receipt.json').read_text()),
        base_Sap_HPO_bank_assets='Exact published assets remain in source plans and prior restored chain; no tensor duplication',
        SHM_explicitly_not_persistent=True,
        originaldev_gold_and_root_quality_not_accessed=True)
    payload = {}
    secrets = [v.encode() for k, v in os.environ.items() if len(v) > 8 and any(s in k.upper() for s in ['TOKEN', 'PASSWORD', 'PASSWD', 'COOKIE', 'SECRET'])]
    for name, path in entries.items():
        m.safe(name); assert path.is_file() and not path.is_symlink() and not path.name.endswith('.gold.jsonl')
        data = path.read_bytes(); assert not any(secret in data for secret in secrets)
        payload[name] = data
    manifest = dict(schema='grounded-full64-dev36-endpoint-members-v1',
        members={name: dict(source_path=str(entries[name].resolve()), bytes=len(data), sha256=m.digest(data)) for name, data in payload.items()},
        external_resources=locators,
        exclusions=['gold bodies and evaluators', 'large public base/adapter/optimizer/RNG/bank', 'prior F1/F0 package contents', 'transport chatter'],
        original_archives_copied_byteexact=True, no_NN_no_score_no_decode=True)
    encoded = m.encoded(manifest); assert len(encoded) <= m.CHUNK
    m.publish(out / 'members-manifest.json', encoded)
    archive = out / 'endpoint-supplement.zip'
    with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for name, value in sorted(dict(payload, **{'members-manifest.json': encoded}).items()):
            info = zipfile.ZipInfo(name, (2026, 10, 24, 0, 0, 0)); info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (stat.S_IFREG | 0o600) << 16
            z.writestr(info, value, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    with archive.open('rb') as f: os.fsync(f.fileno())
    m.sync(out)
    with zipfile.ZipFile(archive) as z:
        assert set(z.namelist()) == set(payload) | {'members-manifest.json'}
        assert z.read('members-manifest.json') == encoded
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
    out = B / 'endpoint-dev36-supplement-v1'; out.mkdir(exist_ok=True)
    lock = (out / 'owner.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if a.stage == 'pack':
        assert a.review and a.review_sha256; pack(out, a.review, a.review_sha256)
    else:
        for p in (out / 'deployment').glob('*.backoff.json'):
            assert time.time() >= json.loads(p.read_text())['retry_not_before_unix'] + 5
        m.deploy(B, out, Path('/workspace/patientphex-tools'))
