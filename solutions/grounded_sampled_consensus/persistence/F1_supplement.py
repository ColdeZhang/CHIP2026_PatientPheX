"""Persist the frozen sampled F1 endpoint; no model, gold, parsing or evaluation.

Only the immutable F1 review-entry allowlist and four closed F1 output directories
are consumed. The three original terminal ZIPs are embedded byte-for-byte.
"""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
import time
import zipfile

S = Path(__file__).resolve().parents[1]
R = Path('/workspace/patientphex-runs/round2-20261023')
B = R / 'research/grounded-sampled-consensus-v1'
G = R / 'research/grounded-generation-v1'
GS = Path('/workspace/patientphex-worktrees/explore-grounded-generation/solutions/grounded_generation')
REVIEW = R / 'validation/grounded-sampled-consensus-F1-final-review'
ENTRY_SHA = '9bd386cc944d81ce8cb52cab86652650498ac50d79a98a65008be67f1eee73fc'
REVIEW_SHA = 'c54828bde312f3f42f4c2fba55ba6203a53b3fd1b643ca59a41e62d969e3186c'
REVIEW_MANIFEST_SHA = 'ddd801e5d30d9e372f3e27ccd5514b1bf82f0914e428aade0c71b7ffb4f8ba91'
TRANSPORT = Path('/workspace/patientphex-worktrees/round2-sequence-crf/solutions/sequence_crf/patient_balanced_loss/endpoint_supplement.py')
assert hashlib.sha256(TRANSPORT.read_bytes()).hexdigest() == 'e40d61d31a948c0426aa4960402f60371c68f615657bd59fe7740c0ad768bcde'
spec = importlib.util.spec_from_file_location('_sampled_F1_samebyte_transport', TRANSPORT)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
m.MOUNT = '5a9955da-5240-41cc-9e52-05d3d8da8a3c'
m.TARGET = 'research/grounded-sampled-consensus-v1/F1-endpoint-supplement-20261024'
m.CHUNK = 131072
m.REMOTE_VERIFY = m.REMOTE_VERIFY.replace('patient-balanced-alpha05-endpoint-persistent-v1', 'grounded-sampled-F1-endpoint-persistent-v1').replace('original_raw_repackaged=False', 'original_raw_archives_copied_byteexact=True')


def read(path): return json.loads(Path(path).read_text())


def checked(item):
    path = Path(item['path']); assert m.ref(path) == item
    return path


def guard_gold():
    forbidden = {str(Path(p).resolve()) for p in [
        '/workspace/patientphex-data/round2-inner4/fold-1/validation16.gold.jsonl',
        '/workspace/patientphex-data/round2-inner4/fold-0/validation16.gold.jsonl',
        '/workspace/patientphex-data/day1/dev16.gold.jsonl']}
    def audit(event, args):
        if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(args[0])).resolve()
            if str(path) in forbidden or path.name.endswith('.gold.jsonl'):
                raise PermissionError('Persistence cannot open/hash gold')
    sys.addaudithook(audit)


def inventory():
    entry_path = B / 'F1-complete-review-entry.json'
    assert m.ref(entry_path)['sha256'] == ENTRY_SHA
    assert m.ref(REVIEW/'review.json')['sha256'] == REVIEW_SHA
    assert m.ref(REVIEW/'manifest.json')['sha256'] == REVIEW_MANIFEST_SHA
    entry = read(entry_path); review = read(REVIEW/'review.json')
    assert review['status'] == 'PASS_COMPLETE_F1_FIXED_INTERSECTION_POSITIVE'
    assert review['entry'] == m.ref(entry_path)
    assert entry['actual_generation']['rows'] == 78 and entry['new_v2'] == 1 and entry['old_v2'] == 0
    entries = {}
    for relative, item in entry['source_refs'].items():
        path = checked(item); assert path.resolve().is_relative_to(S)
        assert not any(part.lower().startswith(('f0', 'postprocess_f0')) for part in path.relative_to(S).parts)
        entries['source/' + relative] = path
    for relative, item in entry['refs'].items():
        path = checked(item); assert path.resolve().is_relative_to(B)
        assert not any('F0' in part for part in path.relative_to(B).parts)
        entries['runs/' + relative] = path
    entries['runs/F1-complete-review-entry.json'] = entry_path
    for folder in ['extraction-v1', 'normalization-preparation-v1', 'normalized-v1', 'quality-F1-v1']:
        for path in sorted((B/folder).rglob('*')):
            if path.is_file(): entries['runs/' + path.relative_to(B).as_posix()] = path
    quality_plan = read(S/'postprocess/quality-plan.json')
    for name, item in quality_plan['sources']['reused'].items():
        entries['inherited-source/' + name] = checked(item)
    for name in ['baseline', 'baseline_metrics', 'baseline_freeze', 'baseline_plan']:
        item = quality_plan['inputs'][name]
        entries['baseline/' + name + Path(item['path']).suffix] = checked(item)
    phases = {}
    for name, count, success_key in [('native-cpu-v1',12,'contract_reported_success'),
                                    ('infer-F1-A30-v1',173,'inference_reported_success'),
                                    ('normalization-runtime-v1',13,'query_reported_success')]:
        directory = B/name; collection = read(directory/'collection.json')
        assert collection['all_actual_SHA_verified'] and len(collection['files']) == count
        if name == 'native-cpu-v1':
            assert collection['wrapper']['returncode'] == 0 and not collection['wrapper']['timed_out']
        else: assert collection[success_key]
        phases[name] = collection
        for suffix in ['collection.json','collection-transfer/archive-binding.json','package.json']:
            entries['runs/'+name+'/'+suffix] = directory/suffix
        entries['runs/'+name+'/member-index.json'] = checked(collection['member_index'])
        archive = checked(collection['archive'])
        entries['runs/'+name+'/original-'+archive.name] = archive
        # Validate exact old container and its original member hashes, never parse raw/model outputs.
        with zipfile.ZipFile(archive) as z:
            index_data = z.read('member-index.json')
            assert index_data == Path(collection['member_index']['path']).read_bytes()
            index = json.loads(index_data)
            assert len(z.namelist()) == len(set(z.namelist()))
            assert set(z.namelist()) == set(index['files']) | {'member-index.json'}
            assert len(index['files']) == count
            for member, item in index['files'].items():
                data = z.read(member)
                assert len(data) == item['bytes'] and m.digest(data) == item['sha256']
    for name in ['review.json','manifest.json','snapshot-selection.json','actual-inputs-check.json','quality-check.json',
                 'check_actual_inputs.py','check_quality.py','build_review.py','check-quality-first-failure.txt']:
        entries['review/'+name] = REVIEW/name
    formal = read(G/'formal-v1/collection.json')
    assert formal['all_actual_SHA_verified'] and formal['formal_reported_success']
    entries['inherited/grounded32-formal-collection.json'] = G/'formal-v1/collection.json'
    entries['inherited/grounded32-F1-receipt.json'] = GS/'persistence/F1-receipt.json'
    for key in ['formal_collection', 'train_target', 'copy_complete', 'original_F1_plan', 'original_same_cache_F0_plan']:
        # The last item is the old grounded32 inference-cache lineage, not this route's new F0 run.
        item = read(S/'a30/infer-plan.json')['input_sources'][key]
        entries['inherited/'+key+'.json'] = checked(item)
    for name in ['normalization/held_query.py','normalization/held_driver.py']:
        entries['inherited-source/'+name] = GS/name
    entries['source/persistence/F1_supplement.py'] = Path(__file__)
    entries['source/persistence/README.md'] = S/'persistence/README.md'
    entries['transport/original_endpoint_supplement.py'] = TRANSPORT
    locators = dict(
        original_grounded32_training=dict(mount=formal['mount'],collection=m.ref(G/'formal-v1/collection.json'),
            complete_state_files=formal['actual_external_checkpoints']),
        persistent_A30_inference_cache=dict(mount=m.MOUNT,copy=read(B/'copy-A30-v1/copy-complete.json'),
            source_service_closed=read(B/'copy-A30-v1/source-service-closed.json'),optimizer_or_RNG_copied=False),
        original_terminal_archives={name:dict(mount=c['mount'],archive=c['remote_archive']) for name,c in phases.items()},
        prior_grounded32_restored_chain=read(GS/'persistence/F1-receipt.json'),
        public_assets=read(S/'normalization_runtime/target-assets.json'),
        model_base_and_tokenizers='Exact SHA, runtime and host locator in frozen a30/infer-plan.json; no model bytes copied',
        inherited_only_F0_name='Original G32 F0 inference-cache plan is lineage only; new sampled F0/94 outputs and sources excluded',
        no_NN_no_score_no_decode=True)
    for name,path in entries.items():
        m.safe(name); assert path.is_file() and not path.is_symlink()
        assert path.suffix not in {'.pt','.safetensors','.npy','.bin'}
        assert 'postprocess_F0' not in name and 'native-F0' not in name
    return entries, locators


def pack(out):
    assert not (out/'package.json').exists(), 'Preserve immutable package; do not repackage'
    entries, locators = inventory(); payload = {}
    secrets = [v.encode() for k,v in os.environ.items() if len(v)>8 and any(s in k.upper() for s in ['TOKEN','PASSWORD','PASSWD','COOKIE','SECRET'])]
    for name,path in entries.items():
        data = path.read_bytes(); assert not any(secret in data for secret in secrets)
        payload[name] = data
    manifest = dict(schema='grounded-sampled-F1-endpoint-members-v1',
        members={name:dict(source_path=str(entries[name].resolve()),bytes=len(data),sha256=m.digest(data)) for name,data in payload.items()},
        external_resources=locators,original_archives_copied_byteexact=True,
        original_phase_member_counts={'native-cpu-v1':12,'infer-F1-A30-v1':173,'normalization-runtime-v1':13},
        exclusions=['new sampled F0/94 preparation, sources and outputs','gold bodies/evaluator execution','model, optimizer/RNG, public bank tensors','transport chatter','prior grounded32 package contents'],
        no_NN_no_score_no_decode=True)
    encoded = m.encoded(manifest); assert len(encoded) <= m.CHUNK
    m.publish(out/'members-manifest.json',encoded)
    archive=out/'endpoint-supplement.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for name,value in sorted(dict(payload,**{'members-manifest.json':encoded}).items()):
            info=zipfile.ZipInfo(name,(2026,10,24,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            info.external_attr=(stat.S_IFREG|0o600)<<16
            z.writestr(info,value,compress_type=zipfile.ZIP_DEFLATED,compresslevel=9)
    with archive.open('rb') as f:os.fsync(f.fileno())
    m.sync(out)
    with zipfile.ZipFile(archive) as z:
        assert len(z.namelist())==len(payload)+1 and set(z.namelist())==set(payload)|{'members-manifest.json'}
        assert z.read('members-manifest.json')==encoded
        for name,value in payload.items():assert z.read(name)==value
    raw=archive.read_bytes();chunks=[]
    for offset in range(0,len(raw),65536):
        piece=raw[offset:offset+65536]
        chunks.append(dict(name=f'chunks/{offset:012d}.bin',offset=offset,bytes=len(piece),sha256=m.digest(piece)))
    package=dict(archive=m.ref(archive),manifest=m.ref(out/'members-manifest.json'),members=len(payload),
        unpacked_bytes=sum(map(len,payload.values())),target_mount=m.MOUNT,target=m.TARGET,chunks=chunks,
        source=m.ref(__file__),transport_source=m.ref(TRANSPORT),external_archives=[],external_resources=locators,
        package_readback_exact=True,all_original_sources_SHA_in_manifest=True,review=m.ref(REVIEW/'review.json'))
    m.publish(out/'package.json',m.encoded(package))
    print(json.dumps(dict(package=m.ref(out/'package.json'),archive=m.ref(archive),manifest=m.ref(out/'members-manifest.json'),content_members=len(payload),ZIP_members=len(payload)+1,chunks=len(chunks))))


if __name__=='__main__':
    assert __debug__
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['inventory','pack','deploy']);a=ap.parse_args()
    os.umask(0o077);guard_gold()
    if a.stage=='inventory':
        entries,_=inventory();print(json.dumps(dict(content_members=len(entries),bytes=sum(p.stat().st_size for p in entries.values()),local_inventory_only=True)));raise SystemExit(0)
    out=B/'endpoint-F1-supplement-v1';out.mkdir(exist_ok=True)
    lock=(out/'owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if a.stage=='pack':pack(out)
    else:
        for p in (out/'deployment').glob('*.backoff.json'):
            assert time.time()>=read(p)['retry_not_before_unix']+5
        m.deploy(B,out,Path('/workspace/patientphex-tools'))
