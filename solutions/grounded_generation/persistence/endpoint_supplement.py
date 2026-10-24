"""Small immutable F1 endpoint backup; same-byte transport only, no GPU or scoring."""
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

S=Path(__file__).resolve().parents[1]
B=Path('/workspace/patientphex-runs/round2-20261023/research/grounded-generation-v1')
TRANSPORT=Path('/workspace/patientphex-worktrees/round2-sequence-crf/solutions/sequence_crf/patient_balanced_loss/endpoint_supplement.py')
assert hashlib.sha256(TRANSPORT.read_bytes()).hexdigest()=='e40d61d31a948c0426aa4960402f60371c68f615657bd59fe7740c0ad768bcde'
spec=importlib.util.spec_from_file_location('_grounded_exact_backup_transport',TRANSPORT)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.MOUNT='5a9955da-5240-41cc-9e52-05d3d8da8a3c'
m.TARGET='research/grounded-generation-v1/F1-endpoint-supplement-20261024'
m.CHUNK=131072  # manifest bound; archive transport pieces stay64KiB below.
m.REMOTE_VERIFY=m.REMOTE_VERIFY.replace('patient-balanced-alpha05-endpoint-persistent-v1','grounded-F1-endpoint-persistent-v1').replace('original_raw_repackaged=False','original_raw_archives_copied_byteexact=True')


def pack(out):
    entries={}
    for path in sorted(S.rglob('*')):
        rel=path.relative_to(S)
        if not path.is_file() or path.suffix not in {'.py','.json','.md','.diff'}:continue
        if rel.parts[0]=='f0' or '__pycache__' in rel.parts:continue
        # Keep actual source/plans/outcomes, not superseded minor readiness snapshots.
        if rel.parts[0]=='evidence' and path.name not in {'F1-result.json','formal-terminal.json','inference-terminal.json','held-normalizer-query-terminal.json','normalization-complete.json','normalization-review.json','engineering-review.json','extraction-normalization-preparation.json','final-F1-review.json'}:continue
        entries['source/'+rel.as_posix()]=path
    for name in ['F1-quotes-v1','F1-normalization-preparation','F1-normalized-v1','quality-F1-metadata-recovery-v1','strong-support-union-F1-v1']:
        for path in sorted((B/name).rglob('*')):
            if path.is_file():entries['runs/'+path.relative_to(B).as_posix()]=path
    phases={}
    for name in ['formal-v1','infer-F1-v1','normalization-held-F1-v1']:
        collection=json.loads((B/name/'collection.json').read_text());phases[name]=collection
        for suffix in ['collection.json','package.json','collection-transfer/archive-binding.json']:
            entries['runs/'+name+'/'+suffix]=B/name/suffix
        archive=Path(collection['archive']['path']);assert m.ref(archive)==collection['archive']
        entries['runs/'+name+'/original-'+archive.name]=archive
    quality=json.loads((S/'quality/metadata_recovery/plan.json').read_text())
    strong=json.loads((S/'strong_support_union/plan.json').read_text())
    for name in ['baseline','baseline_metrics','F1_input','byte_helper','evaluator','validator','baseline_quality_summary','baseline_quality_freeze','baseline_evaluation_bound']:
        entries['fixed-prior/'+name+Path(quality['references'][name]['path']).suffix]=Path(quality['references'][name]['path'])
    for name in ['baseline','baseline_metrics','strong_result_pointer']:
        entries['fixed-strong/'+name+Path(strong['references'][name]['path']).suffix]=Path(strong['references'][name]['path'])
    review=Path('/workspace/patientphex-runs/round2-20261023/validation/grounded-F1-final-review')
    assert m.ref(review/'review.json')['sha256']=='0fcffd7b8ba8b0a7bff5c8f499c6998c9f780c26e1815862b8761e2d47495c72'
    for name in ['review.json','snapshot-manifest.json']:entries['review/'+name]=review/name
    entries['transport/original_endpoint_supplement.py']=TRANSPORT
    payload={};secrets=[v.encode() for k,v in os.environ.items() if len(v)>8 and any(word in k.upper() for word in ['TOKEN','PASSWORD','PASSWD','COOKIE','SECRET'])]
    for name,path in entries.items():
        m.safe(name);assert path.is_file() and not path.is_symlink() and not path.name.endswith('.gold.jsonl')
        data=path.read_bytes();assert not any(value in data for value in secrets),'Sensitive material forbidden in endpoint backup'
        payload[name]=data
    locators=dict(training_checkpoints=phases['formal-v1']['actual_external_checkpoints'],
        original_phase_archives={name:dict(mount=v['mount'],archive=v['remote_archive']) for name,v in phases.items()},
        public_assets='Exact public base/Sap/HPO/bank and input locators remain in bundled source plans/collections; large tensors are not copied',
        original_training_state_kept_on_A100=True,no_claim_of_training_state_on_backup_server=True)
    manifest=dict(schema='grounded-F1-complete-endpoint-members-v1',members={name:dict(source_path=str(entries[name].resolve()),bytes=len(data),sha256=m.digest(data)) for name,data in payload.items()},external_resources=locators,
        exclusions=['F0 preparation or future work','large model/adapter/optimizer/RNG tensors and public bank','training and held gold bodies','credentials and transport chatter'],
        original_three_archives_copied_byteexact=True,local_CPU_endpoints_complete=True,no_NN_no_score_no_native_decode=True)
    data=m.encoded(manifest);assert len(data)<=m.CHUNK;m.publish(out/'members-manifest.json',data)
    archive=out/'endpoint-supplement.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for name,value in sorted(dict(payload,**{'members-manifest.json':data}).items()):
            info=zipfile.ZipInfo(name,(2026,10,24,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=(stat.S_IFREG|0o600)<<16
            z.writestr(info,value,compress_type=zipfile.ZIP_DEFLATED,compresslevel=9)
    with archive.open('rb') as f:os.fsync(f.fileno())
    m.sync(out)
    with zipfile.ZipFile(archive) as z:
        assert set(z.namelist())==set(payload)|{'members-manifest.json'}
        for name,value in payload.items():assert z.read(name)==value
    raw=archive.read_bytes();chunks=[]
    for offset in range(0,len(raw),65536):
        piece=raw[offset:offset+65536];chunks.append(dict(name=f'chunks/{offset:012d}.bin',offset=offset,bytes=len(piece),sha256=m.digest(piece)))
    package=dict(archive=m.ref(archive),manifest=m.ref(out/'members-manifest.json'),members=len(payload),unpacked_bytes=sum(map(len,payload.values())),target_mount=m.MOUNT,target=m.TARGET,chunks=chunks,
        source=m.ref(__file__),transport_source=m.ref(TRANSPORT),external_archives=[],external_resources=locators,
        package_readback_exact=True,all_original_sources_SHA_in_manifest=True,source_snapshot_excludes_F0=True)
    m.publish(out/'package.json',m.encoded(package));print(json.dumps(dict(package=m.ref(out/'package.json'),archive=m.ref(archive),members=len(payload),chunks=len(chunks))))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['pack','deploy']);a=ap.parse_args();os.umask(0o077)
    out=B/'endpoint-F1-supplement-v1';out.mkdir(exist_ok=True)
    lock=(out/'owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if a.stage=='pack':pack(out)
    else:
        for p in (out/'deployment').glob('*.backoff.json'):
            assert time.time() >= json.loads(p.read_text())['retry_not_before_unix']+5
        m.deploy(B,out,Path('/workspace/patientphex-tools'))
