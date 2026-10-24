"""Build the frozen QSC F1 supplement; original terminal archives stay byteexact.

Only immutable entry refs, explicit closed F1 directories, and safe review refs
are selected. No current source-tree traversal, model, parser, gold or scoring.
The owner may later use deploy; this task only prepares one local package.
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
N = R/'research/quote-set-completion-v1'
REVIEW = R/'validation/quote-set-completion-F1-final-review'
ENTRY_SHA = 'a9044674b34aca2b29ba21822eb9496c4f3bbcb39ef5c0b614438141ce8d60e2'
REVIEW_SHA = 'b3656560a212f5fda50e2e58ef5a2cefd87eee272d8912c0dab3cfe55c16914a'
REVIEW_MANIFEST_SHA = '2f2f753ed295bec8af9c8140e868b532f4aab45464fa952c7871d83867d86c74'
PRIOR = Path('/workspace/patientphex-worktrees/explore-grounded-sampled-consensus/solutions/grounded_sampled_consensus/persistence/F1_supplement.py')
spec = importlib.util.spec_from_file_location('_QSC_F1_prior_persistence', PRIOR)
prior = importlib.util.module_from_spec(spec); spec.loader.exec_module(prior)
m = prior.m
m.TARGET = 'research/quote-set-completion-v2/F1-supplement-v1'
m.REMOTE_VERIFY = m.REMOTE_VERIFY.replace('grounded-sampled-F1-endpoint-persistent-v1', 'quote-set-completion-F1-endpoint-persistent-v1')
PHASES = {
    'native-cpu-v2': (15,'CPU_reported_success',True),
    'probe-v2': (10,'probe_reported_success',False),
    'probe-path-recovery-v1': (41,'probe_reported_success',False),
    'probe-two-shape-v1': (23,'probe_reported_success',True),
    'formal-v1': (34,'formal_reported_success',True),
    'inference-F1-v1': (92,'inference_reported_success',True),
    'normalization-F1-v1': (14,'query_reported_success',True),
}


def read(path): return json.loads(Path(path).read_text())


def checked(item):
    path = Path(item['path']); assert m.ref(path) == item
    return path


def inventory():
    entry_path=N/'F1-complete-review-entry.json'
    assert m.ref(entry_path)['sha256']==ENTRY_SHA
    assert m.ref(REVIEW/'review.json')['sha256']==REVIEW_SHA
    assert m.ref(REVIEW/'manifest.json')['sha256']==REVIEW_MANIFEST_SHA
    entry=read(entry_path);review=read(REVIEW/'review.json');review_manifest=read(REVIEW/'manifest.json')
    assert str(review['status']).startswith('PASS') and review['entry']==m.ref(entry_path)
    assert entry['generation']['rows']==39 and entry['training']['updates']==33 and entry['training']['epochs']==1
    assert entry['fixed_endpoint']['formula']=='B4 UNION all completion relations'
    assert entry['result']['new_v2']==1 and entry['result']['old_v2']==0
    entries={'runs/F1-complete-review-entry.json':entry_path}
    for relative,item in entry['source_refs'].items():
        path=checked(item);assert path.resolve().is_relative_to(S)
        assert not any('F0' in part or part.lower().startswith('f0') for part in Path(relative).parts)
        entries['source/'+relative]=path
    for relative,item in entry['refs'].items():
        path=checked(item);assert path.resolve().is_relative_to(N)
        assert not any('F0' in part for part in Path(relative).parts)
        entries['runs/'+relative]=path
    # Explicit closed F1 allowlist only; never rglob the current QSC source tree.
    for folder in ['F1-extraction-v1','F1-normalization-preparation-v1','F1-normalized-v1','quality-F1-v1']:
        for path in sorted((N/folder).iterdir()):
            assert path.is_file() and not path.is_symlink()
            entries['runs/'+path.relative_to(N).as_posix()]=path
    held=read(N/'F1-input-preparation-recovery-v1/manifest.json')
    assert held['rows']==39 and held['known_triples']==328 and held['no_held_gold']
    assert set(held['files'])=={'prompts','known','audit','encoded'}
    for item in held['files'].values():
        path=checked(item);assert path.parent==N/'F1-input-preparation-recovery-v1'
        entries['runs/'+path.relative_to(N).as_posix()]=path
    collections={}
    for name,(count,success,expected) in PHASES.items():
        directory=N/name;c=read(directory/'collection.json')
        assert c['all_actual_SHA_verified'] and len(c['files'])==count and c[success] is expected
        collections[name]=c
        for suffix in ['collection.json','collection-transfer/archive-binding.json']:
            entries['runs/'+name+'/'+suffix]=directory/suffix
        package=checked(c['package']);assert package.parent==directory
        entries['runs/'+name+'/'+package.name]=package
        index_path=checked(c['member_index']);entries['runs/'+name+'/member-index.json']=index_path
        archive=checked(c['archive']);entries['runs/'+name+'/original-'+archive.name]=archive
        # Hash opaque original members; no parser/tokenizer/model invocation.
        with zipfile.ZipFile(archive) as z:
            index_bytes=z.read('member-index.json');assert index_bytes==index_path.read_bytes()
            index=json.loads(index_bytes)
            assert len(index['files'])==count and len(z.namelist())==len(set(z.namelist()))
            assert set(z.namelist())==set(index['files'])|{'member-index.json'}
            for member,item in index['files'].items():
                m.safe(member);info=z.getinfo(member)
                assert not info.is_dir() and not stat.S_ISLNK(info.external_attr>>16)
                data=z.read(member);assert len(data)==item['bytes'] and m.digest(data)==item['sha256']
    for stage in ['extract','finalize','quality']:
        plan=read(N/('postprocess-plans/'+stage+'-plan.json'))
        sources=plan['sources'].get('reused',plan['sources'])
        for name,item in sources.items():
            path=checked(item)
            if path.resolve().is_relative_to(S):continue
            # Source-only references: never recursively follow input/gold/model paths.
            assert path.suffix in {'.py','.json'} and path.stat().st_size<65536
            entries['inherited-source/'+stage+'/'+name]=path
    quality=read(N/'postprocess-plans/quality-plan.json')
    for name in ['baseline','baseline_metrics','baseline_freeze','baseline_plan']:
        item=quality['inputs'][name];entries['baseline/'+name+Path(item['path']).suffix]=checked(item)
    for name in ['review.json','manifest.json']:
        entries['review/'+name]=REVIEW/name
    safe_refs=[review_manifest['selection'],review_manifest['builder']]
    safe_refs += [v['snapshot'] for v in review_manifest['snapshots']]
    safe_refs += [v['projection'] for v in review_manifest['projections']]
    for item in safe_refs:
        path=checked(item)
        relative=path.relative_to(REVIEW).as_posix() if path.is_relative_to(REVIEW) else 'builder/'+path.name
        entries['review/'+relative]=path
    entries['source/persistence/F1_supplement.py']=Path(__file__)
    entries['source/persistence/README.md']=S/'persistence/README.md'
    entries['transport/sampled_F1_supplement.py']=PRIOR
    entries['transport/original_endpoint_supplement.py']=prior.TRANSPORT
    formal=collections['formal-v1'];ack=read(N/'formal-v1/epoch-001-export-ack.json')
    assert ack['status']=='persistent_checkpoint_verified' and ack['global_step']==33 and ack['completed_epochs']==1
    assert ack['persistent_mount_id']==m.MOUNT
    assert ack['fsync_all_files'] and ack['reopened_final_filename_SHA_all_files'] and ack['destination_directories_fsynced']
    assert all(v['actual_ACK_sha256']==m.ref(N/'formal-v1/epoch-001-export-ack.json')['sha256'] for v in formal['actual_external_checkpoints'].values())
    locators=dict(actual_full_checkpoint_ACK=ack,actual_model_optimizer_files=formal['actual_external_checkpoints'],
        original_terminal_archives={name:dict(mount=c['mount'],archive=c['remote_archive'],content_members=len(c['files'])) for name,c in collections.items()},
        public_model_and_bank_assets='Frozen inference plan and normalization target-assets retain exact source/model/public-bank locators; no tensor duplication',
        original_d52_parent_lineage='Warm-start parent exact SHA in formal checkpoint metadata and frozen source plans; old weights not copied',
        source_SHM_not_persistent=True,no_NN_no_score_no_decode=True)
    for name,path in entries.items():
        m.safe(name);assert path.is_file() and not path.is_symlink()
        assert path.suffix not in {'.pt','.safetensors','.npy','.bin'} and not path.name.endswith('.gold.jsonl')
    return entries,locators


def pack(out):
    assert not (out/'package.json').exists(),'Original package immutable; do not repackage'
    entries,locators=inventory();payload={}
    secrets=[v.encode() for k,v in os.environ.items() if len(v)>8 and any(s in k.upper() for s in ['TOKEN','PASSWORD','PASSWD','COOKIE','SECRET'])]
    for name,path in entries.items():
        data=path.read_bytes();assert not any(secret in data for secret in secrets);payload[name]=data
    manifest=dict(schema='quote-set-completion-F1-endpoint-members-v1',
        members={name:dict(source_path=str(entries[name].resolve()),bytes=len(data),sha256=m.digest(data)) for name,data in payload.items()},
        external_resources=locators,original_terminal_ZIPs_copied_byteexact=True,
        original_terminal_content_counts={name:value[0] for name,value in PHASES.items()},
        failed_and_partial_originals_preserved_as_failed=True,
        exclusions=['new F0 source and output','held gold bodies','large weights/optimizer/RNG/public bank','large live training logs and transport chatter'],
        no_NN_no_score_no_decode=True)
    encoded=m.encoded(manifest);assert len(encoded)<=m.CHUNK
    m.publish(out/'members-manifest.json',encoded);archive=out/'endpoint-supplement.zip'
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
        source=m.ref(__file__),transport_source=m.ref(prior.TRANSPORT),prior_protocol_source=m.ref(PRIOR),
        external_archives=[],external_resources=locators,package_readback_exact=True,
        all_original_sources_SHA_in_manifest=True,review=m.ref(REVIEW/'review.json'))
    m.publish(out/'package.json',m.encoded(package))
    print(json.dumps(dict(package=m.ref(out/'package.json'),archive=package['archive'],manifest=package['manifest'],
        content_members=len(payload),ZIP_members_including_embedded_manifest=len(payload)+1,chunks=len(chunks))))


if __name__=='__main__':
    assert __debug__;ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['inventory','pack','deploy']);a=ap.parse_args()
    os.umask(0o077);prior.guard_gold()
    if a.stage=='inventory':
        entries,_=inventory();print(json.dumps(dict(content_members=len(entries),bytes=sum(p.stat().st_size for p in entries.values()),local_only=True)));raise SystemExit(0)
    out=N/'endpoint-F1-supplement-v1';out.mkdir(exist_ok=True)
    owner=(out/'owner.lock').open('a');fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if a.stage=='pack':pack(out)
    else:
        for path in (out/'deployment').glob('*.backoff.json'):
            assert time.time()>=read(path)['retry_not_before_unix']+5
        m.deploy(N,out,Path('/workspace/patientphex-tools'))
