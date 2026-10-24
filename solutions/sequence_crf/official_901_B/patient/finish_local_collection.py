"""Finish byte-exact local collection after a transport-only private bridge."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import zipfile
from collect import ref, sha, read, publish, save, sync_chain, safe_name


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-dir',type=Path,required=True)
    ap.add_argument('--transport-receipt',type=Path,required=True);a=ap.parse_args()
    root=a.run_dir.resolve();binding=read(root/'inference-collection/archive-binding.json')
    plan_path=root/'infer-plan.json';plan=read(plan_path)
    assert sha(plan_path)==binding['plan_sha256']=='1e4e67c2b315e564f7430b35965efab06471d1297db4f365de183f4eb1f73652'
    assert binding['archive']['sha256']=='1df686ceaaf69e55913198482494533ede264dad2effa4561dd4ed4da9593e36'
    assert binding['archive']['bytes']==897210
    assert binding['member_index_sha256']=='643557e709cea3f11a54ce4b9efd7a4084099d57fb353788c0d5ecaae1cb49d1'
    assert binding['mount']==plan['mount']
    assert binding['archive']['path']==str(Path(plan['output']).parent/'complete-collection/patient-complete.zip')
    archive_path=root/'inference-collection/patient-complete.zip'
    assert archive_path.stat().st_size==897210 and sha(archive_path)==binding['archive']['sha256']
    assert a.transport_receipt.is_file();transport=ref(a.transport_receipt)
    local=root/'inference-collected';local.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive_path) as z:
        names=z.namelist();assert len(names)==len(set(names))
        raw_index=z.read('member-index.json')
        assert hashlib.sha256(raw_index).hexdigest()==binding['member_index_sha256']
        index=json.loads(raw_index);files=index['files']
        assert index['plan_sha256']==binding['plan_sha256'] and index['rows']==420
        assert index['remote_mapping']==dict(control=plan['control'],prediction=plan['output'])
        assert index['driver_exit']['no_running_process'] and index['driver_exit']['pid']==229438
        assert set(names)==set(files)|{'member-index.json'}
        for name,item in files.items():
            relative=safe_name(name);info=z.getinfo(name)
            assert not info.is_dir() and not stat.S_ISLNK(info.external_attr>>16) and info.file_size==item['bytes']
            target=local/relative;assert not target.is_symlink() and target.resolve().is_relative_to(local.resolve())
            if not target.exists():
                data=z.read(name)
                assert len(data)==item['bytes'] and hashlib.sha256(data).hexdigest()==item['sha256']
                publish(target,data,root)
            assert target.stat().st_size==item['bytes'] and sha(target)==item['sha256']
            with target.open('rb')as f:os.fsync(f.fileno())
            sync_chain(target.parent,root)
        internal=local/'member-index.json'
        if internal.exists():assert internal.read_bytes()==raw_index
        else:publish(internal,raw_index,root)
    known=set(files)|{'member-index.json'}
    for p in local.rglob('*'):
        assert not p.is_symlink()
        if p.is_file():assert p.relative_to(local).as_posix() in known or p.name.endswith('.pending')
    wrapper=read(local/'control/wrapper-complete.json');manifest=read(local/'prediction/responses.manifest.json')
    assert wrapper['returncode']==0 and wrapper['rows']==420 and wrapper['complete_coverage'] and not wrapper['timed_out']
    assert 'endpoint_error'not in wrapper and sha(local/'prediction/responses.manifest.json')==wrapper['manifest_sha256']
    assert sha(local/'prediction/responses.jsonl')==manifest['output_sha256']==wrapper['responses_sha256']
    for directory in ['raw-rows','generation-rows','decision-rows']:
        assert len(list((local/'prediction'/directory).glob('*.json')))==420
    report=dict(schema='patient-B-complete-local-collection-v1',local=str(local),files=files,
        archive=ref(archive_path),remote_archive=binding['archive'],remote_mapping=index['remote_mapping'],
        driver_exit_before_packaging=index['driver_exit'],inference_plan=ref(plan_path),
        original_collector_source=ref(Path(__file__).parent/'collect.py'),collector_source=ref(__file__),
        transport_receipt=transport,member_index=ref(internal),rows=420,documents=100,patients=244,
        complete_original_recursive_control_and_prediction=True,all_members_SHA_and_size_verified=True,
        files_and_ancestor_directories_fsynced=True,zip_safety_checked=True,
        no_raw_reconstruction=True,no_NN_or_retry_or_score=True,no_remote_calls_in_this_entry=True,wrapper=wrapper)
    final=root/'inference-collection.json'
    if final.exists():assert read(final)==report
    else:save(final,report,root)
    print(json.dumps(dict(collection=ref(final),members=len(files),rows=420)))


if __name__=='__main__':main()
