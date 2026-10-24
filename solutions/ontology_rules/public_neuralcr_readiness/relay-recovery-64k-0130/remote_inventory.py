"""Recheck seven uncertain original and previous-attempt target paths, no writes."""
from pathlib import Path
import datetime,hashlib,json,os,time
R=Path(__file__).resolve().parent
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not (R/'remote-inventory.json').exists()
t=time.monotonic();plan=json.loads((R/'recovery-plan.json').read_text());mbytes=Path(plan['expected_manifest_path']).read_bytes();m=json.loads(mbytes)
plan=json.loads((R/'recovery-plan.json').read_text());previous=Path(plan['prior_inventory_path']).read_bytes();prior=json.loads(previous)
assert hashlib.sha256(previous).hexdigest()==plan['prior_inventory_sha256']
assert prior['expected_manifest_sha256']==hashlib.sha256(mbytes).hexdigest() and not prior['mismatched']
report={'status':'running','at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'expected_manifest_sha256':hashlib.sha256(mbytes).hexdigest(),
        'prior_inventory_sha256':plan['prior_inventory_sha256'],
        'prior_verified_segments_reused_without_rehash':prior['verified_segments'],
        'verified_segments':prior['verified_segments'],'verified_bytes':prior['verified_bytes'],
        'delta_paths_checked':[],'resolved_paths':{},'missing':[],'mismatched':[],
        'original_files_modified':False,'final_assembly_still_rechecks_all_segments':True}
for s in prior['missing']:
    original=Path(s['path']);original=original if original.is_absolute() else Path(m['segment_root'])/original
    previous_targets=[Path(root)/'pieces'/f"pmc_model_new.bin.{s['start']}-{s['start']+s['bytes']}.bin" for root in plan['prior_recovery_roots']]
    found=[]
    for p in [original,*previous_targets]:
        assert not p.is_symlink()
        if not p.exists():report['delta_paths_checked'].append({'path':str(p),'exists':False});continue
        before=p.stat();b=p.read_bytes();after=p.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
        actual={'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
        report['delta_paths_checked'].append({'path':str(p),'exists':True,**actual})
        if actual!={k:s[k] for k in ['bytes','sha256']}:report['mismatched'].append({'path':str(p),'expected':s,'actual':actual})
        else:found.append(str(p))
    if found:
        report['verified_segments']+=1;report['verified_bytes']+=s['bytes'];report['resolved_paths'][str(s['start'])]=found[0]
    else:report['missing'].append(s)
report.update(status='complete_readonly_inventory',elapsed_seconds=time.monotonic()-t,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(R/'remote-inventory.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ['delta_paths_checked','resolved_paths','missing','mismatched']}))
assert not report['mismatched'], 'Unexpected immutable bytes; stop rather than replace'
