"""Read and hash every original FT segment, without modifying original files."""
from pathlib import Path
import datetime, hashlib, json, os, time
R=Path(__file__).resolve().parent
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
assert not (R/'remote-inventory.json').exists()
t=time.monotonic();b=(R/'expected-segments.json').read_bytes();m=json.loads(b)
report={'status':'running','at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'expected_manifest_sha256':hashlib.sha256(b).hexdigest(),'verified_segments':0,
        'verified_bytes':0,'missing':[],'mismatched':[],'original_files_modified':False}
for s in m['segments']:
    p=Path(s['path']);p=p if p.is_absolute() else Path(m['segment_root'])/p
    assert not p.is_symlink()
    if not p.exists():report['missing'].append(s);continue
    st=p.stat();h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    after=p.stat();assert (st.st_size,st.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
    actual={'bytes':st.st_size,'sha256':h.hexdigest()}
    if actual!={k:s[k] for k in ['bytes','sha256']}:
        report['mismatched'].append({'expected':s,'actual':actual});continue
    report['verified_segments']+=1;report['verified_bytes']+=st.st_size
report.update(status='complete_readonly_inventory',elapsed_seconds=time.monotonic()-t,
              source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(R/'remote-inventory.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ['missing','mismatched']}))
assert not report['mismatched'], 'Original immutable segment mismatch; do not replace silently'
