"""Attempt2 patches only the frozen probe's cache instrumentation."""
import hashlib,json,pathlib
import gpu_probe as original
import cache_snapshot_attempt2 as snapshot
manifest=json.loads(pathlib.Path('attempt2-repair-manifest.json').read_text())
for name,digest in manifest['new_sources'].items():
    assert hashlib.sha256(pathlib.Path(name).read_bytes()).hexdigest()==digest,'repair source changed: '+name
original_compare=original.compare_states
original.states=snapshot.capture
original.compare_states=lambda a,b,torch,th:snapshot.compare(a,b,torch,th,original_compare)
if __name__=='__main__':original.main()
