"""One immutable private F1 checkpoint; no model, gold, decoder or evaluator."""
import argparse, fcntl, hashlib, importlib.util, json, os, stat, sys, time, zipfile
from pathlib import Path

S = Path(__file__).resolve().parent
B = Path('/workspace/patientphex-runs/round2-20261023/research/training-seed-ensemble-v1')
OUT = B / 'F1-private-checkpoint-v1'
TRANSPORT = Path('/workspace/patientphex-worktrees/round2-sequence-crf/solutions/sequence_crf/patient_balanced_loss/endpoint_supplement.py')
assert hashlib.sha256(TRANSPORT.read_bytes()).hexdigest() == 'e40d61d31a948c0426aa4960402f60371c68f615657bd59fe7740c0ad768bcde'
spec = importlib.util.spec_from_file_location('_seed_samebyte_backup', TRANSPORT)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
m.MOUNT = '5a9955da-5240-41cc-9e52-05d3d8da8a3c'
m.TARGET = 'research/training-seed-ensemble-v1/F1-private-checkpoint-v1'
m.CHUNK = 131072
m.REMOTE_VERIFY = m.REMOTE_VERIFY.replace('patient-balanced-alpha05-endpoint-persistent-v1', 'seed-F1-endpoint-persistent-v1').replace('original_raw_repackaged=False', 'original_raw_archives_copied_byteexact=True')

def read(p): return json.loads(Path(p).read_text())
def checked(v):
    p = Path(v['path']); assert m.ref(p) == v; return p

def pack():
    assert not (OUT/'package.json').exists()
    entry = read(S/'evidence/F1-result.json')
    assert entry['result']['new_v2'] == 1 and entry['result']['old_v2'] == 0
    assert entry['result']['strict_positive'] and entry['result']['added_TP'] == 6
    entries = {'entry/F1-result.json': S/'evidence/F1-result.json', 'transport/endpoint_supplement.py': TRANSPORT}
    # Explicit source prefixes frozen before any F0 source exists; never include later routes.
    for prefix in ['training', 'inference_helper', 'normalization_A30', 'evidence', 'experiments']:
        for p in sorted((S/prefix).rglob('*')):
            if p.is_file() and p.suffix in {'.py', '.md', '.json', '.diff'} and '__pycache__' not in p.parts:
                entries['source/'+p.relative_to(S).as_posix()] = p
    for name in ['README.md', 'WORKLOG.md', 'NOTES.md', 'proposal.json', 'prepare.py', 'persist_F1.py']:
        entries['source/'+name] = S/name
    for prefix in ['quotes-v1', 'normalization-preparation', 'normalized-v1', 'quality-v1']:
        for p in sorted((B/'F1-postprocess-v1'/prefix).iterdir()):
            assert p.is_file(); entries['postprocess/'+prefix+'/'+p.name] = p
    phase_refs = {}
    for phase, count, field in [('formal-fresh-v1',25,'formal_reported_success'), ('infer-F1-v1',90,'inference_reported_success'), ('F1-Sap-v1',13,'query_reported_success')]:
        c = read(B/phase/'collection.json')
        assert c[field] and c['all_actual_SHA_verified'] and len(c['files']) == count
        for p in [B/phase/'collection.json', checked(c['package']), checked(c['member_index']), checked(c['archive'])]:
            entries['runs/'+phase+'/'+p.name] = p
        with zipfile.ZipFile(checked(c['archive'])) as z:
            index = json.loads(z.read('member-index.json'))
            assert set(z.namelist()) == set(index['files']) | {'member-index.json'}
            for name, v in index['files'].items():
                raw=z.read(name); assert len(raw)==v['bytes'] and m.digest(raw)==v['sha256']
        phase_refs[phase] = dict(collection=m.ref(B/phase/'collection.json'), remote_archive=c['remote_archive'], mount=c['mount'], external_checkpoints=c.get('actual_external_checkpoints', {}))
    for name in ['original-schema-campaign-closed.json', 'formal-recovery-v4/cpu-verify-attempt3.json']:
        entries['history/'+name] = B/name
    payload={}
    sensitive=[v.encode() for k,v in os.environ.items() if len(v)>8 and any(x in k.upper() for x in ['TOKEN','PASSWORD','PASSWD','COOKIE','SECRET'])]
    for name,p in entries.items():
        m.safe(name); assert p.is_file() and not p.is_symlink() and not p.name.endswith('.gold.jsonl')
        raw=p.read_bytes(); assert not any(s in raw for s in sensitive); payload[name]=raw
    manifest=dict(schema='seed-F1-private-checkpoint-members-v1',members={n:dict(source_path=str(entries[n].resolve()),bytes=len(v),sha256=m.digest(v)) for n,v in payload.items()},external_archives=[],phase_refs=phase_refs,no_NN_no_gold_no_score=True,full_weights_optimizer_remain_original_persistent_ACK=True,original_terminal_ZIPs_copied_byteexact=True)
    encoded=m.encoded(manifest); assert len(encoded)<=m.CHUNK
    m.publish(OUT/'members-manifest.json',encoded)
    archive=OUT/'endpoint-supplement.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for name,raw in sorted(dict(payload,**{'members-manifest.json':encoded}).items()):
            info=zipfile.ZipInfo(name,(2026,10,24,0,0,0)); info.external_attr=(stat.S_IFREG|0o600)<<16; info.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(info,raw,compress_type=zipfile.ZIP_DEFLATED,compresslevel=9)
    with archive.open('rb') as h: os.fsync(h.fileno())
    m.sync(OUT)
    with zipfile.ZipFile(archive) as z:
        assert len(z.namelist())==len(payload)+1 and z.read('members-manifest.json')==encoded
        for n,v in payload.items(): assert z.read(n)==v
    raw=archive.read_bytes(); chunks=[]
    for offset in range(0,len(raw),65536):
        piece=raw[offset:offset+65536]; chunks.append(dict(name=f'chunks/{offset:012d}.bin',offset=offset,bytes=len(piece),sha256=m.digest(piece)))
    p=dict(archive=m.ref(archive),manifest=m.ref(OUT/'members-manifest.json'),members=len(payload),unpacked_bytes=sum(map(len,payload.values())),target_mount=m.MOUNT,target=m.TARGET,chunks=chunks,source=m.ref(__file__),transport_source=m.ref(TRANSPORT),external_archives=[],phase_refs=phase_refs,package_readback_exact=True)
    m.publish(OUT/'package.json',m.encoded(p)); print(json.dumps(dict(package=m.ref(OUT/'package.json'),archive=p['archive'],members=len(payload))))

def main():
    a=argparse.ArgumentParser(); a.add_argument('stage',choices=['pack','deploy']);args=a.parse_args()
    os.umask(0o077);OUT.mkdir(exist_ok=True)
    def guard(event,args):
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)) and Path(os.fsdecode(args[0])).name.endswith('.gold.jsonl'): raise PermissionError('Backup cannot open gold')
    sys.addaudithook(guard)
    lock=(OUT/'owner.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if args.stage=='pack': pack()
    else:
        for p in (OUT/'deployment').glob('*.backoff.json'): assert time.time()>=read(p)['retry_not_before_unix']+5
        m.deploy(B,OUT,Path('/workspace/patientphex-tools'))

if __name__=='__main__': main()
