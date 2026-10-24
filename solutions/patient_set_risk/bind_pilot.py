"""Source/input binding, host values supplied explicitly; no GPU/admission."""
import argparse,gzip,json,shutil,zipfile
from pathlib import Path
from common import atomic,checked,ref

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--host',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
    w=Path(__file__).parent;r=Path('/workspace/patientphex-runs/round2-20261023');h=json.loads(Path(a.host).read_text());out=Path(a.output);out.mkdir(parents=True,exist_ok=False);remote=Path(h['remote_root'])
    native=dict(path=str(r/'research/grounded-generation-v1/native-cpu-v2/terminal/actual/encoded.json.gz'),bytes=809334,sha256='4b503cd295ddf84af902efdbcb0df2337ea24ab0bff42c272424ceabbcc7cb06')
    train=dict(path=str(r/'research/grounded-generation-v1/targets-v2/train.jsonl'),bytes=3629191,sha256='bf0091ef057ddb67f9906d035fdaf0ac7b41ae645f77432f0e5a007078291d08')
    encoded=json.loads(gzip.decompress(checked(native).read_bytes()))['train'];targets=[json.loads(l) for l in checked(train).read_text().splitlines()]
    assert len(encoded)==len(targets)==87
    rows=[]
    for i in [76,44,83,19]:
        e,t=encoded[i],targets[i];assert e['row_id']==t['row_id'];rows.append(dict(e,index=i,messages=t['messages']))
    with (out/'rows.json.gz').open('xb') as f:f.write(gzip.compress(json.dumps(rows,ensure_ascii=False,separators=(',',':')).encode(),mtime=0))
    copied={'pilot-plan.json':w/'pilot-plan.json','activation.json':w/'activation.json','model-source-manifest.json':r/'case_retrieval/qwen35-preparation/model-source-manifest.json','normalization-source-manifest.json':w/'normalization-source-manifest.json'}
    sources=('pilot.py','pilot_driver.py','common.py','scst_ops.py')
    for n in sources:copied[n]=w/n
    for n,p in copied.items():shutil.copyfile(p,out/n)
    refs={k:dict(ref(out/n),path=str(remote/n)) for k,n in [('rows','rows.json.gz'),('proposal','pilot-plan.json'),('activation','activation.json'),('model_manifest','model-source-manifest.json')]}
    plan=dict(schema='patient-set-risk-fixed4-probe-v1',source_of_binding=ref(__file__),host_source=ref(a.host),references=refs,sources={n:ref(out/n)['sha256'] for n in sources},
        model=h['model'],adapter=h['adapter'],gpu_uuid=h['gpu_uuid'],python=h['python'],runtime=h['runtime'],
        adapter_model=dict(path=str(Path(h['adapter'])/'adapter_model.safetensors'),bytes=173181072,sha256='d52e2e6c74d3fe874940ddc9a882d3c0faa453ac362b118d1edb2782bae7c6ba'),
        adapter_config=dict(path=str(Path(h['adapter'])/'adapter_config.json'),bytes=1341,sha256='a9e753c405c1a0b0f3811d54224437455629c9cd96d647cfe5f6cf8222dfd307'),
        fixed=dict(indices=[76,44,83,19],max_new_tokens=1600,seed=20261024,shared_seconds=1800,CE_weight=1.,RL_weight=1.,C=11884/87),
        original_native_encoded=native,original_train87=train,no_held_material_in_payload=True,no_formal_or_quality_authorized=True,
        normalizer_manifest_sha256=ref(w/'normalization-source-manifest.json')['sha256'],orchestration_source=ref(w/'remote.py'),reward_bridge_source=ref(w/'reward_bridge.py'),
        recovery_authorization=h.get('recovery_authorization'))
    atomic(out/'execution.json',plan)
    members={p.name:ref(p) for p in out.iterdir() if p.is_file()}
    with zipfile.ZipFile(out/'deployment.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
        for n in members:z.write(out/n,n)
    atomic(out/'binding.json',dict(status='SOURCE_BOUND_NO_ADMISSION_OR_LAUNCH',execution=ref(out/'execution.json'),archive=ref(out/'deployment.zip'),members=members,remote_root=str(remote),mount=h['mount'],environment=h['environment'],python=h['python'],host_source=ref(a.host)))
    print(json.dumps(ref(out/'execution.json')))
if __name__=='__main__':main()
