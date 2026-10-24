"""Build frozen small input/source package; no remote calls or neural model."""
import gzip
import hashlib
import io
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parent
R=Path('/workspace/patientphex-runs/round2-20261023')
D=R/'research/supervised-linker-readiness-20261024/probe-v1'
READY=R/'research/supervised-linker-readiness-20261024/owner-v1'
P='/home/zhangyuheng/projects/CHIP2026_PatientPheX'


def ref(p):
    p=Path(p);return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def main():
    from tokenizers import BertWordPieceTokenizer
    tokenizer=BertWordPieceTokenizer('/workspace/patientphex-runs/day1-20261023/semantic-neural/sapbert-fp16/vocab.txt',lowercase=True)
    train=json.loads((READY/'training-examples.json').read_text());bank=json.loads((READY/'public-rows.json').read_text())
    held=sorted({' '.join(x['text'].lower().split()) for x in json.loads((READY/'F1-eligible.json').read_text())})
    data=dict(train=[dict(text=x['surface'],**{k:v for k,v in x.items() if k!='surface'}) for x in train],
        bank=[dict(text=x['text'],identifiers=x['identifiers']) for x in bank],held=[dict(text=x) for x in held])
    for name,rows in data.items():
        for row,encoded in zip(rows,tokenizer.encode_batch([x['text'] for x in rows])):row['ids']=encoded.ids
    D.mkdir(parents=True,exist_ok=True)
    payload=gzip.compress(json.dumps(data,separators=(',',':'),ensure_ascii=False).encode(),mtime=0)
    (D/'data.json.gz').write_bytes(payload)
    old=json.loads((R/'research/grounded-generation-v1/normalization-v1/deployment/terminal/actual/public-bank-manifest.json').read_text())
    plan=dict(schema='supervised-sapbert-engineering-probe-v1',authorization='Root: sole300+TERM30 engineering; no formal training authorization',
        candidate_plan_sha256=ref(ROOT.parent/'candidate-plan.json')['sha256'],counts=dict(train=1160,bank=41099,held=303),
        model=P+'/semantic-neural/model',model_files=old['model_files_sha256'],
        python=P+'/sequence-crf/.venv/bin/python',gpu_uuid='GPU-07726469-bf7a-b832-9dee-019dc276dc1d',
        environment=dict(PYTHONPATH=P+'/semantic-neural/venv/lib/python3.10/site-packages',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false'),
        launch=dict(prebound_validity_seconds=300,check_at=['entry','before_Popen'],exclusive='job/exclusive-launch.json'),
        budget_seconds=300,term_grace_seconds=30,formal_started=False,
        engineering='Initial full public bank+train retrieval; actual count/length Pareto batches, exact warmup twoAdam and extra accumulated backward with resident state; full model/optimizer/scheduler/scaler/RNG save-reopen; discard all probe state.',
        data=ref(D/'data.json.gz'),source_files={p.name:ref(p) for p in [ROOT/'common.py',ROOT/'native_cpu.py',ROOT/'probe.py',ROOT/'driver.py']})
    (ROOT/'execution-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    members={p.name:p.read_bytes() for p in [ROOT/'common.py',ROOT/'native_cpu.py',ROOT/'probe.py',ROOT/'driver.py',ROOT/'execution-plan.json']}
    members['data.json.gz']=payload
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
        for name,value in members.items():z.writestr(name,value)
    (D/'package.zip').write_bytes(buf.getvalue())
    manifest=dict(archive=ref(D/'package.zip'),members={k:dict(bytes=len(v),sha256=hashlib.sha256(v).hexdigest()) for k,v in members.items()},
        mount='1593f159-19bc-41ca-8b4c-2e7455f5db99',relative='research/supervised-sapbert-linker/probe-v1',remote=P+'/research/supervised-sapbert-linker/probe-v1',plan=ref(ROOT/'execution-plan.json'))
    (D/'package.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(package=manifest['archive'],data_bytes=len(payload),members=len(members))))


if __name__=='__main__':main()
