"""Explicit dataset identity guards only; extraction and normalization policy unchanged."""
import hashlib
import json
from pathlib import Path

F1=dict(documents=16,patients=39,input_sha256="962b76a39b64182502b83a78eae06edac113cf3c1db7dfa7017987c291bc9415",prepared_sha256="32599878940a98d32327541a79ba73fa43869954dfa41f7d040fda15f680d48b")
F0=dict(documents=16,patients=47,input_sha256="ec38c1355f09642f7e8d5af1976c3cabf15a0538bf34d8ed8440ede493459b37",prepared_sha256="17f9607e5a76753e4147f0a8978c889bf3700dfe92e23c4256e17aa2b1c10d72")
def check_dataset(path,input_path,docs,kind):
    data=json.loads(Path(path).read_text()) if path else F1
    assert data in (F1,F0),'Dataset contract must match an explicitly frozen dataset'
    assert hashlib.sha256(Path(input_path).read_bytes()).hexdigest()==data[kind+'_sha256']
    assert len(docs)==data['documents']==len({str(d['pmc_id']) for d in docs})
    key='patients' if kind=='prepared' else 'patient'
    assert sum(len(d[key]) for d in docs)==data['patients']
    return data
