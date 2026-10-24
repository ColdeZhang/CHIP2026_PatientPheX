#!/usr/bin/env python3
"""Actual-token preflight for fixed reversed candidate order; unchanged old runner."""
from pathlib import Path
import datetime, hashlib, json, os, subprocess, sys, time

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()
def save(p,x):
    with Path(p).open('x') as f:json.dump(x,f,indent=2);f.write('\n')

started=time.monotonic();plan=json.loads(Path('plan.json').read_text())
assert not Path('responses.jsonl').exists() and not Path('complete.json').exists()
for name,digest in plan['files'].items():assert sha(name)==digest
assert plan['cap_seconds']==600 and plan['max_input_tokens']==14000 and plan['max_new_tokens']==1600
adapter=Path(plan['adapter'])
actual_adapter={n:dict(bytes=(adapter/n).stat().st_size,sha256=sha(adapter/n)) for n in plan['adapter_files']}
assert {n:r['sha256'] for n,r in actual_adapter.items()}==plan['adapter_files']
import torch,transformers,peft,tokenizers
from transformers import AutoTokenizer
assert transformers.__version__=='5.20.2' and peft.__version__=='0.21.2'
assert not torch.cuda.is_initialized()
tokenizer=AutoTokenizer.from_pretrained(plan['model'],local_files_only=True)
prompts=[json.loads(x) for x in Path('prompts.jsonl').read_text().splitlines()]
expected={r['row_id']:r for r in json.loads(Path('inverse-and-token-audit.json').read_text())}
assert len(prompts)==len(expected)==80
token_rows=[]
for row in prompts:
    assert [m['role'] for m in row['messages']]==['system','user']
    rendered=tokenizer.apply_chat_template(row['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    ids=tokenizer.encode(rendered,add_special_tokens=False)
    r=expected[row['row_id']]
    assert hashlib.sha256(rendered.encode()).hexdigest()==r['full_render_sha256']
    assert len(ids)==r['new_input_tokens']<=14000
    token_rows.append(dict(row_id=row['row_id'],input_tokens=len(ids)))
save('actual-preflight.json',dict(time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    plan_sha256=sha('plan.json'),adapter_files=actual_adapter,tokens=token_rows,
    torch=torch.__version__,transformers=transformers.__version__,peft=peft.__version__,tokenizers=tokenizers.__version__,
    CUDA_initialized=torch.cuda.is_initialized(),runtime_difference=plan['runtime_comparison']))
argv=[sys.executable,'legacy/qwen35_runner.py','--model',plan['model'],'--source-manifest','model-source-manifest.json',
      '--prompts','prompts.jsonl','--output','responses.jsonl','--adapter',plan['adapter'],
      '--max-input-tokens','14000','--max-new-tokens','1600']
completed=subprocess.run(argv,env=dict(os.environ),check=False)
save('runner-exit.json',dict(returncode=completed.returncode,seconds=time.monotonic()-started,argv=argv))
assert completed.returncode==0
responses=[json.loads(x) for x in Path('responses.jsonl').read_text().splitlines()]
assert len(responses)==80 and {r['row_id'] for r in responses}==set(expected)
assert all(r['input_tokens']==expected[r['row_id']]['new_input_tokens'] for r in responses)
manifest=json.loads(Path('responses.manifest.json').read_text())
assert manifest['fingerprint']['adapter_files']==plan['adapter_files']
save('complete.json',dict(completed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    plan_sha256=sha('plan.json'),responses_sha256=sha('responses.jsonl'),manifest_sha256=sha('responses.manifest.json'),
    rows=80,actual_preflight_sha256=sha('actual-preflight.json'),runner_returncode=0,seconds=time.monotonic()-started,
    quality_read=False))
print(json.dumps(dict(complete=True,seconds=time.monotonic()-started)),flush=True)
