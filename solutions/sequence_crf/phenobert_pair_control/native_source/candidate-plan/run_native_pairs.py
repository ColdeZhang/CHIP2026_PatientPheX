"""Bounded public native pair inference; no labels, fit or quality evaluation."""
from pathlib import Path
import gzip
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
import warnings

B = Path(__file__).resolve().parent
t0 = time.monotonic()


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for x in iter(lambda: f.read(1024*1024), b''): h.update(x)
    return h.hexdigest()


def write(name, value):
    (B/name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


plan = json.loads((B/'inference-plan.json').read_text())
assert sha(__file__) == plan['driver_sha256']
target = plan['gpu_uuid']
assert os.environ['CUDA_VISIBLE_DEVICES'] == target
gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,memory.total',
                               '--format=csv,noheader,nounits'], text=True)
apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory',
                                '--format=csv,noheader,nounits'], text=True)
selected = [x.split(',') for x in gpu.splitlines() if target in x]
assert len(selected) == 1 and int(selected[0][0].strip()) == 2
assert int(selected[0][2].strip()) <= 512
assert target not in apps
write('fresh-GPU.json', {'gpu_rows': gpu, 'compute_rows': apps, 'target': target,
                         'target_no_compute_process': True, 'time': time.time()})
modelroot = Path(plan['model_root'])
weight = modelroot/'bert_model_max_triple.pkl'
assert weight.stat().st_size == 434159420 and sha(weight) == plan['weight_sha256']
assert sha(modelroot/'upstream-source-index.json') == plan['source_index_sha256']
for record in json.loads((modelroot/'upstream-source-index.json').read_text())['files']:
    assert sha(modelroot/record['path']) == record['sha256']
input_path = B/plan['input_file']
assert sha(input_path) == plan['input_sha256']
with gzip.open(input_path, 'rt') as f: rows = [json.loads(line) for line in f]
assert len(rows) == plan['rows'] and len({x['pair_key'] for x in rows}) == len(rows)
for row in rows:
    x = row['input']
    assert hashlib.sha256(json.dumps(x,ensure_ascii=False,separators=(',',':')).encode()).hexdigest() == row['pair_key']
    assert len(x['word_nums']) == x['seq_len'] and sum(x['word_nums']) == len(x['word_pieces'])
    assert 0 < len(x['word_pieces'])+2 <= 512


def blocked_network(*a, **kw): raise RuntimeError('No network inside public native inference')


socket.create_connection = blocked_network
import torch
import fastNLP
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always')
    start = time.monotonic()
    model = torch.load(weight, map_location='cpu', weights_only=False)
    load_seconds = time.monotonic()-start
    load_warnings = [{'category':x.category.__name__,'message':str(x.message)[:500]} for x in caught]
assert type(model).__module__ == 'fastNLP.models.Mybert'
assert type(model).__name__ == 'BertForSentenceMatching'
assert sum(p.numel() for p in model.parameters()) == 108312579 and len(model.state_dict()) == 201
assert all(p.dtype == torch.float32 for p in model.parameters())
native = model.bert.model
assert native.auto_truncate is False and native.pooled_cls is True
assert native.pool_method == 'max' and native.include_cls_sep is True and native.layers == [-1]
assert len(native.tokenzier.vocab) == 28996 and model.num_labels == 3
model.eval()


def batch(items, device):
    a = torch.zeros((len(items), max(len(x['input']['word_pieces']) for x in items)), dtype=torch.long)
    b = torch.zeros((len(items), max(len(x['input']['word_nums']) for x in items)), dtype=torch.long)
    c = torch.tensor([x['input']['seq_len'] for x in items], dtype=torch.long)
    for i, row in enumerate(items):
        x = row['input']
        a[i,:len(x['word_pieces'])] = torch.tensor(x['word_pieces'])
        b[i,:len(x['word_nums'])] = torch.tensor(x['word_nums'])
    return a.to(device), b.to(device), c.to(device)


with torch.inference_mode():
    cpu_reference = model.forward(*batch(rows[:2], 'cpu'))['pred'].clone()
assert not torch.cuda.is_initialized()
assert torch.cuda.device_count() == 1
model = model.to('cuda:0')
torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()
stages = []
outputs = []
inference_start = time.monotonic()
with torch.inference_mode():
    for start in range(0,len(rows),plan['batch_size']):
        subset = rows[start:start+plan['batch_size']]
        torch.cuda.synchronize()
        tick = time.monotonic()
        tensors = batch(subset, 'cuda:0')
        logits = model.forward(*tensors)['pred']
        probs = torch.softmax(logits, dim=1)
        assert logits.shape == (len(subset),3) and torch.isfinite(logits).all()
        assert torch.isfinite(probs).all() and torch.allclose(probs.sum(-1),torch.ones(len(subset),device='cuda:0'),atol=1e-6)
        values = probs.cpu().tolist()
        if start == 0:
            first_difference = float((logits[:2].cpu()-cpu_reference).abs().max())
            assert torch.allclose(logits[:2].cpu(),cpu_reference,atol=1e-3,rtol=1e-4)
        torch.cuda.synchronize()
        seconds = time.monotonic()-tick
        stages.append({'offset':start,'rows':len(subset),'max_tokens_with_cls_sep':tensors[0].shape[1]+2,'seconds':seconds})
        outputs.extend({'pair_key':x['pair_key'],'probabilities':p} for x,p in zip(subset,values))
        print(json.dumps({'completed':len(outputs),'total':len(rows),'batch_seconds':seconds}),flush=True)
inference_seconds = time.monotonic()-inference_start
out = B/'probabilities.jsonl.gz'
with open(out,'wb') as raw:
    with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0,filename='') as f:
        for row in outputs: f.write((json.dumps(row,separators=(',',':'))+'\n').encode())
result = {'status':'PASS_NATIVE_PUBLIC_PAIR_INPUT_ONLY_INFERENCE','plan_sha256':sha(B/'inference-plan.json'),
          'driver_sha256':sha(__file__),'input_sha256':sha(input_path),'weight_sha256':sha(weight),
          'rows':len(outputs),'batch_size':plan['batch_size'],'batches':stages,
          'load_seconds':load_seconds,'inference_seconds':inference_seconds,
          'pairs_per_second':len(outputs)/inference_seconds,
          'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
          'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
          'first_two_actual_pairs_CPU_GPU_max_abs_logit_difference':first_difference,
          'probabilities_sha256':sha(out),'probabilities_bytes':out.stat().st_size,
          'classes':['Not Match','Related','Match'],'softmax_temperature':1.0,'dtype':'float32',
          'model_parameters':108312579,'state_tensors':201,'no_source_shims':True,'no_model_conversion':True,
          'no_labels_or_quality_evaluation':True,'no_training':True,'load_warnings':load_warnings,
          'runtime':{'python':sys.version,'torch':torch.__version__,'fastNLP':fastNLP.__version__},
          'elapsed_seconds':time.monotonic()-t0}
write('result.json',result)
print(json.dumps({k:v for k,v in result.items() if k not in {'batches','load_warnings'}}),flush=True)
