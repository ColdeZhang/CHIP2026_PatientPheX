"""One Bioformer CPU256 throughput probe on frozen input-only unigrams."""
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
from datetime import datetime, timezone


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def main():
    start=time.monotonic();signal.alarm(88)
    p=json.loads(Path('plan.json').read_text())
    assert sha(__file__)==p['source_sha256']
    assert os.environ['CUDA_VISIBLE_DEVICES']==''
    assert os.environ['USE_TORCH']=='0' and os.environ['USE_TF']=='1'
    O=Path('output');O.mkdir(exist_ok=False)
    for spec in p['resources']:
        assert sha(spec['path'])==spec['sha256'],spec['path']
    hsh_seconds=time.monotonic()-start
    src=Path(p['native_source']['path']).read_text()
    loader=src[src.index("labels=(R/'lable.vocab')"):src.index('# Complete variable coverage')]
    assert hashlib.sha256(loader.encode()).hexdigest()==p['loader_sha256']
    assert loader==Path('native_loader.py').read_text()
    inputs_all=list(map(json.loads,gzip.open(p['inputs']['path'],'rt')))
    assert len(inputs_all)==13145
    assert [x['lower_tokens'] for x in inputs_all]==sorted(x['lower_tokens'] for x in inputs_all)
    selected=inputs_all[:256]
    assert len(selected)==256 and all(len(x['lower_tokens'])==1 for x in selected)
    R=Path(p['model_root']);SHARED=R.parent/'public-phenotagger-reuse'
    sys.path[:0]=[str(R),str(SHARED/'native-pipeline/src')]
    ti=time.monotonic()
    import numpy as np,tensorflow as tf
    from transformers import AutoConfig,TFAutoModel
    from nn_represent import BERT_RepresentationLayer
    from ml_ner import output_result
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    assert not tf.config.list_physical_devices('GPU')
    assert not tf.config.list_logical_devices('GPU')
    assert str(SHARED/'tf-gpu-overlay') in tf.__file__
    import_seconds=time.monotonic()-ti
    ns=dict(R=R,tf=tf,AutoConfig=AutoConfig,TFAutoModel=TFAutoModel,BERT_RepresentationLayer=BERT_RepresentationLayer)
    tl=time.monotonic()
    with tf.device('/CPU:0'):exec(compile(loader,'native_loader.py','exec'),ns)
    model,rep=ns['model'],ns['rep']
    assert model.count_params()==51380594 and len(model.weights)==137
    assert all('/device:CPU:' in x.device for x in model.weights)
    load_seconds=time.monotonic()-tl
    tr=time.monotonic()
    infer=tf.function(lambda x,y,z:model([x,y,z],training=False))
    with tf.device('/CPU:0'):concrete=infer.get_concrete_function(*[tf.TensorSpec([None,32],tf.int32)]*3)
    trace_seconds=time.monotonic()-tr
    batches=[];outputs=[];array_hash=hashlib.sha256()
    for offset in (0,128):
        batch=selected[offset:offset+128]
        prepared,_=rep.load_data([[tuple([w]) for w in x['lower_tokens']] for x in batch],[],word_max_len=32,training=False)
        for column,key in enumerate(('input_ids','token_type_ids','attention_mask')):
            expected=np.array([x['models']['bioformer'][key]+[0]*(32-len(x['models']['bioformer'][key])) for x in batch],dtype=np.int32)
            assert np.array_equal(prepared[column],expected)
        tfw=time.monotonic()
        with tf.device('/CPU:0'):tensor=concrete(*prepared)
        probs=tensor.numpy()
        elapsed=time.monotonic()-tfw
        assert tensor.device.endswith('CPU:0')
        assert probs.shape==(128,17266) and np.isfinite(probs).all() and np.allclose(probs.sum(1),1,atol=1e-5)
        array_hash.update(probs.tobytes())
        lines=output_result(probs,rep.label_2_index,Top_N=3).strip().splitlines()
        assert len(lines)==128
        for j,(entry,line) in enumerate(zip(batch,lines)):
            cells=[]
            for cell in line.split('\t'):
                ident,rounded=cell.split('|');column=rep.label_2_index[ident]
                cells.append(dict(column=column,hpo_id=ident,probability=float(probs[j,column]),native_probability_5dp=rounded))
            outputs.append(dict(unique_index=offset+j,lower_tokens=entry['lower_tokens'],top3=cells,native_line=line,forward_origin='fixed_CPU256_probe'))
        batches.append(dict(offset=offset,rows=128,seconds=elapsed,device=tensor.device,all_finite=True,sums_close1=True))
    assert not tf.config.list_physical_devices('GPU') and not tf.config.list_logical_devices('GPU')
    with (O/'sample-raw-predictions.jsonl').open('x') as f:
        for x in outputs:f.write(json.dumps(x,ensure_ascii=False)+'\n')
    keys=[x['lower_tokens'] for x in selected]
    fixed=time.monotonic()-start-sum(b['seconds'] for b in batches)
    worst=max(b['seconds'] for b in batches)
    estimate=fixed+math.ceil(13145/128)*worst
    summary=dict(status='complete_CPU256_Bioformer_probe_no_quality',at_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=sha(__file__),plan_sha256=sha('plan.json'),inputs_sha256=p['inputs']['sha256'],
        selection='first256 of frozen lexicographically ordered13145 native lowercase tuples; two128 batches, no label-based selection',
        selected_keys_sha256=hashlib.sha256(json.dumps(keys).encode()).hexdigest(),selected_keys=keys,
        model='public_Bioformer_Ph enoTagger51M'.replace('Ph eno','Pheno'),parameters=model.count_params(),variables=len(model.weights),
        weight_sha256=p['weight']['sha256'],loader_sha256=p['loader_sha256'],physical_GPU_devices=[],logical_GPU_devices=[],
        tf_version=tf.__version__,threads=dict(intra=tf.config.threading.get_intra_op_parallelism_threads(),inter=tf.config.threading.get_inter_op_parallelism_threads()),
        file_hashing_seconds=hsh_seconds,imports_seconds=import_seconds,construct_and_native_load_seconds=load_seconds,
        graph_trace_seconds=trace_seconds,batches=batches,fixed_overhead_seconds=fixed,
        all13145_conservative_CPU_seconds=estimate,estimate_formula='fixed_overhead+ceil(13145/128)*max(actual_two_batch_seconds)',
        allows_prepare_full_plan=estimate<=480,raw_probability_array_sha256=array_hash.hexdigest(),
        elapsed_seconds=time.monotonic()-start,no_gold=True,no_new_acceptance_or_quality=True,
        output=dict(path=str(O/'sample-raw-predictions.jsonl'),sha256=sha(O/'sample-raw-predictions.jsonl'),bytes=(O/'sample-raw-predictions.jsonl').stat().st_size),
        limitations=['CPU input-only throughput measured only first256; extrapolation is not full runtime or quality.','No threshold/hybrid/None acceptance/old-pool change; full forward still requires root decision.','Reuse prior strict137-variable CPU audit; native fullH5 loaded once here, no repeated tensor-wide NaN/equality audit.'])
    assert summary['elapsed_seconds']<90
    (O/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='selected_keys'}),flush=True)


if __name__=='__main__':main()
