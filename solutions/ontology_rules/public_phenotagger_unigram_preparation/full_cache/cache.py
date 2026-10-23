"""Fixed Bioformer unigram CPU cache; preserve prior native resources unchanged."""
import gzip,hashlib,json,os,signal,sys,time
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def alarm(_sig,_frame):raise TimeoutError('480-second CPU budget: preserve partial, no downstream use')


def write_json(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def main():
    start=time.monotonic();signal.signal(signal.SIGALRM,alarm);signal.alarm(475)
    p=json.loads(Path('plan.json').read_text());assert sha(__file__)==p['source_sha256']
    assert os.environ['CUDA_VISIBLE_DEVICES']=='' and os.environ['USE_TORCH']=='0'
    O=Path('output');O.mkdir(exist_ok=False)
    for spec in p['resources']:assert sha(spec['path'])==spec['sha256'],spec['path']
    original=Path(p['native_source']['path']).read_text()
    loader=original[original.index("labels=(R/'lable.vocab')"):original.index('# Complete variable coverage')]
    assert hashlib.sha256(loader.encode()).hexdigest()==p['loader_sha256'] and loader==Path('native_loader.py').read_text()
    data=list(map(json.loads,gzip.open(p['inputs']['path'],'rt')));assert len(data)==13145
    assert [x['lower_tokens'] for x in data]==sorted(x['lower_tokens'] for x in data)
    reuse=list(map(json.loads,Path(p['probe_raw']['path']).read_text().splitlines()));assert len(reuse)==256
    docs={str(d['pmc_id']):d for d in map(json.loads,Path(p['pure_input']['path']).read_text().splitlines())}
    assert len(docs)==64 and all(set(d)=={'pmc_id','pmid','patient','full_text'} for d in docs.values())
    head=json.loads(Path(p['head_map']['path']).read_text())['mapping']
    labels=Path(p['label_file']['path']).read_text().strip().splitlines();assert len(labels)==17266
    assert [r['original_label'] for r in head]==labels
    cmap={r['original_label']:r['current_label'] for r in head}
    def check(row,i):
        assert row['unique_index']==i and row['lower_tokens']==data[i]['lower_tokens']
        assert len(row['top3'])==3
        for c in row['top3']:
            assert labels[c['column']]==c['hpo_id'] and 0<=c['probability']<=1
        assert all(row['top3'][j]['probability']>=row['top3'][j+1]['probability'] for j in range(2))
        assert row['native_line']=='\t'.join(c['hpo_id']+'|'+c['native_probability_5dp'] for c in row['top3'])
    for i,row in enumerate(reuse):check(row,i)
    R=Path(p['model_root']);SHARED=R.parent/'public-phenotagger-reuse'
    sys.path[:0]=[str(R),str(SHARED/'native-pipeline/src')]
    progress=dict(status='preparing_input_only',completed_unique=256,reused_unique=256,started_at_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(__file__),plan_sha256=sha('plan.json'))
    write_json(O/'progress.json',progress)
    raw_path=O/'unique-raw-predictions.jsonl';cache=list(reuse);batches=[]
    try:
        with raw_path.open('x') as out:
            for row in reuse:out.write(json.dumps(row,ensure_ascii=False)+'\n')
            out.flush()
            ti=time.monotonic()
            import numpy as np,tensorflow as tf
            from transformers import AutoConfig,TFAutoModel
            from nn_represent import BERT_RepresentationLayer
            from ml_ner import output_result
            tf.config.set_visible_devices([],'GPU');tf.config.threading.set_intra_op_parallelism_threads(2);tf.config.threading.set_inter_op_parallelism_threads(2)
            assert not tf.config.list_physical_devices('GPU') and not tf.config.list_logical_devices('GPU')
            assert str(SHARED/'tf-gpu-overlay') in tf.__file__
            import_seconds=time.monotonic()-ti
            ns=dict(R=R,tf=tf,AutoConfig=AutoConfig,TFAutoModel=TFAutoModel,BERT_RepresentationLayer=BERT_RepresentationLayer)
            tl=time.monotonic()
            with tf.device('/CPU:0'):exec(compile(loader,'native_loader.py','exec'),ns)
            model,rep=ns['model'],ns['rep'];load_seconds=time.monotonic()-tl
            assert model.count_params()==51380594 and len(model.weights)==137 and all('/device:CPU:' in w.device for w in model.weights)
            tr=time.monotonic();infer=tf.function(lambda x,y,z:model([x,y,z],training=False))
            with tf.device('/CPU:0'):concrete=infer.get_concrete_function(*[tf.TensorSpec([None,32],tf.int32)]*3)
            trace_seconds=time.monotonic()-tr
            for offset in range(256,len(data),128):
                batch=data[offset:offset+128]
                inputs,_=rep.load_data([[tuple([w]) for w in x['lower_tokens']] for x in batch],[],word_max_len=32,training=False)
                for column,key in enumerate(('input_ids','token_type_ids','attention_mask')):
                    expected=np.array([x['models']['bioformer'][key]+[0]*(32-len(x['models']['bioformer'][key])) for x in batch],dtype=np.int32)
                    assert np.array_equal(inputs[column],expected)
                t=time.monotonic()
                with tf.device('/CPU:0'):tensor=concrete(*inputs)
                probs=tensor.numpy();elapsed=time.monotonic()-t
                assert tensor.device.endswith('CPU:0') and probs.shape==(len(batch),17266)
                assert np.isfinite(probs).all() and np.allclose(probs.sum(1),1,atol=1e-5)
                lines=output_result(probs,rep.label_2_index,Top_N=3).strip().splitlines();assert len(lines)==len(batch)
                for j,(entry,line) in enumerate(zip(batch,lines)):
                    cells=[]
                    for cell in line.split('\t'):
                        ident,rounded=cell.split('|');column=rep.label_2_index[ident]
                        cells.append(dict(column=column,hpo_id=ident,probability=float(probs[j,column]),native_probability_5dp=rounded))
                    row=dict(unique_index=offset+j,lower_tokens=entry['lower_tokens'],top3=cells,native_line=line,forward_origin='CPU_unigram_full')
                    check(row,offset+j);cache.append(row);out.write(json.dumps(row,ensure_ascii=False)+'\n')
                out.flush();batches.append(dict(offset=offset,rows=len(batch),seconds=elapsed))
                progress.update(status='running_input_only',completed_unique=len(cache),actual_forward_seconds=sum(x['seconds'] for x in batches),elapsed_seconds=time.monotonic()-start)
                write_json(O/'progress.json',progress)
        assert len(cache)==13145 and len(cache)==len(data)
        assert not tf.config.list_physical_devices('GPU') and not tf.config.list_logical_devices('GPU')
        byword={tuple(row['lower_tokens']):row for row in cache};assert len(byword)==13145
        accepted={};reject=Counter()
        for row in cache:
            c=row['top3'][0]
            if c['hpo_id']=='HP:None':reject['top1_None']+=1
            elif not float(c['native_probability_5dp'])>.95:reject['top1_rounded_not_gt95']+=1
            elif not cmap.get(c['hpo_id']):reject['current_HPO_invalid']+=1
            else:accepted[tuple(row['lower_tokens'])]=row
        assert sum(reject.values())+len(accepted)==13145
        coordinates=0;kept=0;old_dict=0;seen=set();bydoc=Counter()
        with (O/'unigram-sidecar.jsonl').open('x') as out:
            for r in map(json.loads,gzip.open(p['coordinates']['path'],'rt')):
                coordinates+=1;word=tuple(r['lower_tokens']);assert word in byword
                passage=docs[r['pmc_id']]['full_text'][r['passage_index']]
                assert passage['offset']+r['local_start']==r['offset'] and passage['text'][r['local_start']:r['local_end']]==r['text']
                assert r['length']==r['local_end']-r['local_start']
                if word not in accepted:continue
                row=accepted[word];c=row['top3'][0];key=(r['pmc_id'],r['offset'],r['length'],cmap[c['hpo_id']]);assert key not in seen;seen.add(key)
                side=dict(r,passage_offset=passage['offset'],passage_type=passage['type'],raw_hpo_id=c['hpo_id'],hpo_id=cmap[c['hpo_id']],score=float(c['native_probability_5dp']),
                    raw_probability=c['probability'],native_probability_5dp=c['native_probability_5dp'],head_column=c['column'],
                    unique_index=row['unique_index'],source='Bioformer_public_native_unigram_extension',
                    dictionary_added=False,abbreviation_added=False)
                out.write(json.dumps(side,ensure_ascii=False)+'\n');kept+=1;old_dict+=int(r['raw_dictionary_span_present']);bydoc[r['pmc_id']]+=1
        assert coordinates==121413
        progress.update(status='complete_input_only',completed_unique=len(cache),elapsed_seconds=time.monotonic()-start)
        write_json(O/'progress.json',progress)
        files={x.name:dict(sha256=sha(x),bytes=x.stat().st_size) for x in O.iterdir() if x.is_file()}
        summary=dict(status='complete_all13145_unigram_input_only_cache',completed_at_utc=datetime.now(timezone.utc).isoformat(),
            source_sha256=sha(__file__),plan_sha256=sha('plan.json'),inputs_sha256=p['inputs']['sha256'],coordinates_sha256=p['coordinates']['sha256'],
            weight_sha256=p['weight']['sha256'],loader_sha256=p['loader_sha256'],head_map_sha256=p['head_map']['sha256'],
            counts=dict(unique_inputs=13145,reused_CPU_probe=256,new_forward=12889,coordinates=coordinates,accepted_unique=len(accepted),
                accepted_coordinates=kept,accepted_with_old_raw_dict_span=old_dict,**reject),accepted_coordinates_per_document=dict(sorted(bydoc.items())),
            physical_GPU_devices=[],logical_GPU_devices=[],threads=dict(intra=2,inter=2),tf_version=tf.__version__,
            model_parameters=model.count_params(),model_variables=len(model.weights),imports_seconds=import_seconds,load_seconds=load_seconds,trace_seconds=trace_seconds,
            inference_seconds=sum(x['seconds'] for x in batches),batch_count=len(batches),batch_seconds_min=min(x['seconds'] for x in batches),batch_seconds_max=max(x['seconds'] for x in batches),
            elapsed_seconds=time.monotonic()-start,policy=p['policy'],outputs=files,no_gold=True,no_originaldev=True,no_quality_or_old_pool_change=True,
            limitations=['The unigram policy extends the original2-8 generator; quality unmeasured.','Sidecar has not passed downstream fixed negation/nonoverlap assembly; dictionary-presence overlaps retained for accounting.','Exact first256 CPU probe raw reused with the same source/native loader/weights/representation; no PubMed or old multiword logits reused.'])
        assert summary['elapsed_seconds']<480
        write_json(O/'summary.json',summary);print(json.dumps(summary),flush=True)
    except BaseException as e:
        progress.update(status='incomplete_preserved_no_downstream',exception=repr(e),completed_unique=len(cache),elapsed_seconds=time.monotonic()-start)
        if raw_path.exists():progress['partial_raw']=dict(sha256=sha(raw_path),bytes=raw_path.stat().st_size)
        write_json(O/'progress.json',progress)
        raise


if __name__=='__main__':main()
