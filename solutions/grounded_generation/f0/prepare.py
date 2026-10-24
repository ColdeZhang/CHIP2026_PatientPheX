"""All47 F0 pure inputs and same-policy ancestry/local native tokens; no gold/model."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import sys
import time


def ref(path):
    p=Path(path).resolve(); raw=p.read_bytes()
    return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())


def put(path, raw):
    with path.open('xb') as f: f.write(raw);f.flush();os.fsync(f.fileno())
    p=path.parent
    while True:
        fd=os.open(p,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
        if p.parent==p:break
        p=p.parent
    return ref(path)


def save(path,value):return put(path,(json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode())


def digest(value):return hashlib.sha256(json.dumps(value).encode()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    start=time.monotonic();p=json.loads(a.plan.read_text());assert p['budget_seconds']==120
    forbidden={os.path.realpath(v) for v in p['forbidden_gold_paths']}
    def audit(event,args):
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)) and os.path.realpath(os.fsdecode(args[0])) in forbidden:raise PermissionError('No held/originaldev gold in preparation')
    sys.addaudithook(audit)
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('F0 CPU preparation120s')));signal.alarm(120)
    for item in list(p['sources'].values())+list(p['references'].values()):assert ref(item['path'])==item
    assert p['sources']['prepare.py']==ref(__file__)
    sys.path.insert(0,p['source_root']);sys.path.insert(0,p['lightweight_libraries'])
    os.environ.update(CUDA_VISIBLE_DEVICES='',TOKENIZERS_PARALLELISM='false')
    from schema import INPUT_KEYS,messages,units
    from native_tokenizer import FrozenTokenizer
    r=p['references'];docs=[json.loads(v) for v in Path(r['input']['path']).read_text().splitlines()]
    target=json.loads(Path(r['training_target_manifest']['path']).read_text());provenance=json.loads(Path(r['training_split_provenance']['path']).read_text())
    train=set(target['train_ids']);ids=[str(d['pmc_id']) for d in docs]
    assert len(train)==32 and train==set(provenance['training_document_ids'])
    assert len(ids)==len(set(ids))==16 and not train&set(ids) and set(ids)<=set(provenance['excluded_document_ids'])
    assert all(set(d)==INPUT_KEYS for d in docs) and sum(len(d['patient']) for d in docs)==47
    rows=[]
    for d in docs:
        for patient in d['patient']:
            pid=patient['patient_id'];rows.append(dict(row_id=f"{d['pmc_id']}:{pid}:grounded",pmc_id=str(d['pmc_id']),target_patient_id=pid,messages=messages(d,pid),units=units(d)))
    assert len({v['row_id'] for v in rows})==len(rows)==47 and all('answer' not in v for v in rows)
    tokenizer=FrozenTokenizer(r['tokenizer']['path'],r['template']['path']);audits=[];encoded=[]
    for row in rows:
        rendered=tokenizer.apply_chat_template(row['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        tokens=tokenizer.apply_chat_template(row['messages'],tokenize=True,add_generation_prompt=True,enable_thinking=False)
        assert tokens==tokenizer.backend.encode(rendered,add_special_tokens=False).ids
        audits.append(dict(row_id=row['row_id'],pmc_id=row['pmc_id'],input_tokens=len(tokens),input_ids_sha256=digest(tokens),rendered_prefix_sha256=hashlib.sha256(rendered.encode()).hexdigest()))
        encoded.append(dict(row_id=row['row_id'],input_ids=tokens))
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    prompts=put(out/'F0.input47.jsonl',b''.join((json.dumps(v,ensure_ascii=False,separators=(',',':'))+'\n').encode() for v in rows))
    idsref=put(out/'held-local-ids.json.gz',gzip.compress((json.dumps(encoded,ensure_ascii=False,separators=(',',':'))+'\n').encode(),mtime=0))
    auditref=save(out/'F0-input-audit.json',audits);lengths=[v['input_tokens'] for v in audits]
    ancestry=dict(train32_document_ids=sorted(train),F0_document_ids=sorted(ids),disjoint=True,
        only_project_supervised_sources=[r['training_target_manifest'],r['training_split_provenance']],
        normalizer_train32_sha256='38c353ea9920f8917dbc4e0e7c6d58d2130ebcca9779fcbb04cbc0a2abcb479a',
        normalizer_policy=r['normalizer_policy'],normalizer_source_review=r['normalizer_source_review'],
        public_bank_manifest=r['public_bank_manifest'],public_bank_semantics='Pure public41099 text rows; discard all old train64 label IDs/counts and derive mapping only from fixed publicHPO; same public Sap weights, no learned detector',
        model_source='Same fresh train32 grounded epoch2 d52e2e6c; no F1 or F0 supervision update, no retrain',
        train_only_aliases_and_no_held_quote_dictionary_updates=True,
        F0_evaluation_is_known_development_not_blind=True,public_pretraining_corpus_document_exclusion_not_claimed=True)
    ancestryref=save(out/'ancestry.json',ancestry)
    result=dict(status='COMPLETE_F0_PURE47_LOCAL_NATIVE_INPUT_PREPARATION',source=ref(__file__),plan=ref(a.plan),sources=p['sources'],references=r,
        prompts=prompts,local_ids=idsref,audit=auditref,ancestry=ancestryref,documents=16,patients=47,rows=47,
        input_tokens=dict(sum=sum(lengths),minimum=min(lengths),maximum=max(lengths),mean=statistics.mean(lengths),median=statistics.median(lengths)),
        max_new_tokens=1600,all_full_text_no_truncation=True,all47_retained=True,no_gold_no_fit_no_NN_no_quality=True,
        native_actual_Transformers47_replay_pending=True,seconds=time.monotonic()-start)
    print(json.dumps(save(out/'manifest.json',result)))


if __name__=='__main__':main()
