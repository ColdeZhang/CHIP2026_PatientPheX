#!/usr/bin/env python3
"""Freeze 32 labeled documents and 16 strictly input-only documents separately."""
import argparse,collections,datetime,gzip,hashlib,json,random,shutil
from pathlib import Path
def load(p):return json.loads(Path(p).read_text())
def ref(p):
    p=Path(p);return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def rows(p):return [json.loads(x) for x in Path(p).read_bytes().splitlines()]
def save(p,v):
    with Path(p).open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--directory',type=Path,required=True);a=ap.parse_args();b=a.directory;out=b/'formal-v1';out.mkdir(exist_ok=False)
    report=load(b/'packed-report.json')
    for r in report['artifacts'].values():assert ref(r['path'])==r
    docs=load(b/'packed-documents.json');trainids=sorted(x['pmc_id'] for x in docs if x['split']=='train');heldids=sorted(x['pmc_id'] for x in docs if x['split']=='F1_input_only')
    assert len(trainids)==32 and len(heldids)==16 and not set(trainids)&set(heldids)
    maps={x['pmc_id']:x for x in [json.loads(l) for l in gzip.open(b/'packed-token-map.jsonl.gz','rt')]}
    pairs=collections.defaultdict(list);patients=collections.defaultdict(lambda:collections.defaultdict(list));concepts=collections.defaultdict(lambda:collections.defaultdict(list));wins=collections.defaultdict(list)
    for x in load(b/'packed-pairs.json'):pairs[x['pmc_id']].append((x['patient_id'],x['concept']))
    for x in load(b/'packed-occurrences.json'):(patients if x['kind']=='patient' else concepts)[x['pmc_id']][x['key']].append(list(range(x['stream_start'],x['stream_end'])))
    for l in gzip.open(b/'packed-windows.jsonl.gz','rt'):
        x=json.loads(l);wins[x['pmc_id']].append(x)
    trainpath=Path(report['refs']['sft_preparation']['path']).parent/'train.jsonl';assert ref(trainpath)['sha256']=='cb7f67e16959e4ec1357823303915aa44cc070c6f527b4c74ae66e23712d84a8'
    labels={}
    for row in rows(trainpath):
        assert str(row['pmc_id']) in trainids
        ans=json.loads(row['answer']);assert set(ans)==set(row['code_map'])
        for code,concept in row['code_map'].items():
            assert set(ans[code])<=set(row['patient_ids'])
            for pid in row['patient_ids']:
                key=(str(row['pmc_id']),pid,concept);assert key not in labels;labels[key]=int(pid in ans[code])
    assert set(labels)=={(pmc,*pair) for pmc in trainids for pair in pairs[pmc]} and len(labels)==2506 and sum(labels.values())==594
    groups=[]
    for ids in [trainids,heldids]:
        records=[]
        for pmc in ids:
            m=maps[pmc];d=dict(pmc_id=pmc,input_ids=m['input_ids'],valid_mask=[x is not None for x in m['offset_map']],windows=wins[pmc],patient_mentions=patients[pmc],concept_mentions=concepts[pmc],pairs=pairs[pmc])
            assert pairs[pmc] and all(pid in patients[pmc] and cid in concepts[pmc] for pid,cid in pairs[pmc])
            for occs in [*patients[pmc].values(),*concepts[pmc].values()]:
                assert occs
                for positions in occs:assert positions and all(d['valid_mask'][i] for i in positions)
            if pmc in trainids:d['labels']=[labels[(pmc,*pair)] for pair in pairs[pmc]]
            else:assert not {'labels','association','answer'}&set(d)
            records.append(d)
        groups.append(records)
    for filename,group in zip(['train-documents.json.gz','F1-input-documents.json.gz'],groups):
        (out/filename).write_bytes(gzip.compress((json.dumps(group,ensure_ascii=False,separators=(',',':'))+'\n').encode(),mtime=0))
    rng=random.Random(20261023);orders=[]
    for _ in range(8):order=trainids.copy();rng.shuffle(order);orders.append(order)
    save(out/'epoch-orders.json',orders)
    for name in ['train_pair_document.py','probe.py','torch_head.py']:shutil.copyfile(Path(__file__).with_name(name),out/name)
    old=load(b/'probe-v1/probe-plan.json');resource=old['resource'];resource['remote']='sequence-round2/pair-document-formal-v1';save(out/'resource.json',resource)
    plan=dict(schema='patientphex-pair-document-formal-v1',created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),authorization='Root 01:01 UTC approved after one incremental review, source push and fresh GPU check; no repeated probe',model=old['model'],model_files=old['model_files'],gpu_uuid=old['gpu_uuid'],seed=20261023,arms=['uniform','pair'],epochs=8,updates_per_arm=256,train_ids=trainids,heldout_ids=heldids,class_weights=old['class_weights'],class_loss=old['class_loss'],LR=old['LR'],weight_decay=.01,clip_norm=1.,encoder_window_microbatch=4,pairchunk=32,precision='FP32; TF32 disabled; eager; nonreentrant checkpoint',budget_seconds=1800,grace_seconds=30,fixed_probability_cut=.5,CPU_endpoint_budget_seconds=30,policy='same seed/reset and document orders, same all trainable encoder/head parameters; only uniform versus learned pair attention; epoch8 only, no threshold scan/early stopping; both full predictions frozen before F1 gold evaluation',keep_policy='pair must strictly improve Score versus uniform and each same-pool old SFT/rules reference; pair macro must not fall versus any of these references. Otherwise retain mechanism evidence only and stop.',update_check_modules=['encoder.embeddings.word_embeddings.weight','encoder.encoder.layer.0.attention.self.query.weight','encoder.encoder.layer.11.output.dense.weight','head.U','head.W','head.A','head.o'],references=dict(packed_report=ref(b/'packed-report.json'),train_labels=ref(trainpath),probe=ref(b/'probe-v1/terminal/probe-result.json'),probe_review=ref(b.parents[1]/'validation/pair-conditioned-document-probe-review/review.json'),F1_input=report['refs']['F1_input'],F1_candidates=report['refs']['F1_candidates'],F1_prompts=report['refs']['F1_prompts']),remote_files={x.name:ref(x) for x in sorted(out.iterdir()) if x.name in ['train-documents.json.gz','F1-input-documents.json.gz','epoch-orders.json','train_pair_document.py','probe.py','torch_head.py']},quality_evaluation=False)
    save(out/'formal-plan.json',plan)
    check=dict(status='FORMAL_CPU_PACKAGING_INPUT_AND_TRAIN_LABEL_CONTRACT_PASS',created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source=ref(__file__),plan=ref(out/'formal-plan.json'),train_documents=32,heldout_documents=16,train_pairs=2506,positive=594,heldout_pairs=1557,heldout_labels_absent=True,all_endpoint_token_positions_valid=True,all_8_orders_same_set=True,shared_total_budget=1800,probe_not_repeated=True,originaldev_not_accessed=True)
    save(out/'preparation.json',check);print(json.dumps(check))

if __name__=='__main__':main()
