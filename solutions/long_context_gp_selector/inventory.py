"""Read existing label-free pools/metadata; encode F0 pure input only. No fit/NN/gold."""
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from collections import Counter, defaultdict


def ref(path):
    p=Path(path);b=p.read_bytes();return dict(path=str(p.resolve()),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())


def load(path):
    return json.loads(Path(path).read_text())


def main():
    start=time.monotonic();here=Path(__file__).resolve().parent;plan=load(here/'plan.json')
    assert ref(__file__)==plan['source']
    for r in plan['references'].values():assert ref(r['path'])==r
    r=plan['references'];old=load(r['old_gp_plan']['path']);prepared=load(r['prepared_manifest']['path'])['folds'][1]
    fit=load(r['strong_fit_manifest']['path']);terminal=load(r['new_gp_complete']['path'])
    data=json.loads(gzip.decompress(Path(r['new_gp_data']['path']).read_bytes()))
    current=load(r['current_F1_scores']['path']);all_folds={}
    for job in old['jobs']:
        all_folds[job['name']]=set(job['input_ids'])
        assert set(job['train_ids']).isdisjoint(job['input_ids'])
        assert set(job['train_ids']).isdisjoint(old['outer1_held_ids'])
    pools={}
    for role in ('train','input'):
        assert ref(prepared[role]['path'])==prepared[role]
        with gzip.open(prepared[role]['path'],'rt') as f:pools[role]=[json.loads(x) for x in f]
    counts=[]
    for job in old['jobs']:
        role='input' if job['name']=='final48' else 'train'
        rows=[x for x in pools[role] if x['pmc_id'] in set(job['input_ids'])]
        keys={(str(x['pmc_id']),int(c['offset']),int(c['length'])) for x in rows for c in x['candidates']}
        assert len(keys)==job['target_spans'] and len(rows)==16
        counts.append(dict(name=job['name'],training_documents=job['training_documents'],
            training_ids=job['train_ids'],target_ids=job['input_ids'],candidate_rows=sum(len(x['candidates']) for x in rows),
            unique_boundaries=len(keys),train_reference=job['train'],target_reference=job['targets'],
            model_reuse=job['name']=='inner0'))
    input_keys={(str(x['pmc_id']),int(c['offset']),int(c['length'])) for x in pools['input'] for c in x['candidates']}
    existing_keys={(x['pmc_id'],x['offset'],x['length']) for x in current['records']}
    assert existing_keys<=input_keys
    bydoc=defaultdict(list)
    for row in data['held']:
        for b in row['blocks']:
            starts={a+b['passage_offset']:i for i,(a,z) in enumerate(b['offsets'])}
            ends={z+b['passage_offset']:i for i,(a,z) in enumerate(b['offsets'])}
            bydoc[row['pmc_id']].append((starts,ends))
    valid=Counter();missing=Counter()
    for pmc,a,n in input_keys:
        matches=sum(a in ss and a+n in ee and ss[a]<=ee[a+n] for ss,ee in bydoc[pmc])
        if matches:valid['representable']+=1
        elif any(a in ss for ss,ee in bydoc[pmc]) and any(a+n in ee for ss,ee in bydoc[pmc]):missing['no_common_window']+=1
        else:missing['unaligned_endpoint']+=1
    os.environ['CUDA_VISIBLE_DEVICES']='';sys.path.insert(0,plan['cpu_libraries']);sys.path.insert(0,plan['new_gp_source'])
    spec=importlib.util.spec_from_file_location('lcgp_readonly_prepare',r['new_gp_prepare']['path']);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    tok=module.Tokenizer.from_file(r['tokenizer']['path']);tok.no_truncation();tok.no_padding()
    pure=[json.loads(line) for line in Path(r['F0_input']['path']).read_text().splitlines()]
    assert len(pure)==16 and all(set(x)=={'pmc_id','pmid','patient','full_text'} for x in pure)
    rows0,contracts0,_=module.encode_docs(pure,tok,set(),{},False)
    assert not any('positive' in b for row in rows0 for b in row['blocks'])
    rows23=data['train'];ids0=all_folds['inner0'];ids2=all_folds['inner2'];ids3=all_folds['inner3']
    assert {x['pmc_id'] for x in rows23}==ids2|ids3
    groups={'GP23':rows23,'GP03':rows0+[x for x in rows23 if x['pmc_id'] in ids3],
            'GP02':rows0+[x for x in rows23 if x['pmc_id'] in ids2],'GP023':rows0+rows23}
    shape={}
    for name,rows in groups.items():
        shape[name]=dict(documents=len({x['pmc_id'] for x in rows}),windows=len(rows),tokens=sum(len(x['ids']) for x in rows),
            max_tokens=max(len(x['ids']) for x in rows),local_square_cells=sum(x['shape']['local_square_cells'] for x in rows),
            proposed_8epoch_updates=8*len(rows))
    # Derived rows0 are geometry only and never training targets or selector labels.
    epoch_seconds=sum(x['seconds'] for x in terminal['epochs'])
    result=dict(status='STRICT_F1_STRONG48_READINESS_NO_FIT_NO_NEW_MODEL',source=ref(__file__),plan=ref(here/'plan.json'),
        strong_training_documents=fit['training_documents'],strong_features=fit['actual_feature_count'],
        strong_training_rows=fit['training_examples'],strong_input_rows=fit['input_candidates'],
        strict_no_target_or_F1_in_feature_model_training=True,jobs=counts,
        existing_model_training='F2+F3 only; reusable for F0 OOF; not replacement for final48 F1 feature',
        existing_F1_score_records=len(current['records']),existing_F1_score_unique_boundaries=len(existing_keys),
        F1_full_unique_boundaries=len(input_keys),F1_missing_existing_boundaries=len(input_keys-existing_keys),
        F1_representable=valid['representable'],F1_missing_reasons=dict(missing),
        shape_by_model=shape,new_pure_F0_windows=len(rows0),new_pure_F0_max_body_tokens=max(x['body_tokens'] for x in contracts0),
        new_F0_label_reads=0,F1_gold_reads=0,model_calls=0,fit_count=0,
        actual_parent_epoch_seconds=epoch_seconds,actual_parent_program_seconds=terminal['seconds'],
        actual_parent_infer_seconds=terminal['inference_seconds'],
        proposed_features=['lcgp_span_valid','lcgp_span_logit'],
        feature_policy='boolean native-representable + unbounded max raw GP score per exact doc/offset/length; missing false+0 and explicit reason; no ID normalization or GP floor',
        seconds=time.monotonic()-start)
    out=Path(plan['output']);out.mkdir(parents=True,exist_ok=False)
    (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(ref(out/'result.json')))


if __name__=='__main__':main()
