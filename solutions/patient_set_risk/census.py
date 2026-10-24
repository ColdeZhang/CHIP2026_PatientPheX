"""Train32-only full-patient reward supply; cached normalization, no model/fit/quality."""
import argparse, collections, hashlib, importlib.util, json, math, re, sys, time
from pathlib import Path

COMPOUND=re.compile(r'HP:[0-9]{7}(?:;HP:[0-9]{7})+')
def keys(values):
    return {atom for value in values for atom in (value.split(';') if COMPOUND.fullmatch(value) else [value])}
def f1(gold,pred):
    return 2*len(gold&pred)/(len(gold)+len(pred)) if gold or pred else 1.
def ref(p):
    p=Path(p).resolve();b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def put(p,x):
    with Path(p).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n')
def main():
    a=argparse.ArgumentParser();a.add_argument('--grounded-source',required=True);a.add_argument('--run-root',required=True);a.add_argument('--hpo',required=True);a.add_argument('--output',required=True);a=a.parse_args();start=time.monotonic();r=Path(a.run_root);g=Path(a.grounded_source);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    sys.path[:0]=[str(g/'normalization'),str(g),'/usr/lib/python3/dist-packages']
    from strict import StrictNormalizer
    from semantic import load_policy,decide,precheck
    from schema import grounded
    train=r/'ontology_rules/nested-outer0-inner1/train32.jsonl';targets=r/'research/grounded-generation-v1/targets-v2/train.jsonl';scores_path=r/'research/grounded-generation-v1/normalization-v1/deployment/terminal/actual/query-scores.json';audit_path=r/'research/grounded-generation-v1/token-census-v1/train-audit.json'
    assert ref(train)['sha256']=='38c353ea9920f8917dbc4e0e7c6d58d2130ebcca9779fcbb04cbc0a2abcb479a'
    assert ref(targets)['sha256']=='bf0091ef057ddb67f9906d035fdaf0ac7b41ae645f77432f0e5a007078291d08'
    assert ref(scores_path)['sha256']=='ad265eccfce114390c8434ad3e8a79982101371bc4fc2a3927fcb5152dec1056'
    docs={str(d['pmc_id']):d for d in map(json.loads,train.read_text().splitlines())};rows=list(map(json.loads,targets.read_text().splitlines()));audit=json.loads(audit_path.read_text());assert len(docs)==32 and len(rows)==87
    normalizer=StrictNormalizer(a.hpo,train);policy=load_policy(g/'normalization/policy_v1.json');scores={x['quote']:x for x in json.loads(scores_path.read_text())};assert len(scores)==624
    def normalize(triples,unit_by_id):
        found=set()
        for triple in triples:
            grounded(triple,unit_by_id);q=triple[1];s=normalizer.normalize(q)
            if s['reason']=='unmatched' and not precheck(q,policy):assert q in scores,'No new model query permitted'
            s=decide(s,scores.get(q),policy)
            if s['status']=='mapped':found.update(s['ids'])
            elif s['status']=='nil':found.add(q)
        return found
    bydoc=collections.defaultdict(list)
    for i,row in enumerate(rows):bydoc[row['pmc_id']].append(i)
    records=[];normalized=[];triples=[];gold=[]
    for i,row in enumerate(rows):
        d=docs[row['pmc_id']];truth=keys(next(x['phenotype'] for x in d['association'] if x['patient_id']==row['target_patient_id']));ts=json.loads(row['answer']);un={u['unit_id']:u for u in row['units']};pred=normalize(ts,un)
        gold.append(truth);triples.append(ts);normalized.append(pred)
    for i,row in enumerate(rows):
        un={u['unit_id']:u for u in row['units']};base=normalized[i];deletion=None;addition=None;same=None
        for j,t in enumerate(triples[i]):
            altered=normalize(triples[i][:j]+triples[i][j+1:],un)
            if altered!=base:deletion=dict(removed_triple_index=j,predicted=sorted(altered),reward=f1(gold[i],altered));break
        for other in bydoc[row['pmc_id']]:
            if other==i:continue
            for t in triples[other]:
                extra=normalize([t],un)
                if extra and not extra&gold[i] and not extra<=base and addition is None:
                    altered=normalize(triples[i]+[t],un);addition=dict(from_row_id=rows[other]['row_id'],triple=t,predicted=sorted(altered),reward=f1(gold[i],altered),negative_means_unannotated_for_target_not_clinical_absence=True)
        # Different already annotated lexical forms with identical normalized concepts
        # provide a direct check that set reward need not equal literal-quote overlap.
        for t in triples[i]:
            k=normalize([t],un)
            if not k:continue
            for other in bydoc[row['pmc_id']]:
                for alt in triples[other]:
                    if t[1]!=alt[1] and normalize([alt],un)==k:
                        same=dict(first=t,alternative=alt,concepts=sorted(k));break
                if same:break
            if same:break
        records.append(dict(index=i,row_id=row['row_id'],pmc_id=row['pmc_id'],patient_id=row['target_patient_id'],gold_count=len(gold[i]),predicted_count=len(base),gold=sorted(gold[i]),normalized_target=sorted(base),target_reward=f1(gold[i],base),delete=deletion,add_foreign=addition,lexically_different_same_concepts=same))
    put(out/'train-only-reward-cases.json',records)
    byid={x['row_id']:x for x in audit};assert set(byid)=={x['row_id'] for x in rows}
    chosen=[]
    def choose(seq):
        v=next(i for i in seq if i not in chosen);chosen.append(v)
    choose(sorted(range(87),key=lambda i:(-byid[rows[i]['row_id']]['prefix_tokens'],i)))
    choose(sorted(range(87),key=lambda i:(-byid[rows[i]['row_id']]['full_tokens'],i)))
    choose(sorted(range(87),key=lambda i:(-byid[rows[i]['row_id']]['supervised_tokens'],i)))
    choose(sorted(range(87),key=lambda i:(-byid[rows[i]['row_id']]['prefix_tokens']*byid[rows[i]['row_id']]['supervised_tokens'],i)))
    pilot=[dict(index=i,row_id=rows[i]['row_id'],gold_count=len(gold[i]),prefix_tokens=byid[rows[i]['row_id']]['prefix_tokens'],supervised_tokens=byid[rows[i]['row_id']]['supervised_tokens']) for i in chosen]
    put(out/'pilot-selection.json',pilot)
    # Scalar policy-gradient identity on all finite outcomes, including a
    # stop-gradient greedy baseline; finite differences are not a model update.
    logits=[-.7,.1,1.2];rewards=[0.,.4,1.];baseline=.4
    def expectation(z):
        ps=[math.exp(x-max(z)) for x in z];ps=[x/sum(ps) for x in ps];return sum(p*r for p,r in zip(ps,rewards)),ps
    mean,ps=expectation(logits);analytical=[p*(rr-mean) for p,rr in zip(ps,rewards)];scst=[sum(ps[k]*(rewards[k]-baseline)*((1 if j==k else 0)-ps[j]) for k in range(3)) for j in range(3)];fd=[]
    for j in range(3):
        hi=logits.copy();lo=logits.copy();hi[j]+=1e-6;lo[j]-=1e-6;fd.append((expectation(hi)[0]-expectation(lo)[0])/2e-6)
    error=max(abs(x-y) for x,y in zip(analytical,fd));assert error<1e-9 and max(abs(x-y) for x,y in zip(analytical,scst))<1e-15
    # Empty, compound, duplicates and semicolon-containing NIL stay v2-compatible.
    assert f1(set(),set())==1 and f1({'a'},set())==0 and keys(['HP:0000001;HP:0000002','HP:0000001','text;detail'])=={'HP:0000001','HP:0000002','text;detail'}
    report=dict(status='CPU_TRAIN_ONLY_REWARD_SUPPLY_NO_MODEL_NO_FIT',seconds=time.monotonic()-start,source=ref(__file__),sources={n:ref(g/n) for n in ('schema.py','normalization/strict.py','normalization/semantic.py','normalization/policy_v1.json')},inputs={n:ref(p) for n,p in [('train32',train),('targets87',targets),('cached_train624_scores',scores_path),('train_token_audit',audit_path)]},documents=32,patients=87,full_v2_gold_atoms=sum(map(len,gold)),empty_patients=sum(not x for x in gold),train_target_reward_mean=sum(x['target_reward'] for x in records)/87,perfect_target_rows=sum(x['target_reward']==1 for x in records),deletion_changes_reward=sum(x['delete'] is not None and x['delete']['reward']!=x['target_reward'] for x in records),foreign_add_changes_reward=sum(x['add_foreign'] is not None and x['add_foreign']['reward']!=x['target_reward'] for x in records),different_quote_same_concept_rows=sum(x['lexically_different_same_concepts'] is not None for x in records),all_normalization_cached_no_model=True,pilot_selection=ref(out/'pilot-selection.json'),cases=ref(out/'train-only-reward-cases.json'),policy_gradient_contract=dict(analytical=analytical,expected_SCST=scst,finite_difference=fd,max_error=error,no_parameters_fit=True),synthetic_perturbations_are_not_sampled_policy_outputs=True,no_observed_policy_reward_diversity_yet=True,model_calls=0,GPU_calls=0,quality_calls=0,held_gold_reads=0)
    put(out/'result.json',report);print(json.dumps(report,ensure_ascii=False))

if __name__=='__main__':main()
