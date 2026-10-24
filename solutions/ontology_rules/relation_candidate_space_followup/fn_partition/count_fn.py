"""Partition fixed reference FNs using existing legal spaces and positive caches."""
import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import re
import signal
import time


def main(plan_path):
    started=time.monotonic();signal.alarm(120);plan=json.loads(Path(plan_path).read_text())
    # Load only the already frozen stdlib-only accounting helpers; no main call.
    import hashlib
    p=Path(plan['utility']['path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==plan['utility']['sha256']
    spec=importlib.util.spec_from_file_location('frozen_space_helpers',p);u=importlib.util.module_from_spec(spec);spec.loader.exec_module(u)
    assert u.ref(__file__)==plan['source'];out=Path(plan['output_directory']);out.mkdir(exist_ok=False)
    inv=u.read(plan['inventory']);baseline=u.read(plan['baseline']);prior=u.read(plan['prior_result']);spacefreeze=u.read(plan['prior_input_freeze']);spaces=u.read(spacefreeze['spaces'])
    control=u.read(plan['support_plan']);supportfreeze=u.read(plan['support_freeze']);assert supportfreeze['plan']==plan['support_plan']
    evaluator=u.extract(inv['common']['evaluator_v2_normalization_only'],{'COMPOUND_HPO','associations'})
    supportfn=u.extract(inv['common']['prior_public_LGB_support_control_source'],{'entity_support'})['entity_support']
    def rels(docs):return {(str(x['pmc_id']),str(pid),c) for x in docs for pid,cs in evaluator['associations'](x).items() for c in cs}
    allinput=[];running=[]
    for i,f in enumerate(inv['folds']):
        assert f['fold']==i==spaces[i]['fold']==control['folds'][i]['fold']==supportfreeze['folds'][i]['fold']
        inp=u.rows(f['input']);ids=[str(x['pmc_id']) for x in inp];patients={(str(x['pmc_id']),str(p['patient_id'])) for x in inp for p in x['patient']}
        legal={(d,p,v) for d,p,k,v in spaces[i]['raw_legal_pairs']}
        publiclegal={(str(x['pmc_id']),str(p),c) for x in u.rows(f['prompts']) for p in x['patient_ids'] for c in x['code_map'].values()}
        oldlegal={(str(x['pmc_id']),str(p),c) for x in u.rows(f['old_prompts']) for p in x['patient_ids'] for c in x['code_map'].values()}
        assert legal==oldlegal|publiclegal
        refs=control['folds'][i]['inputs'];old=rels(u.rows(refs['baseline']));raw={tuple(x) for x in u.read(refs['raw_additions'])};support=supportfn(u.rows(refs['new_support']))
        retained={tuple(x) for x in u.read(supportfreeze['folds'][i]['new_keys'])};rejected=raw-retained
        assert retained=={x for x in raw if (x[0],x[2]) in support}
        preds=u.rows(baseline['folds'][i]['current_relation_prediction']);assert [str(x['pmc_id']) for x in preds]==ids
        P=rels(preds);assert P==old|retained and not old&raw
        assert all((d,p) in patients for d,p,c in legal|raw|P)
        stats=dict(fold=i,old_positive=len(old),public_raw_positive=len(raw),public_gate_retained=len(retained),public_gate_rejected=len(rejected),final_positive=len(P),old_legal_pairs=len(oldlegal),public_legal_pairs=len(publiclegal),combined_legal_pairs=len(legal),old_positive_outside_legal=len(old-legal),public_raw_outside_public_legal=len(raw-publiclegal),final_identity_and_existing_gate_exact=True)
        allinput.append(stats);running.append((i,f,ids,patients,legal,oldlegal,publiclegal,old,raw,retained,rejected,P))
    u.write(out/'prelabel-freeze.json',dict(at_utc=u.now(),plan=u.ref(plan_path),prior_input_freeze=plan['prior_input_freeze'],folds=allinput,gold_read=False,all_existing_set_identities_checked=True))
    results=[]
    for i,f,ids,patients,legal,oldlegal,publiclegal,old,raw,retained,rejected,P in running:
        gp=Path(f['gold_unread']['path']);assert u.sha(gp)==f['gold_unread']['historical_sha256_from_existing_plan'];gold=[json.loads(x) for x in gp.read_text().splitlines() if x.strip()];assert [str(x['pmc_id']) for x in gold]==ids
        G=rels(gold);FN=G-P;counts=dict(tp=len(G&P),fp=len(P-G),fn=len(FN));assert counts==prior['relationship_counts'][i]['baseline_relation_counts_crosscheck']
        categories=Counter();bytype=Counter();eligible_space=Counter();perdoc=Counter();perdoc_category=Counter()
        for x in FN:
            d,p,c=x;linked=re.fullmatch(r'HP:[0-9]{7}',c) is not None
            if (d,p) not in patients:category='unexplained_patient_alignment'
            elif x in raw and x not in legal:category='unexplained_raw_positive_outside_legal'
            elif x in raw:
                assert x in rejected and x not in retained
                category='public_raw_positive_rejected_by_fixed_LGB_support'
            elif x not in legal:
                category='no_literal_HPO_legal_candidate' if linked else 'nonHPO_or_text_no_literal_legal_match'
            else:
                assert x not in old and x not in raw
                category='legal_but_no_existing_positive_output'
                eligible_space['old_and_public' if x in oldlegal and x in publiclegal else ('old_only' if x in oldlegal else 'public_only')]+=1
            categories[category]+=1;bytype[(category,'literal_HPO' if linked else 'nonHPO_or_text')]+=1;perdoc[d]+=1;perdoc_category[(d,category)]+=1
        assert sum(categories.values())==len(FN)
        proportions={k:dict(count=v,share_of_reference_FN=v/len(FN),documents=sum((d,k) in perdoc_category for d in ids),mean_per_document=v/len(ids),macro_share_within_document_FN=sum(perdoc_category.get((d,k),0)/perdoc[d] if perdoc[d] else 0 for d in ids)/len(ids)) for k,v in sorted(categories.items())}
        results.append(dict(fold=i,baseline_relation_counts_crosscheck=counts,partition=proportions,by_type=[dict(category=k,kind=t,count=v) for (k,t),v in sorted(bytype.items())],legal_without_positive_space=dict(eligible_space),unexplained_count=sum(v for k,v in categories.items() if k.startswith('unexplained')),nonHPO_or_text_separate=categories['nonHPO_or_text_no_literal_legal_match']))
    result=dict(status='COMPLETE_AGGREGATE_FN_PARTITION',at_utc=u.now(),source=u.ref(__file__),plan=u.ref(plan_path),prelabel_freeze=u.ref(out/'prelabel-freeze.json'),seconds=time.monotonic()-started,folds=results,input_counts=allinput,new_v2_calls=0,new_model_calls=0,new_fit_calls=0,original_dev_gold_read=False,limitations=plan['limitations'])
    u.write(out/'result.json',result);signal.alarm(0);print(json.dumps(dict(status=result['status'],seconds=result['seconds'],folds=results)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);main(p.parse_args().plan)
