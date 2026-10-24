"""No-gold synthetic math plus actual complete F1/F0 zero-gate contracts."""
import argparse
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import sys
import numpy as np
import ops


def ref(path):
    p=Path(path);b=p.read_bytes()
    return dict(path=str(p.resolve()),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m)
    return m


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--old-route',type=Path,required=True)
    ap.add_argument('--four-route',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();old=a.old_route;four=a.four_route
    common=load('common',old/'common.py');f0=load('common_f0',old/'common_f0.py')
    # All 16 four-expert patterns and different input contexts.
    cases=[]
    for b in itertools.product([0,1],repeat=4):
        for patients in [1,2,3,7]:
            for disagreement in ([0] if b[1] else [0,1]):
                x=[1.,math.log(patients),float(disagreement)]
                selected=bool(ops.selected(np.zeros(6),[x],[b[:3]])[0])
                assert selected == (sum(b[:3])>=2)
                assert (selected or (b[3] and any(b[:3]))) == (sum(b)>=2)
                cases.append((x,b[:3]))
    x=np.array([v[0] for v in cases]);bits=np.array([v[1] for v in cases])
    y=np.array([i%2 for i in range(len(cases))]);docs=np.array([i//7 for i in range(len(cases))])
    theta=np.array([.2,-.1,.05,-.15,.12,.08]);loss,analytic=ops.objective(theta,x,bits,y,docs)
    numeric=[];h=1e-6
    for j in range(6):
        plus=theta.copy();minus=theta.copy();plus[j]+=h;minus[j]-=h
        numeric.append((ops.objective(plus,x,bits,y,docs)[0]-ops.objective(minus,x,bits,y,docs)[0])/(2*h))
    delta=float(np.max(np.abs(analytic-numeric)))
    assert np.allclose(analytic,numeric,rtol=1e-6,atol=1e-6)
    results={}
    for fold,module in [(1,common),(0,f0)]:
        oldplanpath=old/('fit-plan.json' if fold else 'F0-plan.json')
        plan=json.loads(oldplanpath.read_text());p4=json.loads((four/f'plan-F{fold}.json').read_text())
        # Only explicit input/prediction refs are opened; never all refs/gold.
        u,rows,counts=module.candidates(plan)
        if fold==1:
            base=Path('/workspace/patientphex-runs/round2-20261023/research/public-new-lgb-support-control-20261024/actual-v1/fold1/supported.pred.jsonl')
            base_ref=ref(base)
        else:base_ref=plan['baseline']
        B=module.literal_predictions(base_ref,u);C=module.literal_predictions(p4['refs']['coded'],u)
        old3=module.literal_predictions(p4['refs']['baseline'],u)
        old4_path=Path(p4['output'])/'prediction.jsonl';old4=module.literal_predictions(ref(old4_path),u)
        bits=np.array([r['features'] for r in rows],dtype=np.float64)
        contexts=np.array([[1.,math.log(len(u[r['key'][0]])),0.] for r in rows])
        zero=ops.selected(np.zeros(6),contexts,bits)
        union={tuple(r['key']) for r in rows};majority={tuple(r['key']) for r,v in zip(rows,zero) if v}
        actual=B|majority|(C&union)
        assert B|majority==old3 and actual==old4
        results[str(fold)]=dict(old_plan=ref(oldplanpath),old_four_plan=ref(four/f'plan-F{fold}.json'),
            base=base_ref,coded=p4['refs']['coded'],old_four=ref(old4_path),counts=counts,
            base_relations=len(B),majority_relations=len(majority),coded_relations=len(C),
            fixed_c_intersection=len(C&union),old_three_relations=len(old3),old_four_relations=len(old4),
            semantic_added_at_zero=len(actual-old4),semantic_deleted_at_zero=len(old4-actual),
            serialized_byte_equality_claimed=False)
    out=dict(status='PASS_SYNTHETIC_GRADIENT_AND_FULL_TWO_FOLD_ZERO_GATE_SETS',
        source=ref(__file__),ops=ref(Path(__file__).with_name('ops.py')),numpy_version=np.__version__,
        synthetic_cases=len(cases),synthetic_gradient_max_abs=delta,
        fixed_threshold_operator='>=',S_expert='binary intersection, never average',
        formula='B0 UNION gate_theta(G,S_intersection,Q) UNION (C INTERSECT (G UNION S_intersection UNION Q))',
        folds=results,gold_or_mixed_target_tables_opened=0,fit_calls=0,quality_calls=0,GPU_calls=0)
    with a.output.open('x') as f:json.dump(out,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(dict(status=out['status'],cases=len(cases),gradient_max_abs=delta,
                         folds={k:{n:v[n] for n in ['old_three_relations','old_four_relations','semantic_added_at_zero','semantic_deleted_at_zero']} for k,v in results.items()})))


if __name__=='__main__':main()
