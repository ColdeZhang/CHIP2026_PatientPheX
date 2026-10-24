"""One pure-input eight-pool feasibility pass plus discriminating boundary fixtures."""
import argparse, collections, copy, gzip, hashlib, json, signal, time
from pathlib import Path
from features import relative_features, FEATURES

def ref(p):
    p=Path(p).resolve();return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def checked(r):
    p=Path(r['path']);a=ref(p);assert all(a[k]==r[k] for k in ('sha256','bytes') if k in r);return p
def fixture():
    def c(start,length,identifier,p=None,b=None,z=None):
        out=dict(offset=start,length=length,identifier=identifier)
        for name,score in zip(('public','bioformer','biobert'),(p,b,z)):
            out[name+'_present']=score is not None;out[name+'_score']=False if score is None else score
        return out
    cs=[c(0,4,'HP:0000001',.9),c(0,4,'HP:0000002',.7,.6),c(2,4,'HP:0000001',.8),
        c(4,2,'HP:0000001',.95),c(10,2,'HP:0000001',.99),c(0,4,'HP:0000001;HP:0000002',.65)]
    before=copy.deepcopy(cs);r=relative_features(cs);assert before==cs
    assert r[0]['competition_same_span_other_id_public_max_score']==.7
    assert r[0]['competition_same_span_other_id_public_self_minus_max']==.9-.7
    assert r[0]['competition_same_id_overlapping_span_public_max_score']==.8  # adjacency .95 excluded
    assert r[0]['competition_same_id_overlapping_span_public_self_minus_max']==.9-.8
    assert r[0]['competition_same_span_other_id_bioformer_self_minus_max']==-.6
    assert all(r[4][name]==0 for name in FEATURES)  # own .99 does not fill empty groups
    assert r[5]['competition_same_id_overlapping_span_public_exists']==0  # compound literal not atomized
    assert relative_features([c(0,1,'-1',0.)])[0]==dict.fromkeys(FEATURES,0.)
    invalid=[]
    x=copy.deepcopy(cs);x[0]['selector_probability']=.8;invalid.append(x)
    x=copy.deepcopy(cs);del x[0]['public_present'];invalid.append(x)
    x=copy.deepcopy(cs);x[0]['bioformer_score']=.2;invalid.append(x)
    x=copy.deepcopy(cs);x[0]['public_score']=float('nan');invalid.append(x)
    invalid.append(cs+[copy.deepcopy(cs[0])])
    for x in invalid:
        try:relative_features(x)
        except (AssertionError,KeyError):pass
        else:raise AssertionError('InvalidFixtureAccepted')
    return dict(status='PASS',positive_boundary_checks=8,rejection_cases=len(invalid),input_unchanged=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--contract',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
    started=time.perf_counter();signal.alarm(60)
    contract=json.loads(Path(a.contract).read_text());checked(contract['source']);checked(contract['prepared_manifest'])
    assert contract['CPU_cap_seconds']==60
    tests=fixture();reports=[];global_nonzero=collections.Counter();globmin={n:float('inf') for n in FEATURES};globmax={n:float('-inf') for n in FEATURES}
    for item in contract['pools']:
        path=checked(item['reference']);count=documents=0;nonzero=collections.Counter();digest=hashlib.sha256()
        with gzip.open(path,'rt') as stream:
            for line in stream:
                row=json.loads(line);assert set(row['input'])=={'pmc_id','pmid','patient','full_text'}
                extras=relative_features(row['candidates']);assert len(extras)==len(row['candidates'])
                for values in extras:
                    assert tuple(values)==FEATURES
                    digest.update(json.dumps(values,separators=(',',':'),allow_nan=False).encode())
                    for name,value in values.items():
                        nonzero[name]+=value!=0;globmin[name]=min(globmin[name],value);globmax[name]=max(globmax[name],value)
                count+=len(extras);documents+=1
        assert documents==(48 if item['role']=='train' else 16)
        global_nonzero.update(nonzero)
        reports.append(dict(fold=item['fold'],role=item['role'],pool=item['reference'],documents=documents,candidates=count,
                            feature_values_sha256=digest.hexdigest(),nonzero=dict(nonzero)))
    assert all(global_nonzero[n]>0 and globmin[n]<globmax[n] for n in FEATURES),'DegenerateRelativeFeature'
    result=dict(status='PASS_pure_input_eight_pool_relative_feature_contract',contract=ref(a.contract),source=ref(__file__),feature_source=contract['source'],
                synthetic=tests,features=list(FEATURES),pools=reports,nonzero=dict(global_nonzero),minimum=globmin,maximum=globmax,
                no_gold_or_labels_read=True,no_fit_predict_evaluate_or_GPU=True,no_outputs_mutated=True,seconds=time.perf_counter()-started)
    with Path(a.output).open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({'status':result['status'],'seconds':result['seconds'],'pools':len(reports),'candidates':sum(x['candidates'] for x in reports),'all18_nondegenerate':True}))
if __name__=='__main__':main()
