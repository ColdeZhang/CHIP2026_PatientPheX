#!/usr/bin/env python3
"""CPU synthetic contracts only; no task labels, encoder, trained weights or scores."""
import argparse,datetime,json,time
from pathlib import Path
import numpy as np
from census import ref,save,load
from global_head import initialize,forward,loss_and_grad,merge_window_copies,pair_endpoints


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--directory',type=Path,required=True);a=ap.parse_args();b=a.directory
    plan=load(b/'head-plan.json');start=time.monotonic()
    for name in ['global_head','check_head']: assert ref(Path(__file__).with_name(name+'.py')) == plan['sources'][name]
    p=initialize(4,3,5);rng=np.random.default_rng(7);H=rng.normal(size=(5,4));mask=np.array([True,True,False,True,False]);h=H[0].copy();t=H[1].copy()
    contracts=[]
    # Full finite-difference check of EVERY parameter in both arms and both target classes.
    for mode in ['uniform','pair']:
        for target in [0,1]:
            loss,g,aux=loss_and_grad(p,H,h,t,mask,mode,target);max_error=0.;checks=0
            for key,array in p.items():
                for index in np.ndindex(array.shape):
                    saved=float(array[index]);eps=1e-6
                    array[index]=saved+eps;lp=loss_and_grad(p,H,h,t,mask,mode,target)[0]
                    array[index]=saved-eps;lm=loss_and_grad(p,H,h,t,mask,mode,target)[0]
                    array[index]=saved
                    fd=(lp-lm)/(2*eps);max_error=max(max_error,abs(fd-g[key][index]));checks+=1
            assert max_error<1e-8
            input_error=0.
            for key,array in [('H',H),('patient',h),('concept',t)]:
                for index in np.ndindex(array.shape):
                    saved=float(array[index]);eps=1e-6
                    array[index]=saved+eps;lp=loss_and_grad(p,H,h,t,mask,mode,target)[0]
                    array[index]=saved-eps;lm=loss_and_grad(p,H,h,t,mask,mode,target)[0]
                    array[index]=saved
                    input_error=max(input_error,abs((lp-lm)/(2*eps)-aux[key][index]))
            assert input_error<1e-8
            norms={key:float(np.linalg.norm(value)) for key,value in g.items()}
            assert all(value>0 and np.isfinite(value) for value in norms.values())
            assert np.all(aux['H'][~mask]==0)
            # A single analytic toy update must reduce this toy loss, not a quality claim.
            updated={key:val-.01*g[key] for key,val in p.items()}
            after=loss_and_grad(updated,H,h,t,mask,mode,target)[0];assert after<loss
            contracts.append(dict(name='all_parameter_gradients_and_toy_update',mode=mode,target=target,parameters_checked=checks,max_fd_error=max_error,max_input_fd_error=input_error,gradient_norms=norms,loss_before=loss,loss_after=after))
        clean=forward(p,H,h,t,mask,mode)
        poisoned=H.copy();poisoned[~mask]=np.nan
        dirty=forward(p,poisoned,h,t,mask,mode)
        assert clean[0]==dirty[0] and np.array_equal(clean[1]['c'],dirty[1]['c'])
        assert np.isclose(clean[1]['weights'].sum(),1) and (clean[1]['weights']>0).all()
        contracts.append(dict(name='padding_separator_poison_excluded_before_projection',mode=mode,logit_equal=True,context_equal=True))
    # Same parameter objects/count in the two arms; not a head with dead query parameters.
    assert set(p)=={'U','bq','W','bz','A','ba','o','bo'}
    assert sum(x.size for x in p.values()) == 2*4*3+3+4*3+3+3*3*5+5+5+1
    contracts.append(dict(name='same_parameter_architecture',toy_count=sum(x.size for x in p.values()),production_dimensions=[768,128,128],production_head_parameter_count=344577,all_uniform_parameter_gradients_nonzero=True))
    # Disjoint local supports have zero product; global learned query does not require co-occurrence.
    pp=initialize(2,2,2);pp['W']=np.eye(2);pp['bz'][:]=0;pp['U'][:]=0;pp['U'][:2]=np.eye(2)
    tokens=np.array([[2.,0.],[0.,3.],[1.,1.]])
    m=np.ones(3,dtype=bool);pa,co=pair_endpoints(tokens,'P','C',{'P':[[0]]},{'C':[[1]]},m)
    local=np.array([1.,0.,0.])*np.array([0.,1.,0.]);assert local.sum()==0
    _,c1=forward(pp,tokens,pa,co,m,'pair');_,c2=forward(pp,tokens,tokens[1],co,m,'pair')
    assert np.linalg.norm(c1['c'])>0 and not np.allclose(c1['c'],c2['c'])
    _,u1=forward(pp,tokens,pa,co,m,'uniform');_,u2=forward(pp,tokens,tokens[1],co,m,'uniform')
    assert np.array_equal(u1['c'],u2['c'])
    contracts.append(dict(name='cross_window_endpoints_global_support_and_patient_sensitivity',local_product_sum=float(local.sum()),global_context=c1['c'].tolist(),other_patient_context=c2['c'].tolist(),uniform_context_patient_invariant=True))
    # Overlapping entity token positions stay independently addressable; no destructive markers.
    op,oc=pair_endpoints(tokens,'P','C',{'P':[[0,1],[2]]},{'C':[[1,2]]},m)
    assert np.array_equal(op,np.mean([tokens[[0,1]].mean(0),tokens[2]],0)) and np.array_equal(oc,tokens[[1,2]].mean(0))
    contracts.append(dict(name='overlapping_mentions_independent',patient=op.tolist(),concept=oc.tolist()))
    try: pair_endpoints(tokens,'P','UNKNOWN',{'P':[[0]]},{'C':[[1]]},m)
    except ValueError: contracts.append(dict(name='unknown_candidate_hard_rejection',passed=True))
    else: raise AssertionError('unknown concept accepted')
    try: pair_endpoints(tokens,'P','C',{'P':[]},{'C':[[1]]},m)
    except AssertionError: contracts.append(dict(name='missing_patient_occurrence_hard_rejection',passed=True))
    else: raise AssertionError('missing patient accepted')
    w1=np.array([[np.nan,np.nan],[1.,2.],[3.,4.],[np.nan,np.nan]])
    w2=np.array([[np.nan,np.nan],[5.,6.],[7.,8.],[np.nan,np.nan]])
    merged=merge_window_copies([w1,w2],[[-1,0,1,-1],[-1,1,2,-1]],3)
    assert np.array_equal(merged,np.array([[1.,2.],[4.,5.],[7.,8.]]))
    contracts.append(dict(name='duplicate_window_token_mean_and_no_special_token_context',merged=merged.tolist()))
    # Uniform reduction of equal attention logits is exact for this finite toy.
    pzero={key:value.copy() for key,value in p.items()};pzero['U'][:]=0;pzero['bq'][:]=0
    fp=forward(pzero,H,h,t,mask,'pair');fu=forward(pzero,H,h,t,mask,'uniform')
    assert fp[0]==fu[0] and np.array_equal(fp[1]['c'],fu[1]['c'])
    contracts.append(dict(name='equal_scores_reduce_to_uniform',logit_equal=True,context_equal=True))
    result=dict(status='CPU_SYNTHETIC_CONTRACTS_PASS_NOT_MODEL_QUALITY',completed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),seconds=time.monotonic()-start,plan=ref(b/'head-plan.json'),sources=plan['sources'],numpy=np.__version__,contracts=contracts,contracts_count=len(contracts),no_encoder_forward=True,no_task_labels=True,no_fit=True,no_GPU=True,limits='Independently derived NumPy mathematical contract. No Torch encoder/autograd/mixed-precision performance claim. Future implementation needs actual CPU/GPU parity and cost probe before training.')
    save(b/'head-contracts.json',result);print(json.dumps({k:result[k] for k in ['status','seconds','contracts_count']}))

if __name__=='__main__':main()
