"""Distribution diagnostics from persisted tensors; never a replacement pass gate."""
import math


def compare(left,right,torch,positions):
    assert left.shape==right.shape and left.ndim==3 and left.shape[0]==1 and left.shape[1]==len(positions)
    reports=[]
    for index,position in enumerate(positions):
        a=left[0,index].double();b=right[0,index].double();d=b-a
        if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
            reports.append(dict(position=position,finite=False,statistics=None));continue
        ca=a-a.mean();cb=b-b.mean();cd=cb-ca
        la=a-a.logsumexp(0);lb=b-b.logsumexp(0);pa=la.exp();pb=lb.exp()
        va,ia=torch.topk(a,2);vb,ib=torch.topk(b,2)
        an=float(a.norm());cn=float(ca.norm());epsilon=float(cd.abs().max());margin=float(va[0]-va[1])
        reports.append(dict(position=position,finite=True,raw_mean_left=float(a.mean()),raw_mean_right=float(b.mean()),
            raw_max_abs=float(d.abs().max()),raw_reference_l2=an,raw_error_l2=float(d.norm()),
            raw_relative_l2=float(d.norm())/an if an else None,centered_max_abs=epsilon,
            centered_reference_l2=cn,centered_error_l2=float(cd.norm()),centered_relative_l2=float(cd.norm())/cn if cn else None,
            mean_shift_energy=float((b.mean()-a.mean()).square())*a.numel(),raw_error_energy=float(d.square().sum()),
            KL_left_to_right=float((pa*(la-lb)).sum()),KL_right_to_left=float((pb*(lb-la)).sum()),
            probability_L1=float((pa-pb).abs().sum()),logprob_max_abs=float((la-lb).abs().max()),
            top1_left=int(ia[0]),top1_right=int(ib[0]),top1_equal=bool(ia[0]==ib[0]),
            top1_margin_left=margin,top1_margin_right=float(vb[0]-vb[1]),
            top1_probability_left=float(pa[ia[0]]),top1_probability_right=float(pb[ib[0]]),
            margin_guarantees_unchanged_top1=margin>2*epsilon))
    summary={}
    for key in ('raw_max_abs','centered_max_abs','KL_left_to_right','KL_right_to_left','probability_L1','logprob_max_abs'):
        values=sorted(r[key] for r in reports if r['finite'])
        summary[key]=dict(mean=sum(values)/len(values),maximum=max(values),p50=values[(len(values)-1)//2],
                          p95=values[math.ceil(.95*len(values))-1]) if values else None
    return dict(per_position=reports,summary=summary,finite=all(r['finite'] for r in reports),
        top1_disagreement_positions=[r['position'] for r in reports if r['finite'] and not r['top1_equal']],
        interpretation='Diagnostic only; original failed raw-logit contract unchanged; no new pass/rollout gate.')
