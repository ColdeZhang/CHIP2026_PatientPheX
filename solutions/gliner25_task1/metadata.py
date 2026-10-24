"""Handle the one documented unevaluated native metric sentinel, never loss NaNs."""
import math


def native_summary(summary):
    result=dict(summary)
    best=result.get('best_metric')
    if not result.get('eval_metrics_history') and isinstance(best,float) and best==math.inf:
        result['best_metric']=None
        result['best_metric_status']='not_computed_no_eval'
    return result
