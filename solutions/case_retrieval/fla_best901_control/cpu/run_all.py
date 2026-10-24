"""One <=60s pure-input native tokenizer census; no model, gold or score."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

def ref(p):
    p=Path(p);b=p.read_bytes();return dict(path=str(p.resolve()),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def save(p,v):
    with Path(p).open('x') as f:json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def main():
    a=argparse.ArgumentParser();a.add_argument('--plan',type=Path,required=True);args=a.parse_args()
    started=time.monotonic();p=json.loads(args.plan.read_text());d=args.plan.parent
    assert p['cpu_seconds']==60 and p['flow_order']==['normal','reverse','public-new','patient']
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    def stop(*_):raise TimeoutError('Original single CPU60 budget exhausted')
    signal.signal(signal.SIGALRM,stop);signal.alarm(60)
    result=dict(status='RUNNING_CPU_ONLY',source=ref(__file__),plan=ref(args.plan),flows={},gold_read=False,model_calls=0)
    try:
        import token_audit
        assert ref(token_audit.__file__)['sha256']==p['token_audit_source']['sha256']
        for name in p['flow_order']:
            expected=p['flows'][name];prompts=d/(name+'.prompts.jsonl');out=d/(name+'.native.json')
            actual=ref(prompts);assert all(actual[k]==expected['prompts'][k] for k in ('bytes','sha256'))
            sys.argv=[str(Path(token_audit.__file__)), '--prompts',str(prompts),'--tokenizer',p['tokenizer'],'--template',p['template'],'--output',str(out)]
            token_audit.main();audit=json.loads(out.read_text())
            assert audit['count']==expected['rows'] and audit['total_input_tokens']==expected['actual_old_input_tokens'] and audit['maximum_input_tokens']==expected['actual_old_max_input_tokens']
            order=sorted(audit['rows'],key=lambda row:(row['input_tokens'],row['row_id']))
            assert [dict(row_id=x['row_id'],input_tokens=x['input_tokens']) for x in order]==expected['old_tokens_in_processing_order']
            result['flows'][name]=dict(audit=ref(out),rows=audit['count'],tokens=audit['total_input_tokens'],max_tokens=audit['maximum_input_tokens'],raw_roundtrip_rows=sum(x['native_roundtrip'] for x in audit['rows']),published_normalizer_roundtrip_all=True,old_actual_lengths_and_order_exact=True)
        result.update(status='PASS_ALL234_NATIVE_IDS_INPUTS_OLD_LENGTHS_NO_GPU_GOLD',rows=sum(x['rows'] for x in result['flows'].values()),tokens=sum(x['tokens'] for x in result['flows'].values()))
        assert result['rows']==234 and result['tokens']==715548
    except BaseException as e:
        result.update(status='FAILED_PRESERVED_NO_RETRY',error_type=type(e).__name__,error=str(e));raise
    finally:
        signal.alarm(0);result['elapsed_seconds']=time.monotonic()-started;save(d/'result.json',result)
        fd=os.open(d,os.O_RDONLY);os.fsync(fd);os.close(fd)
        print(json.dumps(dict(status=result['status'],result=ref(d/'result.json'))),flush=True)
if __name__=='__main__':main()
