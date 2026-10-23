#!/usr/bin/env python3
"""Verify all F0 forward scores and freeze them without labels or decoding."""
from pathlib import Path
import argparse,datetime,gzip,hashlib,json,math
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return dict(path=str(p.resolve()),bytes=p.stat().st_size,sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x]
def checked(v):assert sha(v['path'])==v['sha256'] and Path(v['path']).stat().st_size==v['bytes']
def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);a=p.parse_args();b=a.directory.resolve()
    plan=read(b/'run-plan.json');complete=read(b/'complete.json');manifest=read(b/'run_manifest.json')
    assert complete['status']=='all58_F0_binary32_probabilities_complete' and complete['elapsed_seconds']<180
    assert complete['plan_sha256']==sha(b/'remote-plan.json') and complete['manifest_sha256']==sha(b/'run_manifest.json')
    assert complete['probabilities_sha256']==sha(b/'probabilities.jsonl')
    assert manifest['infer_sha256']==sha(b/'infer_outer_slots.py') and manifest['common_sha256']==sha(b/'slot_model.py')
    assert manifest['encoded_sha256']==sha(b/'F0-slot-inputs.json.gz')
    index=read(plan['old_SFT48_source_index']['path']);checked(plan['old_SFT48_source_index'])
    for k in ['F0_prompts','adapter58.responses.jsonl','adapter58.pred.jsonl']:checked(index[k])
    prompts=rows(index['F0_prompts']['path']);by={x['row_id']:x for x in prompts}
    encoded=json.loads(gzip.decompress((b/'F0-slot-inputs.json.gz').read_bytes()))
    values={x['row_id']:x for x in rows(b/'probabilities.jsonl')}
    assert len(by)==len(encoded)==len(values)==58 and set(by)==set(values)
    count=0
    for e in encoded:
        prompt=by[e['row_id']];v=values[e['row_id']]
        keys=[[cid,pid] for cid in prompt['code_map'] for pid in prompt['patient_ids']]
        assert keys==e['keys']==v['keys'] and len(v['logits'])==len(v['probabilities'])==len(keys)
        assert v['input_tokens']==len(e['input_ids'])
        assert v['input_ids_sha256']==hashlib.sha256(json.dumps(e['input_ids']).encode()).hexdigest()
        for pr,z in zip(v['probabilities'],v['logits']):
            assert len(z)==2 and all(math.isfinite(x) for x in z) and 0<=pr<=1
            q=math.exp(z[1]-max(z))/sum(math.exp(x-max(z)) for x in z)
            assert abs(pr-q)<2e-7
        count+=len(keys)
    assert count==1166 and sum(len(x['code_map']) for x in prompts)==415
    release=read(b/'gpu-release.json');assert release['exit_code']==0 and read_json_string(release['stdout'])['empty']
    freeze=dict(status='all58_1166_F0_probabilities_frozen_before_any_new_quality',
        at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source=ref(Path(__file__)),
        plan=ref(b/'run-plan.json'),complete=ref(b/'complete.json'),manifest=ref(b/'run_manifest.json'),
        probabilities=ref(b/'probabilities.jsonl'),inputs=ref(b/'F0-slot-inputs.json.gz'),
        prompts=index['F0_prompts'],cached_SFT48_raw=index['adapter58.responses.jsonl'],
        cached_SFT48_prediction=index['adapter58.pred.jsonl'],GPU_release=ref(b/'gpu-release.json'),
        rows=58,CIDs=415,pairs=1166,no_gold=True,no_new_decoding_policy=True,no_new_predictions=True,
        feature_semantics='Actual raw logits[0,1] at fixedslot end; p=FP32softmax(logits)[1]. Calibrator may use logit1-logit0 without clipping.')
    with (b/'scores-freeze.json').open('x') as f:f.write(json.dumps(freeze,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(ref(b/'scores-freeze.json')))
def read_json_string(s):return json.loads(s)
if __name__=='__main__':main()
