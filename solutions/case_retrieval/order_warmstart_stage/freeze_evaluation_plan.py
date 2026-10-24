"""Materialize only frozen endpoint references after both inference outputs exist."""
from pathlib import Path
import argparse, hashlib, json

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p): return dict(path=str(p.resolve()),bytes=p.stat().st_size,sha256=sha(p))

def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True,type=Path);a=p.parse_args();root=a.directory.resolve()
    plan=json.loads((root/'run-plan.json').read_text())
    assert sha(__file__)==plan['references']['freeze_evaluation_plan']['sha256']
    assert (root/'all-responses-complete.json').is_file()
    template=Path(plan['evaluation_template']['path']);assert ref(template)==plan['evaluation_template']
    evaluation=json.loads(template.read_text());names={
      'training_manifest':'epoch-002/run_manifest.json','training_complete':'train-002/training_complete.json',
      'checkpoint':'epoch-002/checkpoint.json','export_ack':'epoch-002/export-ack.json',
      'responses':'responses.jsonl','response_manifest':'responses.manifest.json',
      'response_freeze':'responses-freeze.json','inference_plan':'inference-plan.json'}
    for arm in ('original','reverse-first'):
        evaluation['arms'][arm]={'references':{key:ref(root/arm/name) for key,name in names.items()}}
    # No gold read or hash. The endpoint freezes both assembled full predictions first.
    out=root/'paired-evaluation';out.mkdir(exist_ok=False)
    with (out/'run-plan.json').open('x') as f:json.dump(evaluation,f,indent=2);f.write('\n')
    print(out)

if __name__=='__main__':main()
