"""Bind complete four-flow evidence; no response parser, gold, model or quality."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from common import MARKER, read, ref, reference_gate, same, save
from inference_gate import relative_member, verify


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--template',type=Path,required=True)
    ap.add_argument('--inference-dir',type=Path,required=True)
    ap.add_argument('--published-plan',type=Path,required=True)
    ap.add_argument('--source',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();plan=read(a.template);reference_gate(plan,a.source,False)
    root=a.inference_dir.resolve()
    assert not a.output.exists() and not Path(plan['output_dir']).exists() and not (root/MARKER).exists()
    same(ref(a.published_plan),plan['expected_gpu_plan']);same(ref(root/'plan.json'),plan['expected_gpu_plan'])
    terminal=read(root/'terminal.json')
    files={name:ref(relative_member(root,name)) for name in terminal['files']}
    for name,identity in terminal['files'].items():same(files[name],identity)
    plan['actual']=dict(root=str(root),plan=ref(root/'plan.json'),terminal=ref(root/'terminal.json'),files=files)
    required={k:plan['fixed_references'][k] for k in ('input','old_candidates','old_HGB_support','task1_template')}
    for arm in ('normal','reverse','public','patient'):
        stem='public_new' if arm=='public' else arm;flow='public-new' if arm=='public' else arm
        required[arm+'_prompts']=plan['fixed_references'][stem+'_prompts']
        required[arm+'_responses']=files['actual/'+flow+'/responses.jsonl']
    plan['assembly_references']=required
    coverage=verify(plan)
    plan.update(execution_ready=True,bound_utc=datetime.now(timezone.utc).isoformat(),
                template_reference=ref(a.template),binder_reference=ref(__file__),
                quality_runner_reference=ref(Path(__file__).with_name('run_quality_once.py')),
                binding_coverage=coverage)
    reference_gate(plan,a.source,True)
    save(a.output,plan);print(json.dumps(ref(a.output)))


if __name__=='__main__':main()
