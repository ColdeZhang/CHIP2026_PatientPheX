"""Bind completed fresh FLA full80 evidence only; no gold, parser or quality call."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import json
from common import ACTUAL_NAMES, read, ref, reference_gate, save
from inference_gate import verify


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--template',type=Path,required=True)
    parser.add_argument('--inference-dir',type=Path,required=True)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();plan=read(args.template);root=args.inference_dir.resolve()
    reference_gate(plan,args.source,False)
    assert not args.output.exists() and not Path(plan['output_dir']).exists()
    assert not (root/'evaluation-started-fulltext-fla-v1.json').exists()
    plan['endpoint_references']={name:ref(root/relative) for name,relative in ACTUAL_NAMES.items()}
    _,_,_,coverage,_=verify(plan)
    plan.update(execution_ready=True,bound_utc=datetime.now(timezone.utc).isoformat(),
        template_reference=ref(args.template),binder_reference=ref(__file__),
        quality_runner_reference=ref(Path(__file__).with_name('run_quality_once.py')),
        actual_binding_coverage=coverage,binding_scope='Full source/member SHA, all80 fresh identities and canonical evidence; no native cache; no tokenizer/response parser/gold/scoring.')
    save(args.output,plan);print(json.dumps(ref(args.output)))


if __name__=='__main__':main()
