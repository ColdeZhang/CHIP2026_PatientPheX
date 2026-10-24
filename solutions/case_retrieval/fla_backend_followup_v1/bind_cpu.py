"""Bind successful local CPU receipt to a separately deployed GPU A plan; no start."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
from evidence import checked,ref,save


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--template',type=Path,required=True)
    parser.add_argument('--cpu-dir',type=Path,required=True);parser.add_argument('--remote-cpu-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    plan=json.loads(args.template.read_text());cpu=json.loads((args.cpu_dir/'result.json').read_text())
    intent=json.loads((args.cpu_dir/'intent.json').read_text())
    assert not plan['execution_ready'] and plan['CPU_result'] is None and plan['fixed_inputs'] is None
    assert intent['plan']==ref(args.template)
    assert cpu['status']=='PASS_CPU_A_ONLY_INPUT_INDEX_AND_SAVE_ORDER' and cpu['elapsed_seconds']<=60
    assert cpu['long_positions']==plan['long_positions'] and cpu['long_input_ids_sha256']==plan['long_input_ids_sha256']
    assert cpu['nonfinite_raw_saved_before_observation_and_assert'] and cpu['actual_long_index_map_exact']
    assert not cpu['model_loaded'] and not cpu['GPU_imported'] and not cpu['gold_read'] and cpu['quality_calls']==0
    assert ref(args.cpu_dir/'fixed-inputs.json')==cpu['fixed_inputs']
    for name,expected in plan['sources'].items():
        actual=ref(Path(__file__).parent/name);assert all(actual[k]==expected[k] for k in ('bytes','sha256'))
    refs={}
    for name,filename in [('CPU_result','result.json'),('fixed_inputs','fixed-inputs.json')]:
        value=ref(args.cpu_dir/filename);value['path']=str(args.remote_cpu_dir/filename);refs[name]=value
    plan.update(execution_ready=True,**refs,bound_utc=datetime.now(timezone.utc).isoformat(),
                template_reference=ref(args.template),local_CPU_intent=ref(args.cpu_dir/'intent.json'),binder_reference=ref(__file__))
    save(args.output,plan);print(json.dumps(ref(args.output)))


if __name__=='__main__':main()
