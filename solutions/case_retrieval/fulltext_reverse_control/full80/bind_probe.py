"""Bind only the completed new reverse probe to fixed80; no model/CPU rerun/quality."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from common import ref, save
from full80_contract import same


def relocated(path, remote):
    value=ref(path);value['path']=str(remote);return value


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--template',type=Path,required=True)
    parser.add_argument('--probe-dir',type=Path,required=True)
    parser.add_argument('--remote-probe-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();plan=json.loads(args.template.read_text());root=args.probe_dir
    assert not plan['execution_ready'] and plan['probe_references'] is None
    terminal=json.loads((root/'terminal.json').read_text());old=json.loads((root/'plan.json').read_text())
    result=json.loads((root/'actual/summary.json').read_text())
    assert terminal['exit_code']==0 and not terminal['timeout_triggered'] and terminal['target_UUID_has_no_compute_process']
    assert result['status']=='PASS_NATIVE_LONGEST_STRICT_COMPLETE' and result['strict_complete'] and result['natural_eos']
    assert result['original_index']==old['probe_index']==19 and result['fresh_rows']==1 and result['quality_calls']==0
    assert not result['hit_token_limit'] and not result['wall_stop']
    assert result['source']['sha256']==old['sources']['runner.py']['sha256']==plan['new_reverse_probe_source_sha256']
    assert result['input_tokens']==14390 and old['input_limit']==14848 and old['generation_limit']==1600
    assert ref(root/'plan.json')['sha256']==plan['reference_probe_plan_sha256']
    assert terminal['plan']['path']==str(args.remote_probe_dir/'plan.json')
    same(ref(root/'plan.json'),terminal['plan']);same(ref(root/'plan.json'),result['plan'])
    for relative,expected in terminal['files'].items():
        value=Path(relative);assert not value.is_absolute() and '..' not in value.parts
        same(ref(root/value),expected)
    for name,expected in old['sources'].items():same(ref(root/name),expected)
    for name in ('model','adapter','adapter_files','CPU_result','CPU_audit','full80_prompts','CPU_source_sha256',
                 'CPU_contract_sha256','runtime_sha256','generation','input_limit','generation_limit'):
        assert old[name]==plan[name]
    refs={key:relocated(root/relative,args.remote_probe_dir/relative) for key,relative in {
        'terminal':'terminal.json','plan':'plan.json','summary':'actual/summary.json',
        'assets':'actual/actual-input-and-assets.json','generation':'actual/generation-config.json',
        'ids':'actual/generated-ids.json','raw':'actual/raw-response.json','format':'actual/format-result.json'}.items()}
    ids=json.loads((root/'actual/generated-ids.json').read_text())
    raw=json.loads((root/'actual/raw-response.json').read_text());form=json.loads((root/'actual/format-result.json').read_text())
    assert ids['input_ids_sha256']==old['row_audit']['reverse']['input_ids_sha256']
    assert raw['row_id']==ids['row_id']==old['row_audit']['row_id']
    assert ids['configured_eos_ids']==[248044,248046] and ids['generated_ids'][-1] in ids['configured_eos_ids']
    assert len(ids['generated_ids'])==raw['output_tokens']==result['generated_tokens']
    assert form['strict_complete'] and form['error'] is None and form['fallback'] is False
    for name,expected in plan['sources'].items():same(ref(Path(__file__).parent/name),expected)
    plan.update(execution_ready=True,status='BOUND_NEW_REVERSE_PROBE_FIXED80_READY_FOR_PARENT_SINGLE_START',
        probe_references=refs,bound_utc=datetime.now(timezone.utc).isoformat(),template_reference=ref(args.template))
    assert not args.output.exists();save(args.output,plan);print(json.dumps(ref(args.output)))


if __name__=='__main__':main()
