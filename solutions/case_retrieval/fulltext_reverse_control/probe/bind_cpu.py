"""Bind successful CPU evidence to one selected-row probe; no tokenizer/model/gold."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from common import ref, save


def jsha(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def located(path, remote):
    value=ref(path);value['path']=str(remote);return value


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--template',type=Path,required=True)
    parser.add_argument('--cpu-dir',type=Path,required=True)
    parser.add_argument('--remote-cpu-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();plan=json.loads(args.template.read_text())
    assert not plan['execution_ready'] and plan['CPU_result'] is None
    cpu=json.loads((args.cpu_dir/'result.json').read_text())
    assert cpu['status']=='PASS_CPU_FIXED_FULLTEXT_REVERSE_NO_MODEL_NO_QUALITY'
    assert cpu['rows']==80 and cpu['documents']==16 and cpu['candidates']==587 and cpu['pairs']==1557
    assert cpu['truncated_rows']==0 and cpu['all_inverse_and_suffix_bytes_exact'] and cpu['raw_Auto_template_IDs_and_roundtrip_all80']
    assert cpu['model_loaded'] is False and cpu['gold_read'] is False and cpu['new_quality_calls']==cpu['GPU_calls']==0
    assert cpu['source']['sha256']==plan['CPU_source_sha256']
    assert cpu['plan']['sha256']==plan['CPU_contract_sha256']
    audit=json.loads((args.cpu_dir/'row-audit.json').read_text())
    assert ref(args.cpu_dir/'row-audit.json')['sha256']==cpu['row_audit']['sha256']
    lengths=[row['reverse']['tokens'] for row in audit]
    index=min(range(80),key=lambda i:(-lengths[i],audit[i]['row_id']))
    assert index==cpu['selected_longest_index'] and audit[index]['index']==index
    assert cpu['maximum_input_tokens']==max(lengths)
    assert cpu['input_cap_ceil512']==((max(lengths)+511)//512)*512
    assert cpu['input_cap_ceil512']+1600<=cpu['model_context']
    rows=[json.loads(line) for line in (args.cpu_dir/'longest-probe.jsonl').read_bytes().splitlines()]
    assert len(rows)==1 and rows[0]['row_id']==cpu['selected_row_id']==audit[index]['row_id']
    assert jsha(rows[0])==audit[index]['reversed_row_sha256']
    for name,expected in plan['sources'].items():
        actual=ref(Path(__file__).parent/name)
        assert all(actual[key]==expected[key] for key in ('bytes','sha256'))
    plan.update(execution_ready=True,status='BOUND_ACTUAL_CPU_LONGEST_READY_FOR_PARENT_UNIQUE_START',
        bound_utc=datetime.now(timezone.utc).isoformat(),template_reference=ref(args.template),
        CPU_result=located(args.cpu_dir/'result.json',args.remote_cpu_dir/'result.json'),
        CPU_audit=located(args.cpu_dir/'row-audit.json',args.remote_cpu_dir/'row-audit.json'),
        probe_input=located(args.cpu_dir/'longest-probe.jsonl',args.remote_cpu_dir/'longest-probe.jsonl'),
        full80_prompts=dict(cpu['artifacts']['prompts']['plain'],path=str(args.remote_cpu_dir/'prompts.jsonl')),
        probe_index=index,row_audit=audit[index],input_limit=cpu['input_cap_ceil512'])
    assert not args.output.exists();save(args.output,plan);print(json.dumps(ref(args.output)))


if __name__=='__main__':main()
