"""No retokenization: bind actual8B IDs to the existing native training interface."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
from common import ref,write,load_jsonl

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-root',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    R=Path(a.run_root);out=Path(a.output);out.mkdir(parents=True,exist_ok=False);W=Path(__file__).resolve().parent
    result_path=R/'qwen3-native-runtime-v1/collected/actual/result.json';native_path=R/'qwen3-native-runtime-v1/collected/actual/native/manifest.json'
    result=json.loads(result_path.read_text());native=json.loads(native_path.read_text())
    assert result['status']=='PASS_ACTUAL_NATIVE_HF_PREFIX_PROCESSOR_NO_MODEL' and result['all126_native_HF_IDs_decode_exact']
    encoded_path=R/'qwen3-native-runtime-v1/collected/actual/native/encoded.json.gz'
    assert ref(encoded_path)['sha256']==result['encoded']['sha256']
    original=json.loads(gzip.decompress(encoded_path.read_bytes()));expanded={}
    remote='/home/zhangyuheng/projects/CHIP2026_PatientPheX/research/canonical-names-v1/qwen3-candidate-v1'
    counts={}
    for phase,inputname in [('train','train.jsonl'),('held','F1.input.jsonl')]:
        targets={r['row_id']:r for r in load_jsonl(R/'targets-v1'/inputname)}
        rows=[]
        for row in original[phase]:
            source=targets[row['row_id']]
            value={**row,'pmc_id':source['pmc_id'],'target_patient_id':source['target_patient_id']}
            assert value['input_ids']==row['input_ids']
            if phase=='train':assert value['labels']==row['labels']
            else:assert 'answer' not in value and 'labels' not in value
            rows.append(value)
        assert len(rows)==(87 if phase=='train' else 39)
        raw=b''.join((json.dumps(v,separators=(',',':'))+'\n').encode() for v in rows)
        path=out/(phase+'.encoded.jsonl.gz');path.write_bytes(gzip.compress(raw,mtime=0));item=ref(path)
        counts[phase]=dict(rows=len(rows),documents=len({r['pmc_id'] for r in rows}),tokens=sum(len(r['input_ids']) for r in rows),
            max_tokens=max(len(r['input_ids']) for r in rows),supervised_tokens=sum(r.get('supervised_tokens',0) for r in rows),
            encoded={**item,'path':remote+'/data/'+path.name})
        expanded[phase]=rows
    assert counts['train']['tokens']==434582 and counts['train']['supervised_tokens']==5406
    assert counts['held']['tokens']==198785
    receipt={**ref(result_path),'path':'/home/zhangyuheng/projects/CHIP2026_PatientPheX/research/canonical-names-v1/qwen3-native-cpu-v1/actual/result.json'}
    cpu=dict(status='ACTUAL8B_IDS_METADATA_ONLY_NATIVE_TRAINING_ADAPTER',counts=counts,native_cpu_receipt=receipt,
        stress_initial8=native['probe_first_real_group'],stress_accumulated3=native['probe_accumulate_next_three'],
        pareto_indices=[x['index'] for x in native['joint_length_supervision_frontier']],
        pareto_shapes=[[x['full_tokens'],x['supervised_tokens']] for x in native['joint_length_supervision_frontier']],
        no_retokenization=True,original_encoded=ref(encoded_path),native_manifest=ref(native_path))
    write(out/'cpu.json',cpu)
    Q=Path('/workspace/patientphex-worktrees/explore-qwen3-relations/solutions/qwen3_relations')
    prior=json.loads((Q/'plan.json').read_text());backend=Q/'full64_component'
    source_refs={}
    for name in ['qwen3_entry.py','phase_driver.py','schema_contract.py']:
        source_refs['own/'+name]={**ref(W/name),'path':remote+'/source/'+name}
    for name in ['common.py','native.py','loss_ops.py']:
        source_refs['backend/'+name]={**ref(backend/name),'path':remote+'/source/backend/'+name}
    assert source_refs['backend/native.py']['sha256']=='41443a75b82bde0dfaad1aa02db2802e22aae4989bcf6d37084b1699d5a0f73c'
    oldnames=R/'qwen3-native-runtime-v1/package.json';pkg=json.loads(oldnames.read_text());names=pkg['files']['names.gz']
    plan=dict(fixed_model='Qwen3-8B-canonical-name-v1',model_id=prior['model_id'],HF_revision=prior['HF_revision'],
        interpretation='Base and target jointly changed; not a target-only ablation. Fixed before new held quality.',
        authorized_phases=['probe','formal','infer'],authorization='root fixed8B and240+30; if all engineering/cost passes fresh87x2/22steps2400+30 and complete39greedy512900+30',
        model_dir=prior['model_dir'],model_files=prior['model_files'],runtime={**prior['runtime'],'tokenizers':'0.21.2'},
        native_forward_sha256=prior['native_forward_sha256'],lora=prior['lora'],seed=prior['seed'],python=prior['python'],
        environment={**prior['environment'],'PYTHONDONTWRITEBYTECODE':'1'},
        gpu_index=0,gpu_uuid='GPU-126293b9-6677-8464-cee7-01e693efbf29',mount_id=prior['mount_id'],
        recipe=dict(rows=87,epochs=2,updates=22,updates_per_epoch=11,last_batch_size=7,warmup_steps=2,
            learning_rate=1e-4,batch_size=1,gradient_accumulation=8,weight_decay=.01,max_grad_norm=1.,rank=16,
            quantization=False,packing=False,truncation=False,loss='Native unchanged indexed standard per-row mean causal CE'),
        budgets=dict(probe=240,formal=2400,infer=900,each_TERM_grace=30),probe_pareto_shapes=cpu['pareto_shapes'],
        probe_generation_train_index=native['longest_row']['index'],probe_generation_uses_training_only=True,
        source_refs=source_refs,backend_source_dir=remote+'/source/backend',run_root=remote+'/run',durability_root=remote,
        cpu_result={**ref(out/'cpu.json'),'path':remote+'/data/cpu.json'},native_cpu_receipt=receipt,
        names_gzip=dict(path='/home/zhangyuheng/projects/CHIP2026_PatientPheX/research/canonical-names-v1/qwen3-native-cpu-v1/names.gz',**names),
        inference=dict(max_new_tokens=512,max_input_reject=9984,maximum_input_plus_output=10435,batch_size=1,do_sample=False,num_beams=1,thinking=False),
        counts=counts,quality_policy='After complete39 and whole-row strict parse, B union all new HPO; preserve1189entities/471relations; completefreeze then1v2/old0; score>.6863389812716092; no single-head quality.',
        no_original_dev_gold=True,NIL_outbranch_limit='All87 retained;14patients have explicit unrepresented NIL/outbranch positives; target is HPO projection only.')
    write(out/'candidate-plan.json',plan)
    write(out/'local-artifacts.json',dict(plan=ref(out/'candidate-plan.json'),cpu=ref(out/'cpu.json'),encoded={k:ref(out/(k+'.encoded.jsonl.gz')) for k in ['train','held']},
        native=ref(native_path),shared_source_commit='f69d88d0',source_refs=source_refs,no_model_GPU_or_quality=True))
    print(json.dumps(ref(out/'local-artifacts.json')))

if __name__=='__main__':main()
