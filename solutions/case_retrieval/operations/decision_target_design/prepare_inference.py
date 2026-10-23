#!/usr/bin/env python3
"""Bind the two fixed F1 arms to an actual complete epoch002; no labels opened."""
from pathlib import Path
import argparse,json,hashlib,shutil,datetime

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return dict(path=str(p.resolve()),bytes=p.stat().st_size,sha256=sha(p))
def save(p,x):
    with p.open('x') as f:f.write(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main():
    p=argparse.ArgumentParser();p.add_argument('--formal',type=Path,required=True)
    p.add_argument('--directory',type=Path,required=True);p.add_argument('--old-control',type=Path,required=True)
    a=p.parse_args();f=a.formal.resolve();b=a.directory.resolve();assert not b.exists();b.mkdir()
    complete=json.loads((f/'training_complete.json').read_text());checkpoint=json.loads((f/'epoch-002/checkpoint.json').read_text())
    assert complete['epochs']==2 and complete['global_step']==32
    assert checkpoint['completed_epochs']==2 and checkpoint['global_step']==32 and not checkpoint['partial_not_for_inference']
    assert complete['checkpoint_sha256']==sha(f/'epoch-002/checkpoint.json') and complete['manifest_sha256']==sha(f/'run_manifest.json')
    formal=json.loads((f/'run-plan.json').read_text());old=json.loads((a.old_control/'run-plan.json').read_text())
    for name in ['slot_model.py','infer_slots.py','qwen35_runner.py','slot-meta-inputs.json.gz','launch_job.py']:
        assert sha(f/name)==formal['references'][name]['sha256'];shutil.copy2(f/name,b/name)
    shutil.copy2(Path(__file__).parent/'evaluate_slots.py',b/'evaluate_slots.py')
    remote='sft-meta1-fusion/decision-target-context-design/'+b.name
    adapter={Path(k).name:v['sha256'] for k,v in checkpoint['files'].items() if k in ['adapter/adapter_model.safetensors','adapter/adapter_config.json']}
    rp=dict(infer_sha256=sha(b/'infer_slots.py'),common_sha256=sha(b/'slot_model.py'),encoded_sha256=sha(b/'slot-meta-inputs.json.gz'),
        gpu_uuid=formal['gpu_uuid'],training_directory=formal['remote_root'],training_complete_sha256=sha(f/'training_complete.json'),
        adapter_files=adapter,arms=['base_disabled_adapter','final_epoch002'],threshold=.5,cap_seconds=300,grace_seconds=30)
    save(b/'remote-plan.json',rp)
    r=old['references'];er={k:r[k] for k in ['parser','input','evaluator','validator','hpo']}
    er.update(entities=r['old_entities'],prompts=r['old_prompts'],old_raw=r['old_raw'],encoded=ref(b/'slot-meta-inputs.json.gz'),
        old_prediction=ref(a.old_control/'baseline.pred.jsonl'),old_summary=ref(a.old_control/'paired-summary.json'))
    labels=json.loads((f.parent.parent/'conditional-subset-likelihood-cpu/all80cid/run-plan.json').read_text())['local_labels_after_freeze']
    # Only filesystem metadata for the unchanged historical label file, not its contents.
    gold=dict(path=old['gold_path'],bytes=Path(old['gold_path']).stat().st_size,sha256=old['historical_F1_gold_sha256'])
    files=['slot_model.py','infer_slots.py','qwen35_runner.py','slot-meta-inputs.json.gz','launch_job.py','evaluate_slots.py','remote-plan.json']
    refs={name:ref(b/name) for name in files}
    refs.update({'formal_complete':ref(f/'training_complete.json'),'formal_manifest':ref(f/'run_manifest.json'),
        'formal_checkpoint':ref(f/'epoch-002/checkpoint.json')});refs.update({'eval/'+k:v for k,v in er.items()})
    remotehash={remote+'/'+name:sha(b/name) for name in files if name not in ['launch_job.py','evaluate_slots.py']}
    for name in ['training_complete.json','run_manifest.json','epoch-002/checkpoint.json']:
        remotehash[formal['remote_root']+'/'+name]=sha(f/name)
    for name,digest in adapter.items():remotehash[formal['remote_root']+'/epoch-002/adapter/'+name]=digest
    plan=dict(status='authorized_same_template_base_and_final_slot_F1_once',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        launcher_sha256=sha(b/'launch_job.py'),evaluator_sha256=sha(b/'evaluate_slots.py'),references=refs,
        mount_id=formal['mount_id'],gpu_uuid=formal['gpu_uuid'],cwd=formal['cwd'],env=formal['env'],remote_root=remote,
        cap_seconds=300,grace_seconds=30,minimum_free_disk_bytes=20000000,remote_sources_sha256=remotehash,
        argv=['/usr/bin/timeout','--signal=TERM','--kill-after=30s','300',formal['argv'][4],'-u',remote+'/infer_slots.py',
            '--directory',remote,'--model','/dev/shm/patientphex-qwen35-1045-c20223623576','--source-manifest','model-source-manifest.json'],
        collect=['run_manifest.json','base.probabilities.jsonl','base.responses.jsonl','adapter.probabilities.jsonl','adapter.responses.jsonl','complete.json'],
        python=old['python'],evaluation_references=er,labels_after_freeze=labels,gold_after_freeze=gold,
        protocol=formal['protocol'],no_quality_or_labels_opened_in_preparation=True)
    save(b/'run-plan.json',plan);print(json.dumps(ref(b/'run-plan.json')))
if __name__=='__main__':main()
