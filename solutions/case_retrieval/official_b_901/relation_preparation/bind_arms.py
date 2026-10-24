"""Bind complete B prompt audits to original-model resources and fixed budgets."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil


RUNNER='fe98e76d3692401df9207f67ace24cef57249944ef846b08bc45f8675e0e1333'
LAUNCHER='4cad4674c60555ddaedbeecc6aac6f7724507fe29f8c083c13de77f2e3e93995'
PREPARER='ed6122ba266ac7a3d2f0f923cfd9ed93ad98aba54768675f17b986dd78a3108a'


def ref(path):
    p=Path(path).resolve();return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def read(path): return json.loads(Path(path).read_text())
def save(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2);f.write('\n')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preparation-directory',type=Path,required=True)
    parser.add_argument('--output-root',type=Path,required=True);parser.add_argument('--artifact-root',type=Path,required=True)
    args=parser.parse_args();src=Path(__file__).resolve().parent;prepared=args.preparation_directory.resolve();root=args.artifact_root.resolve()
    preparation=read(prepared/'preparation.json');assert preparation['status']=='PREPARED_INPUT_ONLY_NOT_INFERRED'
    assert preparation['source']['sha256']==PREPARER and preparation['documents']==100 and preparation['patients']==244
    assert ref(src/'qwen35_runner_raw_first.py')['sha256']==RUNNER and ref(src/'launch_arm.py')['sha256']==LAUNCHER
    deployment=read(src/'deployment-contract.json');inventory=read(src/'inventory.json')
    costs={
        'normal':(341.386,root/'case_retrieval/qwen35-preparation/full64-dev-transfer-preparation/full64-adapter.responses.jsonl'),
        'reverse':(230.897,root/'research/reverse-json-candidate-order-dev/formal/terminal/responses.jsonl'),
        'public-new':(139.727,root/'case_retrieval/qwen35-preparation/sft-meta1-fusion/public-new-relations-dev/responses.jsonl'),
    }
    output=args.output_root.resolve();output.mkdir(parents=True,exist_ok=False);summaries=[]
    for arm,(oldseconds,oldpath) in costs.items():
        oldrows=[json.loads(line) for line in oldpath.read_text().splitlines() if line.strip()]
        prompt=prepared/(arm+'.prompts.jsonl');auditpath=prepared/(arm+'.native-audit.json');audit=read(auditpath)
        assert preparation['artifacts'][prompt.name]==ref(prompt)
        assert audit['status']=='PASS_NATIVE_INPUT_BUDGET' and audit['prompts']==ref(prompt)
        assert audit['count']==preparation['counts'][arm]['prompts']
        rows=audit['count'];assert rows>0 and audit['maximum_input_tokens']<=14000
        ratio=max(rows/len(oldrows),audit['total_input_tokens']/sum(r['input_tokens'] for r in oldrows))
        estimate=oldseconds*ratio
        cap=max(600,int(math.ceil((1.75*estimate+120)/60)*60));assert cap<=10800
        config=deployment['resources']['normal_A30' if arm=='normal' else 'reverse_and_public_A800']
        project='/home/zhangyuheng/projects/CHIP2026_PatientPheX'
        adapter=config['adapter'] if arm=='normal' else '/dev/shm/patientphex-sequence-full64-reverse-dev-1045'
        directory=output/(arm+'-v1');directory.mkdir()
        files={'qwen35_runner_raw_first.py':src/'qwen35_runner_raw_first.py', 'launch_arm.py':src/'launch_arm.py',
               'prompts.jsonl':prompt,'native-audit.json':auditpath,
               'model-source-manifest.json':Path(inventory['source_refs']['old_model_source']['path'])}
        for name,path in files.items():shutil.copyfile(path,directory/name)
        localfiles={name:{k:v for k,v in ref(directory/name).items() if k!='path'} for name in files}
        # Reserve64MiB after a conservative per-row1600 generated-ID JSON budget.
        disk=int(64*1024*1024+rows*(1600*14+2048))
        plan=dict(policy='fixed_old1a703b_official901_B_arm_v1',arm=arm,project_root=project,
            launcher_sha256=LAUNCHER,python=config['python'],model=config['model'],adapter=adapter,
            gpu_uuid=config['env']['CUDA_VISIBLE_DEVICES'],local_files=localfiles,rows=rows,
            input_tokens=audit['total_input_tokens'],maximum_input_tokens=audit['maximum_input_tokens'],
            cap_seconds=cap,term_grace_seconds=30,minimum_disk_free_bytes=disk,
            preparation=ref(prepared/'preparation.json'),native_audit=ref(auditpath),input_contract=preparation['references']['input_manifest'],
            budget=dict(prior_responses=ref(oldpath),prior_rows=len(oldrows),prior_input_tokens=sum(r['input_tokens'] for r in oldrows),
                        prior_output_tokens=sum(r['output_tokens'] for r in oldrows),prior_full_seconds=oldseconds,
                        ratio=max(rows/len(oldrows),audit['total_input_tokens']/sum(r['input_tokens'] for r in oldrows)),
                        predicted_seconds=estimate,formula='ceil60(1.75*priorSeconds*max(newRows/oldRows,newInputTokens/oldInputTokens)+120), min600',
                        limitation='Cost estimate, not guaranteed completion; hard deadline retains all partial evidence without silent repeat.'),
            generation='Unchanged oldgreedy14000/1600, fixed1a703b BF16SDPA, batch1, originalEOS and original decoding; no newfit.',
            resource_source=ref(src/'deployment-contract.json'),actual_adapter_presence=ref(root/'official-b-901'/('normal_A30-asset-readonly-0640.json' if arm=='normal' else 'reverse_and_public_A800-asset-readonly-0640.json')),
            source=ref(__file__),no_quality=True,no_upload=True)
        save(directory/'plan.json',plan)
        relative='case-retrieval/official-901-B-20261024/'+arm+'-v1'
        dep=dict(mcp_scripts='/workspace/CHIP2026_PatientPheX/scripts',mount_id=config['mount_id'],remote_relative=relative,
                 remote_absolute=project+'/'+relative,environment=config['env'],source=ref(__file__))
        save(directory/'deployment.json',dep)
        summaries.append(dict(arm=arm,plan=ref(directory/'plan.json'),deployment=ref(directory/'deployment.json'),rows=rows,
                              total_input_tokens=audit['total_input_tokens'],maximum_input_tokens=audit['maximum_input_tokens'],
                              cap_seconds=cap,term_grace_seconds=30,minimum_disk_free_bytes=disk,
                              raw_roundtrip=sum(r['native_roundtrip'] for r in audit['rows']),native_normalized_roundtrip=sum(r['published_normalizer_roundtrip'] for r in audit['rows'])))
    save(output/'bound-plans.json',dict(status='THREE_COMPLETE_B_INPUT_PLANS_BOUND_NOT_STARTED',source=ref(__file__),arms=summaries))
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':main()
