"""Bind real final receipts, deploy the fixed final64 runner, bind one endpoint.

Preparation only: this never starts a GPU process or evaluates predictions.
Unknown deployment outcomes require a read-only inventory before any retry.
"""
import argparse
import ast
import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys


TEMPLATE_SHA = '20b71f1254c62f4eadbd3fcc3916c54ae84b8142d5ee4934bf8813116239d2be'
TRAINER_SHA = '9c794287de533eb59e02196bc4177a8a808086b373554a7721ee8e3faee1a9ca'


def read(path):
    return json.loads(Path(path).read_text())


def ref(path):
    path = Path(path).resolve(); raw = path.read_bytes()
    return dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def save(path, value):
    path = Path(path)
    with path.open('x') as handle:
        json.dump(value, handle, indent=2); handle.write('\n')
        handle.flush(); os.fsync(handle.fileno())
    descriptor = os.open(str(path.parent), os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(descriptor)
    finally: os.close(descriptor)


def training_refs(b):
    collection = read(b/'training-collection.json'); wrapper = collection['wrapper']
    assert collection['phase'] == 'train' and wrapper['returncode'] == 0 and not wrapper.get('endpoint_error')
    assert wrapper['all_epochs_persistent_and_readback_verified'] and wrapper['no_SHM_or_pending_export']
    result = {'training_collection': ref(b/'training-collection.json')}
    for key, name in [('training_complete','training_complete.json'),
                      ('training_checkpoint','epoch-002/checkpoint.json'),
                      ('training_durability','durability-epoch-002.json')]:
        actual = ref(b/'training-terminal'/name)
        assert all(actual[k] == collection['files'][name][k] for k in ('bytes','sha256'))
        result[key] = actual
    complete = read(result['training_complete']['path']); checkpoint = read(result['training_checkpoint']['path'])
    durable = read(result['training_durability']['path'])
    assert complete['epochs'] == 2 and complete['global_step'] == 68 and not complete['smoke_only']
    assert complete['source_sha256'] == checkpoint['fingerprint']['source_sha256'] == TRAINER_SHA
    assert checkpoint['completed_epochs'] == durable['completed_epochs'] == 2
    assert checkpoint['global_step'] == durable['global_step'] == 68
    assert durable['source_sha256'] == TRAINER_SHA
    assert durable['file_directory_fsync'] and durable['reopened_SHA_all_exact'] and durable['all_members_reopened_and_SHA_verified']
    assert durable['checkpoint_metadata_sha256'] == result['training_checkpoint']['sha256']
    assert wrapper['durability']['2']['receipt'] == durable
    assert wrapper['durability']['2']['receipt_sha256'] == result['training_durability']['sha256']
    assert set(durable['files']) == set(checkpoint['files']) | {'checkpoint.json'}
    for name, digest in checkpoint['files'].items():
        assert durable['files'][name]['sha256'] == digest
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['bind-inference','deploy-inference','bind-endpoint'])
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--mcp-client-dir', default='/workspace/patientphex-tools')
    args = parser.parse_args(); b = args.directory.resolve(); resource = read(b/'resource.json')
    source = Path(__file__).resolve().parent
    refs = training_refs(b)
    if args.mode == 'bind-inference':
        assert not (b/'infer-plan.json').exists() and not (b/'actual-final-infer.py').exists()
        template = source/'infer.py'; assert ref(template)['sha256'] == TEMPLATE_SHA
        text = template.read_text()
        for constant, key in [('TRAINING_COMPLETE_SHA','training_complete'),
                              ('TRAINING_CHECKPOINT_SHA','training_checkpoint'),
                              ('TRAINING_DURABILITY_SHA','training_durability')]:
            token = '__'+constant+'_PENDING__'; assert text.count(token) == 1
            text = text.replace(token, refs[key]['sha256'])
        # Optional module documentation is updated; execution and generation stay fixed.
        text = text.replace('DRAFT ONLY: fixed originaldev64', 'Fixed originaldev64')
        text = text.replace('epoch2/68-step receipt SHA values remain pending and block every execution.',
                            'epoch2/68-step receipt SHA values are bound to the actual persistent final state.')
        ast.parse(text)
        with (b/'actual-final-infer.py').open('x') as handle:
            handle.write(text); handle.flush(); os.fsync(handle.fileno())
        payload = read(b/'payload.json')
        names = ['infer.py','limited_codec.py','patient_first.py','raw_store.py','formal_driver.py',
                 'formal_operations.py','model-source-manifest.json','prompts.jsonl','input-token-audit.json']
        actual = {name: (ref(b/'actual-final-infer.py') if name == 'infer.py' else payload[name]) for name in names}
        for item in actual.values():
            assert ref(item['path']) == item
        remote = resource['remote_preparation_absolute']; output = resource['run_root']+'/final64'
        final = resource['run_root']+'/formal2ep/epoch-002'
        argv = [resource['python'],'-u',remote+'/infer.py']
        for key, value in [('model',resource['model']),('source-manifest',remote+'/model-source-manifest.json'),
                           ('prompts',remote+'/prompts.jsonl'),('token-audit',remote+'/input-token-audit.json'),
                           ('adapter',final+'/adapter'),('training-checkpoint',final+'/checkpoint.json'),
                           ('training-complete',resource['run_root']+'/formal2ep/training_complete.json'),
                           ('training-durability',resource['run_root']+'/formal2ep/durability-epoch-002.json'),
                           ('output',output+'/responses.jsonl')]:
            argv += ['--'+key, value]
        plan = dict(status='frozen_actual_durable_full64_epoch2_originaldev64_all_fresh', phase='infer',
            cap_seconds=900, grace_seconds=30, gpu_uuid=resource['gpu_uuid'], argv=argv,
            files={remote+'/'+name:{k:item[k] for k in ('bytes','sha256')} for name,item in actual.items()},
            output_dir=output, storage_parent=resource['storage_parent'], minimum_free_disk_bytes=1024**3,
            rows=64, documents=16, candidates=466, pairs=1006, cached_rows=0, new_rows=64,
            decode='Unchanged native nonthinking greedy;14000input/1600new;IDs then raw before limited patient codec',
            only_complete_durable_epoch2=True, first_hard_failure_stop=True, no_retry=True,
            no_partial_scoring=True, no_quality=True, actual_training_references=refs,
            endpoint_policy=ref(b/'endpoint-policy.json'))
        save(b/'infer-plan.json', plan)
        save(b/'actual-final-inference-binding.json', dict(at_utc=datetime.now(timezone.utc).isoformat(),
            template=ref(template), runner=ref(b/'actual-final-infer.py'), plan=ref(b/'infer-plan.json'),
            actual_training=refs, only_three_SHA_constants_and_module_docstring_changed=True,
            generation_or_quality_started=False))
    elif args.mode == 'deploy-inference':
        assert not (b/'inference-deployment-intent.json').exists(), 'Unknown outcome: inspect existing files, never blindly redeploy'
        plan = read(b/'infer-plan.json'); assert plan['actual_training_references'] == refs
        runner = ref(b/'actual-final-infer.py'); remote = resource['remote_preparation_absolute']
        assert plan['files'][remote+'/infer.py']['sha256'] == runner['sha256']
        files = {name:Path(path).read_bytes() for name,path in [('infer.py',b/'actual-final-infer.py'),('infer-plan.json',b/'infer-plan.json')]}
        encoded = {name:base64.b64encode(raw).decode() for name,raw in files.items()}
        expected = {name:dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()) for name,raw in files.items()}
        code = '''from pathlib import Path
import hashlib,json,base64,os
root=Path(ROOT);data=DATA;expected=EXPECTED
assert hashlib.sha256((root/'infer.py').read_bytes()).hexdigest()==OLD_SHA
assert not(root/'infer-plan.json').exists()
for name,text in data.items():
 raw=base64.b64decode(text);assert len(raw)==expected[name]['bytes'] and hashlib.sha256(raw).hexdigest()==expected[name]['sha256']
 if name.endswith('.py'):compile(raw,name,'exec')
 temp=root/(name+'.final-bound.tmp')
 with temp.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 if name=='infer.py':os.replace(temp,root/name)
 else:os.link(temp,root/name);temp.unlink()
fd=os.open(str(root),os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
assert all(hashlib.sha256((root/n).read_bytes()).hexdigest()==v['sha256'] for n,v in expected.items())
print(json.dumps(dict(files=expected,all_exact=True,source_compile=True,no_GPU=True)))
'''.replace('ROOT',repr(remote)).replace('DATA',repr(encoded)).replace('EXPECTED',repr(expected)).replace('OLD_SHA',repr(TEMPLATE_SHA))
        save(b/'inference-deployment-intent.json', dict(at_utc=datetime.now(timezone.utc).isoformat(),
            files=expected, code_sha256=hashlib.sha256(code.encode()).hexdigest(), no_GPU=True,
            unknown_outcome='Read exact original two paths and temp files; do not repeat mutation'))
        sys.path.insert(0,args.mcp_client_dir); from mcp_client import Client
        client=Client();client.initialize()
        result=client.request('tools/call',dict(name='run_command',arguments=dict(mount_id=resource['mount'],
            command='python3 -c '+shlex.quote(code),cwd='.',timeout_seconds=40)))
        save(b/'inference-deployment-result.json',result)
        assert not result.get('isError'),result
        assert result['structuredContent']['exit_code']==0,result
        save(b/'inference-deployment.json',result['structuredContent'])
    else:
        assert (b/'infer-job.json').exists() and not (b/'endpoint-plan.json').exists()
        plan=read(b/'endpoint-plan.template.json')
        plan['references'].update(refs,runner=ref(b/'actual-final-infer.py'))
        plan['inference_plan']=ref(b/'infer-plan.json');plan['inference_job']=ref(b/'infer-job.json')
        plan['policy']=ref(b/'endpoint-policy.json');plan['source_sha256']=ref(source/'endpoint.py')['sha256']
        plan['status']='BOUND_ACTUAL_TRAINING_AND_INFERENCE_PRE_QUALITY'
        save(b/'endpoint-plan.json',plan)
    print(json.dumps(dict(mode=args.mode,complete=True,no_model_or_quality_call=True)))


if __name__=='__main__':
    main()
