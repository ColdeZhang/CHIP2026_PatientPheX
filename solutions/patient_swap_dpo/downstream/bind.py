"""Bind only completed DPO/inference metadata to disabled downstream templates."""
import argparse
import copy
import json
from pathlib import Path
from common import checked,ref,save,sha


def sources(plan):
    for name,digest in plan['source_files'].items():
        assert sha(Path(__file__).parent/name)==digest,name


def trained(complete_path,ack_path):
    complete=json.loads(Path(complete_path).read_text());ack=json.loads(Path(ack_path).read_text())
    assert complete['status']=='COMPLETE_FIXED_ONE_EPOCH_DPO_WITH_DURABLE_ACK'
    assert complete['global_step']==10 and complete['completed_epochs']==1
    assert complete['counts']=={'pairs':74,'sequences':148,'tokens':519954,'supervised':9344}
    assert complete['acknowledgement']==ack
    assert ack['status']=='DURABLE_COMPLETE_EPOCH_ALL_ADAPTER_ADAM_RNG_REOPEN_EXACT'
    assert ack['epoch']==1 and ack['global_step']==10 and ack['all_rng_exact']
    assert ack['adapter_tensors']==ack['optimizer_states']==504
    assert set(ack['files'])>={'adapter/adapter_model.safetensors','adapter/adapter_config.json','training_state.pt'}
    assert ack['files']['adapter/adapter_model.safetensors']['sha256']!='543c82a582dedc42f463f44d551257b35231352e5d76dcfdaffc43320097e3c8'
    return complete,ack


def alias(local,remote):
    item=ref(local);item['path']=str(remote);return item


def main(args):
    p=json.loads(args.template.read_text());assert p['enabled'] is False;sources(p)
    _,ack=trained(args.formal_complete,args.persistent_ack)
    if args.mode=='infer':
        epoch=Path(ack['checkpoint']['path']).parent;formal=epoch.parent
        assert epoch.name=='epoch-001' and formal.name=='formal'
        previous=json.loads(args.verified_assets.read_text())
        assert set(previous)==set(p['model_files'])
        for name,expected in p['model_files'].items():
            assert previous[name]['sha256']==expected['sha256'] and previous[name]['bytes']==expected['size']
        p['formal_complete']=alias(args.formal_complete,formal/'complete.json')
        p['persistent_ack']=alias(args.persistent_ack,formal/'epoch-001-persistent-ack.json')
        p['verified_assets']=alias(args.verified_assets,args.remote_verified_assets)
        p['adapter_dir']=str(epoch/'adapter')
        p['adapter_files']={Path(k).name:{'bytes':v['bytes'],'sha256':v['sha256']} for k,v in ack['files'].items() if k in ['adapter/adapter_model.safetensors','adapter/adapter_config.json']}
        p['run_root']=str(args.run_root)
        p['held_encoded']=copy.deepcopy(p['local_inputs']['held_encoded']);p['held_encoded']['path']=str(args.held_encoded_remote)
        p['prompts_gzip']=copy.deepcopy(p['local_inputs']['prompts_gzip']);p['prompts_gzip']['path']=str(args.prompts_gzip_remote)
        for item in p['local_inputs'].values():checked(item)
        p['terminal_inputs']={'held80.jsonl.gz':p['held_encoded'],'prompts80.jsonl.gz':p['prompts_gzip'],
                              'formal-complete.json':p['formal_complete'],'persistent-ack.json':p['persistent_ack'],'verified-assets.json':p['verified_assets']}
    else:
        native=args.native_directory
        complete=json.loads((native/'complete.json').read_text());driver=json.loads(args.driver_result.read_text())
        assert complete['status']=='COMPLETE_FIXED_DPO_F1_ALL80_FRESH' and complete['rows']==80 and complete['cached']==0
        assert driver['status']=='COMPLETE_FIXED_DPO_F1_ALL80' and driver['return_code']==0 and not driver['timed_out']
        assert not driver['release']['target_apps']
        for key in ['responses','canonical']:
            local=ref(native/(key+'.jsonl'));assert all(local[k]==complete[key][k] for k in ['bytes','sha256'])
        assert len(list((native/'generated-ids').glob('*.json')))==80
        assert len(list((native/'raw-responses').glob('*.json')))==80
        assert len(list((native/'canonical-records').glob('*.json')))==80
        p['completed_bindings']={'formal_complete':ref(args.formal_complete),'persistent_ack':ref(args.persistent_ack),
                                'native_complete':ref(native/'complete.json'),'driver_result':ref(args.driver_result)}
        p['collected_native']=str(native.resolve());p['output']=str(args.run_root.resolve())
    p['enabled']=True;p['template']=ref(args.template)
    p['activation_scope']='Existing actual complete metadata only; this binder performs no model, remote call, gold read, or evaluation.'
    save(args.output,p)
    print(json.dumps({'bound_plan':ref(args.output),'mode':args.mode,'enabled':True}))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['infer','quality']);ap.add_argument('--template',type=Path,required=True)
    ap.add_argument('--formal-complete',type=Path,required=True);ap.add_argument('--persistent-ack',type=Path,required=True)
    ap.add_argument('--verified-assets',type=Path);ap.add_argument('--remote-verified-assets',type=Path)
    ap.add_argument('--held-encoded-remote',type=Path);ap.add_argument('--prompts-gzip-remote',type=Path)
    ap.add_argument('--native-directory',type=Path);ap.add_argument('--driver-result',type=Path)
    ap.add_argument('--run-root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args()
    for name in (['verified_assets','remote_verified_assets','held_encoded_remote','prompts_gzip_remote'] if a.mode=='infer' else ['native_directory','driver_result']):
        if getattr(a,name) is None:ap.error('--'+name.replace('_','-')+' required for '+a.mode)
    main(a)
