"""Reproduce actual frozen checkpoint-name guard using only synthetic CPU files."""
from pathlib import Path
import ast,datetime,hashlib,json,tempfile
E=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    tree=ast.parse((E/'train_order_warmstart.py').read_text());names={'verify_fingerprint','validate_checkpoint'}
    scope={'Path':Path,'json':json,'sha256':sha}
    exec(compile(ast.Module(body=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in names],type_ignores=[]),'actual_frozen_trainer_guard','exec'),scope)
    checks=[]
    with tempfile.TemporaryDirectory(prefix='order-ack-name-') as d:
        root=Path(d);cp=root/'epoch-001';cp.mkdir();files={}
        for n in ['training_state.pt','adapter/adapter_model.safetensors','adapter/adapter_config.json']:
            p=cp/n;p.parent.mkdir(exist_ok=True);p.write_text(n);files[n]=sha(p)
        meta={'fingerprint':{'frozen':True},'completed_epochs':1,'global_step':16,'smoke_only':False,'files':files}
        (cp/'checkpoint.json').write_text(json.dumps(meta));bad=root/'epoch-001-export-ack.json';bad.write_text('same ack bytes')
        try:scope['validate_checkpoint'](root,cp,meta['fingerprint'])
        except ValueError as ex:assert str(ex)=='Resume requires latest complete epoch checkpoint';checks.append('actual_old_guard_rejects_epoch_prefixed_ack')
        else:raise AssertionError('expected filename conflict')
        good=root/'export-ack-epoch-001.json';raw=bad.read_bytes();bad.rename(good)
        assert good.read_bytes()==raw and not bad.exists() and scope['validate_checkpoint'](root,cp,meta['fingerprint'])==meta
        checks.append('same_ack_renamed_outside_epoch_glob_actual_guard_passes')
        (root/'epoch-002').mkdir()
        try:scope['validate_checkpoint'](root,cp,meta['fingerprint'])
        except ValueError:checks.append('later_complete_epoch_still_blocks_stale_resume')
        else:raise AssertionError('stale resume accepted')
    result={'passed':True,'checks':checks,'trainer_source_sha256':sha(E/'train_order_warmstart.py'),'original_controller_sha256':sha(E/'run_paired.py'),'corrected_controller_sha256':sha(E/'run_paired_ack_paths.py'),'fixture_sha256':sha(__file__),'only_changes':'two controller ack filename constructions; trainer/data/runtime/optimizer unchanged','no_MCP_GPU_model_gold':True}
    with (E/'ack-name-cpu-contract.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))
if __name__=='__main__':main()
