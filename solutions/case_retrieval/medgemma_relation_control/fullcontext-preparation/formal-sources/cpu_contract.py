"""CPU source/input/resume metadata contracts; no model/tokenizer/MCP/quality calls."""
from pathlib import Path
from types import SimpleNamespace
import argparse
import ast
import copy
import hashlib
import json
import sys
import tempfile

sys.dont_write_bytecode = True


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('/workspace/patientphex-runs/round2-20261023'))
    parser.add_argument('--output',type=Path,required=True)
    opts=parser.parse_args()
    e=opts.root/'research/medgemma-relation-control';d=e/'fullcontext-preparation';here=Path(__file__).resolve().parent
    p=opts.root/'case_retrieval/qwen35-preparation/sft-meta1-fusion'
    sys.path[:0]=[str(here),str(e),str(p)]
    import train_medgemma32 as old
    import train_medgemma32_fullcontext as new
    import infer_medgemma80_fullcontext as infer
    import fullcontext_training_contract as contract
    args=SimpleNamespace(mode='train',train=str(d/'train.fullcontext.jsonl'),original_train=str(p/'train.jsonl'),
                         train_input=str(p/'public-training-candidate-preparation/train32.input.jsonl'),
                         excluded_input=str(p/'excluded.input.jsonl'),data_manifest=str(p/'preparation-manifest.json'),
                         fullcontext_manifest=str(d/'preparation.json'),cpu_reference=str(d/'actual-cpu/summary.json'),
                         cpu_row_audit=str(d/'actual-cpu/row-token-audit.json'),inference_prompts=str(d/'infer.fullcontext.jsonl'),
                         original_inference_prompts=str(p/'meta.prompts.jsonl'),
                         inference_input='/workspace/patientphex-data/round2-inner4/fold-1/validation16.input.jsonl',
                         model_source_manifest=str(opts.root/'research/medical-small-model-readiness/assets/expected-files.json'),
                         resume=None,stop_after_epoch=1)
    checks=[]
    rows,train_ids,excluded=new.verify_sources_and_data(args)
    assert len(rows)==124 and len(train_ids)==len(excluded)==32
    checks.append('real124_original_label_chain_and_append_reconstruction')
    ir,ia=contract.verify_inference_sources(args,new.interface,args.inference_prompts,args.original_inference_prompts,args.model_source_manifest)
    assert len(ir)==len(ia)==80 and len({str(r['pmc_id']) for r in ir})==16
    checks.append('real80_pure_input_append_reconstruction_no_gold')

    def rejects(name,fn):
        try:fn()
        except (ValueError,AssertionError):checks.append(name)
        else:raise AssertionError('Expected rejection: '+name)

    changed=copy.copy(args);changed.data_manifest=args.fullcontext_manifest
    rejects('append_manifest_cannot_replace_original_label_provenance',lambda:new.verify_sources_and_data(changed))
    changed=copy.copy(args);changed.train=args.original_train
    rejects('short_input_data_cannot_enter_fullcontext_run',lambda:new.verify_sources_and_data(changed))
    rejects('original80_cannot_replace_full80_input',lambda:contract.verify_inference_sources(args,new.interface,args.original_inference_prompts,args.original_inference_prompts,args.model_source_manifest))
    assert new.PREPARATION_SHA==old.PREPARATION_SHA and new.TRAIN_SHA!=old.TRAIN_SHA and new.INFERENCE_SHA!=old.INFERENCE_SHA
    assert {k:v for k,v in new.CONFIG.items() if k!='max_seq_length'}=={k:v for k,v in old.CONFIG.items() if k!='max_seq_length'}
    assert old.CONFIG['max_seq_length']==8192 and new.CONFIG['max_seq_length']==14000
    checks.append('only_CONFIG_change_is_explicit14000_cap_original8192_unchanged')
    plans=[new.batch_plan(124,1,8,new.CONFIG['seed'],epoch) for epoch in range(2)]
    assert plans==[old.batch_plan(124,1,8,old.CONFIG['seed'],epoch) for epoch in range(2)]
    assert all(len(g)==16 and [len(x) for x in g]==[8]*15+[4] for g in plans)
    assert all(sorted(i for group in g for batch in group for i in batch)==list(range(124)) for g in plans)
    checks.append('same_two_epoch_order_248exposures_32calls_tail4_seed')
    first=new.semantic_configuration(args)
    resumed=copy.copy(args);resumed.resume='/same-new-output/epoch-001';resumed.stop_after_epoch=2
    assert new.semantic_configuration(resumed)==first
    checks.append('resume_stop_controls_excluded_same_semantic_fingerprint')
    changed=copy.copy(args);changed.train=args.original_train
    assert new.semantic_configuration(changed)!=first
    checks.append('input_path_change_alters_semantic_fingerprint')

    o=(e/'train_medgemma32.py').read_text();n=(here/'train_medgemma32_fullcontext.py').read_text()
    old_ast=ast.parse(o);new_ast=ast.parse(n)
    funcs=lambda tree:{node.name:ast.dump(node,include_attributes=False) for node in tree.body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))}
    old_functions,new_functions=funcs(old_ast),funcs(new_ast)
    names=['semantic_configuration','training_rows','model_files','encode_all','validate_checkpoint','batch_plan',
           'verify_fingerprint','validate_stop_epoch','epoch_outcome','collate']
    assert all(old_functions[name]==new_functions[name] for name in names)
    checks.append('ten_existing_helpers_AST_identical')
    segment=lambda text:text[text.index('        checkpoint=Path(args.resume)'):text.index('\n\ndef main():')]
    assert segment(o)==segment(n)
    assert 'scheduler=get_linear_schedule_with_warmup(optimizer,2,32)' in segment(n)
    checks.append('native_load_LoRA_Adam_RNG_epoch_save_pause_terminal_complete_body_identical')
    a=(e/'infer_medgemma80.py').read_text();b=(here/'infer_medgemma80_fullcontext.py').read_text()
    assert a[a.index('    text_config = AutoConfig'):]==b[b.index('    text_config = AutoConfig'):]
    checks.append('complete_native_inference_load_generation_output_body_identical')
    oldinfer_funcs=funcs(ast.parse(a));newinfer_funcs=funcs(ast.parse(b))
    assert oldinfer_funcs['validate_existing']==newinfer_funcs['validate_existing']
    checks.append('inference_exact_response_resume_guard_AST_identical')

    with tempfile.TemporaryDirectory(prefix='med-fullcontext-own-epoch-contract-') as temp:
        output=Path(temp);epoch=output/'epoch-001';(epoch/'adapter').mkdir(parents=True)
        files={}
        for name in ['training_state.pt','adapter/adapter_model.safetensors','adapter/adapter_config.json']:
            path=epoch/name;path.write_text('CPU fixture only; not a real training state\n');files[name]=contract.proven.sha(path)
        fp={'schema':'patientphex-medgemma15-fullbundle-text-relation-fullcontext-lora-v1',
            'configuration':first,'source_sha256':contract.proven.sha(here/'train_medgemma32_fullcontext.py'),
            'append_only_provenance':contract.provenance()}
        meta={'fingerprint':fp,'completed_epochs':1,'global_step':16,'files':files,'smoke_only':False}
        (epoch/'checkpoint.json').write_text(json.dumps(meta))
        assert new.validate_checkpoint(output,epoch,fp)==meta
        checks.append('own_epoch1_metadata_and_all_file_hashes_accept')
        oldfp=copy.deepcopy(fp);oldfp['schema']='patientphex-medgemma15-fullbundle-text-relation-lora-v1'
        rejects('short_Med_checkpoint_schema_rejected',lambda:new.validate_checkpoint(output,epoch,oldfp))
        wrong=copy.deepcopy(fp);wrong['append_only_provenance']['original_label_training_sha256']=new.TRAIN_SHA
        rejects('rewritten_original_label_chain_rejected_in_resume',lambda:new.validate_checkpoint(output,epoch,wrong))
        (epoch/'training_state.pt').write_text('changed')
        rejects('changed_Adam_RNG_state_bytes_rejected',lambda:new.validate_checkpoint(output,epoch,fp))

    toy={'row_id':'toy','pmc_id':'toy-pmc'};ids=[2,3,4]
    expected={'row_id':'toy','pmc_id':'toy-pmc','input_tokens':3,'token_ids_sha256':contract.construct.jsha(ids),
              'no_answer':True,'rendered_prefix_sha256':hashlib.sha256(b'prompt').hexdigest()}
    contract.verify_inference_encoding(toy,ids,expected,'prompt')
    rejects('same_length_native_input_token_change_rejected',lambda:contract.verify_inference_encoding(toy,[2,5,4],expected,'prompt'))
    rejects('rendered_prefix_change_rejected',lambda:contract.verify_inference_encoding(toy,ids,expected,'changed'))
    assert 'torch' not in sys.modules and 'transformers' not in sys.modules and 'tokenizers' not in sys.modules
    checks.append('source_CPU_only_no_model_or_tokenizer_import')
    for path in here.glob('*.py'):ast.parse(path.read_text(),filename=str(path))
    table=[{'optimizer_call':step+1,'learning_rate_before_step':1e-4*(step/2 if step<2 else (32-step)/30)} for step in range(32)]
    result={'status':'CPU_source_input_resume_contracts_passed','checks_passed':checks,'check_count':len(checks),
            'source_sha256':contract.proven.sha(__file__),'trainer_sha256':contract.proven.sha(here/'train_medgemma32_fullcontext.py'),
            'inference_sha256':contract.proven.sha(here/'infer_medgemma80_fullcontext.py'),
            'input_contract_sha256':contract.proven.sha(here/'fullcontext_training_contract.py'),
            'unchanged_training_load_resume_loop_checkpoint_terminal_bytes':len(segment(n).encode()),
            'unchanged_training_load_resume_loop_checkpoint_terminal_sha256':hashlib.sha256(segment(n).encode()).hexdigest(),
            'identical_helper_functions':names,'schedule':{'epochs':2,'exposures':248,'optimizer_calls':32,'warmup_steps':2,'tail_group':4,'seed':20261023},
            'LR_table_static_fixed_api_formula_not_new_scheduler_execution':table,
            'real_new_model_encoding':False,'model_assets_rehashed':False,'MCP_GPU_quality_gold':False,
            'new_trainer_CPU_meta_run_claimed':False,'actual_existing_native_CPU_summary_sha256':contract.proven.CPU_SHA}
    with opts.output.open('x') as handle:handle.write(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'checks':len(checks),'receipt_sha256':contract.proven.sha(opts.output)}))


if __name__=='__main__':main()
