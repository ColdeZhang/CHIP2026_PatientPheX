#!/usr/bin/env python3
"""Narrow final source/intent/deadline addendum; no MCP, model or quality calls."""
import ast,copy,datetime,hashlib,json,math,os,shlex,subprocess,sys,tempfile,time
from pathlib import Path

R=Path('/workspace/patientphex-runs/round2-20261023')
E=R/'research/candidate-order-warmstart-entry'
O=R/'validation/candidate-order-warmstart-entry-review'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
ref=lambda p:dict(path=str(p),bytes=Path(p).stat().st_size,sha256=sha(p))

def main():
    earlier=read(O/'review.json')
    assert sha(O/'review.json')=='71b9c9b2cef3c0d3219e24ffc4ce667ae206dd423b25b101b584024f918a2046'
    oldsrc=E/'before-intent-persistence/run_paired.py';oldplan=E/'before-intent-persistence/formal/run-plan.json'
    assert sha(oldsrc)=='b78468eefab32a7067ae8c04a3fb6eb50d594373ecc238a609e16afa775d4acb'
    assert sha(oldplan)=='2dadd63a7e771e1a5a107b94424a656cbf1158669ef318fa7e690ed300790c31'
    assert sha(E/'run_paired.py')=='586620a36721d138f29e9432474eb0a2af0e2510c0d37d277fc83feff9a208c5'
    assert sha(E/'formal/run-plan.json')=='3d0acfe84d09dc897efd1d96966ab294fbeec4fbaec46a22d711bb960e3af727'
    plan=read(E/'formal/run-plan.json');previous=read(oldplan)
    normalized=copy.deepcopy(plan);normalized.pop('prelaunch_revision')
    normalized['controller_sha256']=previous['controller_sha256']
    normalized['references']['run_paired']=previous['references']['run_paired']
    assert normalized==previous
    for record in plan['references'].values():assert ref(record['path'])==record
    assert sha(E/'evaluate_paired.py')=='563da3a19527b28e96e07418eb2fee063762301faf9492630292d61615e8a79d'
    assert sha(E/'train_order_warmstart.py')=='93301f6070f624672d8331b8fda7304089a2af44b95c83424513ef4895873ab4'
    def getclass(p):return next(n for n in ast.parse(Path(p).read_text()).body if isinstance(n,ast.ClassDef) and n.name=='Run')
    oc,nc=getclass(oldsrc),getclass(E/'run_paired.py')
    oldf={n.name:n for n in oc.body if isinstance(n,ast.FunctionDef)}
    newf={n.name:n for n in nc.body if isinstance(n,ast.FunctionDef)}
    assert set(oldf)==set(newf)
    unchanged=[n for n in oldf if n not in {'save','start_bounded'}]
    assert all(ast.dump(oldf[n],include_attributes=False)==ast.dump(newf[n],include_attributes=False) for n in unchanged)
    ns=dict(Path=Path,json=json,math=math,time=time,datetime=datetime,shlex=shlex,os=os)
    pure=ast.ClassDef(name='Boundary',bases=[],keywords=[],body=[newf[n] for n in ['save','remaining','start_bounded']],decorator_list=[])
    exec(compile(ast.fix_missing_locations(ast.Module(body=[pure],type_ignores=[])),'<actual-boundary-only>','exec'),ns)
    checks=[]
    with tempfile.TemporaryDirectory(dir=O,prefix='final-boundary-') as tmp:
        b=ns['Boundary']();b.deadline=time.time()+15;out=Path(tmp)/'job.json'
        def simulated_call(mount,name,args):
            assert name=='start_process' and mount=='synthetic-only'
            intent=read(out.with_suffix('.intent.json'))
            assert intent['absolute_deadline_unix']==b.deadline and intent['payload_argv']==[sys.executable,'-c','print("BOUNDARY_REVIEW")']
            checks.append('real_fsynced_intent_exists_before_mutating_call')
            result=subprocess.run(shlex.split(args['command']),capture_output=True,text=True,timeout=5)
            assert result.returncode==0 and result.stdout.strip()=='BOUNDARY_REVIEW'
            checks.append('actual_python_timeout_benign_payload_pass')
            return dict(id='synthetic-job')
        b.call=simulated_call;b.start_bounded('synthetic-only',[sys.executable,'-c','print("BOUNDARY_REVIEW")'],'.',{},out)
        record=read(out);assert record['job']['id']=='synthetic-job'
        checks.append('returned_job_ID_persisted')
        try:b.start_bounded('synthetic-only',[sys.executable,'-c','raise AssertionError("no duplicate")'],'.',{},out)
        except FileExistsError:checks.append('existing_intent_blocks_duplicate_before_call')
        else:raise AssertionError('duplicate allowed')
        expired=record['argv'][2].replace(repr(b.deadline),repr(time.time()-1),1)
        result=subprocess.run(['python3','-c',expired],capture_output=True,text=True,timeout=5)
        assert result.returncode!=0 and 'BOUNDARY_REVIEW' not in result.stdout
        checks.append('server_expired_deadline_refuses_payload')
    boundary=read(E/'start-boundary-cpu-contract.json')
    assert boundary['source_sha256']==sha(E/'run_paired.py') and boundary['fixture_sha256']==sha(E/'test_start_boundary.py') and boundary['passed']
    paired=read(E/'paired-evaluation-cpu-contract.json')
    assert paired['source_sha256']==sha(E/'evaluate_paired.py') and paired['harness_sha256']==sha(E/'paired_evaluation_cpu_contract.py')
    assert paired['count']==len(paired['cases'])==32 and paired['synthetic_event_trace']==['assembly','assembly','format','format','gold_bytes','v2','v2']
    assert all(paired[k]==0 for k in ['actual_quality_calls','actual_gold_bytes_read','actual_parser_calls','actual_model_loads','actual_network_calls'])
    inventory=read(E/'entry-source-manifest.json')
    for name,item in inventory['files'].items():
        p=E/name;assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256']
    # Original numeric source/protocol reports remain intact. The latest explanatory
    # interface document is a versioned supplement, never evidence of old byte identity.
    olddoc=next(x for x in earlier['references'] if x['path'].endswith('/evaluate_paired_interface.md'))
    selected=[O/'review.json',O/'data-and-schedule.json',O/'trainer-and-ack-source.json',oldsrc,oldplan,E/'before-intent-persistence/entry-source-manifest.json',E/'run_paired.py',E/'formal/run-plan.json',E/'entry-source-manifest.json',E/'test_start_boundary.py',E/'start-boundary-cpu-contract.json',E/'paired_evaluation_cpu_contract.py',E/'paired-evaluation-cpu-contract.json',E/'evaluate_paired_interface.md',E/'evaluate_paired.py',Path(__file__)]
    report=dict(status='PASS_MATCHED_WARMSTART_FINAL_ENTRY_WITH_DURABLE_INTENT_AND_ABSOLUTE_SERVER_DEADLINE',
        at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='Narrow final revision over full source/data/CPU entry71b9c9b2; no additional training, inference or quality.',
        decision='No remaining source-contract blocker. Root-authorized execution uses the final plan and existing runtime/resource checks; this report does not claim a launch or durable production checkpoint export.',
        prior_complete_entry=ref(O/'review.json'),final_plan=ref(E/'formal/run-plan.json'),final_controller=ref(E/'run_paired.py'),
        unchanged=dict(trainer=True,paired_evaluator=True,all_data_and_supervision=True,two_arms_OO_RO=True,calls32_each=True,four_epoch_exports=True,two_original80_inferences=True,total3900_plus_TERM30=True,all_other_controller_methods_AST_exact=unchanged,all_other_plan_fields_exact=True),
        boundary=dict(fsynced_intent_before_every_start=True,exclusive_intent_prevents_duplicate=True,jobID_saved_after_response=True,server_recomputes_remaining_against_same_absolute_deadline=True,expired_server_deadline_refuses_launch=True,remote_timeout_TERM_grace_seconds=30,reviewer_actual_CPU_checks=checks,owner_actual_CPU_checks=boundary['checks'],owner_CPU_seconds=boundary['seconds']),
        endpoint_fixture=dict(final_source_sha256=paired['source_sha256'],actual_cases=32,seconds=paired['seconds'],event_order=paired['synthetic_event_trace'],incomplete_second_arm_blocks_both_assemblies_and_gold=True,synthetic_only=True,no_real_parser_evaluator_gold_model_or_network=True),
        superseded_reference_resolution=dict(old_controller=ref(oldsrc),old_plan=ref(oldplan),old_interface_record=olddoc,current_interface=ref(E/'evaluate_paired_interface.md'),note='The explanatory interface file was expanded to record the reproducible final32case fixture after initial review. It is not asserted byte-identical to the old9097B document; the current fixture source/receipt and unchanged563da3a1 executable establish the current contract. Initial review is retained as chronology and its old controller/plan bytes are preserved separately.'),
        limitations=earlier['limitations']+['Earlier explanatory interface prose changed after initial review. This addendum explicitly supersedes that document reference; it does not rewrite the initial report or treat changed prose as old bytes.'],
        references=[ref(p) for p in selected],no_MCP_GPU_model_or_quality=True)
    (O/'final-boundary-addendum.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],report=ref(O/'final-boundary-addendum.json'))))

if __name__=='__main__':main()
