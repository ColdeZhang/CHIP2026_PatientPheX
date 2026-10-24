"""CPU-only command boundary: durable intent precedes start and remote deadline wins."""
from pathlib import Path
import hashlib,importlib.util,json,subprocess,sys,tempfile,time
E=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('paired',E/'run_paired.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def main():
    started=time.monotonic();checks=[]
    with tempfile.TemporaryDirectory(prefix='order-start-boundary-') as d:
        root=Path(d);run=m.Run.__new__(m.Run);run.deadline=time.time()+10
        def fake_call(mount,name,args):
            assert name=='start_process' and (root/'job.intent.json').is_file()
            intent=json.loads((root/'job.intent.json').read_text());assert intent['status']=='intent_before_start_no_known_job_yet'
            checks.append('intent_exists_before_mutating_start')
            import shlex
            result=subprocess.run(shlex.split(args['command']),capture_output=True,text=True)
            assert result.returncode==0 and result.stdout.strip()=='BOUNDARY_PASS';checks.append('actual_remote_wrapper_executes_benign_command')
            return {'id':'synthetic-job'}
        run.call=fake_call;run.start_bounded('synthetic',[sys.executable,'-c','print("BOUNDARY_PASS")'],'.',{},root/'job.json')
        record=json.loads((root/'job.json').read_text());assert record['job']['id']=='synthetic-job';checks.append('known_job_persisted_after_return')
        try:run.start_bounded('synthetic',[sys.executable,'-c','raise Exception("must not launch")'],'.',{},root/'job.json')
        except FileExistsError:checks.append('duplicate_intent_refuses_second_start')
        else:raise AssertionError('Repeated intent was not rejected')
        # Delayed delivery: execute the exact emitted server wrapper with an already-past deadline.
        wrapper=record['argv'][2].replace(repr(run.deadline),repr(time.time()-1),1)
        result=subprocess.run(['python3','-c',wrapper],capture_output=True,text=True)
        assert result.returncode!=0 and 'BOUNDARY_PASS' not in result.stdout;checks.append('expired_server_deadline_refuses_payload_even_if_client_once_had_time')
    result={'source_sha256':hashlib.sha256((E/'run_paired.py').read_bytes()).hexdigest(),'fixture_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'checks':checks,'passed':True,'seconds':time.monotonic()-started,'no_MCP_or_GPU_or_model_or_gold':True}
    with (E/'start-boundary-cpu-contract.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))

if __name__=='__main__':main()
