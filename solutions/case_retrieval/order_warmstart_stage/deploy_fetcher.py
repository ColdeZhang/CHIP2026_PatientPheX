"""Deploy the reviewed tiny fetcher on the persistent mount, without starting work."""
from pathlib import Path
import hashlib,json,shlex,sys
E=Path(__file__).resolve().parent
sys.path[:0]=['/workspace/CHIP2026_PatientPheX/scripts','/workspace/patientphex-runs/round2-20261023/case_retrieval/setup']
import remote_io as io

def main():
    p=json.loads((E/'formal/run-plan.json').read_text());io.M=p['persistent_mount_id']
    target=p['persistent_tools']+'/fetch_checkpoint.py';source=E/'fetch_checkpoint.py'
    assert hashlib.sha256(source.read_bytes()).hexdigest()==p['fetcher_sha256']
    assert not (E/'fetcher-deployment.json').exists()
    check=io.call('run_command',{'command':'python3 -c '+shlex.quote('from pathlib import Path;assert not Path('+repr(target)+').exists()'),'cwd':'.','timeout_seconds':10})
    assert check['exit_code']==0;io.upload(str(source),target)
    result=io.call('run_command',{'command':'python3 -c '+shlex.quote('from pathlib import Path;import hashlib,json;p=Path('+repr(target)+');s=hashlib.sha256(p.read_bytes()).hexdigest();assert s=='+repr(p['fetcher_sha256'])+';print(json.dumps(dict(path=str(p),sha256=s,bytes=p.stat().st_size)))'),'cwd':'.','timeout_seconds':10})
    assert result['exit_code']==0
    with (E/'fetcher-deployment.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print('FETCHER_SOURCE_DEPLOYED_NO_JOBS')

if __name__=='__main__':main()
