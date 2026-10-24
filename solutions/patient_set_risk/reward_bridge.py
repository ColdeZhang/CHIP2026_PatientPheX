"""All8 raw -> exact necessary public queries -> frozen reward delivery.

Runs under the original main deadline. A query UNKNOWN is saved and never
restarted. No automatic retries, new trajectories, masks or fallback mapper.
"""
import argparse,json,shlex,subprocess,time
from pathlib import Path
from common import atomic,exclusive,ref,sha
from remote import Remote,deploy,P

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--binding',required=True);ap.add_argument('--raw-dir',required=True);ap.add_argument('--output',required=True);ap.add_argument('--sap-release');a=ap.parse_args()
    w=Path(__file__).parent;out=Path(a.output);out.mkdir(parents=True,exist_ok=False);rawdir=Path(a.raw_dir);request=json.loads((rawdir/'reward-request.json').read_text());binding=json.loads(Path(a.binding).read_text());execution=json.loads(Path(binding['execution']['path']).read_text())
    assert request['execution']['sha256']==binding['execution']['sha256'] and sha(rawdir/'responses.json')==request['responses']['sha256']
    assert sha(__file__)==execution['reward_bridge_source']['sha256']
    assert sha(w/'normalization-source-manifest.json')==execution['normalizer_manifest_sha256']
    for n,x in json.loads((w/'normalization-source-manifest.json').read_text()).items():assert sha(w/n)==x['sha256']
    def command(argv,label):
        exclusive(out/(label+'-intent.json'),dict(argv=argv,source=ref(__file__),original_TERM=request['TERM_unix']))
        with (out/(label+'.stdout')).open('xb') as stdout,(out/(label+'.stderr')).open('xb') as stderr:
            done=subprocess.run(argv,stdout=stdout,stderr=stderr,timeout=max(.001,request['TERM_unix']-time.time()-5))
        assert done.returncode==0,(label,done.returncode)
    prep=out/'preparation'
    command(['/usr/bin/python3',str(w/'normalization/prepare.py'),'--responses',str(rawdir/'responses.json'),'--output',str(prep)],'prepare')
    prepared=json.loads((prep/'preparation.json').read_text());query_args=[]
    if prepared['query_count']:
        assert a.sap_release,'Necessary new quotes require actual assigned Sap whole-chain release'
        release_path=Path(a.sap_release);estimate=20+.02*prepared['query_count']
        atomic(out/'waiting-for-Sap-owner.json',dict(expected_release=str(release_path),gpu_uuid='GPU-fc5d0525-3ec1-074a-9fa3-35d92a9da539',
            query_count=prepared['query_count'],shared_TERM=request['TERM_unix'],admission_not_yet_created=True))
        while not release_path.exists():
            assert time.time()+estimate+5<request['TERM_unix'],'Original shared window insufficient while waiting for real GP release'
            time.sleep(2)
        rel=json.loads(release_path.read_text());assert rel['whole_chain_released'] is True and rel['gpu_uuid']=='GPU-fc5d0525-3ec1-074a-9fa3-35d92a9da539'
        qr=P/'research/patient-set-risk-v1/Sap100-v1';bound=out/'query-bound'
        command(['/usr/bin/python3',str(w/'normalization/bind_query.py'),'--preparation',str(prep/'preparation.json'),'--remote-root',str(qr),
                 '--shared-term',str(request['TERM_unix']),'--shared-kill',str(request['KILL_unix']),'--output',str(bound)],'bind-query')
        package=json.loads((bound/'package.json').read_text());qp=json.loads((bound/'query-plan.json').read_text());api=Remote(package['mount'],qp['python'],out/'query-operations')
        atomic(out/'query-deployment.json',deploy(api,package['members'],qr))
        assert time.time()<qp['admission_expires_unix'] and time.time()+qp['cost_estimate_seconds']+5<request['TERM_unix']
        exclusive(bound/'unique-start-intent.json',dict(plan=ref(bound/'query-plan.json'),prior_release=ref(a.sap_release),original_TERM=request['TERM_unix']))
        started=api.call('start_process',dict(command=shlex.join(package['argv']),cwd=str(qr.relative_to(P)),env=package['environment']),'single-query-start');atomic(out/'query-start.json',started)
        while time.time()<request['KILL_unix']:
            observation=api.code("import pathlib,json;p=pathlib.Path('wrapper-complete.json');print(json.dumps({'complete':json.loads(p.read_text()) if p.exists() else None}))",qr,'query-status')
            if observation['complete'] is not None:break
            time.sleep(min(3,max(0,request['KILL_unix']-time.time())))
        else:raise TimeoutError('Shared original deadline reached; query UNKNOWN not relaunched')
        assert observation['complete']['success'],observation['complete']
        collected=out/'query-collected';collected.mkdir()
        for name in package['collect_all']:api.get(qr/name,collected/name,'collect-'+name.replace('/','_'))
        wrapper=json.loads((collected/'wrapper-complete.json').read_text());assert wrapper['success'] and wrapper['child_absent'] and wrapper['target_has_no_process']
        query_args=['--query-output',str(collected/'output'),'--query-plan',str(bound/'query-plan.json')]
    command(['/usr/bin/python3',str(w/'normalization/finalize.py'),'--preparation',str(prep/'preparation.json'),'--output',str(out/'reward')]+query_args,'finalize')
    frozen=ref(out/'reward/rewards.json');atomic(out/'freeze.json',dict(rewards=frozen,raw=ref(rawdir/'responses.json'),normalizer_manifest=ref(w/'normalization-source-manifest.json'),source=ref(__file__),original_TERM=request['TERM_unix'],no_fit=True,no_quality=True))
    command(['python',str(w/'remote.py'),'deliver','--binding',a.binding,'--rewards',frozen['path'],'--output',str(out/'delivery')],'deliver')
    atomic(out/'complete.json',dict(status='ALL8_FROZEN_REWARDS_DELIVERED',rewards=frozen,query_count=prepared['query_count'],source=ref(__file__),original_TERM=request['TERM_unix']))
    print(json.dumps(json.loads(Path(frozen['path']).read_text())['pairs']))
if __name__=='__main__':main()
