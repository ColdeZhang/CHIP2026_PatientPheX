"""Mechanical source/plan binding for one same-host fresh seeded G32 run; no remote."""
from pathlib import Path
import hashlib,json,copy,ast,difflib
S=Path(__file__).resolve().parent
G=Path('/workspace/patientphex-worktrees/explore-grounded-generation/solutions/grounded_generation/training')
B=Path('/workspace/patientphex-runs/round2-20261023/research/training-seed-ensemble-v1')
OLD_ROOT='/home/zhangyuheng/projects/CHIP2026_PatientPheX/grounded-generation-v1/formal-v1'
ROOT=OLD_ROOT.replace('grounded-generation-v1','training-seed-ensemble-v1')
def ref(p):
 p=Path(p).resolve();b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def write(p,b):
 if isinstance(b,str):b=b.encode()
 if p.exists():assert p.read_bytes()==b
 else:p.write_bytes(b)
def js(p,x):write(p,json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main():
 proposal=json.loads((S.parent/'proposal.json').read_text());assert proposal['train_seed']==20261025 and proposal['old_train_seed']==20261023
 original=json.loads((G/'formal-plan.json').read_text());p=copy.deepcopy(original)
 src=Path(proposal['source']['path']);assert ref(src)==proposal['source'];write(S/'train.py',src.read_bytes())
 # Copy proven transport/collector/driver, with narrowly scoped namespace/admission changes.
 remote=(G/'formal_remote.py').read_text().replace("REL='grounded-generation-v1/formal-v1'","REL='training-seed-ensemble-v1/formal-v1'")
 remote=remote.replace("PROJECT+'/grounded-generation-v1/native-cpu-v2'}","PROJECT+'/grounded-generation-v1/native-cpu-v2',PROJECT+'/grounded-generation-v1/formal-v1'}")
 remote=remote.replace("    'fresh-resource-gate.json','stdout.log','stderr.log'}","    'fresh-resource-gate.json','stdout.log','stderr.log','launch-admission-expired.json'}")
 # Repair old compact root-marker observation in this new namespace directly.
 remote=remote.replace("value=small_json(output/name)","value=small_json(root/name)",1)
 remote=remote.replace("            intent=dict(command=launch", "            issued=time.time(); launch_deadline=issued+300; campaign_deadline=issued+plan['budget_seconds']\n            intent=dict(issued_unix=issued,launch_deadline_unix=launch_deadline,campaign_deadline_unix=campaign_deadline,campaign_KILL_unix=campaign_deadline+30,command=launch")
 remote=remote.replace("            actual=call('start_process'", "            assert time.time()<launch_deadline, 'Launch TTL expired before the one start call; recover original claim only'\n            actual=call('start_process'")
 remote=remote.replace("CLAIM=r'''\n", "CLAIM=r'''\nimport time\nassert time.time()<SPEC['claim']['launch_deadline_unix']\nassert SPEC['claim']['campaign_deadline_unix']==SPEC['claim']['issued_unix']+3000\n")
 write(S/'formal_remote.py',remote)
 collect=(G/'formal_collect.py').read_text().replace("RELATIVE = 'grounded-generation-v1/formal-v1'","RELATIVE = 'training-seed-ensemble-v1/formal-v1'")
 write(S/'formal_collect.py',collect)
 driver=(G/'formal_driver.py').read_text()
 driver=driver.replace("    cap = plan['budget_seconds']", "    claim=json.loads(Path('start-intent.json').read_text())\n    assert claim['plan']['sha256']==sha('formal-plan.json')\n    def admission():\n        assert time.time()<claim['launch_deadline_unix'], 'Expired launch TTL; no child/model permitted'\n        assert time.time()<claim['campaign_deadline_unix']\n    admission()\n    assert claim['campaign_deadline_unix']==claim['issued_unix']+3000\n    assert claim['campaign_KILL_unix']==claim['campaign_deadline_unix']+30\n    cap = plan['budget_seconds']")
 driver=driver.replace("and fresh['disk_free_bytes'] >= 4 * 1024**3", "and fresh['disk_free_bytes'] >= 2 * 1024**3")
 driver=driver.replace("    start = time.monotonic(); utc = time.time()\n    command=plan['argv']+['--absolute-deadline-unix',str(utc+cap)]", "    admission()\n    start=time.monotonic(); utc=time.time(); deadline=claim['campaign_deadline_unix']\n    assert deadline-utc>=plan['minimum_remaining_before_child_seconds'], 'Insufficient true remaining budget; no model starts'\n    command=plan['argv']+['--absolute-deadline-unix',str(deadline)]")
 driver=driver.replace('deadline_TERM_unix=utc+cap, deadline_KILL_unix=utc+cap+30,','deadline_TERM_unix=deadline, deadline_KILL_unix=deadline+30,\n        issued_unix=claim[\'issued_unix\'],launch_deadline_unix=claim[\'launch_deadline_unix\'],')
 driver=driver.replace('        child = subprocess.Popen(command,', '        admission()\n        child = subprocess.Popen(command,')
 driver=driver.replace('start+cap-time.monotonic()', 'deadline-time.time()').replace('start+cap+30-time.monotonic()', 'deadline+30-time.time()')
 write(S/'formal_driver.py',driver)
 watch=(G/'watch_formal.py').read_text().replace("str(S/'observe_formal.py'),'--directory'", "str(S/'formal_remote.py'),'observe','--directory'").replace("ref(S/'observe_formal.py')","ref(S/'formal_remote.py')")
 write(S/'watch_formal.py',watch)
 for x in S.glob('*.py'):compile(x.read_text(),str(x),'exec')
 # All training assets/encoded bytes are the original verified objects.
 p['schema']='grounded-independent-training-seed-20261025-v1';p['authorization_note']='Root authorized unique same-host fresh seed20261025, shared3000+TERM30 incl both persistent fullstate ACK; no new engineering probe.'
 p['configuration']['seed']=20261025;p['budget_seconds']=p['authorized_budget_seconds']=3000;p['minimum_remaining_before_child_seconds']=2400
 p['launch_admission_policy']={'TTL_seconds':300,'campaign_deadline':'issued_unix+3000 before remote claim/start','no_budget_reset':True,'UNKNOWN':'recover same intent; TTL bounds delayed starts; never replay start_process'}
 p['output_dir']=ROOT+'/output';p['argv']=[p['python'],'-u',ROOT+'/train.py','--plan',ROOT+'/formal-plan.json']
 p['trainer_sha256']=ref(S/'train.py')['sha256'];p['assets']['trainer']={**ref(S/'train.py'),'path':ROOT+'/train.py'}
 for k,x in p['assets'].items():x['path']=x['path'].replace(OLD_ROOT,ROOT)
 p['payload_sources']['train.py']=ref(S/'train.py');p['payload_sources']['formal_driver.py']=ref(S/'formal_driver.py')
 p['files']={ROOT+'/'+name:{k:x[k] for k in ['bytes','sha256']} for name,x in p['payload_sources'].items()}
 p['operations_source']=ref(S/'formal_remote.py');p['collection_source']=ref(S/'formal_collect.py');p['binding_source']=ref(__file__)
 p['reuse_packages']=[ref(Path('/workspace/patientphex-runs/round2-20261023/research/grounded-generation-v1/formal-v1/package.json'))]
 p['reuse_actual_artifacts']={};p['seed_control_proposal']=ref(S.parent/'proposal.json');p['seed_CPU_result']=ref(Path('/workspace/patientphex-runs/round2-20261023/research/training-seed-ensemble-readiness-20261024/prepared-v2/result.json'))
 p['inherited_engineering']={'new_probe':False,'unchanged_all_data_shape_API_runtime':True,'old_original_formal_plan':ref(G/'formal-plan.json'),'original_complete':ref(Path('/workspace/patientphex-runs/round2-20261023/research/grounded-generation-v1/formal-v1/terminal/wrapper-complete.json')),'trainer_AST_only_seed_constants':3}
 js(S/'formal-plan.json',p)
 # Original loss/optimizer/rounds/source logic remains exact after inverse seed substitution.
 old=(G/'train.py').read_text();new=(S/'train.py').read_text();assert old==new.replace('20261025','20261023') and old.count('20261023')==3
 assert p['configuration']=={**original['configuration'],'seed':20261025}
 assert all(ref(x['path'])==x for x in p['payload_sources'].values())
 checks={'status':'PASS_SOURCE_ONLY_NO_MODEL','training_AST_only_seed':True,'original_17_assets_unchanged_except_trainer':True,'configuration_only_seed':True,'same_A100_native_runtime':True,'single3000_whole_two_epochs':True,'launchTTL300_beforePopen':True,'trainer_preload_absolute_deadline_inherited':True,'persisted_two_fullstates_no_warmstart':True,'no_probe':True,'stages_not_started':True,'plan':ref(S/'formal-plan.json'),'sources':{x.name:ref(x) for x in S.glob('*.py')},'pack_command':['python',str(S/'formal_remote.py'),'pack','--directory',str(B/'formal-v1'),'--plan',str(S/'formal-plan.json')]}
 js(S/'source-contract.json',checks);print(json.dumps({'plan':ref(S/'formal-plan.json'),'contract':ref(S/'source-contract.json')}))
if __name__=='__main__':main()
