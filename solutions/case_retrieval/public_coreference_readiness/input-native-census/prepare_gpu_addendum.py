"""Freeze device-only full48 adaptation after first CPU probe, before any full48 run."""
import pathlib,json,hashlib,datetime,difflib
P=pathlib.Path(__file__).parent;cpu=(P/'native_census.py').read_text();uuid='GPU-fc5d0525-3ec1-074a-9fa3-35d92a9da539'
gpu=cpu.replace('One frozen CPU input-only census.','One frozen A30 input-only census.').replace("assert os.environ.get('CUDA_VISIBLE_DEVICES')==''","assert args.mode=='full48' and os.environ.get('CUDA_VISIBLE_DEVICES')=="+repr(uuid)).replace("(P/'plan.json')","(P/'gpu-plan.json')").replace("device='cpu'","device='cuda:0'").replace(".device.type=='cpu'",".device.type=='cuda'")
gpu=gpu.replace("clinical_attribution_evaluated=False);save()","clinical_attribution_evaluated=False,cuda_peak_memory_allocated_bytes=torch.cuda.max_memory_allocated(),cuda_peak_memory_reserved_bytes=torch.cuda.max_memory_reserved(),cuda_device_uuid=os.environ['CUDA_VISIBLE_DEVICES'],cuda_device_name=torch.cuda.get_device_name());save()")
(P/'native_census_gpu.py').write_text(gpu);(P/'native-gpu-device-only.diff').write_text(''.join(difflib.unified_diff(cpu.splitlines(True),gpu.splitlines(True),fromfile='native_census.py',tofile='native_census_gpu.py')))
control=(P/'remote_control.py').read_text().replace("(P/'plan.json')","(P/'gpu-plan.json')").replace("+['plan.json']","+['gpu-plan.json']").replace(' -u native_census.py --mode ',' -u native_census_gpu.py --mode ').replace("(P/'native_census.py')","(P/'native_census_gpu.py')")
control=control.replace("assert args.mode and args.owner_commit and len(args.owner_commit)==40","assert args.mode=='full48' and args.owner_commit and len(args.owner_commit)==40")
needle="env=json.loads((R/'cpu-attempt2-job.json').read_text())['env'];cap=60 if args.mode=='probe' else 600"
replacement="""# Fresh resource check immediately before starting only the allocated UUID.
 selected_uuid=SELECTED
 code="import subprocess,json,datetime;selected="+repr(selected_uuid)+";lines=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines();rows=[x.split(', ') for x in lines];gpu=next(x for x in rows if x[0]==selected);apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True).splitlines();owned=[x for x in apps if x.startswith(selected+',')];assert gpu[1]=='NVIDIA A30' and int(gpu[2])==0 and int(gpu[3])==0 and not owned,(gpu,owned);print(json.dumps({'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'gpu':gpu,'compute_apps_for_uuid':owned}))"
 fresh=command(code);(P/'gpu-fresh-before-full48.json').write_text(json.dumps(fresh,indent=2)+'\\n')
 env=json.loads((R/'cpu-attempt2-job.json').read_text())['env'];env['CUDA_VISIBLE_DEVICES']=selected_uuid;cap=600""".replace('SELECTED',repr(uuid))
assert needle in control;control=control.replace(needle,replacement)
# Check the fresh GPU before creating the output directory so a busy device leaves no false attempt.
mk="command(\"import pathlib,json;P=pathlib.Path('.');o=P/\"+repr(args.mode)+\";assert not o.exists(),'refuse overwrite';o.mkdir();print(json.dumps({'ready':True}))\")"
control=control.replace(' '+mk+'\n','');control=control.replace("cmd='timeout --signal=TERM",' '+mk+"\n cmd='timeout --signal=TERM")
# Above replacement adds one extra space to the already indented line; normalize exactly.
control=control.replace('  '+mk,' '+mk)
control=control.replace("(P/'deployment.json')","(P/'gpu-deployment.json')")
(P/'remote_control_gpu.py').write_text(control)
for name in ['native_census_gpu.py','remote_control_gpu.py']:compile((P/name).read_text(),str(P/name),'exec')
d=json.loads((P/'plan.json').read_text());d.update(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='frozen_device_only_addendum_before_full48',CPU_parent_plan_sha256=hashlib.sha256((P/'plan.json').read_bytes()).hexdigest(),CPU_probe_state_sha256=hashlib.sha256((P/'probe/state.json').read_bytes()).hexdigest(),full48_device_reason='CPU probe total13.013s, doc7.844s/11197chars; corpus842842chars gives near595s linear estimate before length-squared effects; use authorized fresh-empty A30 within same600s cap',GPU=True,selected_gpu_uuid=uuid)
d['runtime']['device']='cuda:0 (single visible A30 UUID '+uuid+')';d['runtime']['CPU_probe_same_native_dtype']='BF16 but CPU and CUDA kernels may differ numerically; no equality/quality inference';d['source_sha256'].update({n:hashlib.sha256((P/n).read_bytes()).hexdigest() for n in ['native_census_gpu.py','remote_control_gpu.py','native-gpu-device-only.diff','prepare_gpu_addendum.py','gpu-readonly-preflight.json','plan.json','probe/state.json']});(P/'gpu-plan.json').write_text(json.dumps(d,indent=2)+'\n')
names=['native_census_gpu.py','remote_control_gpu.py','native-gpu-device-only.diff','prepare_gpu_addendum.py','gpu-readonly-preflight.json','gpu-plan.json','probe/state.json','probe/stderr.log','probe-job.json','probe/collection.json','deployment.json']
(P/'gpu-owner-backup-filelist.json').write_text(json.dumps({'files':[{'path':'input-native-census/'+n,'bytes':(P/n).stat().st_size,'sha256':hashlib.sha256((P/n).read_bytes()).hexdigest()} for n in names]},indent=2)+'\n');print(json.dumps({'gpu_plan_sha256':hashlib.sha256((P/'gpu-plan.json').read_bytes()).hexdigest(),'files':len(names)}))
