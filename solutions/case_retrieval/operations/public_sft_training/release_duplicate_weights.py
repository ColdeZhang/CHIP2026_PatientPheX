#!/usr/bin/env python3
"""Release only two explicitly authorized A800 duplicate LoRA weight files.
Requires freshly rehashed readable A30 copies, SHA-identical source, no path user,
and an empty allocated GPU. Retains configs, optimizer states, and all other files.
"""
import argparse,datetime,hashlib,json,os,subprocess,time,shutil
from pathlib import Path
WEIGHTS={
 'case-retrieval/qwen35-preparation/relation-training/formal2ep/epoch-002/adapter/adapter_model.safetensors':('case-retrieval/qwen35-preparation/dev-relation-transfer/adapter-epoch002/adapter_model.safetensors','8e32daeea3c078af25a7d01e9d9cebf1fe8d99bbfaf2ca5e2c172e643d346513'),
 'case-retrieval/qwen35-preparation/full64-relation-cpu-preparation/formal2ep/epoch-002/adapter/adapter_model.safetensors':('case-retrieval/qwen35-preparation/full64-dev-transfer/adapter-epoch002/adapter_model.safetensors','1a703b244b17c1378b383e4d6feefe2486b1a888e78cbd6f9f25c3fa6e46f45b')}
UUID='GPU-f516a193-de9c-8552-7ac5-4725f24cc0fc'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--backup-check',required=True);p.add_argument('--backup-sha',required=True);p.add_argument('--output',required=True);p.add_argument('--delete-authorized-two-weights',action='store_true');a=p.parse_args()
 assert not Path(a.output).exists() and sha(a.backup_check)==a.backup_sha
 backup=json.loads(Path(a.backup_check).read_text());assert backup['mount_id']=='5a9955da-5240-41cc-9e52-05d3d8da8a3c' and 0<=time.time()-backup['server_epoch']<1800
 expected={v[0]:v[1] for v in WEIGHTS.values()};assert set(backup['files'])==set(expected)
 for n,h in expected.items():assert backup['files'][n]==dict(sha256=h,bytes=173181072,readable=True)
 actual={};paths={str(Path(p).resolve()) for p in WEIGHTS};dirs={str(Path(p).resolve().parents[1]) for p in WEIGHTS}
 for name,(_,h) in WEIGHTS.items():
  p=Path(name);assert p.is_file() and p.stat().st_size==173181072 and sha(p)==h;actual[name]=dict(sha256=h,bytes=p.stat().st_size)
 matches=[];restricted_fd_maps=[]
 for pd in Path('/proc').iterdir():
  if not pd.name.isdigit() or int(pd.name)==os.getpid():continue
  try:
   cmd=(pd/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
   if any(d in cmd or (d[len(str(Path.cwd()))+1:] if d.startswith(str(Path.cwd())+'/') else d) in cmd for d in dirs):matches.append(dict(pid=int(pd.name),kind='command_references_old_checkpoint'))
   for fd in (pd/'fd').iterdir():
    try:
     if os.readlink(fd) in paths:matches.append(dict(pid=int(pd.name),kind='open_weight_descriptor'))
    except OSError:pass
   maps=(pd/'maps').read_text()
   if any(p in maps for p in paths):matches.append(dict(pid=int(pd.name),kind='mapped_weight'))
  except PermissionError:
   # proc cmdline is checked first; some same-UID services deny fd/maps via
   # dumpability. Record the visibility limit; GPU2 must independently be empty.
   restricted_fd_maps.append(int(pd.name))
  except (FileNotFoundError,ProcessLookupError):continue
 assert not matches,matches
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader'],text=True)
 assert not any(line.split(',')[0].strip()==UUID for line in apps.splitlines())
 before=shutil.disk_usage('.').free;will_release=sum(v['bytes'] for v in actual.values())
 assert before+will_release>=680000000 and before+will_release-520000000>=150000000
 if a.delete_authorized_two_weights:
  for name in WEIGHTS:Path(name).unlink()
 after=shutil.disk_usage('.').free
 report=dict(at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_sha256=sha(__file__),backup_check_sha256=sha(a.backup_check),backup_mount=backup['mount_id'],old_files=actual,backup_paths={p:v[0] for p,v in WEIGHTS.items()},active_command_or_visible_fd_matches=matches,restricted_fd_maps_pids=restricted_fd_maps,process_visibility="All accessible cmdlines checked; restricted fd/maps explicitly listed, no claim of global descriptor visibility",gpu_uuid=UUID,gpu_empty=True,deleted_only_exact_two_weights=a.delete_authorized_two_weights,deleted_bytes=will_release if a.delete_authorized_two_weights else 0,free_before=before,free_after=after,min_free_guard=680000000,final_checkpoint_estimate=520000000,final_residual_estimate=after-520000000,configs_optimizer_other_checkpoints_untouched=True,backup_restore_tested=False)
 with Path(a.output).open('x') as f:f.write(json.dumps(report,indent=2)+'\n')
 print(json.dumps(report,indent=2))
if __name__=='__main__':main()
