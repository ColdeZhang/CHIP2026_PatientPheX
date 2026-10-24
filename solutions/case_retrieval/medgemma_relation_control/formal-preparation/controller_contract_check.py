"""Narrow synthetic control-flow checks only. No MCP, GPU, tokenizer or labels."""
from pathlib import Path
import importlib.util,json,tempfile,hashlib,time,ast,contextlib,io
T=Path(__file__).resolve().parent;spec=importlib.util.spec_from_file_location('med_formal',T/'run_formal.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
with tempfile.TemporaryDirectory() as tmp:
 root=Path(tmp);checks=[]
 class Dry(m.MedRun):
  def __init__(self,root,fail=False):
   self.root=root;root.mkdir();self.deadline=None;self.saved={};self.events=[];self.fail=fail
   self.plan={'controller_sha256':sha(T/'run_formal.py'),'prior_controller_source_sha256':sha(m.W/'run_paired_ack_paths.py'),'training_cap_seconds':1200,'inference_cap_seconds':600,'grace_seconds':30,'references':{},'remote_source_sha256':{},'training_output':'/dev/shm/mock/formal2ep','remote_inference_output':'mock/responses.jsonl','persistent_root':'mock-persist','persistent_mount_id':'mock100','mount_id':'mock800','persistent_tools':'mock-tools','fetcher_sha256':'a'*64}
   (root/'run-plan.json').write_text(json.dumps(self.plan))
  def gpu_empty(self,label):self.events.append(('gpu',label));return {}
  def command(self,*args,**kwargs):self.events.append(('read-only-command',));return {}
  def save(self,path,obj):self.saved[str(path.relative_to(self.root))]=obj
  def train_epoch(self,epoch):
   self.events.append(('train-export-ACK',epoch,self.deadline))
   if self.fail:raise RuntimeError('synthetic first ACK failure')
   return {'epoch':epoch}
  def infer(self,ack):self.events.append(('infer-after-two-ACK',ack))
 d=Dry(root/'normal');before=time.time()
 with contextlib.redirect_stdout(io.StringIO()):d.execute()
 after=time.time()
 events=[e for e in d.events if e[0]=='train-export-ACK'];assert [e[1] for e in events]==[1,2] and events[0][2]==events[1][2] and before+1200<=events[0][2]<=after+1200
 assert d.events[-1]==('infer-after-two-ACK',{'epoch':2}) and d.saved['training-campaign-complete.json']['optimizer_calls']==32
 checks.append('two epochs share one exact1200 deadline; inference only after both durable-ACK stage returns')
 d=Dry(root/'failed',True)
 try:d.execute();raise AssertionError('failure did not propagate')
 except RuntimeError as e:assert str(e)=='synthetic first ACK failure'
 assert len([e for e in d.events if e[0]=='train-export-ACK'])==1 and not any(e[0]=='infer-after-two-ACK' for e in d.events)
 checks.append('epoch1/ACK failure propagates and prevents epoch2/inference')
 d=Dry(root/'duplicate');(d.root/'execution-state.json').write_text('{}')
 try:d.execute();raise AssertionError('duplicate not refused')
 except AssertionError as e:assert 'Recover existing' in str(e)
 assert not d.events;checks.append('existing execution-state refuses before any source/GPU/start action')
 # Transport loops stay byte-for-byte AST equal; only root/source inventory and
 # durability directories changed. This is not a fresh HTTP/GPU probe.
 old=ast.parse((m.W/'export_checkpoint_server.py').read_text());new=ast.parse((T/'export_checkpoint_server.py').read_text())
 assert ast.dump(next(n for n in ast.walk(old) if isinstance(n,ast.ClassDef)))==ast.dump(next(n for n in ast.walk(new) if isinstance(n,ast.ClassDef)))
 old=ast.parse((m.W/'fetch_checkpoint.py').read_text());new=ast.parse((T/'fetch_checkpoint.py').read_text())
 select=lambda tree:next(n for n in ast.walk(tree) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='attempt')
 assert ast.dump(select(old))==ast.dump(select(new));checks.append('HTTP Handler and Range/fullSHA attempt loop AST unchanged')
result={'status':'PASS_SYNTHETIC_CONTROL_NO_GPU_NO_MCP_NO_GOLD','checks':checks,'source_sha256':sha(T/'run_formal.py'),'exporter_sha256':sha(T/'export_checkpoint_server.py'),'fetcher_sha256':sha(T/'fetch_checkpoint.py'),'fixture_source_sha256':sha(__file__)}
(T/'controller-contract-check.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
