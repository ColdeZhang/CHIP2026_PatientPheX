"""Fix actual spaCy CLI import failure in a separate tiny overlay; preserve original environment."""
import pathlib,sys,subprocess,json,time,os
P=pathlib.Path.cwd();assert os.environ.get('CUDA_VISIBLE_DEVICES')=='';t=time.monotonic()
cmd=[sys.executable,'-m','pip','install','--target',str(P/'overlay-repair'),'--no-deps','--no-compile','--no-cache-dir','--disable-pip-version-check','--report',str(P/'runtime-repair-pip-report.json'),'click==8.1.8','typer==0.16.1']
try:
 r=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=55);result={'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
except subprocess.TimeoutExpired as e:result={'returncode':124,'stdout':(e.stdout or b'').decode() if isinstance(e.stdout,bytes) else e.stdout,'stderr':(e.stderr or b'').decode() if isinstance(e.stderr,bytes) else e.stderr}
(P/'runtime-repair.stdout.log').write_text(result['stdout'] or '');(P/'runtime-repair.stderr.log').write_text(result['stderr'] or '')
(P/'runtime-repair-state.json').write_text(json.dumps({'status':'complete' if result['returncode']==0 else 'failed','seconds':time.monotonic()-t,'argv':cmd,'returncode':result['returncode'],'original_overlay_unchanged':True},indent=2)+'\n');raise SystemExit(result['returncode'])
