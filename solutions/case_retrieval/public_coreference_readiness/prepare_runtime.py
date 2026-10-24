"""Owned spaCy overlay only, no shared environment writes, model loading or GPU."""
import json,os,pathlib,subprocess,sys,time
P=pathlib.Path.cwd();assert os.environ.get('CUDA_VISIBLE_DEVICES')=='';start=time.monotonic();target=P/'overlay';report=P/'runtime-pip-report.json'
cmd=[sys.executable,'-m','pip','install','--target',str(target),'--report',str(report),'--disable-pip-version-check','--no-compile','--no-cache-dir','spacy==3.8.7','numpy==2.2.6']
r=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=170)
(P/'runtime-install.stdout.log').write_text(r.stdout);(P/'runtime-install.stderr.log').write_text(r.stderr)
(P/'runtime-state.json').write_text(json.dumps({'status':'complete' if r.returncode==0 else 'failed','argv':cmd,'returncode':r.returncode,'seconds':time.monotonic()-start,'shared_environment_modified':False,'neural_model_downloaded':False},indent=2)+'\n');raise SystemExit(r.returncode)
