"""Actual no-child checks for expired, wrong-bound, and already-fenced launches."""
import argparse,hashlib,json,subprocess,sys,time
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False);driver=Path(__file__).with_name('pilot_driver.py').resolve();reports=[]
 for mode in ('expired','wrong_binding','fenced'):
  d=out/mode;d.mkdir();e={'sources':{'pilot_driver.py':hashlib.sha256(driver.read_bytes()).hexdigest()},'python':sys.executable};ep=d/'execution.json';ep.write_text(json.dumps(e));h=hashlib.sha256(ep.read_bytes()).hexdigest();now=time.time();adm={'execution_sha256':h if mode!='wrong_binding' else '0'*64,'valid_from_unix':now-10,'valid_until_unix':now-1 if mode=='expired' else now+60};ap=d/'admission.json';ap.write_text(json.dumps(adm))
  if mode=='fenced':(d/'launch.accepted.json').write_text('{"FENCED_UNKNOWN_LAUNCH":true,"resource_PASS":false}\n')
  run=subprocess.run([sys.executable,str(driver),'--execution',str(ep),'--admission',str(ap)],capture_output=True,text=True,timeout=5)
  (d/'stderr').write_text(run.stderr);assert run.returncode!=0 and not (d/'child.json').exists() and not (d/'driver-start.json').exists()
  if mode!='fenced':assert not (d/'launch.accepted.json').exists()
  else:assert json.loads((d/'launch.accepted.json').read_text())=={'FENCED_UNKNOWN_LAUNCH':True,'resource_PASS':False}
  reports.append(dict(case=mode,returncode=run.returncode,no_child=True,no_resource_or_model_launch=True))
 result=dict(status='PASS3_ACTUAL_ADMISSION_REJECTIONS_NO_CHILD',source_sha256=hashlib.sha256(driver.read_bytes()).hexdigest(),cases=reports,model_calls=0,GPU=False);(out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
