"""Expired admission must reject before resource access or child creation; CPU only."""
import ast
import os
from pathlib import Path
import tempfile
import time
from common import read,ref,save
import driver

S=Path(__file__).resolve().parent


def main():
    before=Path.cwd();temp=Path(tempfile.mkdtemp(prefix='adapted-admission-',dir='/workspace/patientphex-runs'))
    def forbidden(*args,**kwargs):raise RuntimeError('Unexpected resource/child access')
    original=driver.resources;original_popen=driver.subprocess.Popen
    driver.resources=forbidden;driver.subprocess.Popen=forbidden
    p=dict(execution_authorized=True,launch_valid_until_unix=time.time()-1,budget_seconds=180,grace_seconds=30)
    save(temp/'held-query-plan.json',p)
    try:
        os.chdir(temp)
        try:driver.main()
        except AssertionError:pass
        else:raise AssertionError('Expired admission accepted')
        assert read('admission-rejected.json')['no_new_child_or_model_loaded']
        assert not any((temp/n).exists() for n in ['fresh-resource-gate.json','child-started.json','stdout.log'])
    finally:
        os.chdir(before);driver.resources=original;driver.subprocess.Popen=original_popen
    for path in S.glob('*.py'):ast.parse(path.read_text())
    save(S/'evidence/controller-contract.json',dict(status='PASS_EXPIRED_BEFORE_GPU_RESOURCE_OR_POPEN',
        driver=ref(S/'driver.py'),source=ref(__file__),evidence=ref(temp/'admission-rejected.json'),all_python_AST_pass=True,no_model_or_GPU_or_quality=True))
    print('PASS expired admission before resource/Popen')


if __name__=='__main__':main()
