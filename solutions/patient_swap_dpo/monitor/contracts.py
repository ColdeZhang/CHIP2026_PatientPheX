"""Only two operational fault contracts: unknown start and corrupted ZIP."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import time
import zipfile


def main(output):
    tick=time.monotonic();sp=importlib.util.spec_from_file_location('stages',Path(__file__).with_name('stages.py'));m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
    with tempfile.TemporaryDirectory(prefix='dpo-monitor-contract-') as td:
        D=Path(td);S=m.state(D);(S/'deployment.json').write_text('[]\n');(S/'inference-plan.json').write_text('{}\n')
        m.check_receipt=lambda *a:{}
        m.deployment_items=lambda *a:[]
        m.core_plan=lambda *a:dict(python='/python-not-executed',application=m.control.PREFIX+'synthetic-not-deployed',environment={})
        class Unknown:
            def __init__(self):self.calls=0
            def call(self,*a,**k):self.calls+=1;raise RuntimeError('Synthetic lost start reply')
        remote=Unknown()
        try:m.start_infer(D,remote);raise AssertionError('Expected synthetic RPC failure')
        except RuntimeError:pass
        assert (S/'infer-start-intent.json').exists() and remote.calls==1
        try:m.start_infer(D,remote);raise RuntimeError('Second start should fail before RPC')
        except AssertionError:pass
        assert remote.calls==1
        good=b'{"synthetic":true}\n';inventory={'native/example.json':dict(bytes=len(good),sha256=hashlib.sha256(good).hexdigest(),path='/synthetic')}
        for label,body in [('valid',good),('corrupt',b'{"synthetic":false}\n')]:
            z=D/(label+'.zip')
            with zipfile.ZipFile(z,'w') as f:f.writestr('native/example.json',body);f.writestr('inventory.json',json.dumps(inventory))
            dest=D/label
            if label=='valid':m.unpack_verified(z,dest,2);assert (dest/'native/example.json').read_bytes()==good
            else:
                try:m.unpack_verified(z,dest,2);raise RuntimeError('Corruption was not rejected')
                except AssertionError:pass
                assert not dest.exists()
    result=dict(status='PASS_UNKNOWN_START_NO_SECOND_RPC_AND_CORRUPT_ZIP_REJECT_BEFORE_EXTRACT',seconds=time.monotonic()-tick,
                synthetic_start_calls=1,actual_remote_calls=0,actual_GPU_calls=0,actual_quality_calls=0)
    Path(output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);main(p.parse_args().output)
