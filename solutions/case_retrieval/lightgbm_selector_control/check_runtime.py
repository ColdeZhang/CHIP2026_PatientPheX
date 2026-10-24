"""One synthetic native/probability/serialization contract; no task data."""
from pathlib import Path
import hashlib
import json
import os
import pickle
import signal
import subprocess
import sys
import time
import zipfile

import configuration


def ref(path):
    p=Path(path)
    return dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)


def main():
    start=time.monotonic();signal.alarm(30)
    root=Path(sys.argv[1]);dep=json.loads((root/'dependency-proposal.json').read_text());installation=json.loads((root/'deps/installation.json').read_text())
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    import numpy as np
    import scipy
    import sklearn
    import lightgbm
    from lightgbm.basic import _LIB
    assert lightgbm.__version__=='4.6.0'
    assert Path(lightgbm.__file__).resolve().is_relative_to(Path(installation['target']).resolve())
    wheel=Path(installation['wheel']);assert ref(wheel)['sha256']==dep['wheel']['digests']['sha256']
    verified=[]
    with zipfile.ZipFile(wheel) as z:
        for info in z.infolist():
            if info.filename.startswith('lightgbm/') and not info.is_dir():
                actual=Path(installation['target'])/info.filename
                assert hashlib.sha256(z.read(info)).digest()==hashlib.sha256(actual.read_bytes()).digest()
                verified.append(ref(actual))
    native=Path(_LIB._name);ldd=subprocess.run(['ldd',str(native)],capture_output=True,text=True,timeout=5)
    assert ldd.returncode==0 and 'not found' not in ldd.stdout
    target=root/'runtime-contract';target.mkdir(exist_ok=False)
    # Deterministic artificial binary fixture; never a task training row.
    values=np.arange(128,dtype=np.float64)
    X=np.column_stack((values,values%7,np.sin(values),np.ones_like(values)))
    y=(values>=64).astype(np.int64)
    model=configuration.estimator();assert model.get_params()['objective']=='binary'
    model.fit(X,y)
    assert np.array_equal(model.classes_,[0,1])
    probabilities=model.predict_proba(X)
    assert probabilities.shape==(128,2) and np.isfinite(probabilities).all()
    assert np.array_equal(probabilities[:,1],model.booster_.predict(X))
    assert np.allclose(probabilities.sum(axis=1),1.,rtol=0,atol=1e-15)
    assert probabilities[y==1,1].mean()>probabilities[y==0,1].mean()
    path=target/'synthetic.pkl';path.write_bytes(pickle.dumps(model,protocol=5))
    reloaded=pickle.loads(path.read_bytes());assert np.array_equal(reloaded.predict_proba(X),probabilities)
    result=dict(status='PASS_synthetic_runtime_only',seconds=time.monotonic()-start,
                versions={'python':sys.version.split()[0],'lightgbm':lightgbm.__version__,'sklearn':sklearn.__version__,'numpy':np.__version__,'scipy':scipy.__version__},
                source=ref(__file__),configuration=ref(configuration.__file__),parameters=configuration.PARAMETERS,
                private_package_files_verified_against_wheel=verified,native_library=ref(native),ldd=ldd.stdout,
                fixture=dict(rows=128,columns=4,classes=model.classes_.tolist(),actual_iterations=model.n_iter_,
                             positive_column=1,positive_column_exact_native_predict=True,pickle_roundtrip_exact=True),
                artificial_data_only=True,task_gold_read=False,task_fit=False,gpu_used=False,global_install=False)
    (target/'complete.json').write_text(json.dumps(result,indent=2)+'\n');signal.alarm(0)
    print(json.dumps({k:result[k] for k in ('status','seconds','versions','fixture','native_library')}))


if __name__=='__main__':main()
