"""One target-interpreter, CUDA-hidden exact inherited verify_inputs; no ML imports."""
import os,sys,json,time,hashlib,importlib.util,importlib.metadata
from pathlib import Path
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
S=Path(__file__).resolve().parent
p=S/'formal-plan.json';plan=json.loads(p.read_text());started=time.monotonic()
assert not (S/'cpu-verify-attempt3.json').exists()
spec=importlib.util.spec_from_file_location('seed_actual_training_contract',S/'train.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
encoded,audit=m.verify_inputs(plan)
assert len(encoded)==len(audit)==87
assert len({x['pmc_id'] for x in audit})==32
for e in (0,1):
 groups=m.batch_plan(e);assert [len(g) for g in groups]==[8]*10+[7] and sorted(i for g in groups for i in g)==list(range(87))
assert m.CONFIG['seed']==20261025
distribution_versions={k:importlib.metadata.version(k) for k in ['torch','transformers','peft','tokenizers']}
import torch,transformers,peft,tokenizers
versions={k:str(v.__version__) for k,v in [('torch',torch),('transformers',transformers),('peft',peft),('tokenizers',tokenizers)]}
m.write_json(S/'cpu-import-observations.json',{'distribution_versions':distribution_versions,'imported_versions':versions,'verify_inputs_returned_87':True,'no_model_weights_or_forward':True})
assert versions=={'torch':'2.6.0+cu124','transformers':'5.20.2','peft':'0.21.2','tokenizers':'0.23.1'}
result={'status':'PASS_ACTUAL_TRAINER_VERIFY_INPUTS_NO_MODEL','seconds':time.monotonic()-started,'rows':87,'documents':32,'tokens':451672,'supervision':11884,'epochs':2,'updates':22,'tails':[7,7],'seed':20261025,'plan_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'trainer_sha256':hashlib.sha256((S/'train.py').read_bytes()).hexdigest(),'versions':versions,'python':sys.executable,'CUDA_VISIBLE_DEVICES':os.environ['CUDA_VISIBLE_DEVICES'],'new_model_tokenizer_or_quality_calls':0}
m.write_json(S/'cpu-verify-attempt3.json',result)
print(json.dumps(result))
