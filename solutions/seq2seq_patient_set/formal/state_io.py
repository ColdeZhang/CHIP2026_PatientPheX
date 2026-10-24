"""BioClinical full-state fsync/reopen/restore contract, parameterized for LongT5."""
import os
import random
import time
from pathlib import Path
from common import ref, save, tensor_compare, fsync_dir


def rng_state():
    import numpy as np
    import torch
    nr=np.random.get_state()
    return {'python':random.getstate(),'numpy':(nr[0],torch.from_numpy(nr[1].astype('int64')),nr[2],nr[3],nr[4]),
            'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []}


def restore_rng(value):
    import numpy as np
    import torch
    random.setstate(value['python']);ns=value['numpy']
    np.random.set_state((ns[0],ns[1].numpy().astype('uint32'),ns[2],ns[3],ns[4]))
    torch.set_rng_state(value['torch'])
    if value['cuda']:torch.cuda.set_rng_state_all(value['cuda'])


def save_reopen_restore(model, optimizer, scheduler, output, metadata):
    import torch
    out=Path(output);tick=time.monotonic();optimizer.zero_grad(set_to_none=True)
    state={'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},
           'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'rng':rng_state(),
           **metadata,'gradients_cleared':True}
    cp=out/'final-full-state.pt'
    with cp.open('xb') as f:torch.save(state,f);f.flush();os.fsync(f.fileno())
    fsync_dir(out);saved=torch.load(cp,map_location='cpu',weights_only=False);matched=tensor_compare(state,saved)
    model.load_state_dict(saved['model'],strict=True);optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
    restore_rng(saved['rng'])
    tensor_compare(model.state_dict(),saved['model']);tensor_compare(optimizer.state_dict(),saved['optimizer'])
    tensor_compare(scheduler.state_dict(),saved['scheduler']);tensor_compare(rng_state(),saved['rng'])
    assert model.shared.weight is model.encoder.embed_tokens.weight is model.decoder.embed_tokens.weight
    assert model.lm_head.weight is not model.shared.weight
    result={'status':'FULL_STATE_FSYNC_REOPEN_RESTORE_PASS','checkpoint':ref(cp),'matched_tensors':matched,
            'actual_adam_updates':metadata['actual_adam_updates'],
            'every_optimizer_step':sorted({int(s['step'].item()) for s in optimizer.state.values()}),
            'shared_input_identity_preserved':True,'independent_lm_head':True,
            'seconds':time.monotonic()-tick,'resume_equivalence_claimed':False}
    save(out/'final-state-ACK.json',result)
    return result
