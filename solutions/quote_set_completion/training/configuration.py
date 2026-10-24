"""One fixed supervised-denoising pass; original G32 adapter, fresh optimizer."""
from pathlib import Path
import random
EXPECTED=dict(epochs=1,batch_size=1,gradient_accumulation=8,seed=20261023,
 learning_rate=1e-5,weight_decay=.01,max_grad_norm=1.,rank=16,lora_alpha=32,dropout=.05,
 dtype='bfloat16',attention='sdpa',gradient_checkpointing={'use_reentrant':False},packing=False,quantization=False,
 loss='standard per-row mean causal CE; divide each row by actual group size',rows=261,documents=32,held_rows=0,
 steps_per_epoch=33,optimizer_calls=33,warmup_steps=2,tail_group_size=5,
 full_training_tokens=1380333,supervised_tokens=12406,maximum_full_tokens=9307)
def validate(c,route):
 if route!='quote-set-completion-v2' or c!=EXPECTED:raise ValueError('Fixed warm-d52 one-epoch recipe differs')
 return c
def batch_plan(c,epoch):
 if epoch!=0:raise ValueError('Exactly one complete epoch')
 indices=list(range(c['rows']));random.Random(c['seed']).shuffle(indices)
 groups=[indices[i:i+8] for i in range(0,len(indices),8)]
 assert len(groups)==33 and [len(g) for g in groups]==[8]*32+[5]
 assert sorted(i for g in groups for i in g)==list(range(261))
 return groups
def validate_storage(plan):
 output=Path(plan['output_dir']);parent=Path(plan['storage_parent']);persistent=Path(plan['persistent_checkpoint_root'])
 assert output.parent==parent and output.name=='output' and parent.parent==Path('/dev/shm')
 assert parent.name=='patientphex-quote-set-completion-v2-formal'
 assert persistent==Path('/home/zhangyuheng/projects/CHIP2026_PatientPheX/research/quote-set-completion-v2/checkpoints')
 assert plan['persistent_mount_id']=='5a9955da-5240-41cc-9e52-05d3d8da8a3c'
 assert plan['minimum_SHM_free_bytes']>=4*1024**3 and plan['minimum_persistent_free_bytes']>=2*1024**3
 return True
