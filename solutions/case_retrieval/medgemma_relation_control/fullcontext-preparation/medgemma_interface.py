"""Fixed native MedGemma1.5 relation interface; text prompts, full conservative bundle."""
from pathlib import Path
from collections import Counter
import hashlib,json,re
MODEL='google/medgemma-1.5-4b-it'
MS_REVISION='9bca96ca3e3049f6a084d8c876078cedb3069f18'
HF_REVISION='91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b'
BASE_PARAMETERS=4300079472
TEXT_PARAMETERS=3880263168
ADAPTER_PARAMETERS=29802496
TARGETS=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']
TARGET_PATTERN=r'model\.language_model\.layers\.\d+\.(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|mlp\.(?:gate_proj|up_proj|down_proj))'
CPU_FILES=['config.json','generation_config.json','chat_template.jinja','tokenizer.json','tokenizer.model','tokenizer_config.json','special_tokens_map.json','added_tokens.json']
EOS_IDS=[1,106]
CHAT_EOS='<end_of_turn>'
CHAT_EOS_ID=106

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()

def model_files(model_dir,source_manifest,require_weights):
 source=json.loads(Path(source_manifest).read_text())
 if source.get('model_id')!=MODEL or source.get('MS_revision')!=MS_REVISION or source.get('HF_revision')!=HF_REVISION:
  raise ValueError('Require fixed reviewed MS/HF MedGemma model revisions')
 expected={x['name']:x for x in source['files']}
 if len(expected)!=19 or any(Path(n).name!=n for n in expected):raise ValueError('Unexpected or unsafe Med asset list')
 actual={}
 for name in sorted(expected if require_weights else CPU_FILES):
  p=Path(model_dir)/name;record={'bytes':p.stat().st_size,'sha256':sha(p)}
  if record!={k:expected[name][k] for k in ['bytes','sha256']}:raise ValueError('Asset SHA/size mismatch: '+name)
  actual[name]=record
 return {'model':MODEL,'MS_revision':MS_REVISION,'HF_revision':HF_REVISION,'source_manifest_sha256':sha(source_manifest),'actual_files':actual,'all19_files_verified':require_weights}

def render_prefix(row,tokenizer):
 # Pass original message objects unmodified. Native template combines the system
 # string with the first user using two newlines and calls assistant role model.
 text=tokenizer.apply_chat_template(row['messages'],tokenize=False,add_generation_prompt=True)
 ids=tokenizer.apply_chat_template(row['messages'],tokenize=True,return_dict=False,add_generation_prompt=True)
 if not text.endswith('<start_of_turn>model\n') or not isinstance(ids,list) or not all(isinstance(i,int) for i in ids):
  raise ValueError('Unexpected native assistant prefix')
 if tokenizer.convert_tokens_to_ids(CHAT_EOS)!=CHAT_EOS_ID or tokenizer.bos_token_id!=2 or tokenizer.pad_token_id!=0:
  raise ValueError('Fixed Gemma special token contract differs')
 # Original data must never be interpreted as image input.
 if any(i in ids for i in [255999,256000,262144]):raise ValueError('Image token in text-only relation prompt')
 if ids!=tokenizer(text,add_special_tokens=False)['input_ids']:
  raise ValueError('Chat tokenizer injected duplicate BOS or altered native text')
 return text,ids

def encode(row,tokenizer):
 prefix_text,prefix=render_prefix(row,tokenizer)
 full_messages=row['messages']+[{'role':'assistant','content':row['answer']}]
 full_text=tokenizer.apply_chat_template(full_messages,tokenize=False,add_generation_prompt=False)
 if not full_text.startswith(prefix_text) or full_text[len(prefix_text):]!=row['answer'].strip()+CHAT_EOS+'\n':
  raise ValueError('Native assistant completion changed target JSON or end-of-turn: '+row['row_id'])
 full=tokenizer.apply_chat_template(full_messages,tokenize=True,return_dict=False,add_generation_prompt=False)
 if full[:len(prefix)]!=prefix or len(full)<=len(prefix):raise ValueError('Native token prefix mismatch')
 suffix=full[len(prefix):]
 if suffix.count(CHAT_EOS_ID)!=1 or 1 in suffix:raise ValueError('Train exactly one native chat end-of-turn, no artificial EOS')
 labels=[-100]*len(prefix)+suffix
 if not any(i!=-100 for i in labels[1:]):raise ValueError('No causal labels')
 return {'row_id':row['row_id'],'input_ids':full,'labels':labels,'prefix_tokens':len(prefix),'supervised_tokens':len(suffix),
         'eos_token_id':CHAT_EOS_ID,'eos_supervised_position':len(prefix)+suffix.index(CHAT_EOS_ID),
         'rendered_prefix_sha256':hashlib.sha256(prefix_text.encode()).hexdigest(),
         'rendered_full_sha256':hashlib.sha256(full_text.encode()).hexdigest()}

def family(name):
 if '.self_attn.' in name:return 'attention'
 if '.mlp.' in name:return 'MLP'
 raise ValueError('Unexpected trainable module: '+name)

def attach_adapter(base,torch,peft):
 if sum(p.numel() for p in base.parameters())!=BASE_PARAMETERS:raise ValueError('Native full bundle parameter count differs')
 if base.lm_head.weight is not base.model.language_model.embed_tokens.weight:raise ValueError('Native head/embedding must remain tied')
 modules={n:m for n,m in base.named_modules() if re.fullmatch(TARGET_PATTERN,n)}
 counts=Counter(n.rsplit('.',1)[-1] for n in modules)
 if dict(counts)!={n:34 for n in TARGETS} or not all(isinstance(m,torch.nn.Linear) for m in modules.values()):
  raise ValueError('Require all238 language projections and no visual/projector targets')
 config=peft.LoraConfig(task_type='CAUSAL_LM',r=16,lora_alpha=32,lora_dropout=.05,target_modules=TARGET_PATTERN,bias='none')
 model=peft.get_peft_model(base,config)
 named={n:p for n,p in model.named_parameters() if p.requires_grad}
 if any(('.lora_A.' not in n and '.lora_B.' not in n) or '.model.language_model.layers.' not in n for n in named):
  raise ValueError('Only language LoRA may train')
 if sum(p.numel() for p in named.values())!=ADAPTER_PARAMETERS:raise ValueError('Text LoRA parameter count differs')
 if sum(p.numel() for p in model.parameters())!=BASE_PARAMETERS+ADAPTER_PARAMETERS:raise ValueError('Total shared-weight count differs')
 if base.lm_head.weight.requires_grad or base.model.language_model.embed_tokens.weight.requires_grad:raise ValueError('Head/embed must remain frozen')
 families=Counter()
 for n,p in named.items():families[family(n)]+=p.numel()
 coverage={'base_parameters':BASE_PARAMETERS,'text_parameters':TEXT_PARAMETERS,'unused_visual_and_projector_parameters_counted':BASE_PARAMETERS-TEXT_PARAMETERS,
           'adapter_parameters':ADAPTER_PARAMETERS,'total_parameters':BASE_PARAMETERS+ADAPTER_PARAMETERS,'target_counts':dict(counts),'target_module_names':sorted(modules),
           'trainable_parameter_families':dict(families),'all_trainable_names':sorted(named),'all_parameter_devices':sorted({p.device.type for p in model.parameters()}),
           'head_embedding_tied':base.lm_head.weight is base.model.language_model.embed_tokens.weight,'head_embedding_frozen':True,'no_modules_to_save':True}
 return model,named,coverage

def forbid_vision_forward(base):
 # Full bundle stays intact for native checkpoint compatibility and conservative
 # parameter accounting. In this text-only experiment either vision call is a bug.
 def reject(*args,**kwargs):raise RuntimeError('Vision/projector forward forbidden for text-only relation control')
 return [base.model.vision_tower.register_forward_pre_hook(reject),base.model.multi_modal_projector.register_forward_pre_hook(reject)]

def greedy_config(transformers,max_new_tokens=1600):
 # The published v4 generation config requests legacy 'hybrid'. In native TF5,
 # use its default dynamic cache (mixed sliding/full layers from the config),
 # not a handwritten cache. Decoding/stops are fixed and recorded explicitly.
 return transformers.GenerationConfig(do_sample=False,num_beams=1,max_new_tokens=max_new_tokens,
     eos_token_id=EOS_IDS,pad_token_id=0,bos_token_id=2,use_cache=True,cache_implementation=None)
