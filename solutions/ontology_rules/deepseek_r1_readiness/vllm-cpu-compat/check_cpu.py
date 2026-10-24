"""Native function-only CPU comparison; no vLLM import, model load or config edit."""
from pathlib import Path
import argparse,ast,copy,hashlib,inspect,json,math,os,time,types

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 start=time.monotonic();ap=argparse.ArgumentParser();ap.add_argument('--source',required=True);ap.add_argument('--config',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
 source=Path(a.source);config=Path(a.config);assert sha(source)=='9c6323a6e1114573b53adbae70043d1363a4d027a9ba70e71a12227ddc00c8a5';assert sha(config)=='1b1cbbba841c180e94192f7a25d2d97e21498fc1a0a0b4785942158af5eee183';raw=json.loads(config.read_text());rope=raw['rope_scaling']
 import torch,transformers
 from transformers import AutoConfig
 from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
 torch.set_num_threads(1);assert not torch.cuda.is_initialized()
 tree=ast.parse(source.read_text());names={'_yarn_find_correction_dim','_yarn_find_correction_range','_yarn_linear_ramp_mask','_yarn_get_mscale'};nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names];assert len(nodes)==4
 klass=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='YaRNScalingRotaryEmbedding');freq=next(n for n in klass.body if isinstance(n,ast.FunctionDef) and n.name=='_compute_inv_freq');nodes.append(freq)
 ns={'torch':torch,'math':math};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),ns)
 obj=types.SimpleNamespace(rotary_dim=raw['head_dim'],base=raw['rope_theta'],max_position_embeddings=rope['original_max_position_embeddings'],beta_fast=rope.get('beta_fast',32),beta_slow=rope.get('beta_slow',1),extrapolation_factor=rope.get('extrapolation_factor',1),scaling_factor=rope['factor'])
 init=next(n for n in klass.body if isinstance(n,ast.FunctionDef) and n.name=='__init__');scale_stmt=[n for n in init.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Attribute) and isinstance(t.value,ast.Name) and t.value.id=='self' and t.attr=='mscale' for t in n.targets)];assert len(scale_stmt)==1
 exec(compile(ast.Module(body=scale_stmt,type_ignores=[]),str(source),'exec'),dict(ns,self=obj,attn_factor=rope.get('attn_factor',1)))
 native=ns['_compute_inv_freq'](obj,rope['factor']);cfg=AutoConfig.from_pretrained(config.parent,local_files_only=True,trust_remote_code=False);fn=ROPE_INIT_FUNCTIONS['yarn'];assert sha(inspect.getsourcefile(fn))=='6a0727073e080e163d0287e15fc1acb90035e9a9df9cb19a13b632a40abfac24'
 original,original_scale=fn(cfg,device=torch.device('cpu'));derived=copy.deepcopy(cfg);derived.rope_parameters['attention_factor']=obj.mscale;compatible,compatible_scale=fn(derived,device=torch.device('cpu'))
 assert all(x.dtype==torch.float32 and x.device.type=='cpu' and tuple(x.shape)==(64,) and bool(torch.isfinite(x).all()) for x in [native,original,compatible]);assert torch.equal(native,original) and torch.equal(native,compatible);assert original_scale==1.138629436111989 and compatible_scale==obj.mscale==1.0
 hashes={n:hashlib.sha256(x.numpy().tobytes()).hexdigest() for n,x in [('vllm_native',native),('HF_original',original),('HF_derived_attention_factor',compatible)]};assert hashes['HF_original']=='427ed49dc1d6e18683800eca336467c70fc6817bfdae8a61166cbc7e01b05cb1'
 assert sha(config)=='1b1cbbba841c180e94192f7a25d2d97e21498fc1a0a0b4785942158af5eee183' and not torch.cuda.is_initialized()
 result={'status':'PASS_CPU_NATIVE_FUNCTION_FREQUENCY_AND_SCALE_COMPATIBILITY_ONLY','program_seconds':time.monotonic()-start,'source_sha256':sha(__file__),'fixed_vllm_source_sha256':sha(source),'original_config_sha256':sha(config),'original_config_unchanged':True,'native_functions_exact_AST_extracted':sorted(names)+[klass.name+'.'+freq.name,klass.name+'.__init__:self.mscale assignment'],'vllm_scale':obj.mscale,'HF_original_scale':original_scale,'derived_in_memory_recognized_attention_factor':obj.mscale,'HF_derived_scale':compatible_scale,'all64_frequency_values_exact':True,'inv_freq_shape':[64],'inv_freq_dtype':'torch.float32','frequency_bytes_SHA':hashes,'maximum_frequency_abs_difference':float((native-compatible).abs().max()),'torch':torch.__version__,'transformers':transformers.__version__,'CUDA_initialized':torch.cuda.is_initialized(),'model_loaded':False,'weight_values_read':False,'vllm_installed_or_imported':False,'scope':'Only exact native frequency/scale functions; not a complete vLLM engine, logits, format, or quality comparison. Original1024/4096HF probes unchanged; no GPU or further budget authorization.'}
 with Path(a.output).open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(result))
if __name__=='__main__':main()
