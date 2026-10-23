"""Public complete-object native restore; independent config/head coverage; finite synthetic pairs only."""
from pathlib import Path
import os,sys,json,hashlib,time,ast,socket,warnings,resource
B=Path(__file__).parent;t0=time.monotonic();assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
weight=B/'bert_model_max_triple.pkl';assert weight.stat().st_size==434159420 and sha(weight)=='353fc58889b2e59ddb9bf845c037283247268812ff3959cead118f35db90bea8'
index=json.loads((B/'upstream-source-index.json').read_text())
for x in index['files']:assert sha(B/x['path'])==x['sha256']
# Native torch complete-object load only after official-source/SHA and static metadata-global audit.
# No model conversion, checkpoint rewriting, pretrained constructor, shared installation, or network fallback.
def blocked_network(*args,**kwargs):raise RuntimeError('Network forbidden during CPU native model readiness')
socket.create_connection=blocked_network
import torch,fastNLP
from fastNLP.modules.encoder.bert import BertModel,BertConfig,BertTokenizer
from fastNLP import DataSet,DataSetIter
assert not torch.cuda.is_initialized();torch.set_num_threads(1);torch.set_num_interop_threads(1)
with warnings.catch_warnings(record=True) as ws:
 warnings.simplefilter('always');load0=time.monotonic();model=torch.load(weight,map_location='cpu',weights_only=False);loadsecs=time.monotonic()-load0
 warning_rows=[{'category':x.category.__name__,'message':str(x.message)[:500]} for x in ws]
assert type(model).__module__=='fastNLP.models.Mybert' and type(model).__name__=='BertForSentenceMatching'
assert model.num_labels==3 and model.classifier.in_features==768 and model.classifier.out_features==3
native=model.bert.model;enc=native.encoder;expected_config=json.loads((B/'biobert-config.json').read_text())
for k,v in expected_config.items():assert getattr(enc.config,k)==v,(k,getattr(enc.config,k),v)
assert len(enc.encoder.layer)==12
external_tok=BertTokenizer(str(B/'biobert-vocab.txt'));assert dict(native.tokenzier.vocab)==dict(external_tok.vocab) and len(native.tokenzier.vocab)==28996
assert native.pool_method=='max' and native.include_cls_sep is True and native.layers==[-1]
state=model.state_dict();inventory=[]
for name,x in state.items():
 assert x.device.type=='cpu' and torch.isfinite(x).all().item(),name
 inventory.append({'name':name,'shape':list(x.shape),'dtype':str(x.dtype),'numel':x.numel(),'sha256':hashlib.sha256(x.detach().contiguous().numpy().tobytes()).hexdigest()})
# Derive expected encoder keys/shapes from independent original-class config constructor, not from the restored object shape alone.
fresh_encoder=BertModel(BertConfig.from_dict(expected_config));fresh_head=torch.nn.Linear(768,3)
expected_keys={'bert.model.encoder.'+k for k in fresh_encoder.state_dict()}|{'classifier.'+k for k in fresh_head.state_dict()};assert set(state)==expected_keys,(sorted(set(state)-expected_keys),sorted(expected_keys-set(state)))
with torch.no_grad():
 for m in [fresh_encoder,fresh_head]:
  for p in m.parameters():p.fill_(float('nan'))
e=fresh_encoder.load_state_dict(enc.state_dict(),strict=True);h=fresh_head.load_state_dict(model.classifier.state_dict(),strict=True);assert not e.missing_keys and not e.unexpected_keys and not h.missing_keys and not h.unexpected_keys
for name,x in fresh_encoder.state_dict().items():assert torch.equal(x,state['bert.model.encoder.'+name]) and torch.isfinite(x).all().item()
for name,x in fresh_head.state_dict().items():assert torch.equal(x,state['classifier.'+name]) and torch.isfinite(x).all().item()
del fresh_encoder,fresh_head
(B/'native-state-inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
# Frozen original pure preprocessing fixtures; verify with actual restored tokenizer, never import my_bert_match (which constructs another encoder).
fixture=json.loads((B/'pair-tokenizer-result.json').read_text());pairs=fixture['cases'];prepared=[]
for x in pairs:
 actual=[native.tokenzier.convert_tokens_to_ids(native.tokenzier.wordpiece_tokenizer.tokenize(w)) for w in x['p_words']]
 assert actual==x['t_words'];assert [i for a in actual for i in a]==x['word_pieces'];prepared.append(x)
dataset=DataSet({'word_pieces':[x['word_pieces'] for x in prepared],'word_nums':[x['word_nums'] for x in prepared],'seq_len':[x['seq_len'] for x in prepared]})
for k in ['word_pieces','word_nums','seq_len']:dataset.field_arrays[k].is_input=True
batch,labels=next(iter(DataSetIter(batch_size=len(prepared),dataset=dataset,sampler=None)));assert labels=={}
def make_batch(rows):
 a=torch.zeros((len(rows),max(len(x['word_pieces']) for x in rows)),dtype=torch.long);b=torch.zeros((len(rows),max(len(x['word_nums']) for x in rows)),dtype=torch.long);c=torch.tensor([x['seq_len'] for x in rows],dtype=torch.long)
 for i,x in enumerate(rows):a[i,:len(x['word_pieces'])]=torch.tensor(x['word_pieces']);b[i,:len(x['word_nums'])]=torch.tensor(x['word_nums'])
 return {'word_pieces':a,'word_nums':b,'seq_len':c}
manual=make_batch(prepared)
for k in batch:assert torch.equal(batch[k],manual[k]),k
model.eval()
with torch.inference_mode():
 t=time.monotonic();logits=model.forward(batch['word_pieces'],batch['word_nums'],batch['seq_len'])['pred'];firstsecs=time.monotonic()-t;p=torch.softmax(logits,dim=1)
 assert logits.shape==(len(prepared),3) and torch.isfinite(logits).all() and torch.isfinite(p).all() and torch.allclose(p.sum(-1),torch.ones(len(prepared)),atol=1e-6)
 timed=make_batch(prepared[:3]*4);times=[]
 for _ in range(2):
  t=time.monotonic();y=model.forward(timed['word_pieces'],timed['word_nums'],timed['seq_len'])['pred'];times.append(time.monotonic()-t);assert y.shape==(12,3) and torch.isfinite(y).all()
assert not torch.cuda.is_initialized()
r={'status':'PASS_NATIVE_CPU_COMPLETE_MODEL_AND_SYNTHETIC_PAIRS','scope':'Complete public pair model only;5 synthetic pairs +2x12 repeated synthetic timed rows; no competition input/gold, training, GPU, CNN, fastText, Stanza, second pretrained weight or model conversion','weight_sha256':sha(weight),'weight_bytes':weight.stat().st_size,'source_index_sha256':sha(B/'upstream-source-index.json'),'checker_sha256':sha(__file__),'runtime':{'python':sys.version,'torch':torch.__version__,'fastNLP':fastNLP.__version__,'cuda_initialized':torch.cuda.is_initialized()},'load':{'weights_only':False,'seconds':loadsecs,'warnings':warning_rows,'no_compatibility_source_shim':True},'model':{'class':type(model).__module__+'.'+type(model).__name__,'parameters':sum(p.numel() for p in model.parameters()),'state_tensors':len(state),'state_elements':sum(x.numel() for x in state.values()),'encoder_layers':12,'config':enc.config.to_dict(),'num_labels':model.num_labels,'column_semantics':['Not Match','Related','Match'],'column_semantics_basis':'fixed official training/generator and inference source; published weights have no separate training-data manifest','pool_method':native.pool_method,'include_cls_sep':native.include_cls_sep,'pooled_cls':native.pooled_cls,'layers':native.layers,'auto_truncate':native.auto_truncate},'coverage':{'independent_config_encoder_and_classifier_keyset_exact':True,'strict_missing':[],'strict_unexpected':[],'all_parameters_nan_poisoned_then_restored_and_equal':True,'restored_vocab_equal_official28996':True,'original_DataSetIter_vs_manual_batch_tensors_exact':True,'state_inventory_sha256':sha(B/'native-state-inventory.json')},'synthetic':{'cases':[{ 'raw_words':x['raw_words'],'logits':logits[i].tolist(),'probabilities':p[i].tolist()} for i,x in enumerate(prepared)],'first5_seconds':firstsecs,'timed_batch_rows':12,'timed_batch_wordpiece_max_without_added_CLS_SEP':timed['word_pieces'].shape[1],'timed_batch_seconds':times,'not_quality_evidence':True},'seconds':time.monotonic()-t0,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
(B/'cpu-native-result.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k not in ['synthetic','load']}),flush=True)
