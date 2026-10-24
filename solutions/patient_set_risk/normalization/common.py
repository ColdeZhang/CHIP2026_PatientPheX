"""SCST train-only reward bridge; original grounding and normalizer are unedited."""
import hashlib, json, math, os, re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SELECTED = [(76,'5793168:P1:grounded'), (44,'12628480:OP1:grounded'),
            (83,'6134758:P1:grounded'), (19,'12059664:O4:grounded')]
COMPOUND = re.compile(r'HP:[0-9]{7}(?:;HP:[0-9]{7})+')
POLICY_SHA = 'b59eb3f8845ff297578241ed61192fb8ae0153587deab924c51b0a33a5e8bd20'

def ref(path):
    p=Path(path).resolve(); h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
    return dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest())

def verify(item,path=None):
    got=ref(path or item['path'])
    assert all(got[k]==item[k] for k in ('bytes','sha256')), (got,item)
    return got

def save(path,value):
    path=Path(path)
    with path.open('x',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
    return ref(path)

def load(path): return json.loads(Path(path).read_text())

def original():
    for name,item in load(HERE/'reuse-manifest.json').items(): verify(item,HERE/name)
    sys.path.insert(0,str(HERE/'reuse'))
    from solutions.grounded_generation.inference.decode import interpret
    from solutions.grounded_generation.schema import units
    from solutions.grounded_generation.normalization.strict import StrictNormalizer
    from solutions.grounded_generation.normalization.semantic import load_policy,precheck,decide
    policy_path=HERE/'reuse/solutions/grounded_generation/normalization/policy_v1.json'
    assert ref(policy_path)['sha256']==POLICY_SHA
    return interpret,units,StrictNormalizer,load_policy(policy_path),precheck,decide

def atoms(values):
    return {a for v in values for a in (v.split(';') if COMPOUND.fullmatch(v) else [v])}

def f1(gold,pred): return 2*len(gold&pred)/(len(gold)+len(pred)) if gold or pred else 1.

def validate_responses(rows):
    expected={(r,a) for _,r in SELECTED for a in ('greedy','sample')}
    assert isinstance(rows,list) and len(rows)==8,'Need exactly all8 raw responses'
    seen=set()
    for x in rows:
        key=(x['row_id'],x['arm']); assert key in expected and key not in seen,'Unknown/duplicate row-arm'
        seen.add(key)
        assert isinstance(x['response'],str) and type(x['EOS']) is bool and type(x['cap']) is bool
        assert isinstance(x['generated_ids'],list) and x['generated_ids'] and all(type(i) is int and i>=0 for i in x['generated_ids'])
    assert seen==expected

def supply_gate(rows):
    by={(x['row_id'],x['arm']):x for x in rows};pairs=[]
    for _,row_id in SELECTED:
        g,s=by[row_id,'greedy'],by[row_id,'sample'];a=s['reward']-g['reward']
        assert math.isfinite(a)
        pairs.append(dict(row_id=row_id,greedy_reward=g['reward'],sample_reward=s['reward'],advantage=a,
                          greedy_valid=g['valid'],sample_valid=s['valid'],gold_count=g['gold_count'],
                          semantic_difference=bool(g['gold_count'] and g['valid'] and s['valid'] and a!=0)))
    nonzero=sum(x['advantage']!=0 for x in pairs);semantic=sum(x['semantic_difference'] for x in pairs)
    return dict(pairs=pairs,nonzero_groups=nonzero,nonempty_gold_valid_semantic_groups=semantic,
                supply_gate_pass=nonzero>=2 and semantic>=1,
                gate='Pre-output activation: >=2 nonzero groups, >=1 nonempty-gold/both-valid concept-F1 difference; other may be format validity',
                rewards_and_advantages_detached=True)
