"""Exact canonical-name list, row-bound NIL literal, or empty NO list."""
import json
from reuse.canonical.schema_contract import CanonicalList

SYSTEM='''Normalize the marked clinical phenotype mention using its supplied context.
The marked occurrence is the only target. Return either:
- A nonempty JSON array of exact English HPO canonical names equivalent to the mention; use several names only for explicitly expressed compound findings.
- {"NIL": "the exact marked mention"} if this is a positive phenotype with no equivalent HPO concept.
- [] if the marked occurrence is negated, normal, or is not a phenotype.
Do not infer additional findings from a disease. Keep specificity, anatomy, direction and negation. Return no HPO identifiers, explanation, or additional fields.'''


def answer(target, quote):
    status=target['status'];names=target['names']
    if status=='HPO':
        assert names and len(names)==len(set(names));return '[\n'+',\n'.join(json.dumps(x,ensure_ascii=False) for x in names)+'\n]'
    assert names==[] and status in ['NIL','NO']
    return json.dumps({'NIL':quote},ensure_ascii=False,separators=(',',':')) if status=='NIL' else '[]'


def strict_parse(raw, quote, name_to_id, *, eos, capped):
    if not eos or capped:raise ValueError('Incomplete generation: whole row rejected')
    def obj(pairs):
        value={}
        for k,v in pairs:
            if k in value:raise ValueError('Duplicate JSON key')
            value[k]=v
        return value
    value=json.loads(raw,object_pairs_hook=obj)
    if isinstance(value,list):
        if any(type(x)is not str for x in value):raise ValueError('Expected canonical string names')
        if len(value)!=len(set(value)) or any(x not in name_to_id for x in value):raise ValueError('Unknown or repeated canonical name')
        return {'status':'HPO' if value else 'NO','ids':[name_to_id[x] for x in value]}
    if type(value)is dict and set(value)=={'NIL'} and type(value['NIL'])is str and value['NIL']==quote:
        return {'status':'NIL','ids':['-1']}
    raise ValueError('NIL must contain only the exact input mention; no repairs')


class MentionCodec:
    def __init__(self, tokenizer, names, eos_id):
        self.public=CanonicalList(tokenizer,names,eos_id);self.tokenizer=tokenizer;self.names=names;self.eos_id=eos_id
    def nil_ids(self,quote):
        text=answer({'status':'NIL','names':[]},quote);ids=self.tokenizer.encode(text,add_special_tokens=False).ids
        if self.tokenizer.decode(ids,skip_special_tokens=False)!=text:raise ValueError('NIL literal roundtrip mismatch')
        return ids
    def allowed(self,generated,quote):
        generated=list(generated);nil=self.nil_ids(quote);allowed=set()
        try:allowed.update(self.public.allowed(generated))
        except ValueError:pass
        if generated==nil or generated==nil+[self.eos_id]:allowed.add(self.eos_id)
        elif len(generated)<len(nil) and generated==nil[:len(generated)]:allowed.add(nil[len(generated)])
        if not allowed:raise ValueError('Invalid prefix, no full-vocabulary fallback')
        return sorted(allowed)
    def callback(self,prompt_width,quotes):
        def callback(batch_id,input_ids):
            ids=input_ids.tolist() if hasattr(input_ids,'tolist') else list(input_ids)
            return self.allowed(ids[prompt_width:],quotes[batch_id])
        return callback
