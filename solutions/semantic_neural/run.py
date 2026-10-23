"""Ontology-grounded neural span detection/linking and patient graph propagation.

All learned resources are fitted only from --train. Predictions never inspect input labels.
The frozen encoder is SapBERT (CLS) or MiniLM ONNX (mean pooled).
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import numpy as np
import ahocorasick
from sklearn.linear_model import LogisticRegression

WORD = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:[-'][A-Za-z0-9]+)*")
STOP = set('a an the of in on at for and or with without is was were be been has have had to from by as that this these those which whom who all any only no not yes both his her their its it we they he she patient patients case cases study studies clinical features phenotype phenotypes abnormality abnormalities findings finding reported report present presented observed identified normal results showed showing including associated history analysis evaluation test tests level levels cell cells gene genes protein proteins variant variants mutation mutations disease diseases syndrome syndromes group groups type types number numbers'.split())
NEGATION = re.compile(r'\b(?:no|without|denied|denies|negative for|absence of|absent|excluded|unaffected by)\b', re.I)
SKIP_SECTIONS = {'REF', 'REFERENCES', 'BIBL', 'AUTH_CONT', 'COMP_INT', 'ACK_FUND'}


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def norm(s):
    return ' '.join(s.lower().split()).strip()


def hpo_terms(path):
    terms = {}
    term = None
    for line in Path(path).read_text().splitlines():
        if line == '[Term]':
            term = {'synonyms': [], 'parents': [], 'alts': [], 'obsolete': False}
        elif line.startswith('['):
            term = None
        elif term is not None:
            if line.startswith('id: '):
                term['id'] = line[4:]; terms[line[4:]] = term
            elif line.startswith('name: '): term['name'] = line[6:]
            elif line.startswith('alt_id: '): term['alts'].append(line[8:])
            elif line.startswith('is_a: '): term['parents'].append(line[6:].split()[0])
            elif line.startswith('is_obsolete: true'): term['obsolete'] = True
            elif line.startswith('synonym: '):
                m = re.match(r'synonym: "(.*?)" EXACT', line)
                if m: term['synonyms'].append(m.group(1).replace('\\"', '"'))
    valid = {'HP:0000118'}
    changed = True
    while changed:
        before = len(valid)
        valid.update(k for k, v in terms.items() if not v['obsolete'] and any(p in valid for p in v['parents']))
        changed = len(valid) != before
    canonical = {k: k for k in valid if k in terms and not terms[k]['obsolete']}
    for k in list(canonical):
        canonical.update({a: k for a in terms[k]['alts']})
    return {k: terms[k] for k in valid if k in terms and not terms[k]['obsolete']}, canonical


class Encoder:
    def __init__(self, model_dir, kind, device, batch, cache_dir, threads):
        self.kind, self.batch, self.cache_dir = kind, batch, Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        if kind == 'sapbert':
            import torch
            from transformers import AutoTokenizer, AutoModel
            torch.set_num_threads(threads)
            self.torch = torch
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
            self.model = AutoModel.from_pretrained(model_dir, local_files_only=True).eval().to(device)
            if device.startswith('cuda'): self.model.half()
            self.inputs = None
        else:
            import onnxruntime as ort
            from tokenizers import Tokenizer
            self.tokenizer = Tokenizer.from_file(str(Path(model_dir) / 'tokenizer.json'))
            self.tokenizer.enable_truncation(max_length=32); self.tokenizer.enable_padding()
            options = ort.SessionOptions(); options.intra_op_num_threads = threads; options.inter_op_num_threads = 1
            self.model = ort.InferenceSession(str(Path(model_dir) / 'model.onnx'), sess_options=options, providers=['CPUExecutionProvider'])
            self.inputs = {x.name for x in self.model.get_inputs()}
        config = Path(model_dir) / 'config.json'
        self.signature = kind + sha(config)

    def encode(self, texts, label):
        digest = hashlib.sha256((self.signature + '\0' + '\0'.join(texts)).encode()).hexdigest()
        cache = self.cache_dir / (digest + '.npy')
        if cache.exists(): return np.load(cache)
        result = []
        started = time.monotonic()
        for start in range(0, len(texts), self.batch):
            subset = texts[start:start + self.batch]
            if self.kind == 'sapbert':
                inputs = self.tokenizer(subset, padding=True, truncation=True, max_length=32, return_tensors='pt')
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with self.torch.inference_mode(): values = self.model(**inputs).last_hidden_state[:, 0].float().cpu().numpy()
            else:
                toks = self.tokenizer.encode_batch(subset)
                inputs = {'input_ids': np.array([x.ids for x in toks], np.int64), 'attention_mask': np.array([x.attention_mask for x in toks], np.int64), 'token_type_ids': np.array([x.type_ids for x in toks], np.int64)}
                output = self.model.run(None, {k: v for k, v in inputs.items() if k in self.inputs})[0]
                mask = inputs['attention_mask'][:, :, None]
                values = (output * mask).sum(1) / np.maximum(mask.sum(1), 1)
            values = values.astype(np.float32)
            values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-9)
            result.append(values)
        array = np.concatenate(result) if result else np.empty((0, 768 if self.kind == 'sapbert' else 384), np.float32)
        np.save(cache, array)
        print(json.dumps({'event': 'encoded', 'label': label, 'phrases': len(texts), 'seconds': round(time.monotonic()-started, 3)}), flush=True)
        return array

    def nearest(self, vectors, bank):
        if len(vectors) == 0: return np.array([], np.int64), np.array([], np.float32)
        scores, ids = [], []
        if self.kind == 'sapbert' and self.device.startswith('cuda'):
            torch = self.torch
            targets = torch.from_numpy(bank).to(self.device)
            for start in range(0, len(vectors), 512):
                query = torch.from_numpy(vectors[start:start+512]).to(self.device)
                values, idx = (query @ targets.T).max(1)
                scores.append(values.cpu().numpy()); ids.append(idx.cpu().numpy())
            del targets
        else:
            for start in range(0, len(vectors), 256):
                dot = vectors[start:start+256] @ bank.T
                idx = dot.argmax(1); ids.append(idx); scores.append(dot[np.arange(len(idx)), idx])
        return np.concatenate(ids), np.concatenate(scores)


class System:
    def __init__(self, train, terms, canonical, encoder, args):
        self.encoder, self.args, self.canonical = encoder, args, canonical
        self.surface = defaultdict(Counter)
        self.train_surface = defaultdict(Counter)
        self.concept_count = Counter()
        for identifier, term in terms.items():
            if identifier == 'HP:0000118': continue
            for label in [term['name']] + term['synonyms']:
                if len(label) >= 3: self.surface[norm(label)][identifier] += 0.05
        self.positive_heads = Counter()
        for doc in train:
            for e in doc.get('entities', []):
                if str(e.get('note', '')).upper() == 'NO': continue
                identifier = self.canonicalize(e.get('identifier', ''))
                text = str(e.get('text', '')).strip()
                if not identifier or len(text) < 2: continue
                self.surface[norm(text)][identifier] += 1
                self.train_surface[norm(text)][identifier] += 1
                self.concept_count[identifier] += 1
                words = WORD.findall(text.lower())
                if words: self.positive_heads[words[-1]] += 1
        self.labels = sorted(self.surface)
        self.label_ids = [self.best_id(k) for k in self.labels]
        self.automaton = ahocorasick.Automaton()
        for text in self.labels:
            self.automaton.add_word(text, text)
        self.automaton.make_automaton()
        self.word_count = Counter(w for text in self.labels for w in set(WORD.findall(text)))
        self.heads = {WORD.findall(x)[-1] for x in self.labels if WORD.findall(x)} - STOP
        self.heads |= set(self.positive_heads) - STOP
        self.bank = encoder.encode(self.labels, 'ontology-and-training-labels')
        self.fit_detector(train)

    def canonicalize(self, identifier):
        ids = str(identifier).split(';')
        if ids == ['-1']: return '-1'
        canonical = {self.canonical[x.strip()] for x in ids if x.strip() in self.canonical}
        return ';'.join(sorted(canonical)) if canonical else ''

    def best_id(self, text):
        return max(self.surface[text], key=lambda i: (self.surface[text][i], self.concept_count[i], i))

    @staticmethod
    def passages(doc):
        return [p for p in doc.get('full_text', []) if str(p.get('section_type', '')).upper() not in SKIP_SECTIONS]

    def negated(self, text, start):
        before = re.split(r'[.;!?]|\bbut\b|\bhowever\b', text[max(0, start-90):start], flags=re.I)[-1]
        return bool(NEGATION.search(before[-65:]))

    def candidates(self, doc):
        exact, proposed = {}, {}
        for pi, p in enumerate(self.passages(doc)):
            text = p.get('text', ''); offset = int(p.get('offset', 0)); lower = text.lower()
            if len(lower) != len(text): continue
            for end, label in self.automaton.iter(lower):
                start = end-len(label)+1; end += 1
                if (start and (text[start-1].isalnum() or text[start-1] == '_')) or (end < len(text) and (text[end].isalnum() or text[end] == '_')): continue
                if len(label) < 3 and not text[start:end].isupper(): continue
                if self.negated(text, start): continue
                exact[(offset+start, end-start)] = {'offset':offset+start, 'length':end-start, 'text':text[start:end], 'exact':True, 'passage':pi}
            tokens = list(WORD.finditer(text))
            for end_i, tok in enumerate(tokens):
                head = tok.group().lower()
                if head not in self.heads or head in STOP: continue
                for size in range(1, min(6,end_i+1)+1):
                    begin = tokens[end_i-size+1]; start, end = begin.start(), tok.end()
                    label = text[start:end]; words = [m.group().lower() for m in tokens[end_i-size+1:end_i+1]]
                    if len(label)<4 or words[0] in STOP or re.search(r'[.,;:!?\n]',label): continue
                    if any(w in {'was','were','has','have','had','is','are','and','or','with','without','by'} for w in words): continue
                    if self.negated(text,start):continue
                    key = (offset+start,end-start)
                    if key in exact:continue
                    rare = max(math.log((len(self.labels)+1)/(self.word_count[w]+1)) for w in words)
                    if rare < 4.7:continue
                    priority = rare + min(2, self.positive_heads[head]*0.05) - abs(size-2)*0.16
                    proposed[key] = {'offset':offset+start, 'length':end-start, 'text':label, 'exact':False, 'passage':pi, 'priority':priority}
        ranked = sorted(proposed.items(),key=lambda kv:(-kv[1]['priority'],kv[0]))[:self.args.max_candidates]
        exact.update(ranked)
        return [exact[k] for k in sorted(exact)]

    def fit_detector(self, train):
        rows, labels = [], []
        rng = np.random.default_rng(self.args.seed)
        for doc in train:
            gold = {(int(e['offset']),int(e['length'])):e for e in doc.get('entities',[]) if str(e.get('note','')).upper()!='NO' and self.canonicalize(e.get('identifier',''))}
            candidates = self.candidates(doc)
            positives = {k:v['text'] for k,v in gold.items()}
            rows.extend(positives.values());labels.extend([1]*len(positives))
            negatives = [c['text'] for c in candidates if (c['offset'],c['length']) not in gold]
            if len(negatives)>self.args.negatives_per_doc:
                selected = rng.choice(len(negatives),self.args.negatives_per_doc,replace=False)
                negatives = [negatives[i] for i in selected]
            rows.extend(negatives);labels.extend([0]*len(negatives))
        vectors = self.encoder.encode(rows, 'detector-training')
        self.detector = LogisticRegression(C=0.6, class_weight='balanced', max_iter=450, random_state=self.args.seed)
        self.detector.fit(vectors,np.array(labels))
        self.training_stats={'positive_spans':sum(labels),'negative_spans':len(labels)-sum(labels),'training_docs':len(train),'semantic_bank':len(self.labels),'detector_iterations':int(max(self.detector.n_iter_))}
        print(json.dumps({'event':'detector_fit',**self.training_stats}),flush=True)

    def detect(self, doc):
        candidates = self.candidates(doc)
        if not candidates:return []
        vectors = self.encoder.encode([norm(c['text']) for c in candidates], 'input-'+str(doc.get('pmc_id')))
        probabilities = self.detector.predict_proba(vectors)[:,1]
        indexes, similarity = self.encoder.nearest(vectors,self.bank)
        accepted=[]
        for c,prob,idx,sim in zip(candidates,probabilities,indexes,similarity):
            key=norm(c['text'])
            if c['exact']:
                if prob<self.args.exact_threshold:continue
                identifier=self.best_id(key)
            else:
                if prob<self.args.neural_threshold or sim<self.args.min_similarity:continue
                identifier=self.label_ids[int(idx)]
            c=dict(c,identifier=identifier,probability=float(prob),similarity=float(sim))
            c['utility']=0.56*prob+0.27*sim+0.08*c['exact']+0.025*min(4,len(WORD.findall(c['text'])))
            accepted.append(c)
        result=[]
        for c in sorted(accepted,key=lambda x:(-x['utility'],-x['length'],x['offset'],x['identifier'])):
            start,end=c['offset'],c['offset']+c['length']
            if any(start<e['offset']+e['length'] and e['offset']<end for e in result):continue
            result.append(c)
        return sorted(result,key=lambda x:(x['offset'],x['length'],x['identifier']))

    def associations(self, doc, entities):
        patients = doc.get('patient',[])
        if not patients:return []
        patient_ids=[p['patient_id'] for p in patients]
        nodes=[]
        for pi,p in enumerate(self.passages(doc)):
            text=p.get('text','');base=int(p.get('offset',0));start=0
            for m in re.finditer(r'[.!?](?:\s+|$)|\n+',text):
                end=m.end()
                if text[start:end].strip():nodes.append({'start':base+start,'end':base+end,'text':text[start:end],'passage':pi,'section':str(p.get('section_type',''))})
                start=end
            if text[start:].strip():nodes.append({'start':base+start,'end':base+len(text),'text':text[start:],'passage':pi,'section':str(p.get('section_type',''))})
        if not nodes:return [{'patient_id':p,'phenotype':[]} for p in patient_ids]
        seed=np.zeros((len(nodes),len(patients)),np.float32)
        alias_owners=defaultdict(set)
        for j,patient in enumerate(patients):
            for mention in patient.get('mention',[]):
                off=int(mention.get('offset',-1));length=int(mention.get('length',0))
                for i,node in enumerate(nodes):
                    if off<node['end'] and node['start']<off+length:seed[i,j]+=1.0
                text=norm(str(mention.get('text','')))
                if len(text)>=4 and text not in {'patient','patients','he','she','his','her','they','their'}:alias_owners[text].add(j)
        for alias,owners in alias_owners.items():
            if len(owners)!=1:continue
            owner=next(iter(owners));pattern=re.compile(r'(?<!\w)'+re.escape(alias)+r'(?!\w)',re.I)
            for i,node in enumerate(nodes):
                if pattern.search(node['text']):seed[i,owner]+=0.6
        graph=np.zeros((len(nodes),len(nodes)),np.float32)
        for i in range(len(nodes)-1):
            weight=1.0 if nodes[i]['passage']==nodes[i+1]['passage'] else 0.12
            graph[i,i+1]=weight;graph[i+1,i]=weight
        degree=graph.sum(1,keepdims=True);graph/=np.maximum(degree,1)
        values=seed.copy()
        for _ in range(20):values=seed+0.75*graph@values
        predicted=[set() for _ in patients]
        for entity in entities:
            locations=[i for i,n in enumerate(nodes) if n['start']<=entity['offset']<n['end']]
            if not locations:continue
            i=locations[0];scores=values[i];total=float(scores.sum())
            if total<0.008:continue
            best=int(scores.argmax());fraction=float(scores[best]/max(total,1e-9))
            chosen=[best] if len(patients)==1 or fraction>=self.args.patient_threshold else []
            explicit=np.flatnonzero(seed[i]>0)
            if len(explicit)>1 and re.search(r'\b(both|all|siblings|brothers|sisters)\b',nodes[i]['text'],re.I):chosen=list(explicit)
            concepts=entity['identifier'].split(';') if entity['identifier']!='-1' else [entity['text']]
            for j in chosen:predicted[j].update(concepts)
        return [{'patient_id':pid,'phenotype':sorted(predicted[i])} for i,pid in enumerate(patient_ids)]

    def predict(self,doc):
        found=self.detect(doc)
        entities=[{'identifier':e['identifier'],'type':'Phenotype','offset':e['offset'],'length':e['length'],'text':e['text'],'note':None} for e in found]
        return {'pmc_id':doc['pmc_id'],'pmid':doc.get('pmid'),'entities':entities,'association':self.associations(doc,found)}


def main():
    parser=argparse.ArgumentParser()
    for name in ['train','input','hpo','output','work-dir','model-dir']:parser.add_argument('--'+name,required=True,type=Path)
    parser.add_argument('--encoder',choices=['sapbert','minilm-onnx'],default='sapbert')
    parser.add_argument('--device',default='cpu');parser.add_argument('--threads',type=int,default=1)
    parser.add_argument('--batch-size',type=int,default=128);parser.add_argument('--seed',type=int,default=20261023)
    parser.add_argument('--max-candidates',type=int,default=1600);parser.add_argument('--negatives-per-doc',type=int,default=300)
    parser.add_argument('--exact-threshold',type=float,default=0.48);parser.add_argument('--neural-threshold',type=float,default=0.72)
    parser.add_argument('--min-similarity',type=float,default=0.87);parser.add_argument('--patient-threshold',type=float,default=0.60)
    args=parser.parse_args();started=time.monotonic();args.work_dir.mkdir(parents=True,exist_ok=True)
    train=read_jsonl(args.train);inputs=read_jsonl(args.input)
    if set(d['pmc_id'] for d in train)&set(d['pmc_id'] for d in inputs):raise ValueError('Training and inference document IDs overlap')
    if any('entities' in d or 'association' in d for d in inputs):raise ValueError('Inference input must have labels removed')
    if not any('entities' in d for d in train):raise ValueError('Training input has no entity annotations')
    terms,canonical=hpo_terms(args.hpo)
    encoder=Encoder(args.model_dir,args.encoder,args.device,args.batch_size,args.work_dir/'embedding-cache',args.threads)
    system=System(train,terms,canonical,encoder,args)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    counters=Counter()
    with args.output.open('w') as output:
        for i,doc in enumerate(inputs):
            prediction=system.predict(doc)
            output.write(json.dumps(prediction,ensure_ascii=False)+'\n');output.flush()
            counters['documents']+=1;counters['entities']+=len(prediction['entities']);counters['links']+=sum(len(x['phenotype']) for x in prediction['association'])
            print(json.dumps({'event':'predicted','index':i+1,'pmc_id':doc['pmc_id'],'entities':len(prediction['entities'])}),flush=True)
    manifest={'experiment':'E-20261023-006','elapsed_seconds':time.monotonic()-started,'parameters':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},'training':system.training_stats,'counts':dict(counters),'source_sha256':sha(__file__),'model_source':json.loads((args.model_dir/'manifest.json').read_text()) if (args.model_dir/'manifest.json').exists() else {'unrecorded':True},'inputs':{str(p):sha(p) for p in [args.train,args.input,args.hpo]},'output':{'path':str(args.output),'sha256':sha(args.output),'bytes':args.output.stat().st_size},'evaluation':'Not performed in prediction; use shared evaluator on independent gold.'}
    (args.work_dir/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'event':'finished','elapsed_seconds':manifest['elapsed_seconds'],**dict(counters)}),flush=True)


if __name__=='__main__':main()
