"""
Build relation training data using DICTIONARY matches (not gold entities),
so training distribution matches inference distribution.
"""
import json, os, sys, random
os.chdir("/dataset/wanggenyi/patient_phex")
import ahocorasick
from collections import defaultdict

random.seed(42)

dd = json.load(open("data/hpo_dict_augmented.json"))
combined = dd["combined"]

ABBREV = {
    "dm": "HP:0000819", "id": "HP:0001249", "hs": "HP:0002370",
    "t2dm": "HP:0005978", "t1dm": "HP:0100651", "mody": "HP:0004904",
    "cp": "HP:0001257", "asd": "HP:0000750", "adhd": "HP:0007018",
}
FP_STOPWORDS = {"severity", "cut", "niv"}

A = ahocorasick.Automaton()
for surf, ids in combined.items():
    if len(surf) < 3: continue
    A.add_word(surf, (surf, set(ids)))
for abbr, hid in ABBREV.items():
    A.add_word(abbr, (abbr, {hid}))
A.make_automaton()

def dict_match_section(text):
    text_l = text.lower()
    hits = {}
    for end_idx, (surf, ids) in A.iter(text_l):
        start_idx = end_idx - len(surf) + 1
        if surf in FP_STOPWORDS: continue
        if len(surf) <= 4 and surf.isalpha():
            before = text_l[start_idx-1] if start_idx > 0 else " "
            after = text_l[end_idx+1] if end_idx+1 < len(text_l) else " "
            if before.isalpha() or after.isalpha(): continue
        key = (start_idx, end_idx+1)
        if key not in hits: hits[key] = set()
        hits[key] |= ids
    return hits

def get_sentence(text, offset, length):
    start = offset
    while start > 0 and text[start-1] not in '.!?:;=\n\t ' and start > offset - 200: start -= 1
    end = offset + length
    while end < len(text) and text[end] not in '.!?\n' and end < offset + length + 200: end += 1
    return text[start:end].strip()

train = [json.loads(l) for l in open("data/train.jsonl", encoding="utf-8")]

samples = []
for d in train:
    patients = d["patient"]
    if not patients: continue
    
    # Gold associations: patient_id -> set of HPO ids
    gold_assoc = {}
    for a in d["association"]:
        gold_assoc[a["patient_id"]] = set(p for p in a["phenotype"] if p.startswith("HP:"))
    
    # Gold entity spans: (offset, length) -> set of HPO ids
    gold_entities = {}
    for e in d["entities"]:
        if e.get("note") == "NO": continue
        hids = set(x.strip() for x in e["identifier"].split(";") if x.strip().startswith("HP:"))
        if hids:
            gold_entities[(e["offset"], e["length"])] = hids
    
    # For each section, get dictionary matches
    for sec in d["full_text"]:
        text = sec["text"]; sec_off = sec["offset"]
        d_hits = dict_match_section(text)
        
        for (s, e), hpo_ids in d_hits.items():
            g_off = sec_off + s
            g_len = e - s
            span_text = text[s:e]
            sent = get_sentence(text, s, g_len)
            
            # Is this span a gold entity?
            is_gold = False
            gold_hpos = set()
            for (goff, glen), ghids in gold_entities.items():
                # Overlap check
                if not (g_off+g_len <= goff or g_off >= goff+glen):
                    is_gold = True
                    gold_hpos |= ghids
                    break
            
            for p in patients:
                pid = p["patient_id"]
                ptext = p["mention"][0]["text"]
                text_inp = f"{ptext} [SEP] {span_text} [SEP] {sent}"
                
                # Label: is this HPO in gold_assoc for this patient?
                label = 0
                if is_gold:
                    # Check if any gold HPO for this span belongs to this patient
                    for h in gold_hpos:
                        if h in gold_assoc.get(pid, set()):
                            label = 1
                            break
                
                samples.append({"text": text_inp, "label": label})

# Split train/dev
random.shuffle(samples)
n_dev = int(len(samples) * 0.1)
dev = samples[:n_dev]
tr = samples[n_dev:]

# Balance: keep all positive, sample negative 2:1
tr_pos = [s for s in tr if s["label"] == 1]
tr_neg = [s for s in tr if s["label"] == 0]
random.shuffle(tr_neg)
tr_neg = tr_neg[:len(tr_pos) * 2]
tr_balanced = tr_pos + tr_neg
random.shuffle(tr_balanced)

os.makedirs("data/rel2", exist_ok=True)
with open("data/rel2/train.jsonl", "w", encoding="utf-8") as f:
    for s in tr_balanced:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")
with open("data/rel2/dev.jsonl", "w", encoding="utf-8") as f:
    for s in dev:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")

print(f"Train: {len(tr_balanced)} (pos={len(tr_pos)}, neg={len(tr_neg)})")
print(f"Dev: {len(dev)} (pos={sum(1 for s in dev if s['label']==1)}, neg={sum(1 for s in dev if s['label']==0)})")
