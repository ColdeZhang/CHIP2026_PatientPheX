"""
v10 B-set: Pure dictionary + rule association, optimized FP filtering.
Based on v10 self-eval improvements.
"""
import json, os, sys
os.chdir("/dataset/wanggenyi/patient_phex")
import ahocorasick
from collections import defaultdict

ABBREV = {
    "dm": "HP:0000819", "id": "HP:0001249", "hs": "HP:0002370",
    "t2dm": "HP:0005978", "t1dm": "HP:0100651", "mody": "HP:0004904",
    "cp": "HP:0001257", "asd": "HP:0000750", "adhd": "HP:0007018",
    "sp": "HP:0001257", "mr": "HP:0001249", "mh": "HP:0000707",
    "oa": "HP:0001263", "rp": "HP:0000580", "rd": "HP:0000556",
    "di": "HP:0001249",
}
FP_STOPWORDS = {
    "severity", "cut", "niv", "anger", "disability",
    "weakness", "mental disorders", "diabetes", "autism",
    "facial features", "cns", "d", "he",
}

dd = json.load(open("data/hpo_dict_augmented.json"))
combined = dd["combined"]
A = ahocorasick.Automaton()
for surf, ids in combined.items():
    if len(surf) < 3: continue
    A.add_word(surf, (surf, set(ids)))
for abbr, hid in ABBREV.items():
    A.add_word(abbr, (abbr, {hid}))
A.make_automaton()

def dict_lookup(text):
    text_l = text.lower()
    hits = {}
    for end_idx, (surf, ids) in A.iter(text_l):
        start_idx = end_idx - len(surf) + 1
        if surf in FP_STOPWORDS: continue
        if len(surf) <= 4 and surf.isalpha():
            before = text_l[start_idx-1] if start_idx > 0 else " "
            after = text_l[end_idx+1] if end_idx+1 < len(text_l) else " "
            if before.isalpha() or after.isalpha(): continue
        ctx_before = text_l[max(0,start_idx-30):start_idx]
        if any(neg in ctx_before for neg in ["no ", "without ", "negative ", "denies ", "absence of ", "resolved"]):
            continue
        key = (start_idx, end_idx+1)
        if key not in hits: hits[key] = set()
        hits[key] |= ids
    return hits

lines = open("data/B.jsonl", encoding="utf-8").readlines()
results = []
import time; t0 = time.time()
for di, line in enumerate(lines):
    d = json.loads(line)
    entities = []
    sec_entity_map = {}
    for si, sec in enumerate(d["full_text"]):
        text = sec["text"]; sec_off = sec["offset"]
        d_hits = dict_lookup(text)
        sec_ents = []
        for (s, e), hpo_ids in sorted(d_hits.items()):
            stext = text[s:e]
            if len(stext) < 3: continue
            hpo = ";".join(sorted(hpo_ids))
            entities.append({
                "identifier": str(hpo), "type": "Phenotype",
                "offset": int(sec_off+s), "length": int(e-s),
                "text": str(stext), "note": None,
            })
            sec_ents.append((sec_off+s, hpo_ids))
        sec_entity_map[si] = sec_ents
    patients = d["patient"]
    pat2phe = defaultdict(list)
    for si, sec in enumerate(d["full_text"]):
        sec_off = sec["offset"]
        sec_end = sec_off + len(sec["text"])
        sec_patients = []
        for p in patients:
            for m in p["mention"]:
                if sec_off <= m["offset"] < sec_end:
                    sec_patients.append(p["patient_id"])
                    break
        if not sec_patients: continue
        for (off, hpo_ids) in sec_entity_map[si]:
            for pid in sec_patients:
                for h in hpo_ids:
                    if h.startswith("HP:") and h not in pat2phe[pid]:
                        pat2phe[pid].append(str(h))
    association = [{"patient_id": p["patient_id"], "phenotype": pat2phe.get(p["patient_id"], [])} for p in patients]
    results.append({"pmc_id": d["pmc_id"], "pmid": d["pmid"], "entities": entities, "association": association})

with open("out/B_v10.jsonl", "w", encoding="utf-8") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False)+"\n")
tot_e = sum(len(r["entities"]) for r in results)
tot_a = sum(len(a["phenotype"]) for r in results for a in r["association"])
print(f"wrote out/B_v10.jsonl n={len(results)} entities={tot_e} links={tot_a}", flush=True)
