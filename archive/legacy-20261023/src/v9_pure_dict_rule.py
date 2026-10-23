"""
v9: Pure dictionary entities (v3-style, no NER) + rule-based association (v8-style).
This combines the best of both: high precision entity detection + good task2.
"""
import json, os, sys
os.chdir("/dataset/wanggenyi/patient_phex")
import ahocorasick
from collections import defaultdict

ABBREV = {
    "dm": "HP:0000819", "id": "HP:0001249", "hs": "HP:0002370",
    "t2dm": "HP:0005978", "t1dm": "HP:0100651", "mody": "HP:0004904",
    "cp": "HP:0001257", "asd": "HP:0000750", "adhd": "HP:0007018",
}
FP_STOPWORDS = {"severity", "cut", "niv"}

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
        key = (start_idx, end_idx+1)
        if key not in hits: hits[key] = set()
        hits[key] |= ids
    return hits

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    lines = open(args.inp, encoding="utf-8").readlines()
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

        # Rule-based association: same section patients
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
            if not sec_patients:
                continue
            for (off, hpo_ids) in sec_entity_map[si]:
                for pid in sec_patients:
                    for h in hpo_ids:
                        if h.startswith("HP:") and h not in pat2phe[pid]:
                            pat2phe[pid].append(str(h))

        association = [{"patient_id": p["patient_id"], "phenotype": pat2phe.get(p["patient_id"], [])} for p in patients]
        results.append({"pmc_id": d["pmc_id"], "pmid": d["pmid"], "entities": entities, "association": association})
        if (di+1) % 20 == 0:
            print(f"{di+1}/{len(lines)} {time.time()-t0:.0f}s", flush=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False)+"\n")
    tot_e = sum(len(r["entities"]) for r in results)
    tot_a = sum(len(a["phenotype"]) for r in results for a in r["association"])
    print(f"wrote {args.out} n={len(results)} entities={tot_e} links={tot_a}", flush=True)

if __name__ == "__main__":
    main()
