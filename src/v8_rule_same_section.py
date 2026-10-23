"""
v8: Rule-based association (same section nearest patient), NO relation model.
This is the v1 rule approach but with v4/v5 improved entity detection.
"""
import json, os, sys, numpy as np, torch
from collections import defaultdict
os.chdir("/dataset/wanggenyi/patient_phex")
from transformers import AutoTokenizer, AutoModelForTokenClassification
import ahocorasick

LABELS = ["O", "B-PHEN", "I-PHEN"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}

ABBREV = {
    "dm": "HP:0000819", "id": "HP:0001249", "hs": "HP:0002370",
    "t2dm": "HP:0005978", "t1dm": "HP:0100651", "mody": "HP:0004904",
    "cp": "HP:0001257", "asd": "HP:0000750", "adhd": "HP:0007018",
}
FP_STOPWORDS = {"severity", "cut", "niv"}

def build_dict(combined):
    A = ahocorasick.Automaton()
    for surf, ids in combined.items():
        if len(surf) < 3: continue
        A.add_word(surf, (surf, set(ids)))
    for abbr, hid in ABBREV.items():
        A.add_word(abbr, (abbr, {hid}))
    A.make_automaton()
    return A

def dict_lookup(text, A):
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

def predict_spans(text, model, tokenizer, max_len=512):
    all_spans = []
    chunk_size = max_len - 50
    for start in range(0, len(text), chunk_size):
        end = min(start + max_len, len(text))
        chunk = text[start:end]
        if len(chunk) < 10: break
        enc = tokenizer(chunk, truncation=True, max_length=max_len, return_offsets_mapping=True, return_tensors="pt").to("cuda")
        offsets = enc["offset_mapping"][0].cpu().numpy()
        with torch.no_grad():
            out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            pred = out.logits.argmax(-1)[0].cpu().numpy()
        cur_start = None
        for i in range(len(pred)):
            p = pred[i]
            if offsets[i][0] == offsets[i][1]: continue
            if p == LABEL2ID["B-PHEN"]:
                if cur_start is not None:
                    s, e = offsets[cur_start][0], offsets[i-1][1]
                    all_spans.append((start+s, start+e))
                cur_start = i
            elif p == LABEL2ID["I-PHEN"]:
                if cur_start is None: cur_start = i
            else:
                if cur_start is not None:
                    s, e = offsets[cur_start][0], offsets[i-1][1]
                    all_spans.append((start+s, start+e))
                    cur_start = None
        if cur_start is not None:
            s, e = offsets[cur_start][0], offsets[-1][1]
            all_spans.append((start+s, start+e))
    if not all_spans: return []
    all_spans.sort()
    merged = []
    for s, e in all_spans:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ner_model", default="out/ner/final")
    args = ap.parse_args()

    dd = json.load(open("data/hpo_dict_augmented.json"))
    A = build_dict(dd["combined"])
    ner_tok = AutoTokenizer.from_pretrained(args.ner_model)
    ner_model = AutoModelForTokenClassification.from_pretrained(args.ner_model).cuda().eval()

    lines = open(args.inp, encoding="utf-8").readlines()
    results = []
    import time; t0 = time.time()
    for di, line in enumerate(lines):
        d = json.loads(line)
        entities = []
        sec_entity_map = {}  # section_idx -> list of (offset, hpo_set)
        for si, sec in enumerate(d["full_text"]):
            text = sec["text"]; sec_off = sec["offset"]
            ner_spans = predict_spans(text, ner_model, ner_tok)
            d_hits = dict_lookup(text, A)
            all_local_spans = set()
            for s, e in ner_spans: all_local_spans.add((s, e))
            for (s, e) in d_hits: all_local_spans.add((s, e))
            sec_ents = []
            for s, e in sorted(all_local_spans):
                stext = text[s:e]
                if len(stext) < 3: continue
                hpo_ids = set()
                if (s, e) in d_hits: hpo_ids |= d_hits[(s, e)]
                if not hpo_ids: hpo_ids = {"-1"}
                hpo = ";".join(sorted(hpo_ids))
                entities.append({
                    "identifier": str(hpo), "type": "Phenotype",
                    "offset": int(sec_off+s), "length": int(e-s),
                    "text": str(stext), "note": None,
                })
                sec_ents.append((sec_off+s, hpo_ids))
            sec_entity_map[si] = sec_ents

        # Rule-based association: assign entities to patients in same section
        patients = d["patient"]
        pat2phe = defaultdict(list)
        for si, sec in enumerate(d["full_text"]):
            # Find patients mentioned in this section
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
            # Assign all section entities to all patients in section
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
