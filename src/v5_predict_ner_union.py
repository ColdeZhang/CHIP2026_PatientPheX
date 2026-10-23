"""
v5 pipeline: NER span detection + dictionary HPO linking + relation model.
- NER finds phenotype spans (better recall than dictionary)
- Dictionary provides accurate HPO id linking
- Relation model assigns phenotypes to patients
"""
import json, os, sys, numpy as np, torch
from collections import defaultdict
os.chdir("/dataset/wanggenyi/patient_phex")
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForSequenceClassification
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
    """Return dict: (start, end) -> set of HPO ids."""
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
        if key not in hits:
            hits[key] = set()
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

def get_sentence(text, offset, length):
    start = offset
    while start > 0 and text[start-1] not in '.!?:;=\n\t ' and start > offset - 200: start -= 1
    end = offset + length
    while end < len(text) and text[end] not in '.!?\n' and end < offset + length + 200: end += 1
    return text[start:end].strip()

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ner_model", default="out/ner/final")
    ap.add_argument("--rel_model", default="out/rel/final")
    args = ap.parse_args()

    dd = json.load(open("data/hpo_dict_augmented.json"))
    A = build_dict(dd["combined"])

    ner_tok = AutoTokenizer.from_pretrained(args.ner_model)
    ner_model = AutoModelForTokenClassification.from_pretrained(args.ner_model).cuda().eval()

    rel_tok = AutoTokenizer.from_pretrained(args.rel_model)
    rel_model = AutoModelForSequenceClassification.from_pretrained(args.rel_model).cuda().eval()

    lines = open(args.inp, encoding="utf-8").readlines()
    results = []
    import time; t0 = time.time()
    for di, line in enumerate(lines):
        d = json.loads(line)
        entities = []
        all_spans_info = []
        for sec in d["full_text"]:
            text = sec["text"]; sec_off = sec["offset"]
            # NER spans
            ner_spans = predict_spans(text, ner_model, ner_tok)
            # Dictionary lookup
            d_hits = dict_lookup(text, A)
            # Merge: NER spans + dictionary spans
            all_local_spans = set()
            for s, e in ner_spans:
                all_local_spans.add((s, e))
            for (s, e), ids in d_hits.items():
                all_local_spans.add((s, e))
            # For each span, find HPO id from dictionary
            for s, e in sorted(all_local_spans):
                stext = text[s:e]
                if len(stext) < 3: continue
                # Check dictionary for HPO id
                hpo_ids = set()
                # Exact match
                if (s, e) in d_hits:
                    hpo_ids |= d_hits[(s, e)]
                # Try lowercase match
                sl = stext.lower()
                for (ds, de), ids in d_hits.items():
                    if text[ds:de].lower() == sl:
                        hpo_ids |= ids
                if not hpo_ids:
                    hpo_ids = {"-1"}  # NER found it but no dictionary match
                hpo = ";".join(sorted(hpo_ids))
                entities.append({
                    "identifier": str(hpo), "type": "Phenotype",
                    "offset": int(sec_off+s), "length": int(e-s),
                    "text": str(stext), "note": None,
                })
                all_spans_info.append((sec_off+s, e-s, stext, sec, s))
        # Relation model
        patients = d["patient"]
        pat2phe = defaultdict(list)
        if patients and entities:
            pairs = []
            for (off, ln, stext, sec, s) in all_spans_info:
                sent = get_sentence(sec["text"], s, ln)
                for p in patients:
                    ptext = p["mention"][0]["text"]
                    text_inp = f"{ptext} [SEP] {stext} [SEP] {sent}"
                    pairs.append((p["patient_id"], stext, off, text_inp))
            if pairs:
                texts = [p[3] for p in pairs]
                all_probs = []
                BS = 256
                with torch.no_grad():
                    for i in range(0, len(texts), BS):
                        batch = texts[i:i+BS]
                        enc = rel_tok(batch, truncation=True, max_length=256, padding=True, return_tensors="pt").to("cuda")
                        out = rel_model(**enc)
                        probs = torch.softmax(out.logits, dim=-1)[:, 1]
                        all_probs.extend(probs.cpu().tolist())
                for i, (pid, stext, off, _) in enumerate(pairs):
                    if all_probs[i] > 0.5:
                        for e in entities:
                            if e["offset"] == off:
                                for h in e["identifier"].split(";"):
                                    h = h.strip()
                                    if h.startswith("HP:") and h not in pat2phe[pid]:
                                        pat2phe[pid].append(str(h))
                                break
        association = [{"patient_id": p["patient_id"], "phenotype": pat2phe.get(p["patient_id"], [])} for p in patients]
        results.append({"pmc_id": d["pmc_id"], "pmid": d["pmid"], "entities": entities, "association": association})
        if (di+1) % 10 == 0:
            print(f"{di+1}/{len(lines)} {time.time()-t0:.0f}s", flush=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False)+"\n")
    tot_e = sum(len(r["entities"]) for r in results)
    tot_a = sum(len(a["phenotype"]) for r in results for a in r["association"])
    print(f"wrote {args.out} n={len(results)} entities={tot_e} links={tot_a}", flush=True)

if __name__ == "__main__":
    main()
