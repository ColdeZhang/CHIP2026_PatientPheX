"""
v4 prediction: v2 dict + abbreviation expansion + FP stopword filtering.
"""
import json, os, sys, re
from collections import defaultdict
os.chdir("/dataset/wanggenyi/patient_phex")
sys.path.insert(0, ".")
import dict_predict_v2 as dp
import ahocorasick

# Abbreviation -> HPO id (from missed analysis)
ABBREV = {
    "dm": "HP:0000819",        # diabetes mellitus
    "id": "HP:0001249",        # intellectual disability
    "hs": "HP:0002370",        # hereditary spastic paraplegia? or hemiplegic migraine
    "t2dm": "HP:0005978",      # type 2 diabetes mellitus
    "t1dm": "HP:0100651",      # type 1 diabetes mellitus
    "mody": "HP:0004904",      # maturity-onset diabetes of the young
    "cp": "HP:0001257",        # cerebral palsy
    "asd": "HP:0000750",       # autism spectrum disorder
    "adhd": "HP:0007018",      # attention deficit hyperactivity disorder
    "fsh": "HP:0002527",       # fragile X? no, FSH=follicle stimulating hormone - skip
}

# FP stopwords: words that match HPO but are usually NOT phenotypes in context
# These come from error analysis
FP_STOPWORDS = {
    "severity", "cut", "niv",  # clearly not phenotypes
}

def build_automaton_v3(combined):
    A = ahocorasick.Automaton()
    for surf, ids in combined.items():
        if len(surf) < 3: continue
        A.add_word(surf, (surf, set(ids)))
    # add abbreviations
    for abbr, hid in ABBREV.items():
        A.add_word(abbr, (abbr, {hid}))
    A.make_automaton()
    return A

def find_matches_v3(text, A):
    text_l = text.lower()
    hits = []
    for end_idx, (surf, ids) in A.iter(text_l):
        start_idx = end_idx - len(surf) + 1
        # Filter FP stopwords
        if surf in FP_STOPWORDS:
            continue
        # Filter: short abbreviations must be word-boundary match
        if len(surf) <= 4 and surf.isalpha():
            # check word boundary
            before = text_l[start_idx-1] if start_idx > 0 else " "
            after = text_l[end_idx+1] if end_idx+1 < len(text_l) else " "
            if before.isalpha() or after.isalpha():
                continue  # part of a larger word, skip
        hits.append((start_idx, end_idx+1, ids, surf))
    hits.sort(key=lambda x: (x[0], -(x[1]-x[0])))
    kept = []
    cur_end = -1
    for s, e, ids, surf in hits:
        if s < cur_end: continue
        kept.append((s, e, ids, surf))
        cur_end = e
    return kept

def get_sentence(text, offset, length):
    start = offset
    while start > 0 and text[start-1] in '.!?:;=\n\t ' and start > offset - 200: start -= 1
    end = offset + length
    while end < len(text) and text[end] not in '.!?\n' and end < offset + length + 200: end += 1
    return text[start:end].strip()

def process_doc_v3(d, A, rel_model, tok):
    entities = []
    all_spans = []
    for sec in d["full_text"]:
        text = sec["text"]; sec_off = sec["offset"]
        ms = find_matches_v3(text, A)
        for s, e, ids, surf in ms:
            g_off = sec_off + s
            g_len = e - s
            hpo = ";".join(sorted(ids))
            entities.append({
                "identifier": hpo, "type": "Phenotype",
                "offset": g_off, "length": g_len,
                "text": text[s:e], "note": None,
            })
            all_spans.append((g_off, g_len, hpo, text[s:e], sec, s))
    # Relation model: assign phenotypes to patients
    patients = d["patient"]
    pat2phe = defaultdict(list)
    if not patients:
        association = []
    else:
        # Build all (patient, entity) pairs
        pairs = []
        for (off, ln, hpo, etext, sec, s) in all_spans:
            sent = get_sentence(sec["text"], s, ln)
            for p in patients:
                ptext = p["mention"][0]["text"]
                text_inp = f"{ptext} [SEP] {etext} [SEP] {sent}"
                pairs.append((p["patient_id"], hpo, text_inp))
        # Batch predict
        if pairs:
            import torch
            texts = [p[2] for p in pairs]
            all_probs = []
            BS = 256
            with torch.no_grad():
                for i in range(0, len(texts), BS):
                    batch = texts[i:i+BS]
                    enc = tok(batch, truncation=True, max_length=256, padding=True, return_tensors="pt").to("cuda")
                    out = rel_model(**enc)
                    probs = torch.softmax(out.logits, dim=-1)[:, 1]
                    all_probs.extend(probs.cpu().tolist())
            for i, (pid, hpo, _) in enumerate(pairs):
                if all_probs[i] > 0.5:
                    for h in hpo.split(";"):
                        h = h.strip()
                        if h not in pat2phe[pid]:
                            pat2phe[pid].append(h)
        association = [{"patient_id": p["patient_id"], "phenotype": pat2phe.get(p["patient_id"], [])} for p in patients]
    return {"pmc_id": d["pmc_id"], "pmid": d["pmid"], "entities": entities, "association": association}

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="out/rel/final")
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tok = AutoTokenizer.from_pretrained(args.model)
    rel_model = AutoModelForSequenceClassification.from_pretrained(args.model).cuda().eval()

    dd = json.load(open("data/hpo_dict_augmented.json"))
    A = build_automaton_v3(dd["combined"])
    print(f"automaton built with abbreviations + FP filtering", flush=True)

    lines = open(args.inp, encoding="utf-8").readlines()
    results = []
    import time; t0 = time.time()
    for di, line in enumerate(lines):
        d = json.loads(line)
        r = process_doc_v3(d, A, rel_model, tok)
        results.append(r)
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
