"""Fast dict predict v2: supports augmented dict format."""
import json, argparse, re, sys, time
from collections import defaultdict
import ahocorasick

def build_automaton(combined):
    A = ahocorasick.Automaton()
    for surf, ids in combined.items():
        if len(surf) < 3: continue
        A.add_word(surf, (surf, set(ids)))
    A.make_automaton()
    return A

def find_matches_ac(text, A):
    text_l = text.lower()
    hits = []
    for end_idx, (surf, ids) in A.iter(text_l):
        start_idx = end_idx - len(surf) + 1
        hits.append((start_idx, end_idx+1, ids, surf))
    hits.sort(key=lambda x: (x[0], -(x[1]-x[0])))
    kept = []
    cur_end = -1
    for s, e, ids, surf in hits:
        if s < cur_end: continue
        kept.append((s, e, ids, surf))
        cur_end = e
    return kept

def assign_patients(spans, patients, sections):
    sec_ranges = [(s["offset"], s["offset"]+len(s["text"]), s["section_type"]) for s in sections]
    def sec_of(off):
        for so, se, st in sec_ranges:
            if so <= off < se: return st
        return None
    pmentions = []
    for p in patients:
        for m in p["mention"]:
            pmentions.append((p["patient_id"], m["offset"]))
    sec_pat = defaultdict(list)
    for pid, off in pmentions:
        sec_pat[sec_of(off)].append((pid, off))
    for st in sec_pat: sec_pat[st].sort(key=lambda x: x[1])
    span2pid = []
    for sp in spans:
        st = sec_of(sp[0])
        cands = sec_pat.get(st, []) or pmentions
        pid = min(cands, key=lambda x: abs(x[1] - sp[0]))[0]
        span2pid.append(pid)
    return span2pid

def process_doc(d, A):
    entities = []
    all_spans = []
    for sec in d["full_text"]:
        text = sec["text"]; sec_off = sec["offset"]
        ms = find_matches_ac(text, A)
        for s, e, ids, surf in ms:
            g_off = sec_off + s
            g_len = e - s
            hpo = ";".join(sorted(ids))
            entities.append({
                "identifier": hpo, "type": "Phenotype",
                "offset": g_off, "length": g_len,
                "text": text[s:e], "note": None,
            })
            all_spans.append((g_off, g_len, hpo))
    pids = assign_patients(all_spans, d["patient"], d["full_text"])
    pat2phe = defaultdict(list)
    for (off, ln, hpo), pid in zip(all_spans, pids):
        for h in hpo.split(";"):
            h = h.strip()
            if h not in pat2phe[pid]: pat2phe[pid].append(h)
    association = [{"patient_id": p["patient_id"], "phenotype": pat2phe.get(p["patient_id"], [])} for p in d["patient"]]
    return {"pmc_id": d["pmc_id"], "pmid": d["pmid"], "entities": entities, "association": association}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dict", required=True)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=1)
    args = ap.parse_args()

    dd = json.load(open(args.dict, encoding="utf-8"))
    combined = dd.get("combined", dd.get("id2info"))
    if "id2info" in dd and "combined" not in dd:
        # original format
        combined = {}
        for hid, info in dd["id2info"].items():
            for name in [info["name"]] + info["synonyms"]:
                n = re.sub(r"\s+", " ", name.strip().lower())
                if len(n) >= 3:
                    combined.setdefault(n, set()).add(hid)
    A = build_automaton(combined)
    print(f"automaton built, rank={args.rank}, surfaces={len(combined)}", flush=True)

    lines = open(args.inp, encoding="utf-8").readlines()
    mine = [json.loads(lines[i]) for i in range(len(lines)) if i % args.world == args.rank]
    results = []
    t0 = time.time()
    for di, d in enumerate(mine):
        r = process_doc(d, A)
        results.append(r)
        if (di+1) % 10 == 0:
            print(f"[rank {args.rank}] {di+1}/{len(mine)} ent={len(r['entities'])} {time.time()-t0:.0f}s", flush=True)
    with open(f"{args.out}.rank{args.rank}", "w", encoding="utf-8") as f:
        for r in results: f.write(json.dumps(r, ensure_ascii=False)+"\n")
    print(f"[rank {args.rank}] DONE", flush=True)

if __name__ == "__main__":
    main()
