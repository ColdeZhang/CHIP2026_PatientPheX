"""
基于词典的 NER + 链接：  1. 加载 HPO dict(id -> name/synonymouss)。  2.对于每个文档，用最长匹配项扫描全文，覆盖 HPO 表面形式。  3. 具有精确偏移/长度和 HPO id 的输出实体。  4. 患者关联：同一切口优先，最近提及的回退。    
"""
import json, argparse, re
from collections import defaultdict

def norm(s):
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s

def build_surface_list(hpo_dict):
    items = []
    for hid, info in hpo_dict["id2info"].items():
        for name in [info["name"]] + info["synonyms"]:
            n = norm(name)
            if len(n) >= 3:
                items.append((n, name, hid))
    # longest first for longest match
    items.sort(key=lambda x: -len(x[0]))
    return items

def find_matches(text, surfaces):
    text_l = text.lower()
    text_l = re.sub(r"\s+", " ", text_l)
    matches = []
    used_spans = set()
    for surf_norm, surf_orig, hid in surfaces:
        pattern = re.escape(surf_orig)
        pattern = pattern.replace(r"\ ", r"\s+")
        for m in re.finditer(pattern, text_l, re.IGNORECASE):
            s, e = m.start(), m.end()
            # skip if overlap with existing
            if any(not (e <= os_ or s >= oe) for os_, oe in used_spans):
                continue
            pass
    matches = []
    used = []  # list of (s, e)
    for surf_norm, surf_orig, hid in surfaces:
        pattern = re.escape(surf_orig).replace(r"\ ", r"\s+")
        for m in re.finditer(pattern, text, re.IGNORECASE):
            s, e = m.start(), m.end()
            if e - s < 3: continue
            overlap = False
            for os_, oe in used:
                if not (e <= os_ or s >= oe):
                    overlap = True; break
            if overlap: continue
            used.append((s, e))
            matches.append((s, e, hid, text[s:e]))
    matches.sort(key=lambda x: x[0])
    return matches

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hpo_dict", required=True)
    args = ap.parse_args()

    hd = json.load(open(args.hpo_dict, encoding="utf-8"))
    surfaces = build_surface_list(hd)
    print(f"surfaces loaded: {len(surfaces)}")

    results = []
    for line in open(args.inp, encoding="utf-8"):
        d = json.loads(line)
        entities = []
        all_spans = []
        for sec in d["full_text"]:
            text = sec["text"]
            sec_off = sec["offset"]
            ms = find_matches(text, surfaces)
            for s, e, hid, txt in ms:
                g_off = sec_off + s
                g_len = e - s
                entities.append({
                    "identifier": hid,
                    "type": "Phenotype",
                    "offset": g_off,
                    "length": g_len,
                    "text": txt,
                    "note": None,
                })
                all_spans.append((g_off, g_len, hid))
        # patient association
        pids = assign_patients(all_spans, d["patient"], d["full_text"])
        pat2phe = defaultdict(list)
        for (off, ln, hid), pid in zip(all_spans, pids):
            for h in hid.split(";"):
                h = h.strip()
                if h not in pat2phe[pid]:
                    pat2phe[pid].append(h)
        association = []
        for p in d["patient"]:
            pid = p["patient_id"]
            association.append({"patient_id": pid, "phenotype": pat2phe.get(pid, [])})
        results.append({
            "pmc_id": d["pmc_id"],
            "pmid": d["pmid"],
            "entities": entities,
            "association": association,
        })
        print(f"pmc={d['pmc_id']} entities={len(entities)} patients={len(association)}")

    with open(args.out, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {args.out}, n={len(results)}")

if __name__ == "__main__":
    main()
