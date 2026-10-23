"""
纯词典规则管线（v10 逻辑，纯 CPU，无需 GPU 模型）。

实现 v10 的规则链：
  增强词典 AC 匹配 + 扩展缩写 + FP 黑名单 + 短缩写词边界 + 否定上下文过滤
  + 同 section 规则关联。

用法（在归档根目录下）：
  python src/local_predict.py --inp data/B/PatientPheX-V1-B.jsonl --out outputs/local_B_v10.jsonl
  python src/local_predict.py --inp data/train/PatientPheX-train.jsonl --out outputs/local_train_v10.jsonl --self

"""
import json, argparse, os
from collections import defaultdict
import ahocorasick

# v10 扩展缩写（训练集漏报审计得到）
ABBREV = {
    "dm": "HP:0000819", "id": "HP:0001249", "hs": "HP:0002370",
    "t2dm": "HP:0005978", "t1dm": "HP:0100651", "mody": "HP:0004904",
    "cp": "HP:0001257", "asd": "HP:0000750", "adhd": "HP:0007018",
    "sp": "HP:0001257", "mr": "HP:0001249", "mh": "HP:0000707",
    "oa": "HP:0001263", "rp": "HP:0000580", "rd": "HP:0000556",
    "di": "HP:0001249",
}

# v10 FP 黑名单（训练集误报审计得到，非表型词）
FP_STOPWORDS = {
    "severity", "cut", "niv", "anger", "disability",
    "weakness", "mental disorders", "diabetes", "autism",
    "facial features", "cns", "d", "he",
}

# 否定上下文：症状前 30 字符出现以下词则丢弃
NEG = ["no ", "without ", "negative ", "denies ", "absence of ", "resolved"]


def build_automaton(dict_path):
    dd = json.load(open(dict_path, encoding="utf-8"))
    combined = dd["combined"]
    A = ahocorasick.Automaton()
    for surf, ids in combined.items():
        if len(surf) >= 3:
            A.add_word(surf, (surf, set(ids)))
    for abbr, hid in ABBREV.items():
        A.add_word(abbr, (abbr, {hid}))
    A.make_automaton()
    return A


def dict_lookup(text, A):
    text_l = text.lower()
    hits = {}
    for end_idx, (surf, ids) in A.iter(text_l):
        s = end_idx - len(surf) + 1
        if surf in FP_STOPWORDS:
            continue
        # 短缩写必须独立成词，避免命中长词内部
        if len(surf) <= 4 and surf.isalpha():
            before = text_l[s - 1] if s > 0 else " "
            after = text_l[end_idx + 1] if end_idx + 1 < len(text_l) else " "
            if before.isalpha() or after.isalpha():
                continue
        # 否定上下文过滤
        ctx_before = text_l[max(0, s - 30):s]
        if any(n in ctx_before for n in NEG):
            continue
        key = (s, end_idx + 1)
        if key not in hits:
            hits[key] = set()
        hits[key] |= ids
    return hits


def process_doc(d, A, self_eval=False):
    entities = []
    sec_entity_map = {}
    for si, sec in enumerate(d["full_text"]):
        text = sec["text"]
        sec_off = sec["offset"]
        d_hits = dict_lookup(text, A)
        sec_ents = []
        for (s, e), hpo_ids in sorted(d_hits.items()):
            stext = text[s:e]
            if len(stext) < 3:
                continue
            hpo = ";".join(sorted(hpo_ids))
            entities.append({
                "identifier": hpo, "type": "Phenotype",
                "offset": int(sec_off + s), "length": int(e - s),
                "text": stext, "note": None,
            })
            sec_ents.append((sec_off + s, hpo_ids))
        sec_entity_map[si] = sec_ents

    # 同 section 规则关联：本节提到的患者配本节实体
    patients = d.get("patient", [])
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
        for (off, hpo_ids) in sec_entity_map.get(si, []):
            for pid in sec_patients:
                for h in hpo_ids:
                    if h.startswith("HP:") and h not in pat2phe[pid]:
                        pat2phe[pid].append(h)

    association = [
        {"patient_id": p["patient_id"], "phenotype": pat2phe.get(p["patient_id"], [])}
        for p in patients
    ]
    out = {"pmc_id": d["pmc_id"], "pmid": d["pmid"],
           "entities": entities, "association": association}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dict", default="dict/hpo_dict_augmented.json")
    ap.add_argument("--self", action="store_true", help="train 自拟合自检，不要求金标")
    args = ap.parse_args()

    A = build_automaton(args.dict)
    results = []
    for line in open(args.inp, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        results.append(process_doc(d, A, args.self))

    with open(args.out, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    tot_e = sum(len(r["entities"]) for r in results)
    tot_a = sum(len(x["phenotype"]) for r in results for x in r["association"])
    print(f"wrote {args.out} n={len(results)} entities={tot_e} links={tot_a}")


if __name__ == "__main__":
    main()
