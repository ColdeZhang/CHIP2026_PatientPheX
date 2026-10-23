"""
统一评测脚本：唯一评测入口
输入：预测jsonl + 金标jsonl
输出：menF1 / docF1 / micF1 / macF1 / 总Score
"""
import json, sys
from collections import defaultdict

def load_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]

def entity_match(pred_ent, gold_ent):
    """实体匹配：offset+length完全匹配"""
    return (pred_ent["offset"] == gold_ent["offset"] and 
            pred_ent["length"] == gold_ent["length"])

def evaluate_entities(pred_docs, gold_docs):
    """实体级F1 (menF1) 和文档级F1 (docF1)"""
    tp_total = fp_total = fn_total = 0
    doc_f1s = []
    
    for pred, gold in zip(pred_docs, gold_docs):
        pred_ents = pred["entities"]
        gold_ents = gold["entities"]
        
        # 匹配
        matched_gold = set()
        tp = 0
        for pe in pred_ents:
            for i, ge in enumerate(gold_ents):
                if i not in matched_gold and entity_match(pe, ge):
                    tp += 1
                    matched_gold.add(i)
                    break
        
        fp = len(pred_ents) - tp
        fn = len(gold_ents) - tp
        
        tp_total += tp
        fp_total += fp
        fn_total += fn
        
        # docF1: 每个文档单独算F1
        if tp + fp > 0 and tp + fn > 0:
            p = tp / (tp + fp)
            r = tp / (tp + fn)
            f1 = 2*p*r/(p+r) if p+r > 0 else 0
            doc_f1s.append(f1)
    
    # menF1: 全局
    men_p = tp_total / (tp_total + fp_total) if tp_total + fp_total > 0 else 0
    men_r = tp_total / (tp_total + fn_total) if tp_total + fn_total > 0 else 0
    men_f1 = 2*men_p*men_r/(men_p+men_r) if men_p+men_r > 0 else 0
    
    # docF1: 文档平均
    doc_f1 = sum(doc_f1s) / len(doc_f1s) if doc_f1s else 0
    
    return men_f1, doc_f1

def evaluate_association(pred_docs, gold_docs):
    """关联级F1: micF1 (全局微平均) 和 macF1 (每患者宏平均)"""
    tp_total = fp_total = fn_total = 0
    patient_f1s = []
    
    for pred, gold in zip(pred_docs, gold_docs):
        # 构建患者->表型集合
        pred_assoc = {}
        for a in pred.get("association", []):
            pred_assoc[a["patient_id"]] = set(a["phenotype"])
        
        gold_assoc = {}
        for a in gold.get("association", []):
            gold_assoc[a["patient_id"]] = set(a["phenotype"])
        
        # mic: 全局统计
        for pid, gold_hpos in gold_assoc.items():
            pred_hpos = pred_assoc.get(pid, set())
            tp = len(pred_hpos & gold_hpos)
            fp = len(pred_hpos - gold_hpos)
            fn = len(gold_hpos - pred_hpos)
            tp_total += tp
            fp_total += fp
            fn_total += fn
            
            # mac: 每患者F1
            if tp + fp > 0 and tp + fn > 0:
                p = tp / (tp + fp)
                r = tp / (tp + fn)
                f1 = 2*p*r/(p+r) if p+r > 0 else 0
                patient_f1s.append(f1)
            elif tp == 0:
                patient_f1s.append(0.0)
    
    mic_p = tp_total / (tp_total + fp_total) if tp_total + fp_total > 0 else 0
    mic_r = tp_total / (tp_total + fn_total) if tp_total + fn_total > 0 else 0
    mic_f1 = 2*mic_p*mic_r/(mic_p+mic_r) if mic_p+mic_r > 0 else 0
    
    mac_f1 = sum(patient_f1s) / len(patient_f1s) if patient_f1s else 0
    
    return mic_f1, mac_f1

def main():
    if len(sys.argv) < 3:
        print("Usage: python evaluate.py pred.jsonl gold.jsonl")
        sys.exit(1)
    
    pred = load_jsonl(sys.argv[1])
    gold = load_jsonl(sys.argv[2])
    
    men_f1, doc_f1 = evaluate_entities(pred, gold)
    mic_f1, mac_f1 = evaluate_association(pred, gold)
    
    score = 0.25 * (men_f1 + doc_f1 + mic_f1 + mac_f1)
    
    print(f"menF1: {men_f1:.4f}")
    print(f"docF1: {doc_f1:.4f}")
    print(f"micF1: {mic_f1:.4f}")
    print(f"macF1: {mac_f1:.4f}")
    print(f"Score: {score:.4f}")

if __name__ == "__main__":
    main()
