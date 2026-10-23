# CHIP2026 PatientPheX 患者表型概念识别

---

## 1. 任务与评分

给定一篇医学全文（含多个患者 mention），需完成两件事：

1. **实体抽取（Task1）**：定位表型短语的字符区间 `offset/length` 并链接到 HPO 编号；
2. **患者关联**：把每个表型归给正确的患者。

四项指标（`src/evaluate.py`）：

| 指标        | 含义                                       |
| --------- | ---------------------------------------- |
| menF1     | mention 级实体 F1（字符区间匹配）                   |
| docF1     | 文档级实体 F1（区间 + HPO ID）                    |
| micF1     | 关联微平均 F1（全局汇总）                           |
| macF1     | 关联宏平均 F1（每患者等权）                          |
| **Score** | `0.25 × (menF1 + docF1 + micF1 + macF1)` |

> 口径提示：本地 `evaluate.py` 的 men/doc 只校验 `offset+length`；官方平台的 men/doc 会同时校验 HPO ID，
> 因此本地自拟合分不能直接等同于官方分。B 集无金标，官方分以竞赛平台提交为准。

## 2. 目录结构

```
CHIP2026_PatientPheX_v1-v10_archive/
├── README.md
├── data/
│   ├── train/PatientPheX-train.jsonl      # 80 篇训练集（带金标 entities/association）
│   ├── B/PatientPheX-V1-B.jsonl           # 100 篇 B 测试集（无金标，官方提交评）
│   └── hp.obo                             # HPO 本体原始文件
├── dict/
│   ├── hpo_dict.json                      # v1 原始词典（id2info 结构）
│   └── hpo_dict_augmented.json           # v2+ 增强词典（21,155 个 surface forms）
├── src/
│   ├── local_predict.py                   # v10 纯词典规则复现（纯 CPU）
│   ├── evaluate.py                        # 统一评测脚本
│   ├── v1_dict_predict.py                  # v1 纯 HPO 词典 + 就近规则关联
│   ├── v2_dict_predict_ac.py             # v2 增强词典 AC + 就近规则关联
│   ├── v3_train_rel.py                    # v3 PubMedBERT 关系二分类器训练
│   ├── v3_build_rel_data.py             # v3 关系训练数据构建
│   ├── v4_predict.py                      # v4 缩写补全 + FP 过滤（沿用关系模型，未提交）
│   ├── v5_predict_ner_union.py           # v5 NER∪词典
│   ├── v5_train_ner.py                     # v5 PubMedBERT NER 训练
│   ├── v6_predict_ner_rel04.py           # v6 NER+词典+关系模型阈值0.4（失败，官方0.394）
│   ├── v8_rule_same_section.py           # v8 NER∪词典 + 同 section 规则关联
│   ├── v9_pure_dict_rule.py             # v9 纯词典 + 规则关联
│   ├── v10_predict.py                      # v10 原始规则脚本
│   └── build_hpo_dict_v2.py             # 增强词典构建脚本
├── outputs/                               # 各版本 B 集原始输出 jsonl
└── docs/                                  # 竞赛总结、完整方案总结、任务台账
```

## 3. 快速开始（纯 CPU，无需 GPU）

环境：Python 3.10+，仅依赖 `pyahocorasick`：

```bash
pip install pyahocorasick
```

在归档根目录下运行：

```bash
# 1) 对 B 集跑 v10 规则，生成可提交输出
python src/local_predict.py \
  --inp data/B/PatientPheX-V1-B.jsonl \
  --out outputs/local_B_v10.jsonl \
  --dict dict/hpo_dict_augmented.json

# 2) train 80 篇自拟合自检（仅证明管线可运行，分数不外推）
python src/local_predict.py \
  --inp data/train/PatientPheX-train.jsonl \
  --out outputs/local_train_v10.jsonl \
  --dict dict/hpo_dict_augmented.json --self

# 3) 用金标评测（仅 train 有金标）
python src/evaluate.py outputs/local_train_v10.jsonl data/train/PatientPheX-train.jsonl
```

已验证：`local_predict.py` 在 B 集输出与原始 `outputs/B_v10.jsonl` 完全一致
（6838 实体 / 2317 关联）。

## 4. 各版本改动要点（层层承接）

- **v1**：HPO 本体 name + EXACT synonym 建词典，AC/正则字面匹配，实体随机选首个 ID，关联全给首患者。
- **v2**：词典扩到 21,155 surface（UMLS 同义词 + 变体），关联改同 section 就近患者。
- **v3（证伪）**：PubMedBERT 关系二分类器判归属；随机 split dev 0.86 系同篇患者句子泄漏，
  group split 仅 0.32，弃用，关联回规则。
- **v4**：补 9 组缩写（dm/asd/cp/adhd/t1dm/t2dm/mody/hs/id）、FP 黑名单（severity/cut/niv）、
  短缩写独立成词校验、多候选 ID 分号保留；关联沿用关系模型（当时未证伪）。
- **v5（失败）**：PubMedBERT 字符级 BIO NER 与词典并集；80 篇小数据假阳性超过召回，退回纯词典。
- **v6（失败）**：v5 实体 + 关系模型阈值降到 0.4 求召回，低分错配灌入，Score 跌至 0.394。
- **v8**：实体仍 NER∪词典，关联改同 section 规则；men/doc 与 v6 相同（只改关联），证明规则优于模型。
- **v9**：实体也退回纯词典，保留规则关联，precision 回升。
- **v10（达标）**：在 v9 干净底座上扩缩写（sp/mr/mh/oa/rp/rd/di）、扩大 FP 黑名单、
  加否定上下文过滤（症状前 30 字符出现 no/without/negative/denies/absence of/resolved 即丢弃）。
