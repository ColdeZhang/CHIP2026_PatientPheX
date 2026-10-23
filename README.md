# CHIP2026 PatientPheX 患者表型概念识别

新任务先读 [HANDOFF.MD](HANDOFF.MD)，再读 [工作台账](WORKLOG.md)；执行约定见 [AGENTS.md](AGENTS.md)。实验记录入口为 [docs/experiments/](docs/experiments/README.md)，规则的参考依据见 [工作方法说明](docs/workflow/README.md)。

比赛官方资料已保存至 [docs/competition/README.md](docs/competition/README.md)（2026-10-23 快照）：包含规则、数据字段、评分、提交要求、赛程、完整 A/B/赛后榜单及官方原文。后续可先查本地资料。

接手审查见 [docs/REPOSITORY_REVIEW.md](docs/REPOSITORY_REVIEW.md)：包含复现结果、评测和验证问题、规则与模型缺陷，以及保留和重建建议。下方版本实验结论属于原归档记录，证据核验情况以审查报告为准。

---

## 1. 任务与评分

给定一篇医学全文（含多个患者 mention），需完成两件事：

1. **实体抽取（Task1）**：定位表型短语的字符区间 `offset/length` 并链接到 HPO 编号；
2. **患者关联**：把每个表型归给正确的患者。

官方四项指标（现有 `src/evaluate.py` 存在口径差异，见下方说明）：

| 指标        | 含义                                       |
| --------- | ---------------------------------------- |
| menF1     | mention 级实体 F1（字符区间 + HPO ID 匹配）         |
| docF1     | 每文档 HPO ID 去重集合匹配，全局汇总 TP/FP/FN 后计算 F1 |
| micF1     | 关联微平均 F1（全局汇总）                           |
| macF1     | 关联宏平均 F1（每患者等权）                          |
| **Score** | `0.25 × (menF1 + docF1 + micF1 + macF1)` |

> 口径提示：本地 `evaluate.py` 的 men/doc 都只匹配字符跨度，docF1 还是逐文档 F1 平均；官方 menF1 要求跨度与 HPO ID 同时正确，docF1 则比较各文档的 HPO 概念集合，不看位置。
> 本地脚本在否定、复合实体和空患者集合等处理上也有差异，详见 [本地核对记录](docs/competition/LOCAL_NOTES.md)。B 集无金标，本地分不能直接等同于官方分。
>
> 本体版本提示：比赛要求 HPO v2026-06-23；仓库当前 `data/hp.obo` 为 2026-09-01，后续实验需核对版本。

## 2. 目录结构

```
CHIP2026_PatientPheX/
├── AGENTS.md                              # 稳定执行规则
├── HANDOFF.MD                             # 新任务必读：已确认的坑与避免方法
├── WORKLOG.md                             # 当前状态、任务台账、下一步
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
│   ├── evaluate.py                        # 旧评测脚本，存在官方口径差异
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
└── docs/
    ├── competition/                       # 比赛资料、官方原文与榜单快照
    ├── experiments/                       # 实验索引、模板和关键记录
    ├── review/                            # 接手审计证据与诊断脚本
    ├── workflow/                          # 工作方法的参考来源与适配理由
    └── REPOSITORY_REVIEW.md                # 接手审查与重建建议
```

## 3. 快速开始（纯 CPU，无需 GPU）

环境：Python 3.10+，仅依赖 `pyahocorasick`；本次复现验证使用 2.3.1：

```bash
pip install pyahocorasick==2.3.1
```

在仓库根目录下运行，使用独立产物目录保留旧输出：

```bash
# 每次运行创建独立目录；正式实验按实验 ID / run_id 组织
run_dir="../patientphex-runs/manual-v10-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$run_dir"

# 1) 对 B 集跑 v10 规则，复现旧基线作为对照
python src/local_predict.py \
  --inp data/B/PatientPheX-V1-B.jsonl \
  --out "$run_dir/B.jsonl" \
  --dict dict/hpo_dict_augmented.json

# 2) train 80 篇自拟合自检（仅证明管线可运行，分数不外推）
python src/local_predict.py \
  --inp data/train/PatientPheX-train.jsonl \
  --out "$run_dir/train.jsonl" \
  --dict dict/hpo_dict_augmented.json --self

# 3) 旧评测器训练自检；不能作为官方分或独立验证结果
python src/evaluate.py "$run_dir/train.jsonl" data/train/PatientPheX-train.jsonl
```

已验证：`local_predict.py` 在 B 集输出与原始 `outputs/B_v10.jsonl` 的实体及患者表型集合一致
（6838 实体 / 2317 关联）；关联列表顺序可能不同，不保证文件字节一致。

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
