# 2026-10-23 接手审计证据

> 历史证据，已冻结归档。以下入口仅用于复核旧项目；新实验从活动区重新建立。`manifest.json` 的 `source_inputs` 路径相对 `archive/legacy-20261023/`，`artifacts` 路径相对本目录；原始 JSON、诊断脚本及其校验值保持不变。

结论和解释见 [仓库审查报告](../../REPOSITORY_REVIEW.md)。这些文件保存本次审查的观察结果和诊断方法，供复核；不是正式评测器或新版本预测方案。

## 文件索引

| 文件 | 内容 |
|---|---|
| [evaluation_results.json](evaluation_results.json) | 原评测分数、公开规则近似诊断、误差分解、敏感性分析及极小反例 |
| [rule_audit.json](rule_audit.json) | 边界、缩写、否定、黑名单及患者关联归因与例子 |
| [rule_ablations.json](rule_ablations.json) | 八组规则消融的 TP/FP/FN；只在训练集回看 |
| [dictionary_overlap.json](dictionary_overlap.json) | 基础/增强词典与训练标注词面的重合统计 |
| [abbreviations.json](abbreviations.json) | 缩写硬编码 ID、词典原始 ID、训练金标及长度过滤情况 |
| [relation_samples.json](relation_samples.json) | 关系样本划分重建、文档重叠、重复输入及标签问题 |
| [model_diagnostics.json](model_diagnostics.json) | 同一关系样本诊断、NER 末尾解码 stub 反例、输入校验值 |
| [output_audit.json](output_audit.json) | 全部历史输出的结构、坐标、患者、HPO 检查及 CPU 复现记录 |
| [manifest.json](manifest.json) | 被审查源码、数据、词典、输出和诊断文件的 SHA-256 |

## 统计口径

- **所有有标签的分数来自 80 篇训练集**，不构成独立验证或 B 榜成绩。增强词典新增词条与该训练集标注高度重合。
- `evaluation_results.json` 处理 NO、复合 ID、空患者；纯 `-1` 实体按跨度匹配，文档级使用原文，并报告其他假设的敏感性。使用全部输入段落，官方精确正文范围和 `-1` 文档计分细节未获得。
- `rule_ablations.json` 使用全部段落，排除 NO，拆分复合 ID，以严格 `(offset,length,ID)` 集合比较；`-1` 不做通配匹配。故其计数与上一文件略有差异，不能混用。
- `output_audit.json` 的 HPO 分支及废弃项检查基于仓库 **2026-09-01** 本体，并解析 `alt_id`，不能代替官方指定 2026-06-23 版本的判定。`COMP_INT`、`AUTH_CONT` 等仅按元信息类别计数，不把它们全部认定为官方 FP。
- 词典新增 surface 全部与训练标注重合是观察事实；缺少构建记录，不能据此证明作者的完整处理过程或外部资源来源。

## 重跑诊断

以下命令先从仓库根目录进入归档根，再执行旧诊断。CPU 复现使用 Python 3.12.14、`pyahocorasick==2.3.1`；不需要 torch、transformers、模型文件或网络推理。可在现有 Python 环境中安装 pyahocorasick。脚本默认把生成结果写到 `/tmp`，不会覆盖归档已有预测或本目录快照。

```bash
# 从当前仓库根目录进入冻结的旧项目；禁止在归档内写入 Python 缓存
cd archive/legacy-20261023
export PYTHONDONTWRITEBYTECODE=1

# 现有规则管线复现
python src/local_predict.py --inp data/B/PatientPheX-V1-B.jsonl --out /tmp/review_B.jsonl --dict dict/hpo_dict_augmented.json
python src/local_predict.py --inp data/train/PatientPheX-train.jsonl --out /tmp/review_train.jsonl --dict dict/hpo_dict_augmented.json --self

# 原评测器自检；请保留其与官方指标的区别
python src/evaluate.py /tmp/review_train.jsonl data/train/PatientPheX-train.jsonl

# 评分诊断：结果 /tmp/patientphex-evaluation-audit/results.json
PYTHONDONTWRITEBYTECODE=1 python docs/review/2026-10-23/scripts/diagnose_evaluation.py

# 规则归因和消融；消融预测另存 /tmp/patientphex_rule_ablations/
PYTHONDONTWRITEBYTECODE=1 python docs/review/2026-10-23/scripts/audit_rules.py > /tmp/review_rule_audit.json
PYTHONDONTWRITEBYTECODE=1 python docs/review/2026-10-23/scripts/ablate_rules.py > /tmp/review_rule_ablations.json

# 历史输出结构检查：结果 /tmp/patientphex-output-audit.json
PYTHONDONTWRITEBYTECODE=1 python docs/review/2026-10-23/scripts/audit_outputs.py

# 关系样本重建和 NER stub；只在内存中修正旧路径和抽取待测函数
PYTHONDONTWRITEBYTECODE=1 python docs/review/2026-10-23/scripts/diagnose_models.py --repo . --out /tmp/review_model_diagnostics.json
```

诊断脚本为本次审查的辅助代码，最初保存审计证据时仅调整仓库定位方式及临时目录创建；本次整体搬迁未修改脚本，保留的目录层级使其相对路径仍指向旧项目归档根。它们不修改项目源码。实体或关联输出中集合的遍历顺序、错误样例的选择顺序可能随 Python 进程变化；比较时应使用集合语义和汇总统计。

`output_audit.json` 的 `_verification` 字段记录单独执行的 CPU 重跑对比，结构检查脚本本身不重新生成该补充字段。诊断结果适用的原文件版本见 manifest；未来修改模型或数据后应另存新审计日期，不覆盖本次证据。
