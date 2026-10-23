# E-20261023-003：字符近似候选检索与监督排序

状态：进行中，首版实现与合成输入检查。分支 `day1/fuzzy-ranker`；起始 commit `ac8a752`。主线程统一安排数据划分、评测和平台提交，本路线不操作网站。

## 目标、假设与约束

目标是提高四项比赛分数，不要求原创。假设：基于指定 HPO 的近似候选召回能覆盖词形和表述变化，再用训练折监督的稀疏上下文排序降低模糊匹配误报；独立学习患者归属可避免段内笛卡尔分配。

仅使用 HPO v2026-06-23 的 HP:0000118 非废弃后代，以及指定训练折的实体/关联标注。验证输入不读取标签；B 无金标。不得使用旧实现、旧增强词典或旧预测。当前 CPU 单线程，不依赖 GPU，不使用外部人工表型标注数据。

成功标准：在统一 64/16 文献划分上跑通，输出通过覆盖、原文切片及 HPO 检查，并按首日计划与本体精确基线比较；近似分数不能称官方成绩。派生资源仅来自本轮 --train 文件。

## 实际调研与方法取舍

2026-10-23 实际读取以下公开实现：

- [scispaCy candidate_generation.py](https://raw.githubusercontent.com/allenai/scispacy/main/scispacy/candidate_generation.py)：使用 TF-IDF 表示与近邻索引检索知识库别名。读取文件 SHA-256 `5d402c6cabf23f4d00c1cee90e23fb999b8d8b253df222b1d659227e13316a66`。
- [scispaCy linking.py](https://raw.githubusercontent.com/allenai/scispacy/main/scispacy/linking.py)：候选相似度阈值筛选，默认阈值 0.7；并未把字符串相似度当患者关联证据。读取 SHA-256 `350e846a6fb664b8926542979f6b038a1f707f8facd10fde595c5f7474601f57`。
- [QuickUMLS README](https://raw.githubusercontent.com/Georgetown-IR-Lab/QuickUMLS/master/README.md)：SimString 近似匹配、Jaccard/Dice/cosine 等相似度与窗口候选。读取 SHA-256 `277eff6c3f7b203a28d338a558fa00060e343c2fedbd3b10187f69997328fe86`。

采用上述检索思想，重新实现 HPO 专用索引；不下载其过期 HPO 或 UMLS 索引，也不复制第三方实现。使用 sklearn（BSD-3-Clause）、RapidFuzz（MIT）等通用库。

首版流程：

1. 从指定 HPO 重新解析 name、EXACT/BROAD/NARROW synonym、alt_id 和后代范围；添加当前训练折实体表面别名。
2. 扫描最长 8 词窗口，保持原始全局字符坐标；精确及简单复数归一匹配，并对同词头候选计算字符 trigram IDF cosine、编辑相似度和词序无关相似度。
3. 用训练折候选的精确 `(offset,end,HPO)` 标签训练稀疏 LogisticRegression；特征包括检索分数、HPO/训练资源来源、上下文、段落类别、否定、字形及跨文档别名支持。训练特征的别名支持剔除当前文档计数。
4. 按实体置信度解决重叠；对每个候选–患者对训练第二个分类器，使用患者 mention 距离、同句/同段、最近前序指称、竞争患者和上下文。实体–HPO 标签必须与候选 HPO 一致。
5. 输入包含的 entities/association 字段不参与预测；最终只输出实体与给定患者的去重关联。

固定初始阈值：实体 0.60，关联 0.55。它们是首版先验，不是通过验证集反复寻找的最优值。最终 B 运行仅在主线程确认有希望后使用全量训练重新拟合。

## 运行与产物

实现：[run.py](../../solutions/fuzzy_ranker/run.py)；依赖：[requirements.txt](../../solutions/fuzzy_ranker/requirements.txt)。

统一入口：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python solutions/fuzzy_ranker/run.py \
  --train TRAIN.jsonl --input INPUT.jsonl --hpo hp.obo \
  --output RUN_DIR/predictions.jsonl --work-dir RUN_DIR
```

每次保存 `model.joblib`、`run.json` 和预测到仓库外 `/workspace/patientphex-runs/day1-20261023/fuzzy-ranker/`。`run.json` 保存真实命令、数据/HPO/源码/输出 SHA-256、阈值、候选数、耗时和最大 RSS。缓存仅在 `--reuse-model` 且训练/HPO/实现哈希一致时使用。

## 检查与结果

- Python 语法检查已通过。
- 合成边界/否定/患者覆盖检查：待运行。
- 统一验证与 B 预测：等待主线程发布数据路径，尚未运行。
- 官方提交：无，子代理不提交。

## 已知局限与后续判定

模糊召回使用词头阻塞，完全不同的同义表达会漏检；首版没有神经语义编码器。训练集小，候选特征与同文档训练误差可能导致概率过于乐观。否定和重叠处理为简化规则；跨段患者共指仍依赖有限 mention 特征。需要用独立验证逐项检查，失败时区分候选召回不足、链接错误和归属误报。
