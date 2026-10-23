# E-20261023-005：病例记忆检索与患者标签迁移

状态：CPU 首版完成，GPU 检索增强版运行准备中；2026-10-23。分支 `day1/case-retrieval`，从共同基点 `ac8a752` 创建。所有本地分数将标为近似独立验证，不冒充官方结果。

## 问题、约束与成功标准

目标是用训练病例中的真实表型语境和患者归属作为可检索证据，减少纯共现规则的混人错误。首版采用稀疏 kNN 病例记忆；不要求原创，优先验证成熟检索方法能否提高官方四项分数。

约束：仅用官方训练折标注建记忆；HPO 固定 v2026-06-23，限制 HP:0000118 分支；不使用外部人工标注表型数据。CPU 默认单线程。主线程已分配远端 A30 GPU1（24 GiB）用于 Qwen3-8B 推理，其他卡不使用。验证输入不含答案，B 只作推理与格式检查。

成功标准：完成可复现的独立验证、全部输入和患者覆盖、合法字符位置及 HPO；按首日统一门槛比较总分与分项，决定是否提交。无收益或失败也保留原因。

## 已实际查阅的成熟方法

访问日期均为 2026-10-23，原文 HTTP 200；未复制大型网页、模型或第三方数据进仓库。

| 来源 | 确认内容 | 本方案采用与限制 |
|---|---|---|
| [BioSyn README](https://github.com/dmis-lab/BioSyn) / [论文](https://arxiv.org/abs/2005.00239) | 同义词边际化、迭代候选检索；MIT | 借鉴医学实体链接的候选检索与同义词记忆思想；未使用其外部标注集或训练权重 |
| [scikit-learn 最近邻说明](https://scikit-learn.org/stable/modules/neighbors.html) | 基于实例的最近邻预测 | 以稀疏病例上下文和患者相对位置特征检索训练实例，再迁移关联标签 |
| [官方 PatientPheX](https://github.com/DUTIR-BioNLP/PatientPheX) | PhenoTagger + Qwen3-8B baseline，模型不超过 10B | 后续 GPU 可用时，将病例检索证据供 LLM 判定；当前不把尚未运行的 LLM 当成果 |
| [Qwen3-8B 模型卡](https://huggingface.co/Qwen/Qwen3-8B) | 8.2B 参数，Apache-2.0，支持关闭 thinking | 满足参数上限；尚未下载或运行，不声明其本地成绩 |

## 首版方法

1. 从指定版本 HPO 的有效后代名/同义词，以及仅训练折的标注词面构建候选入口。
2. 用原文字符坐标匹配候选；从训练病例实际出现的边界、归一化 ID 与上下文作实例记忆和歧义消解，避免直接取任意 ID。
3. 将每个患者—候选概念的证据组织为病例片段：目标/其他患者指称、最近患者、同句/同段、章节与原句语境。利用相似训练实例的关联标签进行 kNN 投票。
4. 输出指定患者的去重概念集合，记录实际数据/代码/配置版本及产物 SHA-256。

与其他路线的实质区别：患者关系由训练病例记忆的最近邻标签决定，而不是只按同段共现、全给最近患者或训练一个参数分类器。

## 运行与结果

共同评测已 cherry-pick 主分支 `90d52d0`，本分支对应 `abaa18a`。数据为共享 seed=20261023 的 64/16 文献划分；训练 173 位患者，验证 36 位。输入来源和划分清单见 `docs/data/`。

### CPU 首版

代码初始提交 `ca6ef42`；Python 3.12、numpy 2.3.5、scipy 1.17.0、scikit-learn 1.8.0；无训练权重。运行参数固定 k=11、每个训练文献最多 2 个近邻、关系阈值 0.48，没有用 dev 搜索阈值。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python solutions/case_retrieval/run.py \
  --train /workspace/patientphex-data/day1/train64.jsonl \
  --input /workspace/patientphex-data/day1/dev16.input.jsonl \
  --hpo /workspace/patientphex-data/hp.obo \
  --output /workspace/patientphex-runs/day1-20261023/case-retrieval/dev-cpu/predictions.jsonl \
  --work-dir /workspace/patientphex-runs/day1-20261023/case-retrieval/dev-cpu
```

| 运行 | menF1 | docF1 | micF1 | macF1 | 总分 | 成本 |
|---|---:|---:|---:|---:|---:|---|
| CPU kNN，16 篇独立验证，approx-v1 | 0.731657 | 0.749220 | 0.546985 | 0.435663 | 0.615881 | 含建库 2.066 秒，单 CPU 线程 |
| 完整 80 篇拟合 → B 100 篇 | 无金标 | 无金标 | 无金标 | 无金标 | 未评测 | 9.569 秒，单 CPU 线程 |

验证 843 个实体、266 个关联，全部 16 文献/36 患者；B 5466 个实体、1745 个关联，全部 100 文献/244 患者。两份输出均通过统一格式/字符坐标/HPO/患者覆盖检查，无错误或警告。4 个行为反例检查通过（词内匹配、全局坐标、否定作用域、患者专属相对位置，以及复合 ID 规范化）。

首版主要短板在患者关联，尤其 macro F1 低于实体项。CPU kNN 是可用候选，不据此否定病例检索路线；下一版保留相同实体输出，仅替换关联模块，以区分收益来源。

### GPU 检索增强版（进行中）

采用实际查阅并核实参数的 `Qwen/Qwen3-8B`（8.2B，Apache-2.0），BF16、单张 A30、关闭 thinking、greedy 解码。按全文病例 TF-IDF 相似度检索 2 个**训练折**标注示例，输入给定患者、候选概念及临床原文，生成患者→候选代码映射；代码只能映回已有实体候选，不允许凭空造 HPO 或坐标。无效 JSON/遗漏患者会显式记录并回退 CPU 关联，不能静默当成功。

远端挂载 `5a9955da-5240-41cc-9e52-05d3d8da8a3c`，项目内 `case-retrieval/`，GPU UUID `GPU-fc5d0525-3ec1-074a-9fa3-35d92a9da539`，启动前已核对空闲。创建环境时发现系统 `ensurepip` 缺失；改为私有 `venv --without-pip` + 官方 get-pip，自有目录安装，不修改已有环境。首次环境作业 `proc_59da4887-66d1-42dd-bf1e-3f78f59fcc3b` 因此失败；恢复作业 `proc_6dbc492f-8bed-42de-abab-3bcf56d16481` 正在安装依赖和下载模型。未启动正式 GPU 预测，未提交平台。

### 产物与版本

大产物统一位于 `/workspace/patientphex-runs/day1-20261023/case-retrieval/`，完整输入/代码/输出哈希和成本在各运行 `manifest.json`；评测与校验为相邻 `metrics.json`、`validation.json`。

- `dev-cpu/predictions.jsonl` SHA-256：`7835dcc35003c52939becb3b76f5aaedef3e7d34c38493eac9149236f950a91f`。
- `B-cpu/predictions.jsonl` SHA-256：`f558c6d6843a50d59ab120ec5ed74b36a1a6426ffccf89f5b5267e6a418f6b8d`。
- 指定 HPO SHA-256：`a5092cbdf605f568403cf7380d9173014015692433b2cc631bc5c1b053876b1b`。
- CPU 结果可由上述命令重建；模型与全量预测不写入 Git。官方分数尚不存在。
