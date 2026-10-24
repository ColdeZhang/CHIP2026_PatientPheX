# 未来48／64篇：纯输入规模盘点

2026-10-24 09:04 完成一次 CPU 统计，native 23.247648秒／wrapper 23.896289秒、rc0；CUDA隐藏、单线程、未加载模型、未读取答案／原dev gold、未fit／推断／评分。仅为当前质量若正时估计资源；没有创建新正式训练入口或修改当前训练计划。

| 既有旧SFT无标签输入 | 行／文献 | 候选键／文档患者 | 全文输入tokens | 最长输入 | 超13824训练cap | 超14400推断cap |
|---|---:|---:|---:|---:|---:|---:|
| 原train48，nested32候选 | 204／48 | 1477／126 | 1,678,769 | 14,390，idx19 | 1行 | 0行 |
| 原full64，train48 OOF候选 | 265／64 | 1908／173 | 2,197,146 | 16,245，idx137 | 2行 | 1行 |

只统计 `add_generation_prompt=True` 的输入前缀，不读取或计入 assistant 监督，因此这是完整训练序列成本的下界。现有13,824上限已经不足，后续完整长度／mask和资源需另核，不能据此自动扩cap、预算或迁移。两规模候选生成训练量不同，也不是简单增加相同分布的行数。均无截断／无超过32768，实际模型context262144并非显存可行保证。

48输入来自 `sequence_crf/relation-sft-fold0/inner{1,2,3}/prompts.jsonl`，与原204行训练manifest提示SHA一致；64输入来自 `case_retrieval/qwen35-preparation/full64-relation-cpu-preparation/prompts.jsonl`（b100e137，265行）。全文来自四折现成 input-only 文件。复用已验 `construct.append_reports(training=False)`／legacy marker；候选code_map、患者、原schema／局部证据／其他字段完整保持，删追加后缀可复原原行；完整文章重构及每患者marker通过。原生发布tokenizer、Auto、chat template的ID与decode逐行相同。

首次CPU在分词前2.066999847秒退出：新启动env漏了原runtime的PYTHONPATH，严格版本assert失败，未读取tokenizer或产生计数。独立 `attempt-runtime-env` 仅补原完整env，源码／数据／计划SHA不变，另获120+TERM5新CPU预算；原失败及其窗口保持。恢复严格Torch2.7.1+cu126／Transformers5.20.2／tokenizers0.23.1。原GPU任务及其8400时钟未动。

外存根 `R=/workspace/patientphex-runs/round2-20261023`，完整原件 `R/research/qwen35-fulltext-future-scale-input-census-v1/`：plan cde87549，census.py f448eea6，deployment c16178d2；恢复 `attempt-runtime-env/result.json` SHA **d538d126f7120bb90a6adc1ae636be92543d6ae6e4d56a0d413c98ede620a29e**，101945 B；terminal 728abb25／134 B。主数据与源码已在A800私有新目录逐SHA/fsync保存，原与恢复收件都保留。Git仅存这份摘要。
