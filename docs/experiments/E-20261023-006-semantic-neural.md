# E-20261023-006：神经语义概念识别与患者图传播

状态：MiniLM CPU独立验证运行中；SapBERT权重已下载并转换，中转服务器后等待GPU时段。分支 `day1/semantic-neural`，起点 `ac8a752`。目标是提供区别于精确词典、模糊排序、监督 BIO、病例检索迁移的首日方案。

## 假设与首版设计

冻结的语义编码器能将未见过的表型短语映射到指定版本 HPO；在训练折拟合小型 span 判别器过滤泛化词/背景描述，再将提及证据沿患者指称的句图传播，可以获得可提交的独立方案。新 HPO 本体和训练折实体构建候选词表，但最终候选识别与不精确链接使用 dense embedding，而不是字符匹配分数。

主版本模型为 `cambridgeltl/SapBERT-from-PubMedBERT-fulltext`，约110M参数，revision `090663c3ae57bf35ffe4d0d468a2a88d03051a4d`，CLS表示与L2 normalization。MiniLM (~22M、revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`) 的量化ONNX后端仅保留作CPU可行性对照，当前没有用其报告独立效果。HPO v2026-06-23，限制非 obsolete 的 HP:0000118 分支。训练数据只用共享训练折；不使用外部人工标注表型数据集、归档词典/代码/输出或验证标签构建资源。

预设首版：候选由新本体名称/同义词、训练折别名和医学词素结尾 n-gram 产生；语义表型判别器只拟合训练折；新词用最高余弦概念链接，精确匹配采用相同新版词表的概念映射。患者归属使用句间衰减传播，不把同段所有概念机械分给所有患者。首版阈值在运行前固定，验证后仅修正实现错误并记录。

## 已调研与取舍

| 来源 | 核验内容 | 采用或暂缓原因 |
|---|---|---|
| https://github.com/ncbi-nlp/PhenoTagger | 官方 v1.2 README：词典+本体远监督神经识别，无需人工表型标注；提供重建本体词表和训练入口 | 借鉴整体结构；现成模型为 HPO2023-10-09，不能直接假定适配2026-06-23；TensorFlow依赖较重，本轮先不下载其旧分类头 |
| https://huggingface.co/cambridgeltl/SapBERT-from-PubMedBERT-fulltext | UMLS2020AA 自对齐+PubMedBERT，CLS医学实体表示，Apache-2.0 | 主线程分配A30后选为正式首版encoder；不微调其基础参数，仅训练当前比赛span判别器 |
| https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 | 通用句对对比学习、384维、Apache-2.0；官方提供量化 ONNX | CPU对照采用，其训练集不属于外部人工表型标注数据集；模型卡列出的通用句对资源与本任务限制区分记录 |

公开网页已于2026-10-23实际读取，研究原文与模型文件保存在仓库外研究/模型目录；小型来源及哈希见模型 manifest。未复制第三方源码，使用公开模型及标准依赖。

## 环境与最小验证

GPU已由主线程分配：10.5.2.3 的 A30 GPU0（24GB），开始前核验无计算进程；CUDA_VISIBLE_DEVICES固定对应UUID。服务器仅使用项目 semantic-neural 子目录，独立venv只读继承既有torch/transformers。默认CPU单线程。独立 venv 位于 `/workspace/patientphex-runs/day1-20261023/semantic-neural/venv`。

2026-10-23 CPU smoke：ONNX输出为 `(batch, tokens, 384)`；320 个短语（10批×32）单核耗时约0.213秒。该检查只验证资源可行性，不是效果评估。

共享64/16划分与无标签B已由主线程重新取得并冻结，使用 `/workspace/patientphex-data/day1/` 中对应文件；模型拟合只接触train64，验证输入无金标。共同评测器为新 `scripts/evaluate.py` 的公开规则近似，指标不称为官方分。

## 运行与结果

2026-10-23 运行过程：远端直连 Hugging Face 报 `Network is unreachable`，IPv4连接和 hf-mirror 也超时。因此改为本机下载固定版本，再由 MCP 分片传输；大请求曾触发 HTTP413，384KiB 原始块转base64成功。远端独立venv只读继承既有torch2.7.0/transformers4.50.0，新装sklearn1.6.1和pyahocorasick2.3.1，不改共享环境。该环境初始化与网络失败不会计为模型效果失败。

SapBERT原始权重437,955,508字节，使用 `prepare_model.py` 的numpy/safetensors转换为FP16副本218,990,904字节，派生SHA-256为 `dc5e5071ed867afd44cc2fdb21974f2c4ca2288442ac765be353eec80c2e4814`；转换不改变训练内容。GPU尚未占用，由sequence路线先使用、结束后再交接。CPU MiniLM 验证开始，已完成42,575条本体/训练别名编码，单核耗时71.153秒。

外部目录根为 `/workspace/patientphex-runs/day1-20261023/semantic-neural/`：`dev-minilm/` 为CPU对照，`sapbert-model/`为原始权重，`sapbert-fp16/`为转换权重；远端位于10.5.2.3项目挂载的 `semantic-neural/` 子目录。权重、缓存和完整日志均不进入Git。

待运行完成补充实际命令、输入/模型/代码校验、四项指标、格式检查、误差分析、运行成本、预测 manifest 及提交建议。
