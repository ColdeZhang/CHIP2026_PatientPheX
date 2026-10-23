# 单词神经候选供给准备

全64无标签输入，仅CPU：逐原token坐标复放不可达的1gram分支，原pun/pos过滤不变；检查全部坐标、原dict覆盖、两公开模型实际unique缓存及tokenizer输入。旧2–8gram源与缓存不修改。

资源与执行冻结见 `plan.json`；`remote_io.py deploy-start/status/collect` 部署及回收，`run_cpu.py` 保证120秒硬限。输出全坐标与词串小缓存仅放外部目录，不进Git。原作者排除unigram的意图未核实，不把策略扩展称为原生默认或bug修复。

无gold/NN/权重/GPU/fit/新预测/质量；下一步由真实规模、缓存与运行环境证据决定。
