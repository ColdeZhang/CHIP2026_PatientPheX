# 单词神经候选供给准备

全64无标签输入，仅CPU：逐原token坐标复放不可达的1gram分支，原pun/pos过滤不变；检查全部坐标、原dict覆盖、两公开模型实际unique缓存及tokenizer输入。旧2–8gram源与缓存不修改。

资源与执行冻结见 `plan.json`；`remote_io.py deploy-start/status/collect` 部署及回收，`run_cpu.py` 保证120秒硬限。输出全坐标与词串小缓存仅放外部目录，不进Git。原作者排除unigram的意图未核实，不把策略扩展称为原生默认或bug修复。

初始普查阶段无gold/NN/权重/GPU/fit/质量。后续 `cpu_probe/` 与 `full_cache/` 复用原Bioformer loader完成仅CPU的全13145词缓存；`control/` 固定追加到已审nested四fold基底，完整预测先冻结后唯一评测。四fold均负，直接追加已关闭，不继续扫门槛/换基底或拟合。恢复入口和结果见上级 `experiments/E-20261023-PUBLIC-PHENO-UNIGRAM.md`；大缓存/预测仅在外存。
