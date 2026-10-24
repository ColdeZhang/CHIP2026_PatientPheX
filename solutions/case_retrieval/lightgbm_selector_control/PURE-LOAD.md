# 当前最佳的纯加载复现入口

目标是证明已经保存的模型和冻结输入能重建当前 `0e2c26f8…` 预测，不重新训练或评分。`predict_existing.py` 只加载实际 LightGBM/103列向量器 `803b6467…`，计算原特征，沿原 `.3` greedy＋proper-nested 解码，只替换 a179 模板的实体数组；其他字节原样保留。

唯一 CPU 运行于 2026-10-24 05:20:21–05:20:24 UTC 完成：入口 **3.272秒**、外层 **3.463秒**，rc0。16篇、13,246候选，完整140,228字节输出 SHA 与既有 `0e2c26f8a546af6e8ab343f9883f212792d5b32ad01f4db8d28ac1ee27a2345e` 精确相同，391条关系保持。没有fit、读取gold、调用v2、GPU或官方操作；**分数仍为既有 .748583440，并未重新评分**。原生feature-name warnings保存在外部stderr，固定列序、正类列选择和完整输出校验通过。

所有文件路径由CLI指定：`--contract`、`--model`、`--input-pool`、`--input`、`--relation-template`、`--hpo`、`--dependency-root`、`--runtime-root`、`--output-dir`。需要固定的私有LightGBM4.6.0环境，CUDA隐藏、CPU线程为1。实际命令/环境见[冻结计划](pure-load-results/plan.json)，输入和源码SHA见[合同](pure-load-results/contract.json)，结果见[摘要](pure-load-results/result.json)，[统一终审](pure-load-results/terminal-review.json) `53a91d2f…` 已通过。恢复到不同目录时可修改CLI路径，合同按文件字节/SHA校验；输出目录须为新的目录，不能覆盖原 `actual-v1`。计划中的 `baseline_prediction_source` 名称误指模型，执行合同的 `inputs.model` 正确，原计划与字节均保留，解释见terminal manifest。

这是**固定16篇缓存复现入口**：合同固定文献/候选数、当前模型及期望预测SHA。它不生成上游候选或患者关系，不能称任意新文献的原文端到端系统。源中的两段literal替换helper与原实现AST一致，既有特征/decoder依赖逐SHA复用；没有为“简化”而换用旧97列HGB/greedy入口。

持久依赖分层：原58件包已有当前模型、a179模板、源码闭包与固定wheel；纯input `c0fc…` 在上游a179的72件包中。候选池 `18ac…` 与HPO `a509…` 已放入[唯一补包](pure-load-results/persistent-supplement.json)：16成员/2,908,699字节，archive `6aa98c35…`、manifest `b897bfbd…`。服务器已核全部成员SHA并文件/祖先链fsync；服务器到本地恢复由主线程另核，本文不提前宣称。避免重打已有模型或混入官方训练gold。HPO另有官方固定版本恢复来源；仅在已检查12份备份manifest中未找到它，不能宣称所有服务器都无副本。准确上游资源缺口见[库存说明](pure-load-results/upstream-inventory.md)，完整源/过程在仓库外 `research/lightgbm-selector-readiness/pure-load-readiness/`。
