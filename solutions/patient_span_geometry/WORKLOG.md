# 工作台账

2026-10-24：从main `9527768e`独立建分支`explore/patient-span-geometry`。旧PCS8轮与24轮强B保护并集分别.678196/.669185，均低B .686339，已关闭；本线检验新的显式归属几何，不继续加轮数。

当前：CUDA隐藏CPU增量合同已全部通过：零初始旧logit/loss/grad与RNG exact，全部32/16映射；109187763预期参数。下一步固定完整24轮/F1/规范化共享900+30；不新NN探针，不截数据。只采用预写两端点、共同freeze后2v2，旧0；未有本线质量结果。

执行/恢复：外部`research/patient-span-geometry-v1/`，唯一start意图与实际job会保存于该目录。未知MCP返回只恢复同作业，不重启。GPU0须fresh空闲，GPU1外部任务不触碰。
