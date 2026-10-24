# 监督式 SapBERT 链接器

状态：**已获唯一 300+30 GPU 工程授权，源码已准备；正式训练/质量尚未授权。**

目标是检验真实 train32 词面监督是否改善共享 Sap 编码器的概念距离，固定旧强实体跨度，仅替换非精确匹配的 ID。原 F1 强基线 Score `.7286975364690524`；成功门是完整同口径 Score 严格提高，不能用训练检索召回替代。

- 分支 `research/supervised-sapbert-linker`，起点 main `ff7ad16777e3907ed3b4b4231794439017aa5bf8`；只写本目录。
- 查重、供给及固定未执行配方见 [实验记录](experiments/E-20261024-SAP-001.md)；原件在仓库外 `research/supervised-linker-readiness-20261024/owner-v1/`。
- 已实际：本地 train32/public CPU 汇总、F1 输入与旧预测结构统计、6 项纯数学合同；0 模型加载、0 fit、0 新 gold、0 v2、0 remote、0 GPU、0 下载。
- 供给：2222 HPO 行、2246 原子、1160 词面、783 ID；public 非 exact 部分1040原子，旧 Sap top1 639/top8 908。这些均为训练侧供给。
- 应用：16 篇/39 患者/1313 旧实体，保护873 unique exact 和18 ambiguous exact，只允许422实体/303词面进入新链接；保持全部关系。
- 下一步：完成全部1160/41099/303 native IDs合同后，唯一300+30工程。真实初始bank与hard20决定最大候选/长度批次；原warmup两Adam、已有state累积反向、全state保存恢复，状态弃掉。实际成本回报root后再定正式预算。启动双时钟按H69预绑定5分钟有效期，不重发UNKNOWN。
- 实际 .3 初库存15:41:49 UTC：GPU0 4MiB/24160MiB空闲、无应用，项目盘166.65GB；GPU1/2他人进程不碰。来源外部 probe-v1/initial-inventory.response.json。start前另fresh门，旧库存不替代。

2026-10-24：旧 Gmean F1 正/F0 负结案与本线独立。最终旧 F0 终审 `R/validation/grounded-weight-average-F0-final-review/review.json`，SHA `eb46a4ee532db0caa24baf1431259b7d35125e0c9025686174b92cbf5c037b6b`。不因此复活旧配方。
