# 监督式 SapBERT 链接器

状态：**真实工程通过；正式600+30与唯一完整F1端点已授权，源已冻结，尚未start。**

目标是检验真实 train32 词面监督是否改善共享 Sap 编码器的概念距离，固定旧强实体跨度，仅替换非精确匹配的 ID。原 F1 强基线 Score `.7286975364690524`；成功门是完整同口径 Score 严格提高，不能用训练检索召回替代。

- 分支 `research/supervised-sapbert-linker`，起点 main `ff7ad16777e3907ed3b4b4231794439017aa5bf8`；只写本目录。
- 查重、供给及固定未执行配方见 [实验记录](experiments/E-20261024-SAP-001.md)；原件在仓库外 `research/supervised-linker-readiness-20261024/owner-v1/`。
- 已实际：本地 train32/public CPU 汇总、F1 输入与旧预测结构统计、6 项纯数学合同；0 模型加载、0 fit、0 新 gold、0 v2、0 remote、0 GPU、0 下载。
- 供给：2222 HPO 行、2246 原子、1160 词面、783 ID；public 非 exact 部分1040原子，旧 Sap top1 639/top8 908。这些均为训练侧供给。
- 应用：16 篇/39 患者/1313 旧实体，保护873 unique exact 和18 ambiguous exact，只允许422实体/303词面进入新链接；保持全部关系。
- 下一步：完成全部1160/41099/303 native IDs合同后，唯一300+30工程。真实初始bank与hard20决定最大候选/长度批次；原warmup两Adam、已有state累积反向、全state保存恢复，状态弃掉。实际成本回报root后再定正式预算。启动双时钟按H69预绑定5分钟有效期，不重发UNKNOWN。
- 实际 .3 初库存15:41:49 UTC：GPU0 4MiB/24160MiB空闲、无应用，项目盘166.65GB；GPU1/2他人进程不碰。来源外部 probe-v1/initial-inventory.response.json。start前另fresh门，旧库存不替代。

2026-10-24：旧 Gmean F1 正/F0 负结案与本线独立。最终旧 F0 终审 `R/validation/grounded-weight-average-F0-final-review/review.json`，SHA `eb46a4ee532db0caa24baf1431259b7d35125e0c9025686174b92cbf5c037b6b`。不因此复活旧配方。

CPU v1 在0模型/0CUDA阶段拒绝合法整数buffer：199个F16权重＋`embeddings.position_ids` I64[1,512]。只读真实header核过，修正仅接受该已知buffer及其合法loader unexpected；原v1包/rc1保留，独立v2部署，未消耗GPU窗口。ops层原返回0而内层rc1分开记录，现保留结果后assert内层exit。

15:48:39.262891→15:48:58.366413唯一GPU工程完整19.104秒，4FB/2Adam/197非pooler全部更新，793state张量实比，峰5.218GB；GPU0真实4MiB/noapps释放。完整native1160/41099/303为6.319秒，0截断。必要6小metadata已收，1.309GBprobe state仅远端保留且禁止续训。正式按已授权fresh3ep219/4bank/finalACK/303，同源核心AST一致；成本273.263代理、600+30共享时钟，动态候选增长披露。实际入口 `formal/ops.py deploy|start|observe|collect`，完整后 `formal/quality.py formal/quality-plan.json`，无原dev/官网。质量gold只定位已有正确文件名/SHA，冻结前不读取。

正式v1 15:54:09.943514→15:54:42.500011实际32.556秒rc1，epoch1完整73/1160/9.235s、epoch2候选已建，后续batch的AMP梯度非有限；前向loss有限，final state尚未产生。原GPU已释放，原source/namespace保留。root授权独立fresh数值恢复：仅GradScaler初值65536→4096及失败前批次/finite诊断，FP16/目标/seed/LR/3epoch219均不改，仍原16:04:09.943514 TERM/+30绝对时钟，不默默跳批或复用失败Adam。源码 `formal_recovery/`；成功也不声称证明原具体根因。数值非有限实际实现为整链拒绝，区别于旧提案“单条保留旧ID”；全finite输出时替换公式不变。
