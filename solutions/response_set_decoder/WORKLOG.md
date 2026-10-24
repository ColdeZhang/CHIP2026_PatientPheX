# 响应集合解码工作台账

- 目标：检验合法患者代码列表的重复值，是否应按关系集合语义稳定去重，而非整行弃权。
- 分支：`explore/response-set-decoder`，从main `62ffab49`创建，独占本目录。
- 固定变化：统一处理原80 raw，只折叠每CID list中完全相同合法PT值，保留首次顺序。不改原text/token，不补闭合、不修未知值；duplicate JSON keys／未知code／缺CID／类型错／nonEOS／cap仍拒绝。
- 原路径：response-code F1原79严格通过／1弃权及正在运行F0使用的codec均不修改。原79 canonical行必须逐字节保持；原coded-alone预测须用旧记录重建并逐字节对齐。
- 当前：24项边界合同通过，待source/固定计划推送后统一重解80。计划见 `evidence/fixed-plan.json`。
- 唯一质量：新coded-alone完整16文档／39患者／1189实体，共同freeze所有中间后至多1v2，基线已有 .6866189430196797；完全相同则0gold/0v2。不做union/子集/阈值扫描。
- 性质：观察strict失败之后的开发迭代，不是前瞻盲验。0NN、不加载模型/tokenizer、不调用远端，不读F0／原dev／官方。
- 大产物：`/workspace/patientphex-runs/round2-20261023/research/response-set-decoder-20261024/`。关键源码/配置/结果入Git；完整raw已由原路线持久保存，新结果可由固定源CPU重建。

收益严格为正才交root决定F0同政策复验；本线程不自动扩展。
