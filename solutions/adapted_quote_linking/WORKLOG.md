# Adapted quote linking 台账

状态：已关闭，完整F1预测与B4逐字节相同、0新增/0删除、0新v2/未读gold，不进入F0。GPU已释放，统一终审入口和小备份已全部完成回读。旧Copy完整负结果保留，不在本线续试。

目标：检验同train32监督适配Sap能否改善普通Grounded原合法引文的映射，唯一primary为 `B4 ∪ [Gadapted ∩ (S∪Q∪C)]`，明确不含旧G自证。最终验收仍为原train64/dev16 v2≥.8；内部F1只决定是否进入固定F0。

约束和成功标准：[预锁decision](decision.json)。完整16/39/1189、保524旧关系与全部非关系字节；全量freeze后至多1新v2/旧0，byteexact0。GPU分配待root；不抢.3或.100当前任务。

- 从main bc8084e7建独立分支 `explore/adapted-quote-linking` /同名worktree；读当前AGENTS/HANDOFF（含H69）及机制地图。
- 已向新Sap owner核实际资产：最终state b0482af6不是HF目录；公共bank fa312ba7由FP32参数+FP16 autocast/CLSfloat+L2产生，不能改model.half冒称同数值。原G完整573合法anchors、254必要queries全部复用，含旧semantic弃答项。
- 下一步：原S/Q/C真实纯集合/继承B审计、新model+bank身份版本、CPU合同及唯一查询成本计划，推送后报root。

- 完整准备manifest7a3a5314：原573anchors/254queries；S505/pureQ416/C395、继承B471⊆B4，支持union595/B外71。普通G逐item策略保留，oldG不进入支持。
- 新policy/新bank provenance及owner证书be945454已绑定；FP32参数＋FP16 AMP/CLSfloat torchL2，不套旧权重SHA或no_task_fit。CPU有区分力合同通过，所有Python AST通过；query/finalize/assemble尚未实际运行。
- 仅查询成本草案bf27989f：254条1538tokens/max15，180+30、300秒admission，60秒成本代理，现成资产无需传输。controller/正式plan/collector/score在root分配后绑定，不自行抢卡；唯一终点不变。最近核远端预锁commit e7d5c903，当前源码检查点随后push。

- root明确授权.3 GPU0唯一254查询、180+30、freshUUID/process门和300秒启动有效期；无需新增模型probe。正式execution-plan ffb0b2bd，admission截止 2026-10-24T16:15:37.255692+00:00；sourceplan先push再唯一start。controller过期合同PASS，在GPU查询/Popen前阻断。原入口多保存实际254IDs后才比较，不改数值。完整收件→版本化normalize→全16/39/1189/B524 freeze/push→唯一F1；完成release通知root/sequence。

- 唯一实际启动proc_dd549ce6-c862-4f5d-a143-af1ecb79d2f9，driver1223586/child1223598，child16:11:28.179715；TERM16:14:28.177424/KILL16:14:58.177424，不延期或重试。恢复只用remote.py observe --directory .../F1-query-v1，禁止第二start；启动证据evidence/actual-start.json。

- 原254 child1223598在16:11:42.086147完整rc0/no-timeout，13.909秒（native12.731），actualrelease16:11:42.095268 GPU0=4MiB/noapps，已交root/sequence。必要全scores/254实际nativeIDs/终态完整收件；原0.78MBqueryvectors在server同SHA18994851持久，未冒称本地已下载。
- 完整573anchors已按新fab55644 policy归一化，无原始响应改动；唯一16/39/1189/B524预测freeze403a67ea，新增0、删0，纯新G249仅无分审计，byteexact=True。先push此freeze再独立score入口最多1新v2/旧0。

- freeze403a67ea/预测5ad5cfe3在d63d8119已push后执行score入口，判byteexact：继承B4 .701325377212172，0newv2/0oldv2/未读gold。严格正条件未满足，不进入F0、不改cut或加入旧G自证。统一终审 `research/adapted-quote-linking-v1/final-review-entry.json`，仓内evidence/F1-result.json同字节。

- 最小持久包28成员/227091B，SHAc3205f40，已写.3 `research/adapted-quote-linking-v1/backups/F1-artifacts.zip`，文件/祖先fsync、全包实际回读和每成员SHA通过。统一入口 `completion-entry.json` 引用原final-review与备份receipt；原query-vectors仍准确标远端持久/未本地回读。当前无自有GPU/collector任务、无质量或训练后继。最近已核远端a3a5f865，此台账自身提交可领先一步。
