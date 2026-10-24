# Adapted quote linking 台账

状态：CPU准备和合成合同已通过，查询草案已固定，待root分配GPU与绑定正式入口。0新增训练/9B推断/GPU/held gold/质量。旧Copy完整负结果保留，不在本线续试。

目标：检验同train32监督适配Sap能否改善普通Grounded原合法引文的映射，唯一primary为 `B4 ∪ [Gadapted ∩ (S∪Q∪C)]`，明确不含旧G自证。最终验收仍为原train64/dev16 v2≥.8；内部F1只决定是否进入固定F0。

约束和成功标准：[预锁decision](decision.json)。完整16/39/1189、保524旧关系与全部非关系字节；全量freeze后至多1新v2/旧0，byteexact0。GPU分配待root；不抢.3或.100当前任务。

- 从main bc8084e7建独立分支 `explore/adapted-quote-linking` /同名worktree；读当前AGENTS/HANDOFF（含H69）及机制地图。
- 已向新Sap owner核实际资产：最终state b0482af6不是HF目录；公共bank fa312ba7由FP32参数+FP16 autocast/CLSfloat+L2产生，不能改model.half冒称同数值。原G完整573合法anchors、254必要queries全部复用，含旧semantic弃答项。
- 下一步：原S/Q/C真实纯集合/继承B审计、新model+bank身份版本、CPU合同及唯一查询成本计划，推送后报root。

- 完整准备manifest7a3a5314：原573anchors/254queries；S505/pureQ416/C395、继承B471⊆B4，支持union595/B外71。普通G逐item策略保留，oldG不进入支持。
- 新policy/新bank provenance及owner证书be945454已绑定；FP32参数＋FP16 AMP/CLSfloat torchL2，不套旧权重SHA或no_task_fit。CPU有区分力合同通过，所有Python AST通过；query/finalize/assemble尚未实际运行。
- 仅查询成本草案bf27989f：254条1538tokens/max15，180+30、300秒admission，60秒成本代理，现成资产无需传输。controller/正式plan/collector/score在root分配后绑定，不自行抢卡；唯一终点不变。最近核远端预锁commit e7d5c903，当前源码检查点随后push。
