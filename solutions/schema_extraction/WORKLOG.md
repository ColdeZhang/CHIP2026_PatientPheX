# Schema extraction 工作台账

- 目标：为原dev≥.8提供真正新增的患者—表型关系；当前路线没有质量结果。
- 分支explore/schema-extraction，起点main3078f54；只修改本路线。
- E-20261024-SCHEMA-001：公开GLiNER2.5 boundary+relation预训练模型，先CPU接口/数据、精确资产和最短工程，再完整F1固定对照。原F1未读gold，train32只用于已冻结规范化。
- 资产：HF54785d51/代码ca4f6823，ModelScope精确镜像；清单[asset-plan.json](asset-plan.json)。.3现Torch2.7/TF4.50符合上游声明，先测试实际stock库，不修改共享环境。下载无GPU，预算600秒。
- 原GPU队列由root安排；尚无新GPU/模型推断。原dev和官网均不执行。
- 规范化将复用grounded strict train32与public-only41099 Sap bank，不重编码。患者输入固定validation16.input962b76a3，强normalN f33a30f1，Score.676546355；另历史完整支持.686338981作只读参照。
- 输入准备和六项坐标合同通过；首版schema/阈值/完整两端点已冻结，代码尚待真实CPU load。见[实验](experiments/E-20261024-SCHEMA-001.md)。
- .3资产500MiB截断已识别，原partial Range续传中；后续GPU转root预留的.100 GPU1，尚未占卡。
- 最近已核远端检查点8e33710；本阶段保存prepare/run_joint/protocol与失败恢复证据定位，不包含大资产。
- .3资产已全SHA通过；CPU strict load和全164窗原生tokenizer独立IDs合同11.788s通过。未质量/未GPU。root授权.100 probe120+30、满足真实估计后full900+30；.100 runtime合同与复制进行中。
- .3私网transfer首次start_process返回UNKNOWN_OUTCOME；原意图和token已持久化，只读恢复原server状态/port，不重启第二服务。
- .100实际依赖已核，10:37:32独立同SHA资产传输启动proc_ddfd0c60-864e-46b3-891d-15ef1076a625，600+30s，原.3已完成资产不重启。原私网服务状态由span统一恢复；无.100 fetch并发。
- CPU摘要证据[evidence/cpu-readiness.json](evidence/cpu-readiness.json)已本地保存；真正质量仍为0。原dev/官网均未操作。
