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
