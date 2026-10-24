# F-coref readiness：已授权的范围与输入接口

本目录只准备公开权重、CPU原生接口与无标签输入适配；不运行赛事文本NN/评分/训练/GPU。既有研究见外部 `research/public-coreference-feasibility/REPORT.md`。下载600秒、私有依赖准备180秒、native CPU合成检查180秒分别由GNU timeout及入口内部检查约束；失败原样保存，不无限重试。

`census_inputs.py`只读F2/F3/F1的`validation16.input.jsonl`，先拒绝entities/association，再对照SFT32 manifest的文献ID：train32=F2+F3、F1独立。已得train32 87患者/255原mentions、F1 39患者/162 mentions；都没有缺锚点患者或多患者共享同一区间。但32篇有5篇/6个字面alias由多个患者使用，F1有2篇/2个。因此必须按文献+原文区间锚定，不能字面alias广播。完整出处/聚合在anchor-census.json，没有读gold或模型cluster。

`anchor_mapping.py`有四个接口：`original_anchors(doc)`校原文；`text_view(doc)`建立合成全文→原passage全局offset的显式映射；`map_clusters(...)`只接受与原始mention区间精确一致的唯一patient ID；`inject_mentions(...)`在副本中追加新mentions并返回可逆sidecar。无锚点、多患者冲突、跨passage/插入分隔符span都拒绝。新增span不能覆盖原始mention，也不在多个新增span间默默选赢家。原患者ID、原mentions顺序、full_text及实体候选必须保持；任何存在/肯定表型判定仍交原关系器。cluster不等于临床归属，未知不自动改成阴性。

旧9B `prepare_pair_rag.py` SHA16345e096b462ae9ba2fc974bb60d14ff15333b09009b4add1f0a0e1bb23ed97可原样复用，**唯一入口变化是`--input`换成经过sidecar验证的派生input-only文件**。`mark_patients`读新增真实坐标，`patient_anchors`仍先取原mentions（最多3条），`evidence`仍原4段/±500/排序规则；新marker可能改变context优先排序，故未来必须保存每row改变的上下文/顺序，并核`code_map`/患者ID/system/实体候选完全一致。不能称“只改marker而上下文必不变”。本准备没有为赛事文本生成任何NN输出/新提示/预测。

每一后续derived mention应绑定：原输入SHA、拼接view SHA、segment map、固定模型revision/全部权重和tokenizer实际SHA、native源码SHA、native输出SHA、cluster ID、原始anchor区间及patient ID、追加/拒绝原因、派生输入SHA。既有8B18条缓存只作历史证据，不混入该链，也不豁免额外8B参数。

`check_integration_contract.py`已经通过4组合成cluster反例及跨passage间隙检查；它只校适配接口，没有测F-coref语义。`cpu_native.py`后续会原生CPU运行同4条合成原文，保存模型真实cluster及坐标往返；不强迫模型符合手工期望身份、不把合法坐标解释为患者关系质量。

.100共有torch2.7.1/Transformers4.51.3已实导入；只在自有overlay安装spaCy3.8.7/numpy2.2.6及其记录SHA的依赖，使用`spacy.blank('en')`而不下载spaCy神经模型。公开fastcoref的package初始化会额外导入训练用wandb/datasets/scipy，因此仅以namespace加载未经修改的native inference模块；不修改公开forward或共享环境。实际独立参数预期90,526,466，须由native实数核验；与当前9,597,467,990合计9,687,994,456，不默认与NCR同时叠加合规。
