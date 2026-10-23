# 第二轮证据索引

保存小型指标、配置、源码/产物哈希、复现命令和少量错误分析。全量预测、模型和原始日志保存在台账指向的仓库外目录；此目录不存凭据或完整中间日志。CPU内部选型、固定dev复评与Qwen负结果均保留对应证据。

- `qwen-download-complete.json`：13个固定版本文件最终size/SHA校验；`llm-smoke-manifest.json`为首个短/长输入GPU可用性证据。
- `llm-full-relations-fold0-*`：全文zero-shot关系负结果（同实体Score .615159→.569358）；包括解析回退、覆盖统计、运行及格式证据，不作为可部署改进。

- `lora-resume-independent-review.json`：真实Adam/scheduler/RNG状态恢复独立审查；连续数值轨迹等价未建立。
- `sft-extract-fold0-formal-manifest.json` / `sft-extract-fold0-training-complete.json` / `sft-extract-fold0-final-checkpoint.json`：train48-only305窗2epoch与最终adapter身份；原始状态/权重外置，推断再做实际流式hash。
- `sft-extraction-raw-paired-summary.json`：原90窗严格paired跨度收益和6条触顶，非Score。
- `qwen-linked-control-summary.json`：零样本实体HPO完整负控制，明确区分固定关系消融与重建k21/.35完整管线。
- `sft-extraction-format-recovery.json`：6失败窗唯一17子窗恢复，101合并请求全合法；补充跨度诊断与原90配对结果分开。
- `sft-linked-filtered-control-summary.json`：SFT完整HPO控制及单一冻结Sap筛选的7组正式内部预测、SHA、分项、失败统计和复现命令；过滤后仅小幅内部收益，未达0.8。
- `sap-filter-control.md` / `sap-filter-classifier-reconstruction.json`：原LR缺失时冻结配方/缓存重建与14,480旧候选全回放，随后只读新跨度评分。

- `patient-set-sft-method.md` / `patient-set-sft-preparation.json`：整患者开放集合方法、合法训练目标、无gold输入、真实token/mask与严格后处理，区别可行性初稿与实际带quote数据。
- `patient-set-training-start.json`：独立数据审查、最长11709token smoke与A800从原base正式2epoch作业/完整配置；运行证据，不是验证收益。
- `extraction-fold1-cpu-preparation.json`：一次固定跨折复验的自身train48/88输入/token预算；实际训练作业更新于台账及外存plan/job。
- `patient-set-training-complete.json`：整患者正式32step完整训练、两个实际checkpoint SHA核验与环境/数据来源；start文件是历史启动快照，当前状态以台账与外存plan为准。
- `patient-set-inference-preflight.json`：最长3条同协议资源/格式分离、实际adapter加载与EOS、完整47×2固定batch3决策；pilot无gold、不当得分。

- `patient-set-paired-control-summary.json`：完整47×2同卡对照与四种raw/grounded负结果；所有失败/空数组保留，实体不变、0错警，未达标。
- `extraction-fold1-fixed-replication.json`：一次固定跨折复验的7组对照、训练/推断/格式恢复/标准化/Sap来源与全预测SHA；实体增益不稳定，不继续阈值扫描。
- `qwen35-source-manifest.json`：固定9B基座15文件来源/size/SHA；来源元数据与实际下载校验分开，不能当得分证据。

- `patient-set-exact-id-diagnosis.json`：无验证gold的原文支持quote/唯一exact别名ID一致性诊断；不修复、不重评，未知不判错。
- `qwen35-runtime.json` / `qwen35-relation-tokenizer-audit.json` / `qwen35-minimal-comparison.md`：CPU可用性、204/58实际新模板预算和有限候选任务提案；尚无新GPU得分。

- `qwen35-interface-preparation.json`：新架构短smoke实现、实际305行CPU mask/选择与9项不变量；GPU步骤明确待验证。
- `qwen35-runtime-wheel-provenance.json`：私有5个包镜像wheel与官方PyPI SHA一致，不修改共享环境。

- `qwen35-download-complete.json` / `qwen35-resource-checkpoint.json`：15文件实际SHA完成；首轮GPU前检他人占用阻断，未启动smoke，资源协调后再试。

- `qwen35-a800-migration.json`：A100卡被占后的只读内网迁移、SHM易失边界/目录权限、隔离runtime真实CPU导入与恢复入口；启动快照，实时状态看台账。
- `qwen35-relation-cost-preparation.json`：完整204行最长5787token、assistant mask/因果loss/12类LoRA meta计数；GPU成本尚未实测。

- `qwen35-full-context-feasibility.json`：全文固定CID/目标患者的纯CPU预算和来源；明确2训练样本超过12288、表示变化/命名空间，未新训或提分。

- `qwen35-smoke-start.json`：A800完整迁移/关闭临时服务后，fresh空闲卡上启动冻结短接口测试的实际job与env；非完成或提分证据。

- `qwen35-loading-info-fix.json`：实际TF5 loading-info set序列化故障的最小修复/源码指纹；v1未到forward，新v2同设置重试，不能当质量负。

- `qwen35-gpu-interface-and-cost.json`：实际短smoke/最长6040推断/5787完整r16单步全部通过，区分资源、格式与质量；正式58新作业与90分钟固定训练预算。

- `qwen35-relation-training-preparation.json`：9B新架构固定204×2epoch入口的真实CPU mask/因果loss/混合层覆盖和7项恢复控制检查；52更新/3warmup，无新GPU resume等价声明。

- `qwen35-base58-control.json`：9B完整58 base原协议Score .633860、18候选回退、0截断，按身份完整回收。
- `qwen35-relation-training-start.json`：fresh204×2固定训练作业、90分钟预算、4.277GB盘前检和独审；历史启动快照，不代表完成。

- `full-context58-preparation.json`：原58只追加给定全文/marker、609段/158mentions实际CPU预算与A30准备/迁移；尚无该控制GPU得分。

- `pair-schema-control-preparation.json`：预设确定性对象包装投影；7合成tests，原raw/原解析不改，待9B base+adapter齐后同规则配对，未提前重评。

- `full-context8b-resource-check.json`：A30最长16142输入实际显存/格式/成本及固定两臂共享45分钟硬预算，未把probe合法性当得分。

- `qwen35-relation-training-complete.json`：正式fresh204×2完整52更新/30.92分钟、allocated/reserved及两epoch实际SHA；原58 adapter已启动，质量待定。
- `full-context8b-paired-result.json`：同A30原58/full58完整负结果、冻结SHA/分项/格式；不是跨硬件比较。
- `full-context9b-cap-equivalence.json`：实际CPU两cap最终encoded输入/decoder全同，支持复用原同A800 base58；全文尚未推断。

- `qwen35-relation-paired-result.json`：原58同设备base→epoch002完整结果、schema次控制和事后错类；SFT .695444仍低rules，原dev未动。

- `llm-feature-tree-preparation.json` / `llm-feature-tree-contract.json`：204严格OOF官方base决策作为四个事实特征，配对HGB固定协议与实际CPU/8合成控制；尚未推理204或拟合树。
- `qwen35-a30-persistent-copy.json`：15文件实际SHA独立复核与源进程/端口关闭，保留持久副本。
- `qwen35-dev-transfer-preparation.json`：原dev64纯输入提示/固定955实体、实际token预算、epoch002复制和probe准备；尚无原dev新分。

- `full-context9b-paired-result.json`：同A800原58/追加完整给定全文负结果、原base cap等价复用及全预测SHA；停止此配置。
- `llm-feature-tree-inference-start.json`：冻结原204官方base纯消息推理作业、25分钟预算和CPU接口独审；双树尚未拟合，不是质量结果。

- `qwen35-dev-transfer-result.json`：原dev64同A30基座/SFT配对与同955实体fullLR；共享组合.711210和原独立检索.625074分开列出。
- `llm-feature-tree-result.json`：204严格OOF官方base完整响应、两固定HGB共同freeze后负结果；未校准阈值/未读原dev。

- `rules-union-qwen35-sft-result.json`：唯一无条件规则∪9B SFT关系并集控制，新增1TP/14FP、内部Score下降，停止不移dev。

- `strict-meta-calibration-cpu-proposal.json`：严格train32/meta16单折校准的来源矩阵、两个缺失CPU CRF16与可复用语义池；仅CPU可行性，未fit/选阈值。

- `strict-meta-calibration-run-plan.json`：运行前冻结的17点F1-only阈值选择、原.5保留与6池来源；实际终点见下项。
- [strict-meta-calibration-result.json](strict-meta-calibration-result.json)：严格train32/meta16选择.30，原full48树外层Score .653192→.669003，仍低同实体规则.711354；CPU44.24秒、冻结顺序与产物SHA，关闭、不移原dev。
- `qwen35-full64-cpu-plan.json`：合法四折265行/6153最长/68更新的新独立协议；cap6400、不继承48adapter，成本probe与60分钟正式预算授权，尚未GPU。

- [fresh full64 正式启动](qwen35-full64-training-start.json)：265×2/68updates，18:55:25 UTC启动、60分钟训练硬cap；仅运行状态，无质量结果。

- [full64 原dev唯一新adapter臂准备](qwen35-full64-dev-single-arm-preparation.json)：同A30单臂900+30，旧三对照按SHA复用；实际权重待训练终点，当前仅CPU准备/10合成检查。源码快照位于`../../operations/full64_dev/`。

- [关系连续置信度只读判断](continuous-relation-confidence.json)：无现成LLM logprobs，训练内4127pair真实CPU预算4.145M tokens；仅后续候选，无新GPU/质量结果。

- [fresh full64正式训练完整终点](qwen35-full64-training-complete.json)：265×2/68updates、40.77min、唯一epoch002 actual SHA1a703b24；A800已释放，尚无质量结果。

- [full64 原dev唯一新臂实际runplan](qwen35-full64-dev-run-plan.json)：实际epoch002/adapter双端SHA与同A30全部15基座/runtime已绑定；固定900+30、64条输入，旧对照不重跑。

- [full64 原dev唯一新臂完整结果](qwen35-full64-dev-result.json)：共享955实体 Score .715759605，较SFT48 +.004549、fullLR +.004435，64/466全合法；预测先freeze、仅新臂评分、GPU已释放，独立终审81493ebe通过。

- [患者身份桥接CPU准备](patient-identity-bridge-cpu-preparation.json)：同4127候选/969正、原384前缀不变；378/579新增目标身份仅结构代理，双512/空bridge独审通过。仅授权下一次成本probe，正式两臂尚未批准。

- [身份桥接唯一A30单步成本](patient-identity-bridge-probe-result.json)：实际.569s/4.096GB allocated，warmup首lr0正确方向梯度且不保存状态，GPU释放；双臂约19.57min仅粗估，未正式训练。

- [身份桥接双fresh正式准备](patient-identity-bridge-formal-preparation.json)：同4127/969训练、无标签1166外层输入；共享训练1500s/每臂900s、各唯一终点推理300s。固定[运行计划](patient-identity-bridge-formal-plan.json)和[双预测freeze后评测计划](patient-identity-bridge-evaluation-plan.json)，尚未启动。

- [身份桥接双fresh正式启动](patient-identity-bridge-formal-start.json)：20:12:09Z、共享训练deadline20:37:08.390Z，已push02e1634/入口独审通过，无质量结果。

- [身份桥接首臂FP16工程失败](patient-identity-bridge-attempt1-failure.json)：无epoch/checkpoint/评分，GPU已释放；原日志/源冻结，后续仅透明BF16数值修复。
