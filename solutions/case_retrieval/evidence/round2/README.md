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
