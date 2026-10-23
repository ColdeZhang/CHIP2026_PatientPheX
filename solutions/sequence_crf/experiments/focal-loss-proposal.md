# 固定 focal 配方的最小对照提案

2026-10-23；E-20261023-009。当前只做来源/既有训练记录核对和纯函数CPU校验，未改变冻结BCE入口，未读新留出或原dev标签、未启动GPU。

**判断：这是尚未试过、有明确上游参数出处的一次loss配方假设，值得低成本对照；不是修复已证实的正例塌缩。**

## 来源与差别

固定BioMed模型 `Ihor/gliner-biomed-large-v1.0@f7977121` 的[配置](https://huggingface.co/Ihor/gliner-biomed-large-v1.0/blob/f79771210ab2111a2081ef31620fdb7342f602e9/gliner_config.json)明确记录 `loss_alpha=.75`、`loss_gamma=2`、`loss_reduction=sum`、`label_smoothing=0`。这与此前通用GLiNER论文/0.2.22 Trainer上层默认alpha=-1/gamma=0不同。实际0.2.22 Trainer按其TrainingArguments传alpha/gamma，不会因保存配置含这些值就自动启用。可用仓库/论文没有完整原始训练调用，不能仅凭配置宣称精确重现其历史权重训练。

本路线8份实际GLiNER训练manifest均声明valid-span mean BCE，已完成medium/large/BioMed和353增强未做显式focal；此前18:47研究只是静态澄清，不是负训练实验。保留旧结论的时间/范围，新研究由BioMed配置提供不同依据，不改写历史。

新函数仅计算：`mean_valid[alpha_t × (1-p_t)^2 × BCEWithLogits(z,y)]`，alpha正例.75、负例.25，梯度经过调制因子。沿用实际有效span分母、完整原负例和稳定BCE，绕过H30坏接口。**同时改变类别权重和难度权重，视为一个固定配方；保留mean，故不是上游sum全配方复现。** 相同分母下标量尺度仍变化，可影响clip/Adam和置信校准；不为归因拆参数网格。

## 训练侧证据与不确定性

原nested32有1,125,538有效span，2,236正/1,123,302负（约1:502.37）。计数极不平衡，但medium32末轮训练probe R=.926，BioMed R=1.0；medium在线原始BCE logit梯度绝对量正424.73、负433.34，未见数量导致负梯度压倒或正例塌缩。在线模型/未按batch归一的标量统计不是参数梯度或泛化结论。

focal假设是减少已容易判负跨度的梯度贡献、相对关注难span，可能改变未知表达的学习；也可能降低收敛速度、改变固定.5解码下召回/误报。现有训练拟合已强，不能预先承诺涨分。

## 唯一推荐控制与实际成本

优先使用已准备的`.100 A30` medium-v2，**同runtime从公开基座fresh BCE和focal两臂**：原1,150窗/32篇、10epoch/2,880step、batch4 FP32、encoder1e-5/rest5e-5、wd.01、warmup.05、clip1、seed20261023；只换上述loss配方。不混353增强、不改患者/标签/window。两臂统一epoch10；同384/width12/phenotype/.5解码、原Sap32 .65/.92及原规则关系全链。

原medium32训练在A800（torch2.7.1/TF4.51.3/gliner.2.22），耗519.03秒/峰6.04GB。`.100`增强训练实际torch2.7.1+cu126/TF4.51.3、1,503窗/3,760step耗896.39秒/峰5.98GB；按步数粗估每新臂686.6秒，即11.44min，两臂约23–30min，建议每臂20min硬cap并先单步测定。此为估算，长度分布与checkpoint IO不同。

`.100`没有原1,150窗BCE正式训练；它只重放过A800旧权重：1378跨度键相同，概率有最大1.72e-5浮点差，最终Sap预测byteexact。**推理一致不能当训练一致，不能省略同卡BCE臂后称loss单变量。** 若root另分配原A800且可复核同环境，则只跑一新focal臂对冻结历史BCE约9–12min最省；不新迁移环境以追求名义同版本。BioMed在`.3`有实际同卡BCE可比，但一臂约27.67min且卡正跑上下文控制，故不优先。

先冻结两臂raw与完整预测，再唯一nested16公共v2比较全Score/四项/新增TPFP；这个inner16已有开发用途，不称盲测试。若Score不升直接关闭，不扫alpha/gamma/epoch/decoder，不扩原dev。当前GPU卡/训练预算仍由root决定，尚未实现新trainer或启动训练。

## CPU合同已通过

独立文件 `focal_loss.py` 不导入模型，`check_focal_loss_cpu.py` 仅合成张量。实际CPU torch2.7.0+cu128/Python3.10.20、CUDA未初始化，10组：闭式loss/解析梯度、上游helper直接关键字调用、中等logit梯度一致；±1000 FP32/64有限；gamma0等价加权BCE及禁权时原BCE；padding含NaN/Inf不参与且零梯度；翻转masked标签无影响；不同文档有效数按总数均值；全负/全正尾批；finite-difference gradcheck；无有效span/非二元/非有限有效logit拒绝。

这只证明纯函数合同，未证明模型backward、吞吐或质量。报告[CPU证据](../evidence/round2/focal-loss-cpu.json) SHA `784f741d6d65f3db5b09a106970a50badec3a91e06da32b9b18678811c458707`；完整出处/旧训练计数/不可变快照和成本见[小提案证据](../evidence/round2/focal-loss-proposal.json)。

### 资源决定与条件授权

根已授权最省可比路径。实际A800 GPU2虽空，磁盘只有2,007,732,224B，不足检查点峰值；不删除旧唯一产物，选择`.100 A30 GPU1`两fresh臂，实际初查451.97GB余盘。固定训练共享1800秒、单臂最多1200秒，超时TERM后30秒KILL；唯一final推断/链接每阶段300秒。独立trainer只加loss-choice/fingerprint，BCE及collate函数不变。已冻结[执行计划](../evidence/round2/focal-pair-plan.json)；先实际同解释器纯函数/1150窗CPU和最长focal单步，状态丢弃，成功后无需重复许可。当前CPU作业准备中，尚无新质量结果。

实际同`.100` CPU原1150窗/2236正/1125538valid及strict195175936参数全部通过。最长focal单步0.9634s、4.459GB、有限非零grad.121550，完整子进程9.94s，状态丢弃。正式managed `proc_483790e9-1628-4c71-904e-ec14cad664df` 于20:30:10Z启动，训练共享截止`2026-10-23T21:00:12.063790+00:00`；先BCE后focal，全部参数/原数据固定，唯一终点pending。精确来源/恢复见[启动证据](../evidence/round2/focal-pair-launch.json)。
