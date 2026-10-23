# CRF 序列标注工作台账

实验 `E-20261023-009`，分支 `round2/sequence-crf`，初始基点 `00f7da8`。当前已同步共享主分支 `a0e9d0d`，干净检查点正常合并 `8a11b87` 已推送；只更新共享文档/备份脚本，方案和评测源码不变。新任务还应读取 `git show main:HANDOFF.MD` / `main:AGENTS.md`，再看本页。

## 目标与可信最佳结果

- 原 train64/dev16、指定 HPO 2026-06-23、公共 evaluator v2 all/text，**Score ≥ .8** 才申请主线程验收合并。本轮不上传官方。
- 本路线原政策 dev **.711324992**（独审PASS）：原CRF+Sap/LR基线保留，增加固定train48 GLiNER终点、train64 Sap筛选跨度，再沿既有LR；源码`4adf9da`，预测`gliner-dev-transfer/assembled/full-lr.jsonl`。对应仅增实体不改关系控制.706111684；独立train64 LR重拟合、三输出/全部指标逐字节复现，格式/HPO零错警。前一独审最佳.700635947及完整证据继续保留。
- 跨路线Pheno实体-only组合候选 **.712586161** 已独审PASS；固定nested48候选.712146627也已独审PASS。两项都只有温和增益，mention分项有下降，均未达.8。
- 内四折规则关系优于 LR，原 dev 相反；LR/rules 都评过原 dev，该选择使用过 dev，不能声称盲验证。尚未达到 .8。

## 正在执行与恢复入口

1. **共指控制全部完成**：fold0 72窗严格定位新增18mentions；原规则/旧alias/offset-only三路输出与原baseline逐字节相同，Score仍.711353522。独立重放10产物一致，18项语义阅读17支持/1输入身份未决，未按复核结果过滤。见 [共指索引](evidence/round2/coreference-fold0.json) 与 [独立审查](evidence/round2/coreference-independent-review.json)。
2. **同adapter SFT小幅提升**：仅changed29重推，完整相同29行复用旧响应且实际权重/decoder/runtime核验一致；固定947实体，.674333431→.686852935，净TP+4/FP-1，仍低于rules .711353522。实际作业已结束，GPU2复查释放，无待恢复GPU任务。见 [SFT结果与完整恢复入口](evidence/round2/coreference-sft-results.json)。
3. **GLiNER训练内控制结案**：nested32→16固定fresh2ep/10ep的纯span非重叠组合为.700614/.702135；固定10ep+Sap全链.644762→.653745，69新增=42TP/27FP，四项均升，独审PASS。见[2ep](evidence/round2/gliner-supervised-results.json)、[10ep](evidence/round2/gliner-long10ep-results.json)、[固定链接](evidence/round2/gliner-link-results.json)。
4. **outer48→fold0同政策已完成**：A30 fresh10/4360steps耗996.64秒、峰值6.028GB；零样本/终点预测先冻结。固定原947实体+Sap筛选新增26项（16TP/10FP），关联7TP/4FP，Score **.711353522→.721504431**、四项均升，纯span/来源/完整输出重放独审均PASS。只保留final模型+latest，实际余477.22GB。详见[终点结果/完整恢复](evidence/round2/gliner-outer0-results.json)，外存gliner-outer0-a30/、gliner-link-outer0/。
5. **原dev迁移两臂已完成**：明确span48原文/标签是linker64子集且dev隔离；不重训64、不扫阈值。1103原始跨度经固定Sap等处理新增32实体（22TP/10FP），entity-only **.706111684**；同LR新增关联12TP/9FP、未移除旧关系，full-LR **.711324992**。两臂17:46:38Z统一冻结、零错警，独审PASS。[结果及全部恢复入口](evidence/round2/gliner-dev-transfer-results.json)；GPU1于17:45:11Z交检索、GPU0于17:46:11Z释放，无活动GPU。等待根预先固定其他有效候选组合，不自行扫顺序/阈值。
6. **固定GLiNER→Pheno组合已结案**：只追加12实体（4TP/8FP），保旧关系臂 **.712586161**；同LR臂 **.712070726**（新增2TP/5FP）。两臂独审PASS；mention分项略降，只有小总分收益，不扫组合顺序/阈值。[完整结果](evidence/round2/gliner-pheno-joint-dev.json)。
7. **全train64固定10已完成、未改善**：fresh64/2372窗/5930steps耗1300.262秒，峰6.021GB，最终model e7985761、真实Adam/RNG state8026c3a8再哈希通过，仅保epoch10+latest。原48同runtime候选byteexact。64主fullLR **.705869445**（较48下降.005456），entity-only.705271539；预先固定nested次臂.706234066，仍较差。所有输出18:18:59Z统一freeze后评，0错警；模型跨度净+2TP/+3FP但关系-1TP/+9FP，不能因全量训练就替换旧模型。GPU于18:18:39Z fresh核空交本体，实余217.22GB，本次任务已结案，后续large-v2任务见第9项。新终点独审PASS、三预测/全部指标独立复现；[完整结果与恢复入口](evidence/round2/gliner-full64-results.json)。
8. **固定嵌套实体控制完成**：nested57字节复现2d24bb99；同proper containment/HPO不交政策，outer48追加31（12TP/19FP），.721504→.723440；原dev48追加35（10TP/25FP），.711325→.712147，mention分下降而doc分升，所有旧关系不变、0错警，三路独立重放PASS。只保留温和收益，不扫规则；64主终点原政策及预声明nested次臂已统一freeze/评，均未超过48。[结果](evidence/round2/gliner-nested-control-results.json)。

9. **large-v2 固定终点完成、不升级**：fresh train48/1744窗/459,495,168参数、固定10ep4360steps耗2279.35秒，峰13.34GiB；工作区中断未重启远端。base/final/raw/Sap均exit0，19:26:36Z完整输出先freeze，格式0错警。固定原947实体+Sap/rules新增24项，Score **.720152028**，较原.711354升.008799但低于medium48 .721504431（-.001352403）；mention/micro较medium升，document/macro降。纯span非重叠93TP96FP、union .769904，不能拿纯span收益代替完整Score。GPU0 19:26:26Z fresh空，实际余203.93GB。保留medium，不扫参/不自行迁移原dev；终点CPU重放及实际谱系独审均PASS；root关闭本次替换、不迁移原dev。[完整结果/源码/全部恢复入口](evidence/round2/gliner-large-terminal-results.json)，外存gliner-large-v2/terminal-summary.json；原训练与最后链接managed均已结束。

10. **GLiNER-BioMed 仅CPU准备**：固定f7977121权重/匹配tokenizer216.88秒完成全SHA，CPU strictload477,981,696参数/128003词表及1744窗28.58秒全部通过，CUDA未初始化、实际独审PASS90b615df。见[预检结果](evidence/round2/gliner-biomed-interface-results.json)。large结案后，19:28:25Z fresh核空启动唯一nested16零样本/.5/phenotype，同旧Sap32阈值：proc_95858070-d380-4fea-bce1-f44c2712be42，推断300s+链接300s，绝不新训练/扫参/原dev。外存gliner-biomed-nested/{job.json,state.json,commands.json,run_fixed.py}为恢复入口；CPU来源外存gliner-biomed/。

## 已完成的近期结论

- 关系 LoRA：204 条严格嵌套 OOF 提示独立审查通过，2epoch/52steps 已完成。同 A800 基座 `.623068` → adapter `.674333`，减少过度归属但低于同实体规则 `.711354`；固定集合投票也未超规则。原 dev 未新增测试。
- 本体校准池的 24 个新概念/26 支持跨度经现有 adapter 判断，保留原实体及规则关系，仅增 14 项关系（4TP/10FP），fold0 `.712942`。边际精度较低，尚不推广。
- 训练锚点诊断独立复算通过：967 个 SFT 候选内正关系中，363 项实际局部证据无目标患者 marker；全局 anchors 仍可见，不能称不可回答率。完整全文 LLM 共指此前没有实际运行。
- 共指首轮训练输入 pilot 因子串定位与身份混淆失败；同 3 篇仅修订一次为唯一原文 context＋完整词边界，严格保留 4 项新增且逐条身份检查通过，召回保守。已按此协议复验fold0，新增18mentions但规则零变化；同adapter SFT有小增益但未超规则，不继续按结果调提示。
- 神经 NER、HPO 重排、零样本关系、thinking 小样本、O 编号锚点、外部表格等正负结果与限制均保留在实验报告。O 编号四折不稳定、表格新增关系均为本地 FP，未升级默认。未使用包含本地 dev 的赛事公开成品权重。

## 证据与下一步

- [实验过程](experiments/E-20261023-009.md)、[研究笔记](NOTES.md)、[证据索引](evidence/round2/README.md)。完整产物根目录：`/workspace/patientphex-runs/round2-20261023/sequence_crf/`；权重/预测/原始响应/完整日志不入 Git。
- [SFT 固定结果与恢复索引](evidence/round2/relation-sft-fold0-results.json)、[新关系池结果](evidence/round2/new-ontology-relation-results.json)、[训练锚点诊断](evidence/round2/patient-anchor-audit.json)。A800 现有 base 和本路线 epoch2 adapter 可只读复用；旧 180 概念/29 提示弱池未运行。
- 共指完成后保留全部控制，包括负结果；如无可靠收益，停止当前原型，不以格式通过替代身份或分数验证。后续再协调检索新跨度与本体候选组合，避免混淆本次单变量对照。
- 历史基线口径：旧 baseline-fold0 未拆复合关系标签 `.613929`，修复后 `.603398`，实体不变；fold1–3 已使用修复版。不要将四折旧关系基线写成同一未修补模型。
