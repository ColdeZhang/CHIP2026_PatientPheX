# 固定 N/R/P 两票控制：已准备，未装配/评分

目标：检验原normal、同adapter reverse、另训patient三份完整缓存中，至少两票的关系能否胜过其原三方并集。只做F1/F0各一臂，不训练、不NN、不扫描票数、不迁原dev。两折均已反复开发，不称盲验证。F0 patient沿原limited-codec恢复，缺8对仅弃答，不补负标签。

父线程转述root最新固定门：**两折Score都严格高于各自triple-union才保留；macro及全部分项完整披露，无额外macro否决。** 基线F1 .6638418206599839、F0 .7095567430848545，旧metrics直接复用；rules仅作并列参考。

按原字面`(pmc_id, patient_id, phenotype)`每模型一票；不拆compound、不改大小写/NIL/原文、不canonicalize。只过滤既有union列表，保留留下元素的顺序；仅替换每个`phenotype`数组的值，所有其余原UTF-8字节、患者metadata与全部实体精确保留。完整HPO compound只在未来公共v2及变化统计中沿原口径拆原子，不影响投票。

`plan.json`绑定两折N/R/P、完整triple预测/metrics、原隔离审查及公共v2源码。gold path/size/SHA只从旧plan复制；准备和装配不打开、不重新hash。audit guard在接收凭据与共同freeze验证前拒绝gold访问；准备阶段用合成open事件验证，没有真的打开文件。无旧训练/解析脚本导入。

阶段独立，均默认拒绝覆盖：

1. `python prepare.py --plan plan.json`：已完成，仅检查冻结缓存/metadata/三方并集一致性、合成边界，输出外部`preparation.json`；不产生新fold预测。`gate-contract.json`另验证未接收、错误方法/赛季、未来时间等被拒绝，没有创建真实accepted凭据。
2. 父线程commit/push后，`python assemble.py --plan plan.json --push-receipt PATH`。push receipt为`schema=majority-source-push-receipt-v1`，包含精确`plan` ref、`sources`、40位`commit`及`remote_verified=true`，只是记录已完成推送。此步产生完整F1/F0预测与单个`assembled/common-freeze.json`，不自动评分。
3. **真实官方901 accepted之前禁止执行评分。** root/父线程归一真实平台收据后，显式执行`python score.py --plan plan.json --postaccept-receipt PATH --postaccept-receipt-sha256 SHA`。无需新增批准，只要求已授权顺序的真实事实。入口先核acceptance与共同freeze，再启动唯一子进程30秒+TERM1；自身进程组超时退出，原错误/partial原留。每折唯一v2 all/text，旧指标0次重评；不自动重试。

postaccept receipt的`schema`为`official-901-accepted-release-v1`；须含`official_accepted=true`、`release_majority_score=true`、competition_id 532512、season_id 1862、method_dev_sha256 `9016863cb3bff53da5dfde25fce634bd60882f057d0c14501be7cb4937e71c67`、B_input_sha256 `878fe248ce5ae2cf9c96cc35c9ed3a7bae929988752cfdc3ea30ef60c64370d4`、真实`platform_submission_id`及带时区`accepted_at_utc`，另含`official_receipt`和`uploaded_prediction`两份实际`{path,bytes,sha256}`。当前不创建填空或假accepted文件；平台原收据schema不受本控制约束，归一凭据只绑定原件。

外部产物根：`/workspace/patientphex-runs/round2-20261023/research/normal-reverse-patient-majority-control`。当前实际准备和合成拒绝合同见[prepared-manifest.json](prepared-manifest.json)；未读取任何gold、未导入/调用evaluator、未生成新fold预测。四入口源与plan先交父线程推送；后续装配/接收/评分分别追加真实收据，不改写本次未执行事实。
