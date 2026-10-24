# 病例检索路线：当前工作索引

更新：2026-10-24 UTC。分支 `round2/case-retrieval`；worktree `/workspace/patientphex-worktrees/round2-case-retrieval`。开始新任务先读主仓 `/workspace/CHIP2026_PatientPheX` 的 AGENTS/HANDOFF/总台账，再看本索引与实际状态。

## 当前：原dev .754216969；支持门升级通过独审

- [R1监督适配](r1_supervised_adaptation/README.md)：CPU13.434秒通过，原124JSON+EOS/80旧IDs exact、tokens401291/max5519、base+LoRA8.234B。唯一A30 [GPU探针](r1_supervised_adaptation/GPU-PROBE.md)29.012秒在最短570真实native/selected梯度容差断言停止，0Adam/最长未执行；非OOM/无质量。实际峰19.496GB、11小件SHA回收、GPU1已释放；差值断言先于落盘的诊断缺口已记，不放宽/重启。源码faa0a11已push，统一终审615e4193通过失败范围；独立[数值诊断](r1_supervised_adaptation/NUMERICAL-RESULT.md)36.693秒完成：同输入/RNG/logits/loss，native自身237tensor不过旧门；同冻结logits两CE监督grad exact/ignored0，原门不变，不归特定kernel。无optimizer/质量，702MB tensor留server、23小件已回收，GPU已释放；数值终审55524bf2通过。独立[H51 stress v2](r1_supervised_adaptation/STRESS-RESULT.md)65.959秒PASS：12步/1Adam+3累积max5519、7类B探针变/basegradnone，peakalloc21.064GB；49小件SHA回收、GPU释放。源ed37586先push，原allclose不改不重测；终审92fab5e1通过，root已授权fresh两轮1800+30，正式源/plan/两质量端点已冻结，先push后fresh启动；每epoch完整state持久reopen，终态后仅1最长input240+30，不自动全80/评分。完整训练尚未授权；冠军SFT细节未知，旧R1组合结案不动。
- [public-new双折LGB支持控制](public_new_lgb_support/RESULT.md)唯一0.316秒完成/keeptrue：F0 .705717148→.711507580、macro+.012265413；F1 .682128804→.686338981、macro+.003399489。原实体/旧关系/nonassoc bytes保持，双pred先共冻、各1v2/旧0，无fit/NN。全部任务fit祖先排各目标fold；新旧gate均train48，已看fold选型/cut，明确开发控制。源79e2ef5已push、result1c10f478/freeze429b69c4；两fold+原dev统一终审5c043574通过。唯一原dev迁移.748583440→**.754216969**（+ .005633530），mic+.011845595/macro+.010688523，实体两项exact；35→54（删2增21）、391→410，保护其余356关系，pred3049d9fe/root正式接受best。准备最小持久补包/纯缓存byteexact入口，不混sequence a179患者臂。
- [独立R1元模型OOF筛查](r1_meta_oof/RESULT.md)唯一完成：已知F1 1557对／16篇SHA四组，各12篇fit／4篇held，11列±3R1共8fit回载1.467秒；双pred共冻后2newv2 .365秒。无R1 .643533208→含R1 .650270692（+.006737484），仍低强对照 .664421522且macro不达门，keepfalse。全基础排除整个F1；无F0/GPU/扫描，维持后备、不声称16篇fit方法证伪。运行前ca003df及结案d3eef75均已push；结果a607f8ab／终点d6569b91，统一源＋终审77802f13已PASS，无活动进程。
- 用户目标仍是原train64/dev16公共v2 Score≥.8；原dev已审当前best **.754216969**（3049d9fe；上一个0e2c .748583440）。内部F1、train64四fold均是多次研发使用的开发结果，不冒称原dev、盲测或官方。用户通知前禁止官方上传。
- [固定LightGBM](lightgbm_selector_control/REPORT.md)：同强BioBERT八池/原feature/label、160轮/15叶/min35/lr.06/L2=5，.3/nested不变；仅将sklearn HGB替成LightGBM4.6.0，分箱/生长数值不宣称等价。实际维数102/101/103/103；4fit/predict/回载59.389秒<180，finalize2.933秒<30，全部160轮、原生正列与保存回载exact。
- pooled **.730993616492→.731545429370**（+.000551812878）；F0/F1/F2正，F3−.000405716，达到事前pooled严格升且≥3/4fold不降。44篇49加55删，非entity/关系literal bytes不变；完整四pred共冻e31cfa67先于eval marker，新5v2/旧0。结果6fc2c9e7、终点087572cf；入口审084923ce通过，完整终审1d6ed9fd已PASS/keeptrue。
- source f71e491已push核验；正式controller session13703已rc0，外部 `research/lightgbm-selector-readiness/formal-control/execution-state.json` 为complete，不重启。原plan d893继承的HGB/120秒旧说明字段未执行，实际LGB/180/Score门以新字段与源为准；[sidecar c19fb7ea](lightgbm_selector_control/results/plan-interpretation-addendum.json)明确，不修改原冻结字节。
- [唯一full64结果](lightgbm_selector_control/full64-results/result.json)：原103列vec transform/同pool/fresh1fit，全部16.901秒<120。原dev **.746081964986→.748583439566**（+.002501474580）；men .786960514→.792801108，doc .824512535→.828677840，关系两项不变。11篇6加17删、最终1061实体/391关系/36患者；全部非entity literal bytes及35public-new保留。完整newpred先冻后唯一v2，旧分数复用；[终审a1b2c622](lightgbm_selector_control/full64-results/terminal-review.json)PASS。model803b6467/pred0e2c26f8/terminal64172581。
- 两个CPU控制均complete且rc0，无运行作业；full64外部 `research/lightgbm-selector-readiness/full64-dev-control/execution-state.json` 为complete，禁止重启。[同一持久包](lightgbm_selector_control/full64-results/persistent-backup.json)288622b9/3.83MB/58成员已服务器逐成员SHA+fsync，manifest168b0c39；root已完成唯一server→local restore及58成员SHA复核。sequence独立患者关系仍使用原a179，不偷偷合成。[固定误差/历史查重](lightgbm_selector_control/research/next-entity-history.md)已完成；原dev只是事后方法选择依据，不用于新控制fit或手工补规则。
- [相对候选18列](relative_candidate_evidence/README.md)完整负：原LGB pooled.731545429370→.730783429343（−.000762000027），仅F1正/1of4，keepfalse；43篇54加40删，关系不变。实际4fit61.879秒+端点2.647秒，全部160树/120,119,121,121列，先共冻再5newv2/old0；result295c2def/terminal6b67cbea。session57662已rc0/state complete，完整终审3f767632通过，不扫子集/cut、不迁dev。
- [固定HGB/LGB均值](hgb_lightgbm_average/README.md)完整负：.731545429370→.731417753008（−.000127676362），仅F3正/1of4，29篇22加22删；关系/非entity bytes不变。唯一4.622秒、0fit/0新模型预测，先共冻5newv2/old0；审908fbc8c，session15356/state complete。固定关闭，不扫权重/cut/迁dev。
- [纯加载缓存复现](lightgbm_selector_control/PURE-LOAD.md)已唯一完成：入口3.272秒/外层3.463秒rc0，16篇13,246候选，完整输出与0e2c逐字节同SHA，391关系及非entity字节保持；0fit/gold/v2/GPU。源0906e144/合同36c056fd/终点9725d919，独立终审53a91d2f通过；16件2.91MB补包6aa98c35已服务器全SHA/fsync，manifestb897bfbd，root负责另做实际恢复。它固定当前16篇合同，尚非任意新文献或原文端到端入口；上游候选/关系仍来自冻结缓存。现无运行CPU/GPU；不跑赛测集/官方或重做旧NN。

## 已关闭或保留的关键控制

- [R1条件学习融合只读研究](research/r1-meta-fusion-feasibility.md)：官方§4.2支持LLM预测＋上下文→LightGBM，旧303HGB／五列LR均未包含R1；严格F1→F0技术上可隔离。固定后备为同11列树±3个R1状态、强对照匹配；F1 1557对中282对R1未知，不填负。F0缺57fresh、历史均值估约42分钟，新增21TP/140FP证据弱，root同意暂列后备先回收在跑方法。只读报告，无gold/fit/新质量/GPU；旧F0关闭不变。
- [全局兼容解码](global_compatible_decoder/REPORT.md)：强B4,633候选/66非平凡组件最大3；精确有理数枚举3.488秒，仅F3一组件删1加2。pooled .730993616→.730822309、F3−.000731780，其余三fold byteexact；统一终审98962250，固定负，保旧nested、不改目标/cut/迁dev。
- [全文Med](medgemma_relation_control/fullcontext-preparation/RESULT-fullcontext-frozen.md)：fresh Med同strict32/124，train+infer追加允许全文，cap14000/7150监督、32calls；oldSFT32 **.660942003**→全文Med **.592418621**、唯一old∪Med **.627966889**，均负。两epoch/两完整持久ACK1264.674秒<2700，final80 307.881秒<900；587CID合法0fallback，1189实体/非关联bytes不变。终审f7e5df86，不F0/dev/官方。
- [短Med](medgemma_relation_control/RESULT-short-input-frozen.md)：.603701549、old并集.644663430，审fcc84401，固定负。短/全文是不同输入控制，旧零样本全文、开放HPO全文SFT另列，不把任一负结果泛化为所有全文方法无效。
- [顺序增强](order_warmstart_stage/RESULT.md)：旧.660942003→O/O .658421563、R/O .645528610，四状态持久/审d253f625，固定负；不否定此前reverse测试时并集的原dev正结果。共指B、public212、value2等已终审关闭，不重做。
- [thinking4096](research/thinking-sampling-followup.md)是未闭think/无strict final的格式终点，后继budget forcing归sequence。FLA raw-logit合同失败见[fused结果](fla_kernel_readiness/GPU_RESULT.md)，不能称任务质量负。

## 资产、资源与恢复边界

- `R=/workspace/patientphex-runs/round2-20261023`。Git只留源、关键结果与SHA；模型、全量概率/预测、正常日志在外部目录。固定LightGBM wheel cb19b5af/3,569,831B仅安装到 `R/research/lightgbm-selector-readiness/deps/runtime`；原生库6a12e1a7/接口1.163秒通过。未修改global/shared依赖，无新模型下载。
- 本路线无运行GPU。GPU资源由root协调，本路线不占卡；A100由sequence新full64使用，R1原F1已完整结束，A800新全文输入控制由ontology接管准备。不得凭旧空卡收据占用。最近[一次四机快照](evidence/round2/gpu-survey-20261024-043924.json)04:39:24未发现新增≥40GiB空卡；A8000仍foreign56706/594MiB，.3 GPU0 24GiB/4MiB/no compute仅小任务候选，未来用前fresh核。A100当次UNKNOWN不等于其训练失败，不重试风暴。
- Med原件19件8.64GB在.100持久 `research/medical-small-model-readiness/models/medgemma-1.5-4b-it-9bca96ca3e30`；SHA已核，勿重复下载。A800 `/dev/shm/patientphex-medgemma15-1045-20261024`仅临时副本，项目盘不足1GB，不放权重。
- 全文两份完整state在.100 `research/medgemma-relation-control/fullcontext-formal2ep/checkpoints/epoch-001,002`；final checkpoint c4044dac/adapter47798260/Adam-RNG0bfa4203，两ACK后均独立回读。raw与双pred另持久归档07b2ca5a/33项638317B；短Med独立归档b66fd723，不混模型或状态。
- 原Qwen/Med参考TF5.20.2/PEFT.21.2，不换FLA/量化。官方限制**每个独立模型≤10B**，不是组合总和；Med完整bundle+LoRA4,329,881,968，未用视觉模块仍计数。[H48实际文本80审查](medgemma_relation_control/tokenizer-family-audit/REPORT.md)只证Med/Qwen本80输入ID与decode一致，不泛化全词表/音频。
- [H49核查](medgemma_relation_control/fullcontext-preparation/H49-raw-first-note.md)：Med逐条raw先写/fsync、回收后才parse，无热改/重跑；不冒称已保存全部generated IDs。监控UNKNOWN保known job/原deadline，不finally杀健康传输；修复模板外部 `medgemma-relation-control/transfer-a800-recovery/run_once.py`0f64a522。

## 记录与接手

只改本路线目录，root维护共享总账/HANDOFF；关键阶段commit/push并核ls-remote。原完整工作索引分别保留[截至0339](history/WORKLOG-through-20261024-0339.md)、[截至0450](history/WORKLOG-through-20261024-0450.md)，只作历史，不作为当前启动命令。新任务先依据当前实验源/manifest核真实状态，禁止重启已完成实验。
