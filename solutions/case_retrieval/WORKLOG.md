# 病例检索路线：当前工作索引

更新：2026-10-24 UTC。分支 `round2/case-retrieval`；worktree `/workspace/patientphex-worktrees/round2-case-retrieval`。开始新任务先读主仓 `/workspace/CHIP2026_PatientPheX` 的 AGENTS/HANDOFF/总台账，再看本索引与实际状态。

## 当前：固定LightGBM四fold小幅正，准备唯一原dev迁移

- 用户目标仍是原train64/dev16公共v2 Score≥.8；原dev主线当前.746081965。内部F1、train64四fold均是多次研发使用的开发结果，不冒称原dev、盲测或官方。用户通知前禁止官方上传。
- [固定LightGBM](lightgbm_selector_control/REPORT.md)：同强BioBERT八池/原feature/label、160轮/15叶/min35/lr.06/L2=5，.3/nested不变；仅将sklearn HGB替成LightGBM4.6.0，分箱/生长数值不宣称等价。实际维数102/101/103/103；4fit/predict/回载59.389秒<180，finalize2.933秒<30，全部160轮、原生正列与保存回载exact。
- pooled **.730993616492→.731545429370**（+.000551812878）；F0/F1/F2正，F3−.000405716，达到事前pooled严格升且≥3/4fold不降。44篇49加55删，非entity/关系literal bytes不变；完整四pred共冻e31cfa67先于eval marker，新5v2/旧0。结果6fc2c9e7、终点087572cf；入口审084923ce通过，完整终审1d6ed9fd已PASS/keeptrue。
- source f71e491已push核验；正式controller session13703已rc0，外部 `research/lightgbm-selector-readiness/formal-control/execution-state.json` 为complete，不重启。原plan d893继承的HGB/120秒旧说明字段未执行，实际LGB/180/Score门以新字段与源为准；[sidecar c19fb7ea](lightgbm_selector_control/results/plan-interpretation-addendum.json)明确，不修改原冻结字节。
- root新增授权：复用强B原full64训练数据/103列及原dev纯features，唯一同配置fresh fit+回载+dev predict≤120秒CPU；仅替换a179实体，旧391关系及全部非entity原字节不变，完整新predfreeze后一次v2、旧.746081965复用，Score严格升才候选best。**当前仅source/lineage准备，尚未full64 fit或dev新评分**；不更改sequence正在进行的患者关系baseline，也不偷偷合成两条变更。

## 已关闭或保留的关键控制

- [全局兼容解码](global_compatible_decoder/REPORT.md)：强B4,633候选/66非平凡组件最大3；精确有理数枚举3.488秒，仅F3一组件删1加2。pooled .730993616→.730822309、F3−.000731780，其余三fold byteexact；统一终审98962250，固定负，保旧nested、不改目标/cut/迁dev。
- [全文Med](medgemma_relation_control/fullcontext-preparation/RESULT-fullcontext-frozen.md)：fresh Med同strict32/124，train+infer追加允许全文，cap14000/7150监督、32calls；oldSFT32 **.660942003**→全文Med **.592418621**、唯一old∪Med **.627966889**，均负。两epoch/两完整持久ACK1264.674秒<2700，final80 307.881秒<900；587CID合法0fallback，1189实体/非关联bytes不变。终审f7e5df86，不F0/dev/官方。
- [短Med](medgemma_relation_control/RESULT-short-input-frozen.md)：.603701549、old并集.644663430，审fcc84401，固定负。短/全文是不同输入控制，旧零样本全文、开放HPO全文SFT另列，不把任一负结果泛化为所有全文方法无效。
- [顺序增强](order_warmstart_stage/RESULT.md)：旧.660942003→O/O .658421563、R/O .645528610，四状态持久/审d253f625，固定负；不否定此前reverse测试时并集的原dev正结果。共指B、public212、value2等已终审关闭，不重做。
- [thinking4096](research/thinking-sampling-followup.md)是未闭think/无strict final的格式终点，后继budget forcing归sequence。FLA raw-logit合同失败见[fused结果](fla_kernel_readiness/GPU_RESULT.md)，不能称任务质量负。

## 资产、资源与恢复边界

- `R=/workspace/patientphex-runs/round2-20261023`。Git只留源、关键结果与SHA；模型、全量概率/预测、正常日志在外部目录。固定LightGBM wheel cb19b5af/3,569,831B仅安装到 `R/research/lightgbm-selector-readiness/deps/runtime`；原生库6a12e1a7/接口1.163秒通过。未修改global/shared依赖，无新模型下载。
- 本路线无运行GPU。A800 GPU2 f516在Med最终04:21:32.942实际释放，已交ontology的R1F0；不得凭旧空卡收据占用。A100 GPU2归sequence full64、.100 GPU1归R1F1。最近[一次四机快照](evidence/round2/gpu-survey-20261024-043924.json)04:39:24未发现新增≥40GiB空卡；A8000仍foreign56706/594MiB，.3 GPU0 24GiB/4MiB/no compute仅小任务候选，未来用前fresh核。A100当次UNKNOWN不等于其训练失败，不重试风暴。
- Med原件19件8.64GB在.100持久 `research/medical-small-model-readiness/models/medgemma-1.5-4b-it-9bca96ca3e30`；SHA已核，勿重复下载。A800 `/dev/shm/patientphex-medgemma15-1045-20261024`仅临时副本，项目盘不足1GB，不放权重。
- 全文两份完整state在.100 `research/medgemma-relation-control/fullcontext-formal2ep/checkpoints/epoch-001,002`；final checkpoint c4044dac/adapter47798260/Adam-RNG0bfa4203，两ACK后均独立回读。raw与双pred另持久归档07b2ca5a/33项638317B；短Med独立归档b66fd723，不混模型或状态。
- 原Qwen/Med参考TF5.20.2/PEFT.21.2，不换FLA/量化。官方限制**每个独立模型≤10B**，不是组合总和；Med完整bundle+LoRA4,329,881,968，未用视觉模块仍计数。[H48实际文本80审查](medgemma_relation_control/tokenizer-family-audit/REPORT.md)只证Med/Qwen本80输入ID与decode一致，不泛化全词表/音频。
- [H49核查](medgemma_relation_control/fullcontext-preparation/H49-raw-first-note.md)：Med逐条raw先写/fsync、回收后才parse，无热改/重跑；不冒称已保存全部generated IDs。监控UNKNOWN保known job/原deadline，不finally杀健康传输；修复模板外部 `medgemma-relation-control/transfer-a800-recovery/run_once.py`0f64a522。

## 记录与接手

只改本路线目录，root维护共享总账/HANDOFF；关键阶段commit/push并核ls-remote。原完整工作索引分别保留[截至0339](history/WORKLOG-through-20261024-0339.md)、[截至0450](history/WORKLOG-through-20261024-0450.md)，只作历史，不作为当前启动命令。新任务先依据当前实验源/manifest核真实状态，禁止重启已完成实验。
