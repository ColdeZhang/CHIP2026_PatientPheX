# 工作台账

分支`explore/grounded-generation`，worktree `/workspace/patientphex-worktrees/explore-grounded-generation`，起点main `0c0672d`。只修改`solutions/grounded_generation/`。旧序列路线已f0785f7完整结案/推送，α=.5固定负结果不重开。

## 当前目标和状态

- 目标：9B原文证据复制＋独立HPO规范化，补充强全文N的患者关系；保持Task1骨架，固定总Score主门。N .676546只是normal全文强对照，历史完整public支持F1 .686338981只读参考；胜N先称方法候选，不称历史或原dev全流程best。原dev最佳.754361398，官方.6922为用户回传；本路线尚无新分数。
- 原32金标38c353ea/969963B和356109e9谱系已核：F2+F3、32文献87患者；F1输入16文献39患者，与训练不交。held/原dev gold没有读取或哈希。
- 历史边界已核：旧8B窗口抽取/SFT直接并集两折不稳；旧患者HP+quote自由编号严重失败。新机制改变为患者条件证据复制、禁止HP生成、独立规范化，不能只把换9B当创新。
- 真实schema已核：association只有patient_id→phenotype；实体独立含offset与identifier。目标quote只是按关系选出的弱证据，不能称人工归属证据。用原passage单位ID保全原始offset，避免按gold切句。
- 目标v2实际85979cdc/.484秒已完成，全87训练行/39held保留：814有效原子键全部有可定位词面，801去重目标，4关系无效/分支外明确列出；原文位置边界7项通过。local token实际c8f29baf/3.253秒：451672完整训练tokens，最长9210，监督11884/max610，held203588/max10080；三Pareto形状9210/399、6521/449、5676/610。原生TF CPU120+5与420+30工程已执行完成，正式训练已唯一完成87×2/22updates；10:47:48实际rc0，两durable ACK通过，held生成/质量未执行。规范化冻结公共银行与624个train-only查询已完整，政策未按scores调整。CPU与14步真实工程均已完整PASS，见下方实际终点；probe状态丢弃，正式87×2/3000+30已批准。
- 资源：A100 GPU2由root保留；实际新input/答案token census后提出工程压力和完整训练预算，不套旧32.693GB，不截最长样本。其他卡/作业由root调度。

## 执行与下一步

1. 固定每个目标患者的完整输入、可定位quote构造、不可构建与歧义统计；所有训练/held患者均在覆盖表。
2. 实际计算训练全文/答案/监督位置与held预算，原生tokenizer核对；纯CPU合同及源/计划先推送。
3. 与root协调一次有界真实最难工程检查，fresh正式训练与完整F1推断；不从工程状态续训。
4. 两完整预测共同冻结后各唯一v2、旧分数复用。主臂N∪new、诊断new-only，Task1/患者/非关系字节保持；另一内部折有益后再full64/原dev验收。

大文件放 `/workspace/patientphex-runs/round2-20261023/research/grounded-generation-v1`。当前native CPU v2已原proc8bd3ffbe完整rc0/16.244秒，18件全部SHA回收237aaa65；87/39全输入与6数值PASS。GPU14步原proc7d1bc518已完整PASS，242.157秒/peak28.182GBallocated、29.067GBreserved，60件e24c8065全收；10:04:31.503GPU2已释放。root已批准正式87×2/22calls/warmup2/尾7 3000+30及39held1200+30/两个固定质量端点；formal-plan已bind，纯metadata preflight核全87/32输入与实际probe成功，19件原部署已SHA/fsync/reopen，package bd4d90a3；10:10:32启动前LUNA后按185秒单owner恢复，10:13:49.227正式唯一proc_e1b482f9/driver75341/child75359已启动。TERM11:03:49.227，KILL11:04:19.227 UTC；87×2/22calls/warmup2/尾7，原两完整ACK后才允许39fresh。未知状态先查原intent，禁止重复启动。


2026-10-24 GPU前澄清：全文先判目标患者的概念，复制quote只作概念词面锚定，不把该处出现当患者归属证据；最近mention只是确定训练代表词面的弱选取规则。首CPU产物targets-v1与源c3b69eb保留，v2仅修正prompt语义，目标/身份/全文不变并逐项对照。170/814处其他患者更近已如实披露。

2026-10-24 CPU首尝试v1在token回放前asset lookup失败5.609秒，完整8件97e8f008已保留；v2只删不存在且不属于tokenizer的generation_config lookup，其他四发布资产及输入/数值不变。独立120+5 source6483fda已push exact。训练器与strict/Sap独立规范化并行准备，后者在DocRE释放.3后才使用已批准120+15 query encode。

2026-10-24 后继推断source：固定39fresh/1600原native，actual epoch2双ACK之后绑定，原CPU held IDs只提取不重编码。raw IDs→native decode→raw落盘→本地exact anchor顺序已写；quality helper只编写固定双端点，不读/哈希gold。正式冻结monitor原来误从output/找driver marker；新增只读`training/observe_formal.py`仅修root路径，trainer/driver/deployment原字节未改。

工程+formal源集中终审9e6a683f已PASS（38安全快照，原CPU8失败+18恢复+GPU60原件）；仅记录pointer，不重运行。10:17:16唯一watcher已看到epoch1/globalstep1/group8成功，后继每180秒。固定双端点quality源码已完成合成边界合同；真实组装/评分仍未执行。actual强N412条均为唯一canonical原子HPO，primary额外断言原关系列表prefix完整，Task1/非关系byteexact。

推断前复用原baf060的FiniteLogitsOnly guard（class AST逐同）：只在NaN/Inf时保留故障并停止，合法logits原样返回；不改变greedy1600/模型/输入/规范化。实际held尚未启动，后续绑定新runner SHA；不为此重跑CPU、probe或已运行训练。

10:32:16原只读观察已见第一完整epoch/step11/尾7，程序epoch耗时982.831秒，mean group loss .3920198；summary绑定durability ACK d51df102/2031B，原第二轮自动继续。此时未收最终ACK全文/两epoch成员，不把它冒充完整训练终审；统一待唯一终态collection。按首轮实耗约10:48前后可能结束，实际硬TERM11:03:49.227不变。

10:38:10.265原A100 readonly观察明确LUNA_OFFLINE；最后真实健康10:35:28为epoch2/step13。唯一watcher144628仍alive，按原backoff最早10:41:15.267再观察；不增加owner、不重start，远端driver绝对TERM11:03:49不变。后继.3 E004已10:38:31完整结束并实际释放，准确local release为stage8-span-normalization/oof-fourfold-completion/gpu-release-observation.json；真正Sap启动仍需当时fresh门。

10:53:29原watch恢复发现正式已10:47:48 rc0完整完成，wrapper2039.028秒/controller2041.959秒，两ACK d51df102/5f8ca5fc，epoch2 adapter d52e2e6c/fullstate47ae7027，GPU实际10:47:51释放；原watch正常退出。唯一完整collector首inventory返回UNKNOWN（只读，未冻结/启动），保留错误后沿原namespace库存恢复，不重训。准确终态与两ACK在[formal-terminal](evidence/formal-terminal.json)；完整本地collection及后继39尚待，不冒称新质量。

原collector已恢复并完整回收25件433659a2，原四个大checkpoint全SHA/fsync核验且留A100持久locator；无新模型。39fresh实际计划7c9c3026已绑定两ACK/epoch2/原39native IDs/固定b59eb规范化，预算1200+TERM30、greedy1600；当前源/计划先推送，实际start后补唯一ID。

10:54:45.757原39fresh唯一proc_e603593d/driver105509/child105533已启动，固定1200+30，TERM11:14:45.757/KILL11:15:15.757；首次观察两PID健康、加载中无响应，未读gold或评分。16件部署80bcdf85已SHA/fsync；本地readonly watcher 150076 / session30679每180秒，准确来源和恢复见[evidence/inference-started.json](evidence/inference-started.json)。原start-intent存在绝不重start；已授权后继fixed-normalization与双质量，无须新阶段审批。

固定normalizer原未覆盖的独审增量已在schema路线统一完成，稳定报告dac8bde8的normalization_increment节核四固定源/624查询/41099银行来源；本路线复用该同字节来源审，不误指9e6a，不重编码。自己的39输出与新normalize实际链仍等最终增量审，见[evidence/normalization-review.json](evidence/normalization-review.json)。

11:13:10.349原39在原窗完整rc0，无补跑；native1103.215秒/wrapper1104.592/controller1104.730，39EOS/cap0/peak21.015GB，GPU释放11:13:10.376。90件完整回收5c32d2f8，全raw/token/end状态SHA核过，extraction实际计划已绑定，尚无质量；见[evidence/inference-terminal.json](evidence/inference-terminal.json)。
