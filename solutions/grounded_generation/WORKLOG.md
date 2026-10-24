# 工作台账

分支`explore/grounded-generation`，worktree `/workspace/patientphex-worktrees/explore-grounded-generation`，起点main `0c0672d`。只修改`solutions/grounded_generation/`。旧序列路线已f0785f7完整结案/推送，α=.5固定负结果不重开。

## 当前目标和状态

- **F1完整且终审PASS**：原GG001 N∪grounded .681717659986（强N .676546354663），new-only .583496662244；独立GG002强支持B∪全部grounded .695950447742（B .686338981272），micro下降.013393436/macro上升.051839302。9B只复制原文词面、独立HPO规范化，quote是弱词面锚定而非人工归属span。完整终审0fcffd7b，F1补包184内容已root实际服务器恢复c48d3776；原dev最高.754361398和官网结果不变。
- **F0原47已完整**：11:48:52.618 rc0，44 EOS/3 cap；native593.562秒/wrapper594.451/controller594.584、peak20.975GB，GPU2实际已释放、原watch退出。完整原包回收与固定normalizer待接，3 cap保原弃答不重试。原唯一身份：同d52模型的F0全文47，A800 GPU2 `proc_1a64fada-6c00-43a3-806e-b9ea4b755862`，driver63172/child63188；11:38:58.166起，TERM12:13:58.166、KILL12:14:28.166。原plan ed3c9297/f657e38已推。唯一只读watch session54389/PID157105每180秒；先核其实际存在，禁止重复start。准确入口[f0/actual-start.json](f0/actual-start.json)、[恢复步骤](f0/RECOVERY.md)。
- **已完成F0工程**：47原生输入5.522秒全部ID/渲染/decode一致，289420/max9935；11件5bcd8c19。173MB d52推断缓存19.422秒完成四SHA/两端fsync/reopen，原fullstate和两ACK仍A100。A800 Torch2.7.1相对训练A1002.6.0如实披露，原native/greedy1600/b59阈值不变。原LUNA与2GiB保守容量门失败保留；按实际cache/raw上界768MiB copy/512MiB infer，未清其他资产。
- **F0固定质量**：全47→原解析/固定train32别名与public41099/Sap→全部新关系。只强B0 .7115075800266846∪全新关系一v2、旧0，保947实体/全部原关系/非关系字节；new-only只输出不评分，完整供root固定2of3与校准。不得因独立F0负删source，不自动原dev/官网。
- **条件full64仅CPU已完成**：[full64_readiness/summary.json](full64_readiness/summary.json)，173患者全覆盖、1711有效关系可构建、1678弱quote，dev纯36；957793训练tokens、监督24997，max10637/646。两轮代理4242–4582秒、候选6000+30不是实测或授权，新形状需后续实际工程。train64含内部F0/F1监督，但这批独立未来资源绝未进入当前已冻结F0模型/别名。无原devgold、新fit、GPU或质量。
- 当前分支检查点`bba8650`已核远端（后续台账提交可在其上）；所有大产物位于仓库外。资源由root调度，A100现给response-code，.3在DocRE QwenF0完成后才轮到本线必要Sap。

## 执行与下一步

1. 原F0已完整且watch退出，沿唯一collector回收原不可变全包；所有EOS、cap、strict失败原样保留，不按质量重试。
2. 固定parser/normalizer完整闭环，先给root/validation纯new关系完整产物，再完整强B0union冻结→唯一v2。逐项实际命令见f0/RECOVERY.md。
3. 完整终点一次增量审，结果和原件最小持久包、及时push；不逐阶段重复审旧链。
4. 仅当独立F0或root固定2of3通过，再由root决定full64工程/训练；当前readiness不能当执行许可。

## 历史阶段记录（准确原时点，非当前作业）

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

完整39一次解析acfbb8a6：573 exact anchors/434 unique quote，126项拒绝，0整行JSON弃答但1行全item被拒，39EOS/cap0。同32/public固定strict准备759e42b8产生254必要query；0f97c594计划绑定最新schema F0实际release，180+30，估25.08秒，既有41099 bank不重编码/不fit。先push再fresh门唯一启动，未使用held gold或评分。

254必要Sap唯一proc9f6935a7已11:15:44.484 rc0，wrapper11.603秒/controller11.779秒，实际GPU释放11:15:44.491。13小件f1720f1c完整逐SHA回收，原queryvectors留持久locator；无重bank/fit。实际finalize计划已绑定原全573 anchors与254 scores，固定policy不变，随后只CPU normalize/双冻结/两v2。

固定normalize完整267cd326：573记录mapped328/NIL0/abstain245，strict244+semantic84；所有原doc/PID/unit/occurrence/offset/length/quote与拒绝理由保留，normalizations786cc560已交root作独立后继供给，本线不改1189实体。双端点质量计划03c88a3f已绑定完整8refs，先push再共同freeze→各唯一v2/旧0。

11:18:08完整双质量收口：主臂 .681717659986 > N .676546354663（+.005171305324），micro +.001808409 / macro +.018876813，Task1不变；new-only .583496662244。新增98=35TP63FP，候选外34=17TP17FP。共同freeze a8591946，两个唯一v2/旧0，.342秒；metadata接口机械恢复保留，模型/规范化/政策未重跑。见[F1-result](evidence/F1-result.json)与完整实验；候选待独审、原devbest与官网不变。

GG-002父线程新授权固定强B2a65056f（.686338981272）∪全部282grounded，源plan baee5ec先push；完整pred ef93ce2d/156658B、freeze31648074→唯一v2/旧0。总分 .695950447742（+.009611466471），micro−.013393436395/macro+.051839302277，Task1不变；+90=26TP64FP，保全部471基底。此为已知首结果后的独立固定CPU控制，无TP筛选。当前准备同权重/normalizer全祖先排F0和47患者全文原生输入合同，预计A8002调度，仍不自动GPU；[strong_support_union/result.json](strong_support_union/result.json)。

11:26恢复：F1完整终审0fcffd7b PASS，训练/39raw/254query/原双质量及后续强B融合一次收口，见final-F1-review指针；不重复运行。F0本地纯输入47全文已1.322秒完整，289420tokens/max9935，manifest37534e8e；F2F3训练32/alias与F0不交，纯公共41099链原审复用，未读取F0gold。下一步A800实际native47合同与173MBd52推断缓存同字节复制，完整训练state仍A100，实际Torch2.7.1与原A1002.6.0差异明示；固定强F0 B .7115075800266846∪全部新关系，不自动启动GPU。F1最小补包源码已准备，原3 ZIP逐字节收进补包、所有大权重另留精确locator。

F1补包62d55b/1006565B已.100全184内容逐SHA/fsync/reopen，root随后实际下载恢复并核184内容＋内嵌manifest，receipt c48d3776；不重复下载。F0 native47实际11:32:16 rc0/5.522秒，全部IDs/渲染/decode exact，11件5bcd8c19已完整回收；A100唯一copy库存11:32:23.677 LUNA，未部署HTTP/未启动复制，统一ontology最早11:35:28.677恢复，A800卡未占。

F0原47已11:38:58.166169唯一启动A8002 proc1a64fada / driver63172 / child63188，TERM12:13:58.166/KILL12:14:28.166，2100+30；planed3c9297/sourcef657e38已push。copy19.422秒/173197471B四件两端SHA/fsync/reopen完成ACK7a3982d3；盘实际1.47GB，原任意2GiB门保留失败，改为按真实cache/raw上界768MiB copy/512MiB infer并获root确认，无清理他人资产。17件部署e1873dda，首观察两PID正常；唯一只读watch session54389/180秒，恢复先查看[f0/actual-start.json](f0/actual-start.json)，不可再start。固定同d52/native/greedy1600，全47后相同b59eb及唯一强B0union评分；不使用部分结果。

F0原106件完整回收32aec970；固定解析d4bdba97得到342 exact anchors/262独特词面，3cap+2strict JSON整行弃答，48项拒绝如实保留。固定train32/public政策准备d5b77b01产生149必要Sap查询；绑定DocRE QwenF0实际11:53:51释放817ab0ca，plan5a347250/a6d2b6b已先推，原180+30同现成bank/encoder，无新fit或质量。
