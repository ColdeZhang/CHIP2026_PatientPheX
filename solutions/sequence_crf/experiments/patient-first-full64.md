# 患者优先 full64：CPU 和最长样本探针

目标是检验两折均有小幅互补的患者优先关联能否迁到原 train64/dev16。三方并集已通过固定双折门槛，但 F1 micro 下降，F0 仍低于 rules；不能预期必然提高原 dev。

## 当前边界

只授权 CPU 数据与单次 A100 最长行更新，240 秒加 TERM30；正式训练尚未授权。后续若运行，用 fresh 两轮 265 行/68 次 optimizer calls/4 warmup，拟 4500+30 含两完整持久状态；dev 原64输入拟另900+30。最终必须在当前 a179 全预测上保留全部旧实体、关系、public-new支持，仅检验新增患者关联，不重新计算或删除旧候选。当前未读 devgold、未评分、未官方上传。

## 实际 CPU 数据

既有冻结265条监督逐行转置，4条48→16候选来源链均排原dev16。旧 native 编码逐行及全量哈希一致，5328对（1229正/4099负）双向恢复原答案精确。新总860405 token、9818监督；唯一最长3563609:relations:0为6166/6044前缀/122监督，小于旧6400上限，零截断。dev64/16篇/1006候选对仅输入，旧IDs逐行精确，新最长5198。纯CPU实际7.988秒，正式训练环境 native265 检查待执行。

## 最小 GPU 验证

复用已审 F0 数值主体，fresh公共9B+r16 LoRA、BF16/nativeGDN/SDPA、非重入checkpoint、原生完整logits CE、单次非零lr1e-4/AdamW .01/clip1。只adapter进optimizer、base冻结；检查三族B真更新，允许零初始化B造成首步A梯度为0。探针不存权重、不作为正式起点，不修改allocator/输入/loss。首次OOM、非有限或超时即停，保留终态，不自行重试。

新源 `patient_first_full64/`；原数值段逐字比对及6输入绑定见 evidence/patient-first-full64。外部入口 `/workspace/patientphex-runs/round2-20261023/research/patient-first-full64-probe/`，原CPU完整交接为同级 `patient-first-full64-cpu/handoff-manifest.json`。小摘要进Git，全数据/日志/后续权重外留。

## 实际 A100 native265 入口

隐藏 CUDA 的实际 TF5.20.2/PEFT.21.2/Torch2.6 CPU预检通过；两个 GDN binding 来自原 native 源 b4b17f，FLA不存在，allocator变量仍null。265条全部编码/labels及聚合SHA与CPU准备精确，实际 8.432秒；最长selected完整文件3beacc4e与原CPU精确。receipt8137d850，未读dev目标/单独dev输入文件或权重，CUDA未初始化。16件源码/数据部署与新receipt绑定后，将执行唯一240+30探针；完整训练仍待结果。

04:33:22.332Z唯一GPU探针已启动 `proc_281ec4d2-bef3-490c-bdd7-4d60f43dccae` / PID113890，fresh GPU2核空、源与实际计划3481172已推送；近似TERM04:37:22/KILL04:37:52，以remote driver实际截止为准。只读回收 `patient_first_full64/operations.py collect --directory <外部probe目录>`，未知状态不得重复start。正式trainer仅绑actual native与既有full64配置，未生成正式可执行plan/未launch。

## 最长行实际终点

唯一探针81.873秒rc0，native80.700秒，前向/反向/真实更新15.167秒。loss .252976、梯度范数3.105489均有限；DeltaNet/full-attention/MLP三族B真更新，248个B梯度非零，A首零符合初始B全零。base无梯度/仅adapter优化，FP32优化器状态。allocated峰38,183,758,336 B，reserved峰39,082,524,672 B，结束测得free2,665,807,872 B；free不是峰值剩余，也不证明accum8和多形状全程。未变更allocator/attention/loss/精度/输入，未重试，未保存权重。

真实04:34:44.426409释放GPU2，0/40960MiB且无进程。8小件逐SHA回收，step2d94cb44、wrapper700a210d；完整探针清单见evidence。第一次回收读取先见complete、未见wrapper；随后原文件回收证实实际成功终点，不重启作业。source与job已d313452推送。

正式计划参考同A100 F0两轮2279.145秒，新token比1.309922，线性参考2985.501秒；拟4500+TERM30包括每epoch完整状态fsync/祖先链持久/原字SHA回读。原dev64另900+30，原max6400不截断，fresh两轮265条/68calls/4warmup。正式trainer已绑定actualnative但尚未运行，full64正式是否启动由root根据实测统一决定。纯推断草稿保留先原始IDs后raw再有限codec，三项epoch2真实收据仍pending。

## 正式事前协议（探针后授权）

主线程已基于实测统一授权fresh265×2/68calls/4warmup的4500+30正式训练，以及durable epoch2后原dev64唯一900+30推断。首OOM/非有限/硬截止停，不降精度/改loss/扩预算、不重复生成。原dev唯一评分为a179全预测literal∪有效患者关联，1072实体/391旧关系/35public-new严格保留；完整raw与最终pred先冻再单次v2，旧.746081965复用，不评分patient-alone或扫组合。Score严格升记新候选best，macro下降必须披露；用户达标是Score≥.8并通过格式/来源/复现，内部折macro筛选不另作用户硬条件。协议a0b2b77b在任何新dev质量前冻结。

正式源/计划661f60f4已准备；推断三个epoch2实际SHA保持pending，未伪造实际模型身份。probe独审02881279 PASS只覆盖工程单步，不当正式训练或质量证明。

正式部署04:38:21原intent后唯一213363B上传块返回UNKNOWN_OUTCOME；mkdir先成功，未解包/未formalCPU/无GPU启动。04:39:24跨路线同A100只读也UNKNOWN，按共享退避到≥04:40:25只核原目录库存，若块已落只复用，不重mkdir/整包/start。这是连接观察失败，不是模型训练失败。

恢复库存实际确认213363B原块未落，3个≤70KiB标准库小调用全部成功、组装原归档388e36不变，20成员全SHA和compile通过。正式CPU10.454秒rc0：265/64、6166 max、860405 full/9818监督，source9c794、manifest41659b1e，CUDA未初始化/无权重或forward。其excluded读取为dev16纯输入核ID，未读dev targets/gold；不混称完全未读dev数据。冻结实际入口清单后将fresh gate启动唯一4500+30训练。

## 唯一正式作业

04:42:33.455619Z `proc_5b00069a-f26c-45c5-a559-c9aa3e7d1d64` / PID120406 已启动，A100 GPU2 fresh UUID/无进程/持久空间门槛通过。4500+TERM30，近似TERM05:57:33/KILL05:58:03，真正截止由remote driver-started确定；源/actualCPU检查点46e6829已push。控制目录 `sequence-round2/patient-first-full64-formal-v1/control/train`，输出同root `/run/formal2ep`。未知观察只读现存job/小状态，不再次start。

单一dev端点已静态冻结87633491/policy901b5b2f，等待真实训练与推断小收据再绑定；203成员、64生成IDs独立decode和codec/strict并列、只a179 literal union一份预测/一次v2。原政策a0b2的Score升best及macro披露口径不变。后续推断仍依完整durable epoch2，不使用probe或部分epoch。

04:45:40首次常规观察：原child120423已完成基座实际指纹/完整编码审计/载模，进入formal_training；无失败或终点，不冒称实际步数。终点控制器 `finish_setup.py` 已准备但未执行，仅在真实epoch2/fullstate齐后绑定3SHA、受限部署两小件并绑定单一dev端点。输入级窄查有2个原候选字面字符串（increased apoptosis、cardiovascular），它们与其余候选同样保持；辅助全HPO假设过严失败单独保存，不改任何训练/推断/端点源码或筛掉候选。

统一正式入口独审468a1f84 PASS：训练数学/来源/20件实际部署/CPU、完整状态祖先fsync、原dev单臂端点和后继收据绑定控制器均无具体阻断。两文件后继部署是逐文件原子，不冒称跨文件事务；UNKNOWN须原路径库存后接续。训练继续原proc/4500+30，不新增探针、早停或评分臂。

04:48:59原作业实际epoch1 step5，loss .304094387、lr1e-4、当前峰allocated35,101,749,248B；step2首非零更新grad7.833115/三族B真变化。04:54下一常规只读返回LUNA_OFFLINE，当前终态未知。保持原proc5b00和remote4500+30，不把连接状态当模型失败，不重启/延长预算；退避后只读原目录恢复。

## 原正式训练 OOM 与独立恢复（05:00 后）

真实 wrapper 557.666s/rc1，04:51:51 在 trainer 第573行 backward OOM，申请5.71GiB。stdout最后仅step10；未保存精确失败row/step或CUDA算子，不把它写成第10步失败或前向float失败。冻结顺序CPU推导最长6166位于第12组第4条，只是候选假设，不是观测身份。GPU实际释放，epoch/staging库存为空，无可exact续训的optimizer/RNG状态；无推断/新dev质量。原最长单步PASS不覆盖累积和Adam状态，见main H51。

[统一失败清单](../evidence/patient-first-full64/failure-terminal-manifest.json)和独审f7668be7保留全部原件，完整trace/日志/zip外存。后继只改完整BF16 head之后的监督位置FP32 CE，目标Torch真实CPU先核loss与梯度，再240+30 fresh真实工程探针覆盖Adam状态＋已有累计梯度；不改数据、精度、attention、optimizer或每row mean/实际group分母，不同时改allocator。5.704GiB完整logit大小相近只是形状推断。完整fresh重训待root依据新证据统一决定。

### 新选监督CE工程入口

目标Torch2.6.0+cu124实测288例与完整tiny head H/W梯度、group1/2/8均通过，1.843745s；原生83db两loss函数沿用，源e6b1无改。第一次传输构造SyntaxError未执行，窄修后真实合同通过；另部署保护因CPU-result已存在而拒绝，库存确认9成员exact后仅补smoke依赖与plan，UNKNOWN结果先查库存。上述均无GPU。小收据见[CPU](../evidence/patient-first-full64-memory-recovery/actual-Torch-CPU-summary.json)。

固定probe只变dfd0759f选监督CE；保留全BF16head与原临时输出生命周期，每rowmean/8，非零LR1e-4先完整第一组8条产生真实Adam状态，再原第12组前三条累积与第4条6166完整backward。诊断不重现已丢失的原失败row、不重放前11组或正式warmup，不存权重、不质量；12条44305tokens，硬240+30。逐行记录stage、内存，最长前实核已有grad与Adam库存，三族B变化、base无梯度。最长通过仍不保证完整两轮。

### 带实际 Adam 与累计梯度的 probe 完整通过

唯一proc_b7a810e6于05:08:43启动，147.295s/rc0完整；实际计算窗口79.901s/44305tokens、12forward+12backward、1次完整group8更新；最长前已有496梯度与496 Adam参数状态，三族B已真变，最长loss .157666758，全部数值有限/base无grad。峰allocated25,031,000,576B、reserved29,976,690,688B；结束free11,769,544,704B是结束值，非峰headroom。05:11:10.524555Z卡0/40960MiB且无目标进程，权重丢弃。45小产物完整SHA核验，见[终点](../evidence/patient-first-full64-memory-recovery/terminal-manifest.json)。

新正式source1203139e仅instance CE hook与失败上下文记录，原编码/批序/优化器/两epoch保存函数不变；22件实际部署/compile/ancestor fsync，free57.076GB。token线性粗估两轮约3103s，另约66s加载及两次完整状态I/O，固定4500+30预算不变，不保证全程。固定原64推断900+30、单一a179并集先freeze再唯一v2不变；没有读新devgold/无正式重训launch，交root一次统一决策。

### 新fresh正式作业已启动

root已依据53e98230与c85fa31c统一授权。精确已推Git dcf7a610直接由git rev-parse传入，fresh UUID/空卡/空间通过；唯一proc_7f51657c / PID141616于05:13:13.317Z启动，4500+TERM30，近似TERM06:28:13/KILL06:28:43，remote driver-started为权威。新run独立、不从probe或旧失败续训；原数值合同与两持久epoch、后继原64/900和唯一a179终点不变。操作与连接退避见[恢复入口](../patient_first_full64/memory_recovery/OPERATIONS.md)。


### 05:18 本地环境重连后的原作业接管

原本地watcher PID108070已不存在；05:17:47实际原状态最后打印epoch1 step5、loss .302332392/峰allocated23.892GB，05:18:35独立只读核driver141616及trainer141633命令仍指向原selected-CE计划，A100 used28305MiB/util68%。未重新start、未改变源/数据/预算，没有完整epoch或新dev评分。新的本地只读watcher使用独立reconnect-0518命名空间，每180秒一次原状态读取、仍于原KILL+300止；恢复证明[小收据](../evidence/patient-first-full64-memory-recovery/reconnect-0518-status.json)。原目录中preparation-time的`formal_execution_authorized_by_root=PENDING`字段保持原字，不作为当前状态；实际统一授权、job和已推源码才是本次运行依据。
