# patient-first F0 固定稳健性复验

## 目标、约束和成功标准

F1 patient-first单独负，oldSFT32∪new的固定secondary仅+.003479519，micro略降、macro升。真实目标是判断该有限互补在另一个预先留出文献折F0能否成立；不以更多训练或多试组合代替成功。

按主线程2026-10-24授权，复用旧strict train48完整204行与原F0纯输入58，转置同一闭集矩阵，fresh基座/2epoch/原优化器和解码不变。只允许primary新方案和secondary旧SFT48∪new，实体与候选池原样；两份完整预测共同freeze后各唯一v2，旧baseline复用不重评。保留条件沿用delta(micro+macro)>0且macro不降。未经后续明确安排不迁原dev/官方、不加第三组合/阈值，不使用F0 targets构建训练/资源。

## 初始授权与状态（历史，当前见台账）

- 已读当前main规则与H46/H47：每个独立模型≤10B，允许模型组合；不能把流水线总和当门槛。
- 新外部目录：`research/patient-first-closed-set-F0/`。自有helper只写其`cpu-preparation/`，做来源locator、先冻结plan、新204/58转换/分词与训练矩阵还原；不改Git、不MCP/GPU、不读F0 gold。
- 下一步核旧48实际native/A100资源与新最长tokens。若新长度或实际显存证据需要，可安排一次fresh A100最长行更新180+30，诊断状态丢弃；不重复已验证旧合约。正式2轮预算依新实际tokens/成本与空间冻结，此时没有新GPUjob或正式训练启动。A800留Med，不抢卡；MCP未知状态只查原intent/partial、不盲重启。
- 204/58及旧来源目前是主线程给定范围，具体SHA/token计数须待实际核验，不能把计划数当结果。F1 frozen源/产物均保持，见[完整F1记录](patient-first-closed-set.md)。


## 实际本地CPU及必要新长度诊断准备

helper纯CPU6.715304s已通过，来源/矩阵/旧native全编码逐字核验。旧train204 eb133c31与原58 d387f5b8，新转置train ec64defd/F0input dae9b5d8；204/48/126患者、1477候选、4127对＝967正3160负全部双逆精确。总full660330→656837，监督11921→7612，prefix逐行+4；最长6063015:relations:0从5787→5769（5695prefix/74监督）。F0纯58/16/47患者415候选1166对，总195587/max6044，无targets；未用旧追加全文58版本。旧native逐行audit只有4字段，本次4字段精确且另核完整encoded SHA，不能冒称旧逐行有10字段。新10字段/全encoded b8d0d766与selected a6f4ba64另冻结。

旧204正式为A80080GiB，1855.103s/52calls/3warmup，peakallocated37.476GB/reserved78.498GB；A100仅其CPU准备，不能视为A100训练已通过。当前新5769比本路线已有A1005556单步更长213，按主线程条件授权需要一次新最长行native full CE更新180+30。同原rank/seed/模型/原生GDN/BF16/SDPA、非零lr1e-4一步；真实数值段从fresh load到optimizer update逐字不变，状态丢弃，不保存adapter、不推断/质量。只新增长度/来源绑定与失败时显存记录；首OOM/数值/预算失败停，不改loss/allocator或截断重试。

独立S0=`patient_first_f0/`，D0外部`research/patient-first-closed-set-F0/probe-preparation/`；native CPU新6输入硬绑定和逐204token/hash/来源核验源1d85789a、probe e72abdc4、policy ebdcc3ab、16件payload已冻结，实际nativeCPU尚待部署。ontology确认GPU2无后继，真正start仍fresh核UUID/空卡；MCP若未知只按intent恢复，不重复上传/start。当前没有新GPUjob或正式2轮源/预算启动。


## 实际A100部署中断（未运行CPU/GPU）

原source/计划已push并核02435386。首次实际run_command仅尝试创建自有`sequence-round2/patient-first-F0-longest-probe-v1`及检查磁盘，返回UNKNOWN_OUTCOME；没有进入任何write_file、native CPU或GPU启动。目录可能已建立，不能把未知当“未执行”重调deploy。D0保留完整cpu-deployment.zip/intent与16成员SHA，已通知root并暂停新增MCP。待统一恢复，先只读同目录库存；不存在再建、已空则原位复用，只续接原冻结数据，禁止重启不存在的GPU任务。此连接故障不构成模型质量或显存失败。[状态](../evidence/round2/patient-first-F0-deployment-unknown.json)。


## 部分上传恢复及离线正式源草案

03:05 同目录库存实际恢复，UID1045、自有目录已创建且为空、磁盘free58,139,037,696B。复用原377023B/16成员archive，无重复mkdir；part000有成功回执，part001在03:06:41返回UNKNOWN，尚未解包/nativeCPU/GPU。03:08:33间隔至少60秒后的单次库存仍UNKNOWN；遵照主线程持续授权按退避后同目录只读核验，仅补实际缺件，不把未知当未发生。原source02435386与docs f376d212已GitHub lsremote实核。

正式204训练草案e1aaf284保持旧9d2 CONFIG和8863B数值循环逐字一致，52calls/3warmup，新增新schema/三chain/656837full/7612监督绑定。nativeCPU实际收据SHA仍显式pending，入口硬拒绝执行；没有启动正式训练或把旧F1收据移用。旧F0原58 baseline仅定位现有pred63aefe68/metricse16ff5b0，Score .6954439475814587，未重评。


## 实际A100 native CPU通过与唯一probe入口

03:11第4次库存确认仅part000；随后只补缺part001成功，原archive9343e282全部16成员SHA/compile通过，无重复已成功块或mkdir。实际runtime Torch2.6.0+cu124 / TF5.20.2 / PEFT0.21.2、native GDN b4b17fe0、FLA不存在、CUDA隐藏且未初始化；当前两allocator环境变量均null，不能回推历史。新204 native tokenizer/六输入来源/全row与encoded哈希实际7.470150秒通过，最长5769/5695/74与本地精确。收据76ae7940、runtime ba808e77、selected55bb276a已完整回收，原件保留并核flat副本。

最终唯一probe计划ef9c6225绑定19个实际成员，180s+TERM30、非零lr1e-4一次fresh native CE、三family B更新/base冻结检查，不改allocator/attention/loss/长度；诊断状态丢弃。计划已明确正式训练未启动/未授权。正式源草案仅用新native receipt替换占位得990dfd0c，推断草案仅绑定此trainer得f1bba655；完整2轮仍待probe实际成本/空间预算及root安排，旧F1源不变。


## 唯一最长行probe实际启动

source/实际CPU/计划已push+lsremote c72d985c6bc006561d31642bf4874988caf32226。fresh GPU2 UUID2c22、0MiB/无计算进程通过后，唯一proc_68557469-8cf0-41f7-abb8-a7b3fb0c94c1/PID55826于03:13:02.591809Z启动；180+TERM30，approx TERM03:16:02.592/KILL03:16:32.592（以远端driver-started为准）。未重复start，诊断状态不保存；终点待完整forward/backward/step、显存/有限值/耗时和实际release，不能将启动记为成功。独立native入口eae78122通过，终点将同一组收口，不重复旧合同。


## 唯一5769更新诊断完整PASS

唯一proc68557469实际native80.301720/wrapper81.688066秒，rc0；真实单步13.935736秒，loss .4758164287、clip前grad5.950649261，三family B实际变化；248个B梯度非零、A首步0符合fresh零B。仅adapter在optimizer，base无梯度，FP32 optimizer states，未保存模型/推断/质量。峰allocated36,893,822,976B、reserved41,748,004,864B；free16,836,263,936B为**结束时**空闲，不能当峰时余量。03:14:24.493747 GPU2核0MiB/无进程，诊断状态丢弃，无自动重试。8件终点SHA全部回收，manifest50acff1b；统一增量独审e300f328 PASS，无新CPU/NN/评分。单最长行通过仍不证明accum8、后续optimizer常驻/多形状碎片稳定。

主线程现已给正式条件授权：同204×2/52calls/3warmup，A100项目持久盘按原epoch边界fsync/重开SHA保存完整adapter+optimizer/scheduler/RNG；3900+TERM30含保存，首OOM/数值/截止停，不改变数学/长度/精度或追加预算。完整plan和保存增量集中审查无具体阻断、freshUUID/空卡后执行；原58另600+30，两完整预测共同freeze再各唯一v2，旧SFT48不重评。此处仅记录授权，正式任务尚未启动。


## 正式固定源/预算与持久恢复合同

已冻结readiness be8e2798、train03e668ce、inferf2cbd7ce、endpoint143638ee。`patient_first_f0/formal/`为新正式源，旧probe引用990df/f1bba保持。trainer f08c只在原epoch保存边界增加逐file/dir fsync、原子rename、重开全成员SHA、durability-epoch-NNN排他收据；原CONFIG/52calls3warmup/连续两轮与整个run()去掉新I/O后逐字相同。唯一标准库真实小文件fixture0.0392s、20file+14dir fsync、3rename：两epoch成功、坏SHA无成功收据、已有目标拒绝，未读取真实weights/state。最终infer6ef4只绑定此trainer并修旧模板头文案，解码/严格parser不变。

3900+30含实际保存；按13.935736s最长5769与2×656837总token线性粗估3173.343s、约727s余量，**不是完整吞吐或显存保证**。A100项目持久盘最近free58.139GB，开跑fresh要求≥4GiB，两完整state估1.04GB；首OOM/数值/截止停，不改allocator/精度/loss/长度、不自动重启。每epoch保留adapter、optimizer/scheduler、Python/NumPy/Torch/CUDA RNG和元数据，恢复必须核完整SHA且另由主线程决定，不能声称GPU resume数值等价已测。

固定原58另600+30，只完整持久epoch2可推断。新policy d2b8已在质量前固定primary/oldSFT48∪new，实体947/415CID/1166对不变；finish92b3/uniondf82先共同freeze两fullpred与原raw/来源，再各唯一v2；primary独立提前evaluate受共同freeze+before-both marker硬阻断。旧baseline .6954439475814587直接复用，保持sum(micro+macro)升且macro不降。gold仅独立后置locator，SHA来自旧baseline既有metrics、此时只stat文件大小，未读gold内容或计算其新SHA。10件已核数据/辅助源在A100本地copy并逐SHA，另上传10件新源/输入/plan；实际部署/新formalCPU尚待完成。


集中审查发现初版f08只fsync checkpoint所在output，未明确持久化本轮新建output父链。保留before-ancestor-fsync原源/plan后窄补：通过显式PATIENTPHEX_DURABILITY_ROOT绑定A100项目根，核output在其下，在ACK前fsync output→项目根全部父目录；收据记录实际父链，driver再核根匹配。新trainer cc707253、完整run()逐字不变；仅一次新祖先链标准库fixture0.031782s通过（209380ef），旧310fixture不重跑。最终readiness1cb194ea、train4b70dbce、infer69c05df1，原endpoint143638ee不变；正式尚未启动。


## 实际正式部署和CPU预检通过

最终source543e98e已push+lsremote。A100自有formal-v1目录全20成员actualSHA/compile通过（10件同机已核副本copy、10件新源/输入/plan；archive5390d934）。actual formalCPU 9.527084s通过，source cc707、204/48/max5769/656837full/7612监督、26steps/epoch×2、warmup3、总参数8,997,081,600，CUDA未初始化、not_training=true。completeabbe4d58、manifest40409ec4、audit881cf950三件回收SHA exact；未加载权重或执行model forward。显式说明本预检读excluded F0纯input用于ID隔离，不读F0 targets/gold或新质量。完整入口独审收口中，按原条件授权在fresh卡和空间核验后启动；这里尚未宣称正式训练已运行。


## 正式F0训练实际启动

完整增量入口bcaf14d3 PASS（保存父链窄修、实际20部署/新CPU及双臂端点一组闭环）。source+actualCPU已push/lsremote9f9de5d。fresh GPU2 UUID2c22空卡与持久盘门槛通过后，唯一proc_ced868e2-fd8c-4ae4-b4b5-da75a8bb06a0/PID64875于03:26:28.486604Z启动。原204×2/52calls3warmup连续单invocation，3900+TERM30；approx TERM04:31:28.487/KILL04:31:58.487，远端driver-started为权威。模型/epoch/质量均待真实结果；不能把入口PASS或历史job running=true当当前完成。恢复看formal-preparation/train-{launch-intent,job}.json及A100自有formal-v1/control/train，不重复start。完整训练/持久epoch2后才启动固定原58，唯一双评；新8B∪9B单独CPU控制不改变本实验方案。


## 第1轮完整且已真实持久化

03:47:58正常只读监控取得首轮完成收据：epoch1/26calls，epoch_seconds1102.546863、mean_group_train_loss .2208504973；publisher于03:46:01完成5成员file/dir/全部祖先链fsync与重新开SHA，checkpoint总519,866,582B。adapter SHA6fc6d2d6、完整optimizer/scheduler/RNG state151bbe39、metadata9e44eb4a；所有receipt值为远端原件读出的JSON，当前观察保留其MCP原回包SHA，原件字节将在最终统一collection回收，不将重序列化观察冒称原件SHA。原invocation继续epoch2，无暂停/重启/新质量，已记录peakallocated37.416GB。此前03:37监控UNKNOWN已在≥60s后单次小读恢复，非模型失败；原截止未变。[首轮观察](../evidence/round2/patient-first-F0-formal-epoch1-observation.json)。


## 正式两轮完整训练与持久回收

原单invocation于04:04:29正常rc0结束：native2279.144899秒/wrapper2280.456254秒，两轮52calls/3warmup，无OOM/数值/预算失败。第2轮1106.754564秒、mean_group_train_loss .1009211968；峰allocated37,416,358,912B/reserved40,802,189,312B。训练指标仅证明训练有效，不代表F0质量。两个epoch的adapter+optimizer/scheduler/RNG+元数据各519,866,582B，全部file/dir/项目祖先链fsync、原子rename、重新开SHA；epoch1原receipt22591ba4、epoch2原receipt23eff58b。final adapter51f8c7a4、state9932ff4d、metadata6a0b9906；原小文件16件226,862B已实际回收并逐SHA核验，collection b9693ce9。04:04:29.167 GPU2空卡，仍保留给此camp的固定58推断。

04:06一次正常监控明确LUNA_OFFLINE，≥60秒后同小只读恢复，得真实完成终态；没有重启训练或把网络错误当模型失败。后续按冻结600+30计划只推断durable epoch2，完整58严格输出后才两预测共同freeze/唯一双评。终点汇集器已离线准备，仅核已有收据链、不读gold/权重、不执行评分，待真正终点后运行。

运行环境如实不同：旧48正式为A800/Torch2.7.1+CUDA12.6/Python3.11.17，新48为A100/Torch2.6.0+CUDA12.4/Python3.10.21；TF5.20.2/PEFT.21.2/native modeling SHA相同。保持旧数值配方并不证明跨硬件逐位等价，不能把极小差值无条件归因schema。main H48已针对R1 tokenizer链独立纠正；检索侧对Qwen3.5原80文本已actual核Auto与发布JSON IDs/双decode一致，不改变或重跑本固定F0。


## 固定原58推断启动

完整训练终点af86985ad2ea已push/lsremote后，按原授权fresh核GPU2空卡、持久epoch2的adapter/config/checkpoint/training_complete实际SHA，唯一proc_22400ec4-e939-41a7-988e-bef7c87ddc23/PID93337于04:08:22.106526Z启动。固定原58、600+TERM30，approx TERM04:18:22.107/KILL04:18:52.107；实际以driver-started为准。无新采样/阈值/旧baseline评分，尚无新F0质量。训练camp仍保留GPU2至完整推断真正释放；不因训练释放收据提前移交。


## 原strict58格式失败与有界后继授权

原推断04:10:32.994 rc1/130.785452秒，23行已持久，实际04:10:33.116 GPU2空卡；无F0评分/完整组装。按固定长度+row_id顺序，失败输入为第24条3563609:relations:8，3073input，允许患者O1P1/O1P2/O1P3/O2P/O3P/O4P/O5P/O6P/O7P，C000–C007。trace在validate_matrix顶层keys检查：可能非object、缺PID或多PID，尚未检查CID值。原runner在parse_final成功后才write/flush，故失败raw/IDs未保存，无法还原具体key差异；只能依据固定控制流推断EOS/计数/非cap/JSON解析已越过，具体EOS ID/输出长度均未知。原23raw SHA73235d91，7小件23939B完整回收，失败诊断0fd182ef；训练成功不受此格式故障否定，也不能将本次记成F0负分数。

主线程在未见新F0质量时明确授权独立恢复：精确复用23缓存，余35新生成（包含丢失那条，明确非精确恢复原响应），同epoch2/输入/greedy数值与1600cap，600+TERM30；先保存唯一raw/IDs/终止统计再解析。新有限codec允许合法PID子集，遗漏键仅弃答、不补模型答案；未知/重复PID、非法/重复CID、非object/错误值类型/cap/无EOS整row弃答，不扩其他格式规则。全58终态后两完整预测共冻再各唯一v2，旧指标复用，保留标准不改。原源/失败/23缓存单独冻结，不改写strict判断；新CPU/源准备中，未启动恢复GPU。


## 恢复源与新有限CPU合同冻结

新目录P/abstain-recovery，远端sequence-round2/patient-first-F0-abstain-recovery-v1；readiness27dc0883绑定新infer/driver/ops/原子raw_store/codec与600+30。模型/204训练/source/epoch2适配器不变，23原8853B精确缓存复用，35新生成含原丢失行；两次均do_sample=False且原源无manual_seed，不声称重新载模/KV后逐位重放原失败结果。只新增只读FiniteLogitsOnly硬停检查，返回原scores不改logits；其他generation参数保持。

新数据保存顺序为generation IDs/真实末EOS/cap/耗时先file+目录链fsync、原子排他发布，随后decode完整文本并持久raw，再调用有限codec；异常不会丢已取得的唯一生成IDs/raw。codecd60731e12在原23缓存全部与strict正集合/observed exact，27项新边界通过；合法非空PID子集partial，{}或任何非法row整体abstain，不补遗漏键。纯CPU0.00572秒，无gold/新模型/旧25或14/15重跑。原子保存新小fixture0.00352秒实际fsync后模拟codec异常，raw保留、重复写拒绝；隐藏CUDA导入新module仅Python，不导入Torch/加载模型。原strict失败独审a09c0e2c PASS，保持无分数。恢复endpoint另源准备，两完整pred先共冻再唯一双v2政策已在质量前freeze；此处尚未启动新GPU。
