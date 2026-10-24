# patient-first F0 固定稳健性复验

## 目标、约束和成功标准

F1 patient-first单独负，oldSFT32∪new的固定secondary仅+.003479519，micro略降、macro升。真实目标是判断该有限互补在另一个预先留出文献折F0能否成立；不以更多训练或多试组合代替成功。

按主线程2026-10-24授权，复用旧strict train48完整204行与原F0纯输入58，转置同一闭集矩阵，fresh基座/2epoch/原优化器和解码不变。只允许primary新方案和secondary旧SFT48∪new，实体与候选池原样；两份完整预测共同freeze后各唯一v2，旧baseline复用不重评。保留条件沿用delta(micro+macro)>0且macro不降。未经后续明确安排不迁原dev/官方、不加第三组合/阈值，不使用F0 targets构建训练/资源。

## 当前授权与状态

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
