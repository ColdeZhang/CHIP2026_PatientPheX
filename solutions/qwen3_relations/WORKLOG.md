# 原生 Qwen3 关系工作台账

- 目标：固定strict F2+F3 train32原124行normal关系监督，原生Qwen/Qwen3-8B非thinking两轮，完整F1全文80推断；新单臂与fullsupport整体并集两端点中至少一个Score严格高于.686338981才考虑F0。原dev.8总目标未达；本线F1固定并集已过内部继续门，另一折F0进行中。
- 来源：`explore/qwen3-relations`，base `ff676a01615cce944b95ea8032e8d9bcd056cfdf`。只维护本目录。DocRE另线四组负结案保持。
- 当前：旧独立并集F0负结案不变。root固定三源2of3两折正（F0 .724457355332，审99e5a6d8），已另授权原生Qwen3 full64作为组合必要组件：新shape240+30→fresh2ep68updates2400+30→64fresh1200+30；只输出901实体壳上的new-only，禁止本线原dev gold/独立quality。CPU合同与源计划已固定，等待.3 fresh门后唯一控制器。
- 原生标准模型与R1-0528-Qwen3-8B为不同发布权重；两者都是全注意力，不称首次attention或冠军复现。历史Qwen3 Task1 train48→F1已做过，strict32关系SFT＋fullF1未见现有终点。
- 固定HF b968826d9c46dd6066d109eabc6255188de91218，ModelScope26028140be3ee69b82b1d1450179ab71bb1121b9，Apache2.0。基座8,190,735,360＋r16七线性LoRA43,646,976=8,234,382,336，每模型<10B。
- CPU11.979837s已核204原生/raw IDs及解码exact：train124/32总403151/max5534/监督7170；F1 full80/16总666591/max14322。源cb7f67e1与2a95bad1不变；no new gold/quality。
- 损失CPU最终actual-v4 1.842823s：4冻结logit/输出头梯度、2小Qwen3全梯度、1实际PEFT r16/dropout.05固定RNG合同，104数值比较maxabs0；28/28 LoRA梯度非零且exact、原全input/mask保留，结果e19d2858。早期依赖路径与观察器错误完整保留于evidence/cpu-loss-recovery.json，没有大模型/GPU/质量。
- 外部调研：R=`/workspace/patientphex-runs/round2-20261023`，`R/research/qwen3-dense-attention-readiness-20261024/REPORT.md` c34f1ae3。新运行将用`R/stage8-qwen3-relations`，GPU`.100`GPU1由root排期，不能依据旧空卡状态跳过fresh门。

- GPU唯一控制器已11:01:33.852 UTC启动proc_d5a8edba/PID2729926；probe子PID2729930，原TERM11:08:33.930/KILL11:09:03.930。freshGPU1 0MiB/noapps/free24164MiB，盘408.48GB。部分只读MCP观察UNKNOWN，已恢复原启动回执，无重复start；结果待原job。

- Probe已78.736652s PASS：1原生FB＋13indexed FB＋1Adam，全部Pareto、原14322最长cache prefill通过；13资产共16,397,443,036B全SHA核完。实际同logits loss差4.768e-7在预写容差内，梯度exact；勿称loss bitwise exact。估2ep973.507s<2400；已discard后fresh正式子PID2730123，11:02:53.552177启动，TERM11:42:53.552177/KILL11:43:23.552177。

- epoch1已11:08:43.488205完成16updates并durable ACK：504adapter、504Adam/1512tensor、scheduler与全部RNG重开exact，checkpoint139e8be0，adapter174655536B/Adam等state349612141B；4.228s。原正式窗内继续epoch2，不评分中间checkpoint。

- 正式两轮完整PASS：32updates、248行曝光/806302tokens/14340监督tokens，两个durable ACK；train686.263s、phase693.697s，11:14:27.312087完成，峰21179588096B。全80fresh infer子PID2731613已11:14:28.408008启动，TERM11:34:28.408008/KILL11:34:58.408008；未读gold/未评分。

- 已完整终态：80/80 fresh、80 strict complete、0弃权、无cap；infer504.491s/child505.297s。11:22:53.883520实际GPU1 0MiB/noapps，已交还root。
- 固定F1两pred共同freeze554e1c92（11:24:22.324593）后各1v2、旧0，格式0err/warn：single .653487056578，fullsupport_union .689312034492，比强.686338981272高.002973053220。union micro .720737327189（−.017910503285）、macro .621571874990（+.029802716166）；Task1不变，471→565关系，新增94=25TP69FP。只有预写union通过F0候选门，尚无F0/原dev泛化提升结论。
- 终包892169B/338件，SHA46d568798e7c2d672d0363a7a065213266621a6a6993f2494a4800085b2d96d0，各件全SHA回收；整包read_file一次UNKNOWN后128KiB分块恢复，无重复train/infer。详见evidence/terminal-pointer.json和外部terminal-manifest.json；根共享记录由root收录。

- 最终source/两质量终点补包108292B/48件，SHA92004344，已在同.100真实重开全SHA并联核原338件46d56879；evidence/quality-backup-ack.json。交validation只读终审；接下来仅CPU核F0原祖先/完整全文CID和可用runtime，未授权/未启动新GPU。

- F0本地祖先/全58 append-only原输入已核：原124训练F2/F3，两train16候选生成器和训练32均排整个F0；F0候选层为train48（与F1层train32不同，明确披露），semantic/CRF实际manifest同trainSHA且排F0。完整58/16篇47患者/415concept保持，无F0gold/fit/GPU/评分。evidence/F0-ancestry.json 44493fcb；仅CPU原生分词脚本与plan4e1706b6已预写。

- F0真实CPU5.807805s PASS：完整58/16篇47患者415concept/1166pairs，503003tokens/max16142，拒绝cap16384/gen1600，原生/raw IDs与双解码全部exact。原两epoch文件/ACK及所有parent目录链已单独纯I/O fsync补证，旧产物不改。root已授权待目标资产/runtime/fresh门后纯58推断900+30，B0保护union唯一1v2/旧0、raw-new-alone仅供root融合。当前.3实际Qwen3同13基座缓存，Torch2.7.0cu128/TF4.50；需复制已核完整TF4.51.3私有runtime+PEFT与174MB原epoch2adapter，未GPU。

- F0迁移完成：原182807574B/b357f150/2617件固定包，public tunnel约103s仅6MiB，已明确停止自己receiver/server并关闭tunnel、保留partial。改.100→.3私网单包bearer/180TTL/20MiB上限，小range身份核通过，16.043463s全SHA+fsync完成；服务已停止、secret删除。目标CPU8.819904s核实际Torch2.7.0cu128/TF4.51.3/PEFT.15.2/tokenizers.21.2等，58原生IDs/双解码与.100编码全exact、504adapter头43.646976M参数，未加载NN值/未CUDA。f0-plan.json与唯一union质量plan已固定，等待fresh GPU0门后唯一900+30。

- F0 唯一推断已实际启动：11:46:42.243320 UTC，proc_8618168a / PID1206030，child1206034；TERM12:01:42.243320、KILL12:02:12.243320。首观察3/58，非训练；启动和全卡只读门见 evidence/F0-start.json。

- 条件full64 CPU完成：265/64原行 native token/双解码与shift mask精确；train866495 tokens/max6155/监督15410，dev509609/max13175。2ep68updates，Pareto(4992,116)/(6155,113)。证据416da6ef，成本推算见 full64_readiness/cost-and-conditional-plan.json；尚未获或执行full64 GPU/原dev质量。

- F0全58实际终态成功：428.808083s、rc0/no-timeout，11:53:51.051205 UTC GPU0回4MiB/noapps，已交生成线Sap；evidence/F0-release.json。收包/冻结质量仍在CPU进行。

- 纯I/O暂阻：terminal.zip 464451B/202件/SHA2549e153，收2/4块后read_file返回LUNA_OFFLINE；完整GPU成功与11:53:51 release有效，不重start。当前.3由生成线Sap owner优先恢复，我暂停重复poll；恢复后运行R/stage8-qwen3-relations/collect_f0.py复用已存块，仅补缺块。尚无F0 gold读取/新质量。

- F0最终：202件464451B/2549e153全SHA恢复，58/58 complete、0弃答/0cap。11:56:03.617963两pred共冻bda8a219后唯一union1v2/旧0，Score−.023320908898，micro−.060380147610、macro−.032903487983。raw75dbdc65已交root/validation，独立负不剔root预锁来源。见 evidence/F0-terminal-pointer.json 与 F0-result.json。

- 最终持久备份：同.3 quality-source.zip 76286B/29件/SHA1ddde88e，与原terminal202件2549e153逐件重开全SHA通过，两个archive和父链fsync、ACK写入后父目录再次fsync；evidence/F0-backup-ack.json。本线独立full64已按F0门关闭，raw继续供root预锁融合，GPU和I/O均无在途任务。

- 新组合full64控制器唯一12:08:35.986798Z启动proc_e5d04680/PID1207543，source f69d88d0/plan5490e564；初次source上传LUNA且实际未落文件，恢复后v2源成功，无先前GPU。freshGPU0 4MiB/noapps/盘169.31GB。精确probe child/截止见 evidence/full64-component-start.json；原240→2400→1200各+30自动有门顺序接续，原dev不独立评分。
