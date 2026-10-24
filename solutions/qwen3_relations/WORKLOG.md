# 原生 Qwen3 关系工作台账

- 目标：为root已通过F1/F0的固定三源2of3提供full64原生Qwen3组件；完整265行两轮68updates及64条fresh全文推断，不单独评价原dev。整体目标仍原dev Score≥.8。独立Qwen并集F0负结案保留。
- 来源：`explore/qwen3-relations`，base `ff676a01615cce944b95ea8032e8d9bcd056cfdf`。只维护本目录。DocRE另线四组负结案保持。
- 当前：组合full64组件完整交付root：预测3d2e4ca3/138474B，16篇36患者1061实体/265关系；64fresh strict完整、0弃答/0cap，非关联字段与901 exact、格式0err/warn。两轮68updates及两完整state ACK已完成，GPU已释放。无本线原dev gold/quality；26件小补包与284件raw终包已逐SHA重开及文件/父链fsync，完整ACK已保存。
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

- 组合full64新shape实际64.748844s PASS：13indexedFB/1Adam，504梯度/504Adam1512状态finite，峰allocated21,628,089,344B，固定成本估2245.071238<2400。probe state丢弃后正式child1207643于12:09:42.129377启动；TERM12:49:42.129377/KILL12:50:12.129377，原两轮完整链继续。读probe子receipt一次UNKNOWN只属I/O，原formal身份已明确，不重启。

- 组合full64 epoch1完整ACK：12:22:09.652993 UTC/34updates，504adapter及504Adam1512tensor、scheduler/RNG重开exact，checkpoint6b681bb2，174655536B adapter df4ef6ca与349612141B state dc56ad04；保存核验3.212588秒（receipt计时不含最后ACK/父链尾操作）。继续原第二轮，无中间quality。

- 组合full64正式完整：train1478.328467s，68updates/1,732,990tokens/30,820监督，峰21.628GB；epoch2 checkpoint a8a02d4e/adapter59fb484a/state45e5c7e4，两ACK及原64推断start见 evidence/full64-component-trained-infer-start.json。

- 组合组件终态：284件/1,076,977B/8d1283bf逐SHA回收；12:43:53.079831完整new-only冻结3d2e4ca3交root，265关系/466候选概念/0质量。完整训练和raw解析出处见 evidence/full64-component-terminal.json；独立旧F0负保留。

- 最终持久备份：.3 terminal-component/component-source.zip 59309B/26件/bcca28c3，与run/small-terminal.zip 1076977B/284件/8d1283bf逐成员重开全SHA；两archive及父链fsync，backup ACK写入后父链再次fsync，见 evidence/full64-component-backup-ack.json。两个完整checkpoint留原.3路径，无额外GPU或质量。

- 补充只读查重已关闭重复提案：旧retrieved-json-example真实train32 TF-IDF top1完整训练输入/答案→Qwen3.5 SFT32/F1全80，Score .660942→.645783（两关系F1均降）。旧query是局部证据而非文章全文，不能外推所有RAG无效，但不据此重复同机制；规则明确允许prompt/RAG并保持train/held隔离。本次0GPU/0fit/0新quality，见 evidence/retrieval-demonstration-duplicate.json 与外部4959B短报告22550ca8。

- 患者跨度几何只读研究：当前PCS仅anchor语义池化+span RoPE，未显式输入锚点距离/同段；旧LR与identity bridge已用可学习几何，旧maxMIL亦完成，不能泛称项目从未尝试。现成train32 32篇/87患者/255锚点，1421/1503多患者实例的全锚点几何可区分患者，720/813正袋同段、另139/560多患者正袋所有实例有更近的其他患者（不是错误计数）。仅建议当前fresh24完成后，一个50参数全锚点距离×同段残差；零Parameter不耗RNG，旧LR3e-5/新表固定1e-2及同schedule，不扫描。外部报告与纯schedule算术见 evidence/patient-span-geometry-readiness.json；0新gold/fit/model/quality，未改owner源码/未启动新GPU。

- Query-token classifier查重结案：发现旧fixed-slot-binary已真实Qwen3.5 fresh LoRA、全部无答案CID×PID查询共享prefill/末hidden两类CE，train124/2506pairs/2ep32steps→完整F1 80/1557pairs。固定.5 Score .630248、expected-F1 .648033均低旧JSON .660942；另F1→F0二特征校准已完整仅+.001102。旧输入局部概念证据+global anchors，非全文；独立可训练head/8B/全文仍是变量但非核心机制未试，不改名BCE/YesNo重训。本次0新gold/fit/model/quality/tokenize/remote；见 evidence/query-token-classifier-duplicate.json。
