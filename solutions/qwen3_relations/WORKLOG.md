# 原生 Qwen3 关系工作台账

- 目标：固定strict F2+F3 train32原124行normal关系监督，原生Qwen/Qwen3-8B非thinking两轮，完整F1全文80推断；新单臂与fullsupport整体并集两端点中至少一个Score严格高于.686338981才考虑F0。原dev.8总目标未达，本线尚无质量结果。
- 来源：`explore/qwen3-relations`，base `ff676a01615cce944b95ea8032e8d9bcd056cfdf`。只维护本目录。DocRE另线四组负结案保持。
- 当前：查重/公共资产/真实CPU分词census和标准损失合同已通过；完整源与两固定端点已预写，已在目标runtime持久化原124/80全编码（9.115569s，train9b5e2fab/heldd4ec43b0），无新gold；两种Pareto为5152/116与5534/106。未GPU/fit；先等待schema固定F0真实释放`.100`GPU1。root已授权CPU通过后唯一420+30探针，discard后根据实测成本允许fresh2ep2400+30、完整80fresh1200+30；24GB若OOM不截输入，待A800另行调度。
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
