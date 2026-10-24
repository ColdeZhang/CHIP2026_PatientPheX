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
