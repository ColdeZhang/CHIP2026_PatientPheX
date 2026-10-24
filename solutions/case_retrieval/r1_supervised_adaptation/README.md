# R1 闭集 JSON 监督适配：CPU 可行性准备

真实目标：改善关系分项，检验同一已缓存R1基座经过任务监督后是否仍有价值。现有R1集合／元模型OOF已结案，本任务不重启它们，也不把微调提案当既成成绩。

官方缓存 `overview.txt` §4.2只确认冠军使用Qwen3-8B、DeepSeek-R1-0528-Qwen3-8B预测、上下文/位置特征与LightGBM，并做五折OOF；**未披露这两个LLM是否SFT或LoRA、输入模板、训练长度/轮数、学习率或适配器目标层**。文中明确描述微调的是第二名Qwen3.5-9B，不能归到冠军。§3的±500字符、每批8候选、greedy与1800/2600输出上限属于官方零样本基线，也不能当冠军细节。

尚未定位到实际R1 SFT运行。当前R1是公开固定基座，正确发布接口的原80输出无项目监督训练；旧Qwen3.5有严格F2+F3 train32的124条闭集JSON、2epoch LoRA。新变量因此是**对同一R1做监督适配**，而不是再换集合组合。

固定研究方向：保留原124训练提示/答案（候选F2→F3、F3→F2两个train16生成）、原80 F1纯输入、原CID与PID字面身份。R1原发布模板生成前缀为BOS＋system＋User＋正文＋Assistant，**不预先打开think**；CPU将核所有124完整编码的prefix边界、JSON＋原生EOS，以及80输入与既有发布合同相等。直接监督现成JSON答案；不添加思维链或空think，不静默改模板/parser/采样。

CPU范围：只在`.100`已有R1缓存所在机器运行一次59＋kill1秒；CUDA隐藏，仅读配置/tokenizer/权重头形状，模型仅meta构造，无真实权重值载入或神经前向。检查LoRA r16/α32/dropout .05、Q/K/V/O及gate/up/down七投影的真实形状与参数；保留冻结head/embedding。通用selected-CE可作为内存候选，但只检查实际Qwen3 forward/loss挂载和CPU合成logits梯度，不能沿用Qwen3.5的GPU结论。

源与CPU计划运行前提交；完整token IDs/labels留服务器自有外部目录，Git只留短合同。当前没有GPU训练/推断或下载新模型；实际CPU结果、A30 24GB可行性、最小真实Adam/累积探针与完整单折预算将在结果出来后另列，禁止把推断19.5GB峰当训练显存。

## CPU完成与唯一GPU工程探针

CPU13.434秒完成，见[CPU结果](CPU-RESULT.md)。原124监督共401291输入tokens／7046答案tokens、max5519，原80输入IDs逐条等于既有正确R1缓存。实际published模板无预开think，监督纯JSON+nativeEOS，未伪造推理。

root已授权`.100 A30 GPU1`一次420秒+TERM30探针，源`gpu_probe.py`、固定`gpu-plan.json`。先最短570样本同参数/同dropout RNG比较原生与selected CE及真实LoRA梯度；随后8条累积完成非零lr Adam更新，再保留3条累积梯度处理最长5519。后者按H51覆盖已建Adam；仅诊断使用lr1e-4无warmup，不冒称重放正式训练顺序。基座BF16、LoRA FP32、252目标模块、fullhead不变，TF32关/SDPA/derived95fb原Yarn；无量化、截断或allocator暗改。

执行前fresh核GPU UUID/空闲/进程/空间和全部9件模型SHA；探针状态随进程丢弃，不保存adapter或产出质量。完整训练尚未授权。工程输出与失败在外部`R/research/r1-supervised-adaptation-readiness-20261024/gpu-probe-v1`；单独原CPU冻结plan不改。
