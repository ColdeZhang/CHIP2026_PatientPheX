# R1 闭集 JSON 监督适配：CPU 可行性准备

真实目标：改善关系分项，检验同一已缓存R1基座经过任务监督后是否仍有价值。现有R1集合／元模型OOF已结案，本任务不重启它们，也不把微调提案当既成成绩。

官方缓存 `overview.txt` §4.2只确认冠军使用Qwen3-8B、DeepSeek-R1-0528-Qwen3-8B预测、上下文/位置特征与LightGBM，并做五折OOF；**未披露这两个LLM是否SFT或LoRA、输入模板、训练长度/轮数、学习率或适配器目标层**。文中明确描述微调的是第二名Qwen3.5-9B，不能归到冠军。§3的±500字符、每批8候选、greedy与1800/2600输出上限属于官方零样本基线，也不能当冠军细节。

尚未定位到实际R1 SFT运行。当前R1是公开固定基座，正确发布接口的原80输出无项目监督训练；旧Qwen3.5有严格F2+F3 train32的124条闭集JSON、2epoch LoRA。新变量因此是**对同一R1做监督适配**，而不是再换集合组合。

固定研究方向：保留原124训练提示/答案（候选F2→F3、F3→F2两个train16生成）、原80 F1纯输入、原CID与PID字面身份。R1原发布模板生成前缀为BOS＋system＋User＋正文＋Assistant，**不预先打开think**；CPU将核所有124完整编码的prefix边界、JSON＋原生EOS，以及80输入与既有发布合同相等。直接监督现成JSON答案；不添加思维链或空think，不静默改模板/parser/采样。

CPU范围：只在`.100`已有R1缓存所在机器运行一次59＋kill1秒；CUDA隐藏，仅读配置/tokenizer/权重头形状，模型仅meta构造，无真实权重值载入或神经前向。检查LoRA r16/α32/dropout .05、Q/K/V/O及gate/up/down七投影的真实形状与参数；保留冻结head/embedding。通用selected-CE可作为内存候选，但只检查实际Qwen3 forward/loss挂载和CPU合成logits梯度，不能沿用Qwen3.5的GPU结论。

源与CPU计划运行前提交；完整token IDs/labels留服务器自有外部目录，Git只留短合同。当前没有GPU训练/推断或下载新模型；实际CPU结果、A30 24GB可行性、最小真实Adam/累积探针与完整单折预算将在结果出来后另列，禁止把推断19.5GB峰当训练显存。
