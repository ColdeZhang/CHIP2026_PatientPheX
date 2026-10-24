# 独立 indexed-head 工程探针

**状态：CPU检查与本地部署包完成，未部署、未启动GPU。** 官方901 B推断优先；本目录不授权或改写既有formal训练。由主线程在官方接收后决定是否执行这一次420+TERM30秒工程控制。

`probe.py`从原已完成H51源`165e9bfc…`机械派生；原12条输入、顺序、105332 tokens、batch1、r16/alpha32/dropout .05、BF16/SDPA、非重入checkpoint、AdamW及一个非零lr1e-4更新均保持。原前8条真实更新，再3条累积，最后最长idx98（13549 tokens/61监督）；这仍是诊断顺序，不冒称正式warmup/完整训练。全部探针状态随进程丢弃，无checkpoint、推断或质量评分。

改动分两部分：

1. 对原12条中最短一条仅捕获一次实际backbone hidden，冻结该局部边界，对同一真实head做native A/B与indexed C。保存logits/loss/hidden梯度、非有限计数、RNG及容差判定后再断言。仅这个局部对照detach；恢复原Python/NumPy/Torch/CUDA RNG后才进入原H51顺序。该比较不证明全骨干逐位一致。
2. H51训练路径保持完整输入与LoRA计算图，用原causal shift生成监督位置，传Tensor `logits_to_keep`，对返回的`[1,K,V]`直接调用原生FP32 `fixed_cross_entropy`。不向模型传labels或额外shift参数，不二次移位；每row mean及`loss/8`不变。空监督、batch>1、输入shape不一致、模型未按索引返回均拒绝。

实际GPU只使用BF16，**事前容差**为logits/hidden梯度同时满足`rtol=.02, atol=.002`和relative L2≤`.005`，loss绝对差≤`.001`；native重复A/B也必须满足同门。FP32 CPU门为`1e-5/1e-6`及relative L2≤`1e-5`；FP64另有更紧门，详见`indexed_head.py`。这只是数值工程门，不是质量证明；失败保存原统计，不自动放宽容差或重试。

局部合同有额外前向，单独记录其峰值；进入H51前只重置峰值**计数器**，不清allocator缓存、不gc。结果分别保存局部/压力阶段及两者最大峰值，不能把局部head节省解释为整个训练等比例加速。

CPU在现成.100 Torch2.7.1+cu126、CUDA隐藏下唯一运行0.157秒：9个真实新helper合成训练梯度对照、9个同hidden局部对照和11个负例通过，包括NaN统计已保存后失败。使用固定native forward函数体与损失，但未执行实际PEFT/模型装饰器/真实GPU；真实PEFT转发将在H51路径覆盖。`source-checks.json`另核41项源码/计划/原压力机制，未改变formal源。

交接入口：

- `plan.json`：冻结输入/依赖、容差、目标UUID及总预算；`launch.py`有新鲜空卡、固定SHA、单次intent和超时保护。
- `entry.json`：待部署的新私有目录和argv，**目前无GPU job ID**。
- `.prepared/deployment.tar.gz`：151511字节、16个成员；仅源码和既有小探针输入，已逐成员核对。目录被Git忽略，owner只提交下面的源与小凭据。
- `source-manifest.json`、`ready.json`：精确源码/旧依赖/研究依据与移交文件清单。
- `CPU-result.json`、`cpu-run-intent.json`：实际CPU证据；重复工具封装留本地且忽略。

`build_probe_source.py`和`prepare_cpu_source.py`仅生成派生源码；`build_payload.py`仅构造本地包，无MCP/GPU调用。已有CPU唯一intent阻止无意重复。执行时由owner沿entry部署并核SHA后，按主线程明确决定运行launch；不调用旧formal/controller，也不把新探针状态用于正式训练。
