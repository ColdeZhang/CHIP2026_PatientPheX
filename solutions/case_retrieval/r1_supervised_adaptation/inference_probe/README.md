# R1最终适配器的单条最长F1探针

当前只有CPU源码准备：`template.json` 的 `adapter_binding=null`、`execution_ready=false`。没有绑定未完成的检查点，没有加载tokenizer/模型、调用GPU或产生新分数。静态检查结果在 `static-contract.json`；五个复用helper的原始SHA及来源在 `helper-origins.json`。

选择只来自原80条纯输入的已核原生token审计：唯一最长是index73、`6063015:relations:0`，5689 tokens，输入IDs SHA `dcebddee156a66fe29243111b0d577adcd2db303c1921af5deb63c04e0d45ca7`。旧6002长度来自已关闭的tokenizer重建接口，不使用。此次最多生成这一条，不自动启动完整80。

固定设置是240秒加TERM30秒、内部210秒软停止、4096新token、temperature0.6、top_p0.95、top_k0、seed20261024。每行独立重置torch及CUDA种子，与未来完整80的缓存语义一致。本次使用训练末轮适配器，不能复用旧base模型的row0/73响应。

绑定必须等待formal进程真实结束：`terminal.json`返回0且无超时，`actual/result.json`为完整124×2/32steps；只接受固定 `epoch-002/checkpoint.json`。核查checkpoint本体、adapter权重、training_state三个实际SHA，并核`checkpoint.files`全部成员，包括adapter_config；校验末轮持久ACK、248行/802582输入token/14092监督token、训练32篇排除整个F1以及源码/计划/数据/模型fingerprint。绑定只进行CPU元数据和文件SHA操作，不反序列化Adam/RNG，不启动推断。

源部署到独立服务器目录并核SHA后，绑定入口为：

```bash
python bind_completed.py --template template.json \
  --training-directory /home/zhangyuheng/projects/CHIP2026_PatientPheX/research/r1-supervised-adaptation-readiness-20261024/formal2ep \
  --output plan.json
```

主线程核绑定并通知执行后，唯一启动入口为 `python launch_probe.py --plan plan.json`。启动器检查A30物理GPU1空闲，只拥有和终止自己的进程组；保存退出码、超时状态和实际释放查询。相同目录有任何启动标记就拒绝重启。

运行时再次核所有80条published原生/generic/chat-template输入IDs及往返解码；直接从发布tokenizer.json构造generic wrapper，拒绝AutoTokenizer旧类重建。derived配置沿用`95fb72c1…`，只增加原已核YaRN attention_factor1.0。fresh BF16 base加载后逐张量核399个形状、dtype与有限值；复用已完成formal的全base SHA及未变化mtime，重哈希小配置，不伪称再次扫描16GB权重。

适配器用PEFT0.21.2 `autocast_adapter_dtype=True`加载，随后逐一对比保存与载入的504张量FP32 dtype、形状和值完全相同。避免先降BF16再升回FP32；不merge、不量化、不更换模型。学习参数全部冻结，推断不含训练loss或优化器。

原`raw_first.py`确保完整生成IDs和EOS状态先fsync，原始解码文本再fsync，最后调用原full80的strict/sparse/normalized解析器。五个helper保持原字节，不扩大整数PID的既有可逆表面规则，也不补缺失CID答案；缺失为弃权，不视作负关系。正常JSON失败或token封顶也是保留原文的最终结果，不重试；异常/墙钟超时保留已完成证据。报告normalized/strict有效性、CID覆盖、EOS、速度及显存，格式通过本身不自动授权80条或质量评测。

新检查点实际SHA尚未知，因此本模板不能直接运行。后续若另行授权完整80，这一条只能在相同末轮checkpoint、输入、tokenizer/config及采样合同下作为1条缓存复用。

## 当前调度暂停

root为本次唯一官方901提交优先，已明确暂缓新R1 index73探针及质量，等待这份官方结果文件接收确认后再恢复。当前只保存CPU准备；不得因formal完成就自动绑定/启动。新R1不参与本次提交。原formal仍按1800+30完成完整state持久。
