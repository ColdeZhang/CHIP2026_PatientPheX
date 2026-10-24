# R1 适配后固定 F1 完整推断

目标是取得原 80 条纯输入的完整输出，供父线程之后独立装配两个质量端点。只使用已完成的原 124×2、32 次更新的 epoch002（adapter `20f15b26`、checkpoint `2fbd4517`、state `f49f32af`），不训练、不读标签、不评分，也不做官方操作。

运行合同为一次 `900 + TERM30`，从 launcher 开始计绝对时间，870 秒软停止用于保留证据；不能在模型加载后重新获得 900 秒。父线程先提交源码和 plan，再负责唯一启动。Launcher 检查实际 A30 GPU1 UUID、空闲进程和磁盘；失败、超时均保留原目录，不自动重试。实际目录及父链至自有项目根目录 fsync，终态包含所有文件 SHA 和释放收据。

原输入顺序固定 0–79、共 249937 tokens。79 条新生成，仅第 73 条复用这次适配器的实际最长样本探针，其他旧基座缓存一律不使用。每条新生成重置 seed20261024；原采样温度 .6、top_p .95、top_k 0、4096 tokens、原 EOS 不变。基座 BF16、保存的 504 个 FP32 LoRA 张量逐一核对，原生 tokenizer/RoPE/模型实际接口验证整段沿用已成功探针；无量化、合并或精度回退。

每条输出依次持久化 `rowN.ids.json`、`rowN.raw.json`、`rowN.json`；每 32 tokens 保存 partial。`raw_first.py` 与三种 codec 保持原字节。缺失 CID 弃答，无效整条弃答，不补负。`full80_contract.py` 整体原样保留以固定 `parse_item`，只调用该函数；其中旧基座 `load_inputs` 包含旧 0/73 缓存与 10800 秒合同，**此入口不调用它**。新来源检查由 `probe_contract.fixed_inputs` 与 `full80_cache.check_cache` 执行。

第 73 条三个原件按字节复制，另写 `row73.cache-use.json` 标明复用。完整 `responses.jsonl` 按原序输出；其中第 73 条的组合包装设 `fresh_NN=false`、`original_probe_fresh_NN=true` 并携带 cache 引用，原件不修改。同时保存固定 mtime0 的 gzip 和解压 SHA。`result.json` 分别报告 79 fresh/1 cache、全部 row/IDs/raw 引用、严格及 normalized 有效数、EOS/cap、CID 覆盖和显存。完整 80 条冻结之前没有质量入口。

`build_plan.py` 只读取已完成 probe 的 plan、terminal/result 和三条原始证据的大小/SHA，生成部署 plan；不加载模型。当前 CPU 准备只做 10 份源码语法检查、6 份原 helper 字节相等、实际 tokenizer→模型→RoPE→LoRA 加载块字节相等。未在本地假冒执行远端资源检查；真正资源和缓存合同在唯一远端入口运行。

示例启动（由父线程部署并确认唯一 intent 后执行）：

```text
<固定 Python> -u <唯一部署目录>/launch_full80.py --plan <唯一部署目录>/plan.json
```

质量政策沿父线程前置合同：primary 为新 R1-SFT，diagnostic 为 oldSFT32∪新 R1-SFT；完整预测先共同冻结，之后分别一次 v2。与适配前 R1 的收益和同池强模型的收益分列，保留按 Score 严格升判断，macro 必须报告，明确两折复验。此推断源码不自动启动质量或下一折。
