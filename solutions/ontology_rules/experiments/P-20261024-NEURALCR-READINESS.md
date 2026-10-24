# NeuralCR 完整库存与原生接口准备

目标是核实一个不同于现有三种 PhenoTagger head 的公开表型抽取资源能否合法、完整地加载并给出准确字符坐标。本阶段不读取赛事输入或标签，不产生质量预测，不使用 GPU。官方来源、训练谱系与 PhenoLinker 有界检索见 [只读研究](../public_neuralcr_readiness/readonly-inventory/review.md)。

固定 `data-team-uhn/NeuralCR` release `1.1`、源码 `1af8851b5e4a1ebeec7ced6c2f50d2dd34d4db0f`。ModelScope 准确身份接口没有可核同 revision 副本，采用官方 `HP.tar.gz`（583,483,718 字节）及 `pmc_model_new.bin`（1,345,465,351 字节）。上游未公布摘要，本地完整 SHA 将在实际下载完成后计算；小 Range 证据不是完整文件校验。

当前生产链已含一个 full64 LoRA，总参数 **9,597,467,990**，余 **402,532,010**。完整 fastText 输入词表及子词矩阵加 output 矩阵共 **333,363,400** 参数；CNN/10 ensemble 的实际 H5 形状待完整文件。默认用完整库存判断是否只能替换 BioBERT 的 121,587,826 参数，不因推断可能未使用 output 矩阵而先行扣除。任何后续裁剪必须是单独明确的部署契约，当前未实现。

## 实际准备状态

- [原下载计划](../public_neuralcr_readiness/download-plan.json)和源码于 `13e45d1` 推送后启动。2026-10-24 00:20:44.810536Z，A100 私有 `public-neuralcr-readiness`，`proc_1d81e43a-1d9a-458d-828a-ce1e6a725ef0`，600 秒截止 00:30:44.810536Z，TERM30；下载不使用 GPU。两资产分别串行 Range，合计两网络线程。原 `.part` 不覆盖。
- [CPU 依赖准备](../public_neuralcr_readiness/runtime-preparation.json)实际 6.786 秒通过，耗时包含小 wheel 网络、安装和 import。仅将精确 SHA 的 fasttext-wheel 0.9.2 装入自有 overlay；复用只读 TF2.12/h5py，GPU 列表为空，三个 TF1 compat 符号存在。没有加载模型或前向。源码与计划在 `a133cc9` 事前推送。
- 00:24:50 的[进度证据](../public_neuralcr_readiness/download-progress-00-25.json)：HP 已写 18,874,368 字节、fastText 134,217,728 字节。后续块也变慢，不能仅解释为首次连接等待；旧程序没有分离握手/首字节/流时间。原 600 秒继续执行，不自动延长。
- [续传准备](../public_neuralcr_readiness/resume-preparation/resume_assets.py)只准备四路总并发、8 MiB 缺口 Range。主线程已授权独立 attempt2 **1800 秒+TERM30**；须先确认原作业退出，再绑定不可变 prefix 的长度/SHA、原 ETag、assetID，写入独立目录，最终全文件拼接核长度/SHA。当前没有正式 `resume-plan.json` 或新网络作业。[冻结准备协议](../public_neuralcr_readiness/resume-preparation/preparation.json)不宣称首轮按时完成，也不授权自动第三轮。0.240 秒本地合成[传输契约](../public_neuralcr_readiness/resume-preparation/transport-check.json)通过中途断流续写、ETag 变更拒绝和原 prefix 不变；它不证明真实远端吞吐。新 wrapper 将保存真实 returncode，原 MCP 仅暴露 running，不能凭超时推测其 rc。

## 完整文件到齐后的最小检查

[已冻结 CPU 计划](../public_neuralcr_readiness/cpu-preparation-plan.json)仅安全提取 `HP/ncr_weights.h5`、`HP/config.json`、`HP/onto.json`，拒绝链接与目录逃逸，不反序列化 pickle、不提取外部标注。核 H5 唯一对象和完整 fastText 字典/两个矩阵至精确 EOF；异常头结构先保留证据，不能把未确认布局直接交 C++ 或私自修改。

原生加载另限总 180 秒，四个事前固定非空合成文本，保持 10 ensemble、原 `.8`、1–7 gram、原 overlap 政策。预先允许的一行兼容改动仅 `tensorflow` → `tensorflow.compat.v1`；数学不变。检查实际权重恢复、概率有限、返回字符坐标与真实打分词串/原文的对应。不把接口通过当质量增益。首次故障保留，必要时仅一次有证据的接口修复；未完整下载/库存不足时不进入赛事质量。
