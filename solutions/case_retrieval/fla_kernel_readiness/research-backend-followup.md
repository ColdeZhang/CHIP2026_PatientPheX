# FLA 独立后续诊断提案（未执行）

2026-10-24。目标是判断 FLA 是否值得作为独立后端继续研究，以及能否节省完整推断/训练成本。当前仅阅读既有源码、结果和纯输入 metadata；没有 GPU、forward、下载、标签读取、重评分或修改现行 native 作业。建议新 namespace `research/fla-backend-followup-v1/`，与原失败目录分离；具体启动由 root 在全文逆序终点及 gap addendum 后决定。

**结论：有一个明确未解决的数值问题，值得先做小型证据采集；现有证据不足以切换后端。** 原 raw-logit 组合门仍然失败，mean KL 通过既不推翻原决定，也不足以永久否定另行注册的后端控制。

## 已证事实及缺口

- 原 probe `b8aca0d2`、plan `f16cc09a`：先单 GDN 块，再完整裸基座的合成 128/255/256/513 输入，最后短 LoRA。attempt1 因缓存字典检查器故障退出；attempt2 `c6351a49` 只修检查器，沿原截止，不是新 600 秒授权。H43 的修复与随后 logits 失败是两个结论。
- 原 128×248320 完整 logits：relative L2 **.0905751226 > .01**、max abs **13.03125 > .25**、mean KL **.0004124003 < .001**。原组合合同失败；后续 teacher-forced/cache/greedy 结果未保存比较，255/256/513 与 LoRA 步未执行。原 logits 没有保存，不能事后重算居中误差或声称差异是共同平移。
- 单块 fused 首次 forward/backward **140.3986 秒**含冷 JIT；三次 warm forward 中位数 native **.00309257 秒**、fused **.00634079 秒**。完整 9B 只留下 128 首轮 prefill native **.131192 秒**/fused **.077762 秒**、8-step decode **.851928/.935022 秒**；没有完整模型三次热测，不能据此宣布稳定加速，更没有长输入或训练吞吐证据。
- 现有私有依赖：`.100` 的 `research/fla-kernel-readiness/private-v1/target`，fla-core **0.6.0** wheel `b0708b70`、einops **0.8.1** `919387eb`；Torch **2.7.1+cu126**、Triton **3.3.1**、TF **5.20.2**。旧 GPU import gate 已证两 GDN 进入 FLA、两 convolution 保持 native。当前远端存在性/空卡未重查；CUDA 隐藏时 FLA 导入会触发 Triton driver 错误，不能把未来 GPU 可见 import 伪称纯 CPU。

## 唯一建议的最小数值采集

先固定两个范围，分别报告，不混成原 probe 重试：

1. **128-token 合成输入、原裸基座**：复用 `synthetic-prompts.json` 的第 0 条，保存全 128 个位置、全部 **248320** 词表 logits。新鲜执行固定 `native A → FLA A → native B → FLA B`，每次相同输入、权重、RNG、shape、精度、SDPA/TF32、cache 初态；分别刻画 native 自身重复和 FLA 自身重复。它是新证据，不恢复旧 raw，不重算或放宽旧失败门。
2. **真实 14390-token 纯输入、固定 old4f48 适配器**：优先复用已完整核验的 reverse index19，`row_id=12455887:relations:40`；CPU `41942c56`、audit `8af1a815`、encoded `dc67ee47`、prompt `6232de0c`，native IDs SHA `3681c64dcc3e458df7d81d07a01b8e3b4c7a288cfaacdfa9f0dde91d10be95eb`。这组权重与裸基座组分开，不能把差异归因跨组比较。5556 仅在旧摘要搜索出现，当前未另核完整 native-ID 链，故不作为默认入口；不截取 14390 冒充 5556。

长输入保留**完整上下文 forward**，仅 LM head 选取预注册的 32 个位置：`floor(i*(L-1)/31), i=0..31`，包含首尾；位置不能看误差后挑选。实际 modeling `b4b17fe0` 第1721–1722行支持 Tensor `logits_to_keep`，在 head 前选 hidden states，不产生 `[1,14390,248320]` 全 logits。另用原合成输入的固定 8 个 teacher-forced token 建立相同续写历史，每步保存完整词表分布；这不是模型生成的任务答案。每臂独立空 cache，按 H43 已修 schema 记录所有层/键/shape/dtype，不能跳过未知状态。

**先保存再统计/断言**：每次 raw tensor 的实际 dtype、shape、bytes、SHA、input/RNG/权重/绑定/执行次序先落盘并 fsync/reopen；非有限值也保存原件和 finite mask 摘要，再给统计 null 并停止。Git 只保存小报告/manifest，张量留仓库外持久目录。128 全量与长输入选择位置的 head 形状不同：两后端必须使用相同选择方式，先在 128 组额外保存原 head 输入及固定 head 权重引用，用同一冻结 hidden 做选择前后 head 检查，避免把投影形状变化混入 GDN 归因；此附加检查属于未来探针，尚未实测。

按 FP32 上界估算：128 全词表每次 **127,139,840 B**；长输入 32 位置 **31,784,960 B**；8 个 decode 位置 **7,946,240 B**。上述各四次合计 **667,484,160 B（636.56 MiB）**，额外小 hidden/state 摘要和元数据另列，设置原始数值包 **1 GiB 硬上限**；缓存大状态不默认保存全部层全部快照。实际输出若 BF16 可原 dtype 无损保存并减少大小，不能有损量化后冒称 raw。完整长 logits 的 FP32 大小为 **14,293,299,200 B**，禁止生成后再靠切片补救。需要更大资产时先停止报具体缺口，不偷偷丢位置。

## 统计与解释边界

- 对每个位置、每对后端分别保存原均值、最大绝对差、L2；再用 FP64 归约计算 `z-mean_vocab(z)` 的误差以及均值偏移贡献。居中统计只能检查“共同偏移”假设，不能提前宣称原因。近零分母单独标识，不除零。
- 用稳定 logsumexp 得到 logprob/softmax，报告每位置 KL 两方向、概率 L1、mean/max/分位数；不能只给全位置平均 KL 掩盖局部差异。完整分布 raw 已保存，可在 CPU 按位置分块重算，不需要再 forward。
- 报告逐位置 top1 是否相同、两臂 top1–top2 margin、相应概率、全部分歧位置；在共同 teacher-forced 历史下比较，避免两条自由生成历史分叉造成不可比。若 native margin 大于该位置居中 logit 误差上界的两倍，则 top1 保持有直接界；其余只报告，不据低 margin 自动原谅分歧。
- 将 native A/B、FLA A/B 与跨后端差异并列。native 自身稳定性未知；不得把别的 R1 backward 重复差异外推为本 Qwen 前向天然噪声。原 `.01/.25/.001` 失败只引用原证据；本提案不设新通过标签、不自动开质量或替换 native。

## 后续成本探针与预算建议（均未获本报告的启动授权）

| 独立阶段 | 最小内容 | 建议上限/结果用途 |
|---|---|---|
| CPU 合同准备 | 固定 input/位置/版本/全部资产引用；合成数组验证常量偏移不改变 softmax、margin 界、NaN 先保存、位置映射及 head 选择契约；不导入会触 GPU 的 FLA | **60 秒 CPU**，先产源码/disabled plan；无金标 |
| A：数值证据 | 上述 128 全量与 14390 固定32位置各 N/F/N/F，真实 input/cache/绑定证据；记录每次冷/暖调用耗时，但每臂只有一个后续重复，不声称稳态吞吐 | **600+TERM30 秒**，含 fresh GPU gate、资产核验、加载、可能 JIT、保存；1 GiB raw 上限。这是保守预算建议，不是实测长速度 |
| B：推断稳态 | A 独审后另定是否做。实际 14390、old4f48，native/FLA各1冷+3暖，三暖顺序交替；正常生成 `logits_to_keep=1`，固定相同8-step teacher-forced历史，分别报 prefill/decode/完整路径与峰显存，I/O计时分列 | **300+TERM30 秒**；仅工程成本，不把32位置测量直接当实际生成速度。新JIT未在预算内完成则止，无自动延长 |
| C：训练可行性与稳态 | 仅在确需训练后端时另提。沿原合成128、后64 token监督、fresh r16/alpha32/dropout0、native CE、非重入checkpoint。每臂同初始LoRA/Adam/RNG；先真更新建立Adam，再3次同形状更新计暖中位数，核base冻结、梯度有限、非零B更新；数值对照要含native自身重复，梯度/更新先持久再判 | **600+TERM30 秒**起；梯度/更新raw可达约1.6 GiB，独立存储额度至少2 GiB。只说明短步稳态，**不能证明实际最长训练可行**；真实训练须另绑现行训练样本/监督位置，并按H51覆盖Adam+累积+最长，不以纯推断14390替代训练合同 |

这些阶段不自动串联。先由 root 决定 A 的源码准备/实际预算；有具体数值证据后再讨论 B/C 与任务质量控制。即使新分布统计接近，也要独立注册真实固定输出/质量比较，不能把工程诊断当已提分。

## 精确证据定位

以下路径以 `R=/workspace/patientphex-runs/round2-20261023` 为根；本次只读核 SHA。

| 证据 | 路径 | SHA-256 |
|---|---|---|
| 原最终独审 | `validation/fla-kernel-terminal-review/review.json` | `b78eec326865b71ac2c41e96b3ae8b7f2423928d3139e95716ae0a4f948ba2c1` |
| 原完整指标 | `research/fla-kernel-readiness/attempt2/gpu-numeric-attempt2/metrics.json` | `efaf1868a12d4d9138ff0ea6b5d3268f6d3204f7282b9f23ec2c98304a7f2365` |
| 原计时 | `research/fla-kernel-readiness/attempt2/gpu-numeric-attempt2/timings.json` | `296d845d017d94127d902e55e839b8af0de14076817018cf8fc7e32b7d54de78` |
| 原计划/源 | `research/fla-kernel-readiness/probe-plan.json` / `gpu_probe.py` | `f16cc09a5cfc8b0dddd8555e93e39a1ce56dd8538f3b51b6ca2c99f1c83267d8` / `b8aca0d289e70623d57e537e23751017c2b6cd260a6ab2cba6a740997bf1e01e` |
| 128 输入集合 | `research/fla-kernel-readiness/synthetic-prompts.json` | `1d57caa3bceaab8b9f024452c5c0a3f25502f62b591b571c1a73f0bccceec4ad` |
| 固定实际建模源 | `research/fla-kernel-readiness/sources/transformers-modeling.py` | `b4b17fe08d8d54da34a0fce78639666279ae7fd0a9e4ec06fce6602774707ca3` |
| 真实80 native IDs | `research/qwen35-fulltext-reverse-20261024/cpu-v1/full-inputs-collected/encoded.jsonl` | `dc67ee47b964eeef45f663f26d10bae17f8d674dd8816de8d998690d98a240ba` |
| 真实80 input audit | `research/qwen35-fulltext-reverse-20261024/cpu-v1/early-collected/actual/row-audit.json` | `8af1a8151605cb12be25278c22b45669a35174460168b839c732ae8663ce8a0f` |

仓库已有 [原GPU结案](GPU_RESULT.md)、[CPU资产与限制](CPU_REPORT.md)、[原计划](probe-plan.json)、[独审副本](independent-terminal-review/review.json)保持不变。当前报告只是后备研究，不是旧失败已修复的声明。
