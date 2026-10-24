# 人工结束思考后续写：公开方法与精确 token 重放可行性

实验 `E-20261023-009-thinking-budget-forcing-readiness`，2026-10-24 UTC。**仅完成 CPU、源码和计划；未启动 GPU、未生成新答案、未读 gold/targets、没有质量分。**

**后续单条诊断已由主线程选择，正在完成部署与备份。** 实际目标改为 .100 A30 GPU1，复用原生 TF5.20.2/PEFT.21.2；CPU 实核 FLA 不可导入、两GDN绑定均原生。实际 generation helper L2935–2936本已自动置`logits_to_keep=1`，新入口显式1沿用语义。15项新增presence位置反例均在真实HF CPU通过，未重做原14项或token重放。A30只读库存未找到旧32 adapter，已用独立两文件HTTP从A800复制：173181072B权重4f4815b6、1287B配置96564441均full SHA，19.224秒，不涉及warm新权重或GPU。实际[运行计划](../evidence/round2/thinking-budget-final-run-plan.json)、[CPU接口](../evidence/round2/thinking-budget-final-presence-cpu.json)、[运行环境](../evidence/round2/thinking-budget-final-runtime-preflight.json)、[传输收据](../evidence/round2/thinking-budget-final-adapter-transfer.json)。源在`thinking_budget_forcing/runner.py`，独立wrapper写真实returncode与资源释放，launch前保存唯一intent以防MCP unknown后重复启动。第一次只读runtime检查因内嵌换行转义SyntaxError在import前退出，失败source/收据保留在外部final-v1；修正后CPU通过，不是模型失败。

结论：原 4096-token 思考可以逐 token 保留，拼接当前模型真实的人工结束边界后进行有限续写。原来的“触顶且未自然闭合”失败保持原样。可精确恢复的是 token 前缀，旧 RNG 与 KV 状态未保存，不能叫同一随机流或原缓存的精确续跑，更不能把人工边界记成模型生成。

## 目标、约束与完成标准

真实目标是用已付出的思考成本验证一个低成本终稿出口，先解决格式失败，不把增大思考预算本身当目标。此次仅核原文来源、模型模板、完整 IDs、边界、上下文长度与资源；不挖掘思考中的中间 JSON 作答案，不接触任何目标答案。完成标准是同原输入 SHA 的实际 CPU 重建、完整旧输出 decode exact，以及一份独立的新输出合同；格式或比赛分数是否提高仍未知。

## 原始方法依据

[s1 原论文 v1 §3.1](https://arxiv.org/html/2501.19393v1#S3.SS1.SSS0.Px1)明确以人工追加 end-of-thinking delimiter 和 `Final Answer:` 强制最大思考预算；追加 `Wait` 是相反的延长机制。[官方代码固定提交](https://github.com/simplescaling/s1/blob/77272c6e925d610257a50b520bad15330b513389/eval/lm-evaluation-harness/lm_eval/models/vllm_causallms.py#L393)在 `finish_reason == "length"` 时保留真实输出 token IDs，追加答案边界，再以 `prompt_token_ids` 调用生成。代码 snapshot 日期 2025-06-25，晚于论文 v1，不能称首发当日实现。

s1 默认边界是 `<|im_start|>answer\nFinal Answer:`；其 README 简例主要展示 `Wait`，不包含完整的缩短路径。**本项目只借鉴人工阶段切换，实际使用 Qwen3.5 的 `</think>`，不添加会破坏纯 JSON 合同的 `Final Answer:`。** s1 基座为 Qwen2.5-32B，未加载或用于本比赛，也不是本次 9B 的质量证据。全部公开下载、固定 URL/SHA 与行号见[公开来源清单](../evidence/round2/thinking-budget-public-sources.json)，详读外部 `s1-sources/REPORT.md`。

## 当前样本与 CPU 实证

输入仍是原固定首条 `10313222:relations:0`，8 个 CID、合法患者仅 P1；原 system/user 提示、原 oldSFT32 epoch2、候选含义不改。原采样 `temperature=1, top_p=.95, top_k=20, presence=1.5`，4096 tokens 实耗 186.896 秒、总 native 209.493 秒，最终未关闭思考。原诊断已经独审，不重新解释成成功。

| 核对项 | 实际结果 |
|---|---|
| tokenizer / template | 原 `tokenizer.json`、`tokenizer_config.json`、`chat_template.jinja`、`config.json` 全 SHA 匹配旧运行 manifest |
| CPU 依赖 | 独立外部 `--target` 安装 tokenizers 0.23.1、Jinja2 3.1.6、MarkupSafe 3.0.3；没有改共享环境，也未加载 torch/transformers/权重 |
| 原输入 | True/False 两种实际模板 render SHA 及完整 input IDs SHA 都与原 CPU 记录一致；本次 True 输入仍 2030，SHA `3c5a740e…` |
| 原输出 | 原 `generated_token_ids` 全 4096 保存；两种 `skip_special_tokens` decode 均与原 response exact，不通过 decoded 文本重编码来近似 |
| 实际边界 | `<think>` 248068、`</think>` 248069；两者是 added token，但 `special=false`，不应凭名字假设会被 decoder 隐藏 |
| 人工追加 | 实际 assistant 模板分隔 `\n</think>\n\n` → **[198, 248069, 271]**；原 2030 + 4096 IDs 原样保留 |
| 重放长度 | 6129；再生成最多 512 则总 6641，低于原输入 cap 14000；模型/tokenizer 上限均 262144，无截断需求 |
| 重放 ID SHA | `e1876ed36ea7db983b90fc6afd81d317867a39a2f40132d22346e163644289a4` |
| 严格格式检查 | 14 个合成反例通过；只检查新 suffix，旧思考中 JSON 不作 fallback；拒绝漏/重复/多 CID、非法/重复 PID、不完整 JSON、markdown/额外文本、新 think/role、无 EOS、prefix 修改 |

精确重建用[prepare_replay.py](../thinking_budget_forcing/prepare_replay.py)，结果见[CPU 报告](../evidence/round2/thinking-budget-replay-readiness.json)。全量重放 IDs 留外部 `replay/replay-input.json`，不入 Git。重新 prefill 6129 tokens 仍有计算成本；不同 prefill/cache 路径的浮点状态不保证与旧4096步自然轨迹逐位一致。

## 建议的单次后续合同（尚未启动）

[机器可读提案](../evidence/round2/thinking-budget-proposal.json)记录一条固定输入、同旧32 adapter、同 native GDN/BF16/SDPA/TF 5.20.2/PEFT 0.21.2，最多新生成512，单 job **180 秒 + TERM30**，涵盖冷加载、核验、prefill 与 final。没有重试、延长思考、扫预算或自动扩成80条/F1/原dev；目前无推断入口运行，无 GPU 占用。

原 EOS 有两个：text config 的248044与 chat 的248046；**248069只结束思考，不结束整个回复**。新 suffix 要自行以真正 EOS 结束，解析仅移除末尾这一个 EOS，不用 `skip_special_tokens=True` 吞掉中间非法角色标记。严格需要8个CID全集与合法PID，完整 JSON。源在[strict_final.py](../thinking_budget_forcing/strict_final.py)；这只是新合同，不会令原输出变合法。

为了尽量保留原采样机制，提案仍用同 temperature/top-p/top-k/presence，并显式新 seed20261024：presence集合包含旧4096个**模型生成** IDs及新 final IDs，排除原2030提示和人工3个边界IDs；未来实现应先用 CPU 对这个集合做有区分力的检查。新 seed 是新实验，不冒称旧 RNG 恢复。若直接把整个6129当新 prompt 并重置 presence 历史，便改变了另一项变量，不能悄悄这么做。

资源复用同 reference A800 环境、既有 `c202236…` 基座和 old32 adapter `4f4815b6…`，8,997,081,600 参数，低于10B。当前旧 GPU 由检索 warm-stage 使用，本研究未探测空闲、更未抢卡；将来实际执行前由主线程安排。不能因 warm-stage 产生了新 adapter 就悄悄替换旧模型，也不切换已失败合同的 FLA。

旧512/4096线性 decode 代理为 **23.362秒**，它忽略了更长 prefix 的 prefill、冷加载、权重核验和上下文变长对吞吐的影响，**不是新运行耗时预测保证**。旧峰18.7245GB也不是新峰值。上述180+30是建议硬上限，新实际时间和内存尚未测量。

## 恢复与边界

外部根：`/workspace/patientphex-runs/round2-20261023/research/thinking-budget-forcing-readiness/`。`replay/`存固定输入与CPU报告；`s1-sources/`存13份原始来源及索引；`proposal.json`存建议合同。源、短证据和本报告入Git；原 prompts/raw/adapter、原失败结案均只读。

精确重建命令（独立输出目录必须尚未存在对应两个文件）：

```bash
PYTHONPATH="$RESEARCH_ROOT/cpu-libs" python solutions/sequence_crf/thinking_budget_forcing/prepare_replay.py \
  --original-run "$QWEN_ROOT/thinking-sampling-followup" \
  --model-small "$QWEN_ROOT/verified-small" --output "$NEW_REPLAY_DIRECTORY"
```

其中 `RESEARCH_ROOT` 为上述外部根，`QWEN_ROOT` 为既有外部 `case_retrieval/qwen35-preparation/`。目前只证明 token/方法/格式合同可准备；单条最终格式成功也不能当训练 fit、F1泛化或比赛收益。后续若主线程选择该控制，再实现小型固定ID续写入口并冻结实际来源，不需要重做旧思考。
