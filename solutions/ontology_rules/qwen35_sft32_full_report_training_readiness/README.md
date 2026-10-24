# Qwen3.5 / strict32 全文训练准备盘点

目前没有定位到 **Qwen3.5-9B＋原 F2/F3 的 124 行＋完整允许文章训练**的真实终点。该结论限定为本地 runs 中全部 `examples=124` 的训练 manifest 与路线台账清点；不能保证所有远端历史均不存在。当前全文 Qwen 对照沿用局部证据训练出来的 SFT32/SFT48，仅改变推断输入。MedGemma 全文 124 行确实训练过，但不能据其负结果排除 Qwen 这项控制。

本轮目标是核对来源、构造完整训练输入、准备真实 token 和监督边界检查。没有读取 F0/F1 gold，没有运行模型、训练、远端调用或评分；后续训练、GPU 探针及迁移均未获本任务授权。

| 已核历史 | 事实与边界 |
|---|---|
| 原 Qwen SFT32 | F2→F3 的 61 行＋F3→F2 的 63 行，32 文献；候选生成器及训练文献均排除 F0/F1。真实两轮 32 次 optimizer 调用，402,594 tokens/轮，最长 5,523，监督 7,170 tokens/轮。 |
| 原 Qwen 成本 | A800 80GB，1,149.460 秒，包括 trainer 内的检查、加载、训练和本地 epoch 保存；峰 allocated 36.633GB，reserved 81.109GB。不能把 reserved 当新增训练需求或把该耗时当纯训练 loop。 |
| Med 全文训练 | 同源 124 行全文，真实完成两轮，1,264.674 秒；Med tokenizer 为 965,436 tokens/轮、最长 12,824。这是另一模型/分词器的数值。 |
| selected CE | patient-full64 已实测目标 loss/梯度合同以及已有 Adam 和累积梯度下的 12 次前后向；最长 6,166，峰 allocated 25.031GB。截止本轮读取，其第一轮已持久保存，第二轮仍在运行；不能写成完整两轮成功。 |

原始训练行的准确结构是 `messages=[system,user]` 加独立 `answer` 字符串。冻结 `qwen35_smoke.encode` 在渲染完整训练序列时才添加 assistant。构造复用全文推断与 Med 控制的同一 `b201fb7f…` / `ae7f5853…` 源，只追加最后的 **user** 内容，完全保留答案 UTF-8、CID、PID、schema、原局部证据和其他 metadata。删除新增后缀可以恢复原 124 行。完整输入实际涉及 32 文献、87 患者、1,122 段、255 给定 mentions；按既定重叠政策插入 254 markers、跳过 1 个重叠 marker，所有患者仍有标记，原文全部重构通过。

`expanded124.jsonl` 为 4,145,866 字节，SHA `9e80e54125d3d46479acb9a3720e8b5f4c13ffacb5f65e82d3d9d10ac839d25e`，与 Med 那次全文输入字节相同。这只证明输入相同，不意味着 Qwen token 数相同。完整数据与逐行审计仅放 runs，仓库保留源码和小摘要。

## Token 成本：已准备，尚未实测

本机没有 torch、transformers、tokenizers、peft。没有安装临时替代版，也没有把字符数或 Med tokens 当 Qwen tokens。user 字符总数从 1,476,209 增至 3,884,700，比例 2.63154 仅作字符成本提示。

`cpu-plan.json` 与 `native_cpu.py` 已准备，要求既有 Qwen runtime 的 Torch 2.7.1+cu126 / Transformers 5.20.2 / tokenizers 0.23.1，以及发布 tokenizer 六资产精确 SHA。入口只做 CPU tokenizer，不需要权重或 adapter。它对原/全文 124 行分别检查：

- 原 frozen encode 的完整 encoded SHA、402,594 总 tokens、最长 5,523、7,170 监督 tokens；这是对原合同的实际复核入口。
- prefix/full 的发布 native、Auto encode、chat template IDs 全等与两套 decode 往返。
- answer 字节、assistant EOS、全部掩码与每行监督 token IDs 在追加前后相同；不截断、不改变 teacher 内容。
- 输出总长、最长行、平方长度总和、各 cap 越界数和按 512 向上取整的**建议拒绝上限**。超过 32,768 或模型上下文时只报告，不能据此启动 GPU。

一次 CPU 预算建议为 60 秒＋TERM 5 秒；具体执行由父线程另行安排。部署清单为 `cpu-plan.json` 加 `files` 中四件，压缩输入合计约 744KB，计划内 basename 不可改。运行前写排他 `native-CPU-intent.json`，状态未知时先核原任务，不重复运行。输出为 `native-row-audit.json` / `native-result.json`；目前均不存在。参考命令在 plan，脚本必须用普通 Python 执行，不能用 `-O` 关闭断言。

## selected CE 可以复用什么

`selected_loss.py` SHA `dfd0759f…` 保留完整 BF16 LM head，只将移位后非 ignore 的监督位置转 FP32 算 native fixed CE。它没有改变样本权重、batch 尾组除数或答案。相同 Qwen3.5 架构下可考虑复用该实例级 loss hook、源码绑定和数值核验方式；不能直接照搬 patient-transposed 数据身份和整个 trainer。

已有 CPU 合同运行在 Torch 2.6.0+cu124，而原 A800 SFT32 是 2.7.1+cu126；不能声称跨运行库已经数值等价。完整 BF16 logits 仍按序列长度×248,320 词表增长，hidden activations、完整注意力、反向与累积梯度仍占显存。6,166-token 探针和 patient265 首轮通过，都不能保证本次全文最长训练行可行。未来若安排 GPU，仍需在实际目标库和具有 Adam/累积状态下核新最长输入；这是待决方案，不是本轮已经批准的后继作业。

## 两轮预算的可审查假设

若保持原配方，仍为 248 次样本暴露、32 次 optimizer 调用、2 次 warmup；batch1、累积8，末组4行除数为4。实际全文 tokens 尚待 native census。

粗成本公式为 `1149.460371 × (全文每轮 tokens / 402594)`。仅以 **假设** token 比例 2.5 / 3.0 举例，两轮粗代理分别为 2,873.651 / 3,448.381 秒；这不是置信区间，也不是运行承诺。原耗时含固定加载和保存，长注意力与反向又可能非线性，selected CE 在此数据上的收益未知。

供统筹审阅的条件性建议是 **4,500＋TERM30 秒**，明确未授权。只有真实 124 行 token/mask 合同和另行授权的目标训练状态探针通过，且实际长度/耗时支持该预算时，才适合冻结新训练计划。若成本不支持，应按新事实重算，不自动延长或截输入。持久 checkpoint 导出另核成本；A100 patient265 或 Med4B 速度不能直接替代 Qwen 此控制。训练检查点、两轮完成及最终质量仍是不同终点，之后的验证方案需事前固定。

机器可读事实、来源 SHA 与限制见 `readiness.json`；有限历史清点见 `history-inventory.json`；本地已做的构造/语法检查见 `local-checks.json`。此目录没有新训练入口。
