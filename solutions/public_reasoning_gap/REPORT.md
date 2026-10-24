# 公开模型思考推断：有界缺口核查

2026-10-24 UTC；从 `main@d648efef` 开始。目标是找尚未尝试且可能改善原 train64/dev16 分数的机制，当前 best .769235669942，目标 .8。只读旧 source、实际模板、运行元数据及已保存生成记录，不调用模型、tokenizer、parser、拟合或评测，不读新 gold。

**结论：不新开“公开权重完整 reasoning，再独立解析最终关系”这条泛化提案。准确公开 R1 已完整做过同类 F1 控制，并非只有后来的 JSON SFT。** 精确的“公开 Qwen3.5-9B＋thinking=True＋完整 F1”在本次有界来源中未找到；不能称其已测，也不能仅因模型名字不同就认定有实质新机制。没有为本核查创建 GPU/训练计划。

## R1：公开权重、完整思考和最终关系解析已经实际完成

实际源 `infer_full80.py` SHA `f1c57dd7` 与原 plan `4a88bd33`、真实 terminal `d7fe7386` 绑定一致。直接 `AutoModelForCausalLM.from_pretrained(modeldir)`，无 PEFT/load_adapter；真实模型 399 参数张量、**8,190,735,360 参数**，准确模型 `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B`，ModelScope revision `5ee5fa4832e658ab84298806de06ed2a167834bd`。本次沿已核资产/实际加载证据，不重读或重新哈希16GB权重。

这是未经本任务微调的公开 R1。历史正确接口修复使用发布的 `tokenizer.json` 全链，以及明确派生的 RoPE attention scale=1；没有隐瞒成原封不动的运行配置。更早错误 AutoTokenizer 下的探针不作为该质量结论。

| 条件 | 实际证据 |
|---|---|
| 输入 | F1 16文献、80提示、39患者、587概念；原 train-only 候选，只有 system/user，没有 assistant 演示。HPO含义、原文局部证据、全局患者锚点；不是本次新建全文/证据检索任务 |
| 模板 | 发布原生模板；代码没有传 `enable_thinking=False`，该 R1 模板本身没有 `enable_thinking` 分支。模板以 User内容＋Assistant标记结束，模型实际生成 think |
| 生成 | BF16/SDPA、temperature .6/top_p .95/top_k0/repetition1、seed20261024，**4096 新token**；78 fresh＋2同配置原缓存，完整80状态 |
| 完整性 | 实际旧原始gzip241553B已按旧receipt核SHA；**80/80 均以 `<think>` 开始且恰一对 think 开闭**，总106142输出tokens；78自然EOS、2cap。不是只根据模型名或模板猜开启思考 |
| 最终解析 | 单独 `normalized_final.py`：严格开闭边界，仅解析关闭后整段JSON；许可整段单围栏和唯一可逆整数P码。非法/截顶整行弃答，缺CID不补负、无语义修补。严格全集 parser 同时保留诊断，实际质量采用预声明codec |
| 实耗与范围 | native3445.468秒，wrapper3446.521秒，完整80；70合法、10弃答，其中8个非法患者字符串、2cap。不是单条工程探针 |

本次只数原记录中的边界并核已有字段，**没有重新解析答案或重新评分**。两条cap也有think关闭，因此“思考闭合”并不替代最终EOS/完整关系检查。

同一既有实体框架上的旧完整质量：旧SFT32 `.660942003`；公开R1 alone **`.605215986`**；旧SFT32∪R1 **`.634402907`**。并集增加161条关系，仅21TP/140FP。两臂按原门关闭，完整F0不启动。已有终审 `05d60bdb` 已核完整响应、冻结和两次质量；这里继承结果，不追加新held分析。此结论只覆盖原候选/上下文/4096/codec配方，不证明所有 reasoning 方法无效。

## Qwen3.5-9B：不能把SFT短探针混称公开基座完整质量

| 已有实际工作 | 身份和结论 |
|---|---|
| 12份旧实际推断manifest | 本次沿冻结目录逐一核SHA/实际 fingerprint，全部 thinking=False，包含公开base及任务adapter；原 runner 明确传 `enable_thinking=False` |
| thinking1600 | 原 full64 JSON SFT epoch2 `1a703b24`，同12训练样本诊断；首条1600token/72.048秒cap、未闭think，1/12后停止；没有fit或质量 |
| thinking4096 | 原 train32 JSON SFT `4f4815b6`，官方sampling＋实际generated-only presence1.5；唯一首条4096token/186.896秒cap、未闭think，无strict final；没有完整F1质量 |
| 人工结束思考续写 | 仍是同SFT32与原4096轨迹，插入真实close边界；单条54新token/完整JSON格式通过，不能当质量收益 |
| 完整80 budget forcing | 固定2304思考＋512 final、原SFT32，CPU准备后因约3小时代理成本暂缓，未运行完整80；不是公开基座对照 |

因此，准确公开9B的完整thinking质量仍是**未核到的模型/配方组合**；当前没有 train-only 新信号证明，仅换该名字重做已完成R1机制不符合本轮“找实质新机制”的目标。若以后提出全文输入、原文证据检索、显式遗漏提问等不同供给，需先说明其与旧局部候选 R1、当前 G 补全和现有 quote 训练的实际差异；本核查不代其证明未试，也不扩展成新任务。

## 官方方法说明的边界

缓存官方 overview §4.2 第二队只写了：在两种微调方案之外，增加 LLM reasoning 识别可能遗漏的关联，再合并三种输出。它没有给出该 reasoning 组件的精确权重、是否微调、提示、思考开关、预算或解析规则。该描述可以作研究动机，不能当成“公开9B＋某种思考预算”的复现配方。

可复现检查：[audit.py](audit.py)；源、真实终点、模板、旧响应与Qwen12份manifest的准确SHA统一在 [audit.json](evidence/audit.json)。本报告关闭泛化重复提案，没有下载、GPU、拟合、新gold或质量调用，也未改变正在运行的G补全/BOUNDARY。
