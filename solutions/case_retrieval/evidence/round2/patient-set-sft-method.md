# 整患者开放集合 SFT：实际数据与校验边界

2026-10-23，E-20261023-008。当前完成数据准备与 CPU 审计，尚未训练或评测。目的是检验整篇患者条件生成能否补足逐候选关系判定的召回上限；此前研究和旧的无 quote 预算见 [可行性研究](patient-set-sft-feasibility.md)。本页采用带 quote 的实际格式，不能沿用旧的 640,031 tokens/epoch。

## 冻结输入与目标

入口为 `solutions/case_retrieval/patient_set.py`。只用内部 fold0 的 train48 标注构造 assistant，heldout 只读取 `validation16.input.jsonl`。按文献隔离；同一患者字符串跨文献不作身份合并。每篇按照给定顺序把患者映射为 P001、P002 等；每行只有一个 TARGET，同时保留整篇原文、全部给定患者的 marker 和原始全文 segment。无词典检索结果、金标实体或候选 HPO 进入输入。训练输入可在删除 entities/association 后逐行重建。

模型目标为：

```json
{"phenotypes":[{"identifier":"HP:0004322","segment_id":"S003","quote":"short stature"}]}
```

未知 HPO 表型以内部 `identifier="-1"` 加原文 quote 表示；最终 association 必须输出该原文字符串，禁止裸 `-1`。合法 HPO 复合项机械拆分，使用指定 HPO 2026-06-23 的 HP:0000118 分支与 canonical ID；4 个分支外和 1 个畸形目标排除，不猜修。

训练 quote 只从同文、同概念、非 NO 的金标实体中选取，要求 offset/length/text 严格匹配指定原文 segment 且满足词边界。距离取实体区间与目标给定 mention 区间的最小间隙，依次以 offset、length、text、segment 解并列。此选择使用金标仅在 assistant 构造阶段。最近 quote 可能来自背景或另一个患者，机械选择没有证明患者归属；literal 支持也没有证明 HPO 语义等价。这是本表示的已知限制，不把 quote 当额外关系金标。

train48 原始 association 列表 1,424 项，逐患者原字符串去重为 1,313 项，仅拆合法 HPO 复合后为 1,338 项。剔除 4 个分支外与 1 个畸形后合法集合 1,333 项；其中 6 项没有可用非 NO 同概念原文实体，逐项记入外部 `target-decisions.jsonl`。最终 **126 行、1,327 个目标（1,303 HPO＋24 原文）、7 个空目标患者**，不删患者或凭空造 quote。此计数与独立审核参考一致；其余详细审查另由 validation 执行。

## 实际 token 与后续固定对照

CPU 使用正式 Qwen3-8B tokenizer，以及冻结 trainer 的 `encode_example(..., 12288)`；CUDA 隐藏，不加载模型。126 行全部前缀一致、assistant-only mask 正确、无截断。训练全文 tokens 最小 1,972、中位 5,306、最大 **11,709**，每 epoch **712,250**；prompt 最大 10,639，answer 最大 **1,188**。最长训练行 `12455887:patient-set:P001` 可供父线程执行最长样本 smoke。

实际 heldout stripped input 16 篇产生 **47** 个患者 prompt，全部按相同模板审核：输入最大 **10,543**，中位 5,613，总计 305,180 tokens，无裁剪。仅据训练答案长度，解码预算取 `max(1024, ceil(1.5 * 1188 / 256) * 256) = 1792`。建议 base 和 adapter 都使用 batch1、greedy、禁用 thinking、max_input_tokens=12288、max_new_tokens=1792。最长推理输入加预算为 12,335，超过训练 maxseq12288 47 tokens，但低于基础模型 context40960；训练序列限制不是推理总上下文限制。生成触顶仍属失败证据，不能事后据 heldout 标签挑预算。

后续父线程按授权执行 A800 最长 smoke，再从同一 base 正式 LoRA 2 epochs；本准备任务没有启动 GPU。训练后的 base→adapter 固定47输入对照只换 adapter，实体固定 CRF+Sap；尚无本方案的 Score。开放 ID 输出能绕过候选缺失，也可能产生 ID 幻觉、语义误配、过度患者归属；必须分别报告 raw 与 grounded 输出及失败率。

## 严格输出与真实失败保留

`merge` 要求推理输入无标签、基线文献集合完全一致、47 prompts 从输入完全重建、response row_id 精确覆盖且无重复。多患者 prompts/responses 使用独立 `read_rows`，不复用强制文献 ID 唯一的文档读取函数。输出保留基线 entities，替换每个给定患者的 association。

- JSON 必须精确满足目标 schema；不修复 markdown/残缺 JSON，不猜患者 ID。解析失败保留显式空预测及失败状态，不能把它解释成患者无表型。
- raw 诊断：正确 schema 下有效分支 HPO ID 可不依赖 quote；NIL 必须有指定 segment 的原文词边界 literal 支持。触顶但能解析的 raw 结果仅作为可能不完整的集合诊断保留。
- grounded 主输出：HPO 或 NIL 均要求同一段原始**未插 marker**文本中至少一个词边界 literal 匹配；不要求唯一位置。整行触顶则整行拒绝并审计。任何输入原文证据检查均不读取评测 gold。
- literal 校验只支持文本存在，不能自动证明对应患者、否定状态或 HPO 含义。比如词 `short` 在 `short stature` 中合法，并不意味着足以证明该概念。

7 项 CPU 测试覆盖金标变化不影响 prompt、训练缺标签拒绝、非法/无证据目标、NIL 原文输出、literal/marker/边界、非法 JSON，以及真实多患者 merge、触顶整行拒绝、遗漏/重复行硬失败。

## 复现与产物

完整命令、源文件/数据/模板 SHA 和实际统计见 [准备 manifest](patient-set-sft-preparation.json)，CPU 脚本见 [token 审计](patient-set-token-audit.py)，远端实际命令与 stdout 见 [命令证据](patient-set-token-command.json)。全量训练输入、47 prompts、目标决策与逐行 token 长度保存在仓库外：

`/workspace/patientphex-runs/round2-20261023/case_retrieval/patient-set-fold0/`

训练数据 SHA `416500b650acac91ed7d24debd71b3b91ff1621c63c7410d77d5ad23637cfc2d`；推理 prompts SHA `ab50b859aefb00cf095372794baae3988e114978c64493f6010eab5a327594f8`；完整 token 审计 SHA `485ea468acb4c84c44c63ac41c9018f8ff417f3edfdda2effe1c3130e167e3c7`。CPU 远端副本位于 `.100` 自有 `case-retrieval/round2/patient-set-fold0/`；父线程另行上传 A800 供正式工作。大文件不入 Git。
