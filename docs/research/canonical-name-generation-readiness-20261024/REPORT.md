# Canonical-name generation：CPU 准备结案

**结论：存在一个未实跑的具体机制差异，可作为后备新方向；当前仅证明目标与原生 token 约束可构建，没有训练、模型输出或质量提升证据。** 当前 F0 与 full64 正路线优先，本任务没有使用远端或 GPU，没有读取 F1／原 dev 金标。

真实假设：患者全文直接生成公开 HPO 标准名称集合，再精确查表，能否绕开旧开放编号的语义错配，同时补充现有候选池漏掉的关系。只改变输出表示与约束，复用现有 Qwen3.5-9B；不下载 GENRE／REBEL 权重。

## 按实际机制查重

- 旧 `patient_set.py` 目标是 `identifier + segment_id + quote`，训练用距离患者 mention 最近的正实体选 quote。E-008 的 r2-11/12 已完整训练 Qwen3-8B 并评完 47 患者：adapter raw Score .397872、grounded .396343，对照 .711354；出现临床短语与无关／递增 HP 编号配对。当前源码 SHA 与历史执行 SHA `6218d5ce…` 分列在 sources，不能混称。
- 当前 `grounded_generation/schema.py` 只允许 `[unit_id, exact_quote, occurrence]`，再经过严格 normalization；quote 只是词汇锚点，不证明患者归属。它仍以最近患者 mention 选择训练 quote。
- 本方向从 association 的合法 HPO 直接取公开 canonical name，**不使用 gold entity、nearest-mention quote 或 marker 选目标**。输入仍保留全文与所有患者给定 mentions。名称由模型生成，既不要求原文逐字出现，也不预测自由 HP 编号。所查四个方案源码没有这一目标的完成终点；已有 `prefix_allowed_tokens_fn` 命中只是 EOS-only 工程 smoke，不能算执行过名称约束生成。
- 收益只是待检验假设。标准名约束不能解决患者归属、否定、过度推断或疾病常见症状幻觉；丢失逐字证据后，这些风险更直接。模型只见 527 个训练目标概念，公开可生成 19,119 名称不代表它会正确使用这些名称。

## train32 的实际覆盖

输入冻结：train32 `38c353ea…`（F2+F3）、hp.obo `a5092cbd…`（2026-06-23）。F1 仅读取纯四字段 input `962b76a3…`，16 篇／39 患者，文献交集 0。

| 检查 | 结果 |
|---|---:|
| 公开活跃 phenotype 名称，排除泛根 HP:0000118 | 19,119；exact／casefold 重名均 0 |
| 这些概念的 alt IDs | 3,864；加载器验证没有冲突映射 |
| 训练文献／完整患者行 | 32／87；5 个真实空集合，0 个阳性但全无可表示 HPO 的患者 |
| 原关系值 | 861 = 839 合法 HPO 值 + 4 分支外 HPO + 18 NIL 原文值 |
| compound／有效 alt-ID 原子 | 9 个 compound／3 个 alt-ID 原子 |
| 合法 HPO 原子与去重 | 853 → 796 个患者–HPO 目标；57 次重复，527 独立 HPO |
| MESH／未知／obsolete／畸形 HP／字面 -1 | 均 0；18 个 NIL 值都能匹配同文 exact `identifier=-1` 实体 |

14 个患者存在被名称模式遗漏的正标签，全部仍在 87 行 inventory。目标明确是 **HPO 部分集合**，不能声称是完整官方答案，不能把 NIL／分支外标签伪造为阴性。4 个分支外 ID 是 HP:0031135、HP:0002686、HP:6000529、HP:0031375。compound 整体检查，任一原子非法时整值排除，不偷偷保留正确部分。训练按 canonical HPO ID 排序、映射标准名；解码允许任意无重复次序，避免把排序错误变成标签遗漏。基线保留其 NIL；本头不单独承担 NIL。

## 成熟来源与实际接口

- **GENRE**, *Autoregressive Entity Retrieval*，论文 arXiv:2010.00904，代码 `facebookresearch/GENRE@08af8d0119ed60692e428791be98a17db73996b7`。实际 `genre/trie.py` 返回 prefix 合法下一 token；README 以 `prefix_allowed_tokens_fn` 配约束 beam search。代码 CC-BY-NC-4.0。它是单实体名称检索范式，并非现成的医学患者集合系统。
- **REBEL**, *REBEL: Relation Extraction By End-to-end Language generation*，Findings EMNLP 2021，代码 `Babelscape/rebel@54ea5fd07dafece420c28c6f71f1c6431f42797c`。它以特殊标记线性化多个关系，提供集合序列化参考；README 的宽松 parser 不适合此处严格拒绝要求。实际许可证在 `LICENSE.md/LICENSE.md`，CC-BY-NC-SA-4.0。没有复制其 parser 或训练代码。
- Transformers 5.20.2 的实际 source revision `18329f1065382e54db698e242cf98bd3f3e5a766`，Apache-2.0：普通 `generate` 构建 PrefixConstrainedLogitsProcessor，按每条 beam 的完整 prefix 调 callback，空 allowed 列表报错；continuous batching 明确不支持。greedy 可用 `num_beams=1` 走此 processor；本次没有执行 HF processor 或 model.generate。不能把 GENRE beam=5 默认直接套到现有 Qwen3.5 hybrid cache，beam 的显存与 cache 重排尚未验证。

所有真实 URL、版本、下载内容 SHA 与本地 source 引用见 `sources.json`；原始网页／源码只在外部目录，未 vendor 到项目。

## 原生 token 的 CPU 证据与边界

`schema_contract.py` 是独立小接口：标准 JSON 字符串数组；公开名称精确查表；未知、重复、非数组、尾随说明、未 EOS 或触顶整行拒绝。无 substring、大小写修复、nearest alias、Sap 或 gold 修正。token trie 同时编码“名称后继续”和“名称后关闭”两个分支；禁止重复，关闭后仅允许 `<|im_end|>`（248046）。每条 beam 从其 prefix 重建状态，不以 batch_id 共享 mutable 已选集合。

现有 Qwen3.5 native `tokenizer.json` SHA `5f9e4d49…`，仅本地 cached 3.3 MB `tokenizers==0.23.1`。CUDA 隐藏、线程 1：19,119 名称 singleton 与全部 87 训练目标 roundtrip 通过；块拼接 IDs 与完整 JSON 原生 encode 的已检查实例均一致；全部 6,167 训练前缀允许其下一 token，8 种严格拒绝、重复路径阻断通过。38,238 entry 变体／173,381 trie nodes，检查约 4 秒，峰值约 207 MiB。训练答案含 EOS 共 6,167 tokens，最长 309。不是 GPU 解码时延、模型质量或任意字串编码等价的证明。

原生 USER 内容：train87 最长 8,607、总 420,996 tokens；F1 input39 最长 9,864、总 195,164。尚未加新 system／chat template，不能直接当完整训练 cap。脚本与精确结果见 `native-contract.json`，完整目标 inventory 留外部。

一次复跑的源码 guard 误用了旧 preparation-plan 的 schema SHA `f1daecc3…`，在长度计算前退出；随后冻结实际已读源码 `460dd7cc…` 并重新通过。旧 plan 仅作为数据路径来源，未把旧 SHA 冒充当前执行源码。

## 最小可证伪配方建议（尚未授权／执行）

现成 Qwen3.5-9B 固定 revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`，复用已核 native 输入与 trainer；原文本体约 8.954B，加既有同结构 LoRA 约 8.997B，独立模型小于 10B。新 LoRA 从同 base 初始化，train32 全 87 行 HPO 投影目标，固定 2 epochs／epoch2；沿已验证 grounded 超参而不做搜索。先完整模板 token/cap 与最短、最长真实 loss 小检查；不因本 CPU 证明直接启动训练。

推断保留完整 F1 39 患者，greedy、普通 batch1、无 thinking，固定 max_new_tokens=512（基于训练最长 309，仍需检查完整上下文上限）。保存 raw IDs／raw 文本／EOS 与 cap 状态再解析，不重试修答案。Task1 与现有强 B 完全固定；**主比较预先冻结为 B 的关系集合 ∪ 新名称头关系集合**，不按 held 标签挑名字或删患者；单头仅辅助诊断。若完整 F1 主比较不能改善同一 v2 all/text Score 且关系 micro/macro 同升，关闭该固定配方；若正，再申请完整独立 F0，不立即转原 dev 扫描。

预计成本量级是另一轮 87 全文患者 LoRA（两轮约 0.85M 文本 tokens，非实测 GPU 时长）＋39 患者、最多约 20k 新 tokens。建议一次有界训练不超过已有同架构 grounded 的 3000+30 秒预算、推断先按短 probe 实测估算，总量由 root 另行安排。名称更短可能节省输出成本，全文训练和 KV cache 成本仍在。F0／full64 现行正路线优先；本准备不承诺达到 0.8。
