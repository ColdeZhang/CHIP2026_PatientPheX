# 原文三元组受限解码：F1 终点与准备依据


**独立 fallback-v2 的完整 F1 Score 为 0.704349598608，较 B4 的 0.701325377212 增加 0.003024221396。** 这是内部 F1 开发对照，不是原 dev16 的最高分更新或 0.8 达标。完整记录：[终点证据](evidence/F1-fallback-v2-result.json)。

原 V1 同权重/完整39/greedy/nonthinking/max1600实际生成完39后，37EOS/37strict、2cap、完整EOS parser失败0，按启动前固定门rc1关闭，0quality；没有改成PASS。首次只看末尾进度误报1cap，已由全39统计更正2cap。原源码/计划/89原件及[关闭证据](evidence/V1-closed.json)保持。

父线程在无新质量标签时另设[后设整行弃答v2](fallback_v2/decision.json)：2cap完全不解析、不提取前缀，其他37严格行用原parser；所有16文档/39患者均保B4。498 anchors/333 unique quotes经原train32 alias+b59政策；仅142必要Sap FP16查询，8.243秒rc0，15:14:37.640730 UTC真实释放GPU0。全16/39/1189 entities先冻结后只1newv2/old0，无子集、单臂或全union质量。

| F1 完整流水线 | B4 | 弃答v2 | 差值 |
|---|---:|---:|---:|
| menF1 | .667191188041 | .667191188041 | 0 |
| docF1 | .747747747748 | .747747747748 | 0 |
| micF1 | .749042145594 | .743494423792 | −.005547721802 |
| macF1 | .641320427466 | .658965034852 | +.017644607386 |
| Score | .701325377212 | .704349598608 | +.003024221396 |

保护原524关系、非关系字节和所有旧字面前缀；固定primary新增32=9TP/23FP。收益来自macro而micro下降，不能概括成整体错误减少。纯new324只作无分审计。32新增中原G支持25，仅其他旧流支持7，证明先前扩大同意支持空间确有实际差别；没有评分G-only替代臂。预测141815B/SHA `0380a1bc…`，freeze `0b5ea454…`；全部大产物见仓外终审入口。

真实运行使用.100 A30、Torch2.7.1而原G为A100/Torch2.6；不能声称纯decoder数值因果。strongB祖先48排F1但含F0，整链不是纯train32。F1严格正仅支持考虑下一次独立F0对照，必须重建排F0候选及匹配旧支持；本线未启动F0或官网提交。

## 历史 CPU 准备（下述为首次 GPU 前结论）

**建议保留为一次有界完整对照候选，尚未运行 GPU。** 这不是重新训练旧方案：固定原 Grounded Qwen3.5-9B train32 epoch2 `d52e2e6c…`、完整 39 prompts，只将自由 token 空间限制为已有原文候选三元组。原 raw 有明确地址错误，供给非零；是否提高分数尚无证据。本次没有读取 held gold、训练、加载模型或评分，不能宣称达到 0.8。

实验 `E-20261024-COPY-GROUNDED-001`；[固定协议](evidence/protocol.json)、[完整身份/CPU 摘要](evidence/readiness.json)、[源资料](evidence/sources.json)。

## 能改变什么

实际答案是 `[["U0013","exact quote",0],…]`，三字段分别为原单元、原文短语、单元内第几次精确出现。不是字符串或 HPO 名称列表。候选由既有 strong Task1 1313 spans 与 GLiNER 886 spans 按原坐标合成 1416 项；全部逐项精确定位成功。16 篇每篇 33–177 项，共 907 个文档内不同 quote；保留不同 occurrence，禁止重复同一完整三元组。候选只用于 logits 约束，不加入提示或附带答案/预测 ID。

原 39 raw 全部 EOS、无 cap：699 项中 573 个合法 anchor、123 个不存在的 quote/occurrence、3 个重复 anchor。有限池能消除地址错误和重复，但会限制自由 span 供给，也可能将原来非法文本的概率推向错误的合法候选。它不能自动解决患者归属、否定、候选漏召回或 HPO 映射。

| 无 gold 供给统计 | 数量 |
|---|---:|
| 原合法 anchors 在池内：精确三元组 / 同文档同 quote | 340 / 345 |
| 原 pure G 关系 / 现有 B4 关系 | 282 / 524 |
| G 在 B4 外 / 其中有原 exact-anchor 池内见证 | 62 / 53 |
| G∪S∪Q∪C 在 B4 外 | 133 |
| 相比 G-only 多出的支持关系 | 71 |
| 133 中与候选 proposer 同文档预测 ID 相交 / 新增 71 中相交 | 122 / 66 |

最后一行仅作供给代理；proposer 的 ID 不进入生成，也不保证旧 Grounded normalizer 会把该 quote 映成同一 ID。53 是既有原 anchor 的见证，不是 TP 数或未来召回上限；133 才是固定同意协议的最大新增集合大小。

初始建议为 `B4 ∪ (constrained_G ∩ pure_G)`。主线程在新输出/质量均为零时修订为唯一 primary：**`B4 ∪ (constrained_G ∩ (pure_G ∪ S ∪ Q ∪ C))`**。原因是 G-only 会排除从原非法 quote 恢复、但受其他旧流支持的关系；供给比较也确有 71 项差异。S 存盘含旧 B，但已核 471 条旧 B 全在 B4 中，故不改变 B4 外支持空间。不声称各流统计独立；不评单臂、不扫权重或阈值。

## 工程合同与已试机制区别

`triple_trie.py` 复用内部 CanonicalList 的 trie/used-set 思路，保持原 compact JSON。不能把 `encode('[')` 与整个三元组的编码随意拼接：这里 opening 为 `[["`，各块从 `U` 开始，继续块以 `],["` 结束，末块以 `]]` 结束。发布 tokenizer SHA `5f9e4d49…` 的原生 tokenizers 0.23.1 对 **1839 完整串、62159 个 callback token、2832 个真实 pretoken 分块边界**检查通过；包括完整候选串正反序、空池/空输出、Unicode/转义、重复 occurrence、两个 EOS，以及拒绝重复/缺 EOS/cap/非法前缀。全流程 CPU **1.430 秒**；16 trie 共 12902 节点，单篇最多 1660。

既有 patient-first CID/PT 改生成目标/表示，9B entity-slot binary 改监督与候选打分，8B canonical-name/trie 生成公共标准名，Top8 多选改 mention→概念归一化。本视图不重训这些机制，而是在原 Grounded 同权重患者全文→原文 anchor 任务上增加有限原文支持域。仍沿用已有有限 trie 技术，不宣称新架构。CanonicalList 的换行布局、旧 cap-prefix 修复均未引入。

祖先方面，Grounded/GLiNER 学习部分为 F2/F3 train32；strong B 提议链含 **F0/F2/F3 train48**，既有 nested 审计排 F1，整链不能称纯 train32。G/S/Q/C/B4 的已有排 F1 审计与全部 refs 在协议中；公共预训练文献成员未知。后继 F0 必须换排 F0 的提议/支持资源，不能复用本 pool。

## 成本与下一步

原同一 39 prompts 共 203588 input tokens，原 10513 output tokens，native 总耗时 1103.215 秒（逐行生成合计 1030.970 秒），最大单行 1272 tokens。仅可作为约 20 分钟级的成本代理，受限生成长度/分布可能改变；建议主线程若采用，先核现成 A100.12 身份和实际 callback 接口，再设独立有界完整 39 行运行，不启动新训练。当前 CPU 准备没有占用 A100。

尚未执行：真实 HF `generate` callback 集成、同 runtime native/HF 双解码、新输出完整 EOS 覆盖、固定 normalizer、最终装配与质量比较。未来先持久化 raw IDs、raw 文本及结束状态，再严格验证完整列表；不得截取合法 prefix、补答案或失败重生成。只装配一个 primary，保留 B4 全部非关系对象和旧关系，完整冻结后由主线程决定是否一次 v2 比较。

集成时还有一个明确门：原 runner 的 all-finite guard 必须置于 trie 的 `-inf` mask **之前**，不能直接加默认 prefix callback 后沿用未经检查的 processor 顺序。合法 logits 不变、非法项屏蔽、原始 NaN/+inf 拒绝，都须在实际 runtime 核实；详见 [NOTES](NOTES.md)。

复现：在本 worktree 运行下列命令；`PATIENTPHEX_WORKSPACE` 和 `GROUNDED_SOURCE_ROOT` 可改根路径，输入 SHA 保持不变。完整候选 143692 B、SHA `f2027e69…` 放仓外且可重建。

```bash
PYTHONPATH=/workspace/patientphex-runs/round2-20261023/research/thinking-budget-forcing-readiness/cpu-libs \
python solutions/copy_grounded_decode/readiness.py --output /path/to/new-output-dir
```
