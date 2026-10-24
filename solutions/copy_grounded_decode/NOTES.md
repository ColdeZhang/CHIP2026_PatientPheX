# 机制与边界

- Grounded 原生输出是三元组列表，不是 canonical-name 字符串列表。原文定位单元和 occurrence 必须保留；同一 quote 的不同原文 occurrence 均可成为候选，不默默改成第一个或最近患者的一处。
- Grounded prompt 明确允许在全文任一位置复制词汇 anchor；词汇出现处不代表患者归属。有限池只解决合法供给/生成，不解决患者归属、否定、HPO 映射或候选遗漏。
- 既有 CanonicalList 的 trie/used-set/EOS 思路可复用，不能照搬其换行串分块。compact JSON 的 `[[` 与 `[["` 编码不同；本实现以 `[["` 开头、各 block 从 `U` 起，并将下一项的 `],["` 放在前项末尾。必须以实际 Qwen3.5 tokenizer 检查整串编码。
- 不把受限生成的 prefix 当预测，不把自由生成的非法 quote 修成合法 quote。准备中的纯预测覆盖统计不是 TP/FP、recall 或分数。
- 已核：原始 schema、候选祖先排 F1、完整 native tokenizer 合同、旧 raw/纯预测供给；结果见 [REPORT](REPORT.md)。实际 callback + HF generate 仍需后续小门。

## 成熟方法来源

- De Cao, Izacard, Riedel, Petroni，**Autoregressive Entity Retrieval**，ICLR 2021：[论文](https://arxiv.org/abs/2010.00904)、[OpenReview](https://openreview.net/forum?id=5k8F6UU39V)。生成实体名称并使用前缀树限制合法下一 token 是既有成熟机制；不能由论文效果直接推断本任务质量。
- [GENRE 官方 README](https://github.com/facebookresearch/GENRE/blob/08af8d0119ed60692e428791be98a17db73996b7/README.md) 的实际例子用 `prefix_allowed_tokens_fn=lambda batch_id, sent: trie.get(sent.tolist())`；[trie.py](https://github.com/facebookresearch/GENRE/blob/08af8d0119ed60692e428791be98a17db73996b7/genre/trie.py) 在有效前缀返回子节点 token，无效前缀返回空列表；[hf_model.py](https://github.com/facebookresearch/GENRE/blob/08af8d0119ed60692e428791be98a17db73996b7/genre/hf_model.py) 通过 `generate(**kwargs)` 接入约束。固定官方 commit `08af8d01…`，取得源码身份见 [sources](evidence/sources.json)。
- GENRE 是 BART encoder-decoder 的实体检索；本任务是 causal 模型的多组三元组 JSON，需剥离完整 prompt、动态禁重复、精确 occurrence 和 EOS 状态。本实现来自项目内部 `mention_candidate_selection/reuse/canonical/schema_contract.py` 的已核模式，未复制 GENRE 代码或权重。GENRE 源仓许可 CC BY-NC 4.0 已查；当前只参考原理/来源。

## 本次工程记录

- 初次本地入口发现 main 尚不含 Grounded 分支模块；改为显式可配置 `GROUNDED_SOURCE_ROOT`，并核原 schema/decode SHA 后复用。
- 初次 CPU 全合同跑完后，文件白名单阻止读取刚写出的候选文件来计算 SHA；只将该实际生成文件加入允许集合。未放宽任何既有输入，更未开放 gold。最终 `final-cpu` 成功；失败临时产物留仓外，不将 tracebacks 入 Git。
- 新增接口风险：原 Grounded `inference/run.py`（SHA `56c3a9aa…`）的 `FiniteLogitsOnly` 要求所有 scores finite；HF 前缀约束会用 `-inf` 屏蔽非法词。不能直接追加默认 `prefix_allowed_tokens_fn` 再假定原 custom guard 顺序兼容。后继必须在目标 runtime 核顺序，保留 raw-logit finite 检查在 trie mask **之前**（例如有序 custom processors），并验证合法 logits 原值保留、非法 logits 为 `-inf`、NaN/+inf 原始故障仍报错。不能删除数值检查作为修复。本准备未执行该真实 HF 门。

## 实际终点补充

- 上述接口风险已由真实.100 TF5.20.2 CPU 234状态确认并解决：FiniteLogitsOnly必须先于trie mask；GPU39原始logits未报告非finite。见actual-cpu。
- 有限合法token树保证每一步合法，不保证在max1600前选EOS；本次2/39cap。完整raw先落盘，V1原全行成功门因此失败。独立v2是看过无标签失败后的新整行弃答政策，不能改写V1历史。
- 末尾进度不是全量统计：最初只见最后一条cap误推总数1，完整39的EOS37/cap2/strict37为准。以后先核complete+逐行audit再汇报总计。
- v2全体cap/nonEOS及原strict失败在JSON解析前贡献空集合；保留其患者和B4，不能只评分37成功行。完整F1微分下降、宏分上升；9TP/23FP不支持“所有项都更准确”的结论。

- MCP `start_process` 的UNKNOWN不等于失败/已执行。F0必要Sap首请求只有claim、无managed或/proc匹配，而原driver没有请求有效期；先沿实际进程/源码核事实。root批准exclusive tombstone占原fresh-resource-gate后，迟到旧driver会在其Popen前的os.link失败；不覆盖真实gate、不抹旧UNKNOWN。独立v3再绑定5分钟admission并在request/driver/模型前核，唯一执行成功。后继47也增加10分钟启动有效期，运行预算保持独立不延长。详见f0_proposer README与fence/actual receipts。
- 完整生成证据和本地全诊断回读不是同一门：原172窗口selected/windows/inventory全部按冻结ZIP manifest核SHA后可推进，不拿部分窗口构池；大raw仍在server持久，pending明确记录。最终701原件全部本地回读，旧accepted manifest不改，补独立完成sidecar。
