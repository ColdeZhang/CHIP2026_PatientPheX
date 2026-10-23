# 病例检索研究笔记

只保存可复用的方法来源、工程经验和待验证假设；试验结果以[实验过程](experiments/E-20261023-008.md)为准。已确认跨方案坑回报主线程更新根 HANDOFF.MD。

首日结果见全局实验索引；第二轮不能只优化关系选择，现有实体/关系候选覆盖的诊断性上限不足 0.8，需要同时改善覆盖和患者归属。此上限是诊断，不是真实候选成绩。

## 本轮研究

- 只读研究在train64内部fold0诊断：1147个有效金标实体中704精确命中；443漏项有410项属于未见词面。建议将原文quote抽取和HPO归一化分开验证；完整研究记录在外部 `research-recommendation.md`，未读取dev金标。
- 当前旧RAG只能选择原候选代码，不能新增实体。新增抽取按文献/段落/窗口/阶段建立唯一row_id，缓存绑定prompt和源码hash，防止多窗口被同一pmc跳过。原文引用回定位不接受模型编造offset。
- 公开同赛实现 `https://github.com/yuanpinz/chip2026-PatientPheX` 值得参考逐occurrence、患者竞争与显式组患者关系；暂无明确许可，只参考思想，不复制代码。其自报严格LOO .68478与当前口径不同，不能直接比较。公开报告自由实体补漏未改善，故保留原候选关系判定先行对照。
- 全文讨论段落可能提供entity，但不能因此归属患者。实体输出和关系候选可分开校准；关系仍必须有原文证据，不凭疾病名臆造。

## 官方综述与后续对照

已读取官方综述 `https://arxiv.org/pdf/2610.29606` 的baseline说明：患者额外称谓/共指识别 → 每批8个表型、各出现±500字符语境判关系，Qwen3-8B greedy。当前新增 `prepare_coreference.py`/`merge_coreference.py` 和 `prepare_pair_rag.py`/`merge_pair_rag.py` 作独立对照；自写实现，非复制未许可代码。先跑最小全文关系基线，避免为准备所有可选模块推迟首个有效分数。

冠军公开LoRA配置可参考r16/alpha32/dropout.05与7个attention/MLP投影层，但其权重训练80篇包括我们dev16，禁止直接用于本地验证。若无监督Qwen收益不足，应只用当前训练折重建LoRA训练；原配置不等于收益证据。

新增quote归一化的候选可并用字符TF-IDF与冻结SapBERT CLS检索；训练别名仍只相应训练折，SapBERT来源沿首日已核验公开预训练权重，不新加人工表型数据。候选模型不会直接决定最终标签，未知/歧义项由Qwen在列出的候选中选取并显式记录无法解析项。

ModelScope对小文件完整Range可能返回200而非206；仅当请求严格覆盖全文件时可以接受200，必须另验长度和全SHA。部分范围仍必须206及准确Content-Range。分片任务失败应保留已下载文件并从缓存恢复，不重下16GB模型。

训练48抽取风格审计（未用于更改当前冻结zero-shot prompt）：3,612正实体中41含syndrome，INTRO+DISCUSS占48.75%，提示不能把“病例归属”边界套进实体识别。至少134正跨度不含紧邻严重程度/方位修饰，compound正跨度仅30；宽泛“complete specific phrase”可能损失精确边界，需要训练示例或SFT单独验证。305窗口有12个同词同时正/NO，去重quote→所有出现回定位不能完整表达逐出现断言差异。证据外存research-train48-style-notes与research-sft-style-diagnostics。

私有训练依赖在独立`training-runtime`安装peft0.15.2/bitsandbytes0.45.5，不修改已验证推理runtime；CPU隐藏GPU的import通过，不代表CUDA量化kernel已验证，仍须获GPU时做single-step。
