# MedGemma 闭集全文独立正式控制（训练前固定）

目标检验在相同 CID→PID 任务、相同候选和严格 train32 的情况下，在训练与推断中追加全部允许原文是否改善旧 9B 强基线。短输入 Med 两端负结案不变。本次 fresh 公共 base，不使用短输入 adapter。

- 唯一研究变量：原 124 行/80 输入消息末尾追加已核全文，训练必要容量明确 14000；原始消息、答案、CID/PID、候选、7150 监督 tokens 不变。124 行两个完整 epoch，238 个文本投影 r16；batch1/acc8/tail4、32 optimizer calls、2 warmup、seed20261023、AdamW LR1e-4/WD.01/clip1、BF16/原生 SDPA/native CE 不变。
- 两次模型加载、两个 epoch 和完整 adapter/Adam/RNG 检查点持久导出共用一次 2700 秒绝对窗口，TERM grace30。每 epoch 导出到 .100 并全 SHA 独立回读 ACK 后才能继续。最后固定全80 greedy、max-new1600 另900+30。第一 OOM/非有限/截止停止，不改精度、删样本或自动扩大预算；监控未知只保留原 job IDs。
- 主要端点 fullMed 单独；唯一预定次要端点 oldSFT32 关联集合并 fullMed。使用原80提示作为固定 parser/candidate 框架，分别核实际 full80 输入的追加等价与 SHA。所有实体和非关联原始行 bytes 不变。两完整预测共同 freeze 之后各唯一 v2 all/text；旧指标复用，无额外组合、阈值或切分。
- 保留判据：相对旧 SFT32 强基线关系 micro+macro 严格增加且 macro 不降。旧 F1 Score .660942002653；短 Med .603701549011 / union .644663430477 仅已冻结背景，不重新评测。本 F1 是反复观察的内部开发折，不是独立或官方结果。
- 预算依据：实际12824最长更新8.469秒、allocated58.364GB、reserved60.677GB；token线性约800秒与248×最长约2100秒均估计，不能保证正式耗时/多形状显存。正式实际记录为准。
- 本次不迁 F0/原 dev/官方，不访问新验证 gold，不下载新模型。单个 Med 基座+LoRA4,329,881,968，原视觉模块冻结未运行；官方限制每个独立模型≤10B，允许组合。
