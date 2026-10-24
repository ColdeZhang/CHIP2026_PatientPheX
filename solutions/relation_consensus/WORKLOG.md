# 关系共识融合台账

- 当前最佳内部F1：强fullsupport **.686338981272 → .690618521538**，+22关系=10TP/12FP；micro+.003701627，macro+.013416534。原dev最高901不变。
- E-20261024-FUSION-001固定 `base ∪ (DocRE30 ∩ schema)`，完整16篇/39患者/1189实体，0fit/NN、1新v2/旧0，原非关联对象保持。两个单模型的负结论保持。
- 先有完整freeze03739415，再唯一评分；预测e09273f6。见[evidence/result.json](evidence/result.json)、[实验](experiments/E-20261024-FUSION-001.md)。
- 工程首次pair-reader错误发生在首pred/gold前，原件actual-v1保留；只改reader后actual-v2完成。不能将此错误当模型质量失败。
- 下一步固定F0复验：原DocRE32所有祖先已证排F0，可仅推断；schema保持相同公开模型与train32规范化政策。辅助仅编辑docre_f0/；主线程维护融合源码/台账。没有原dev/官方操作。
