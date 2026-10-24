# 方法依据

这是选择性删除控制。旧9B entity-slot binary veto已有负结果；当前改用已完成2931行、665 NO监督的Qwen3名称规范化头，训练目标和供给不同。NO指按赛题标注学习的否定/正常/非表型类别，不保证生物学意义上的阴性。HPO/NIL或弃答不具有删除权限；不根据新HPO输出替换旧ID。

父训练、完整推断与公共资源来源继承 explore/mention-hard-negatives；其原Bunion/new-only两端点不变。

旧binary-veto实际证据：`/workspace/patientphex-worktrees/round2-sequence-crf/solutions/sequence_crf/evidence/round2/entity-slot-selected-veto-results.json` SHA afcdca25（完整身份由后续独审核）；当时删152=71TP81FP，.728144787→.720623999。当前固定头删除133=60TP73FP，仍负；无额外子集挑选。
