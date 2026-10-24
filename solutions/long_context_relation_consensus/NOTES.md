# 查重与边界

- 新父模型为BioClinical ModernBERT全文患者关系 `b63af4dc`，完整2874固定logit、epoch30；父并集F1 .664715835189低于B4 .701325377212，父结论不改。
- 旧DocRE三直接控制与meta-OOF使用Biomed短窗/marker，meta +Doc .674069低于.686339；见 `/workspace/patientphex-worktrees/explore-document-relations/solutions/document_relations/WORKLOG.md`。
- 旧DocRE30∩schema F1正、F0负，见 `/workspace/patientphex-worktrees/explore-relation-consensus/solutions/relation_consensus/WORKLOG.md`；既无新父模型，也不是本次G/S/Q/C支持集合。
- seed支持公式与字节helper可复用，原seed F1的evaluate_report会带4个敏感性调用；本次仅直接evaluate一次all/text，不复用其旧评分入口。
- 窄检新父完整source/plan只有无条件保护并集，未见同父同公式实际端点。本次是看到父负后新增假设，F1为研发集，不称盲测。
- S沿既成完整双采样交集预测，减去其继承B；因继承B⊆当前B4，对新增集合数学等价于真实S1∩S2。不是把缺失当拒绝，不重新解析或运行生成模型。
- 新L只有HPO单原子；支持按v2 literal/NIL语义处理，不canonicalize或模糊匹配。
