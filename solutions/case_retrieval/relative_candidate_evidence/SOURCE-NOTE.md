# 相对公共来源18列：独立纯源定义核（2026-10-24）

范围：只读下列实际源码，不读候选/标签数据，不 fit、预测、模型或评分。这里确认语义与建议边界，不声称新 source/全8池 actual census 已通过。

**原分数语义。** `public_score`、`bioformer_score` 从 canonical 的实际 `score` 转 float，逐项要求属于 provenance 的 `actual_hybrid_scores`，再按 exact `(offset,length,identifier)` 取最大值；`biobert_score` 直接对同 key 的实际 hybrid score 集合取 max。均不是 raw classifier posterior，也不是 selector 正确率，dictionary/abbreviation 来源可能参与。每个来源单独做差，不横跨三模型比较数值大小。presence 是该来源确有此 exact key，不是 score>0。

**字段缺失与合法零。** 新 helper 用强索引读取三组 `*_present`/`*_score`，缺任一字段即失败，不用 `.get(...,0)` 把 schema 损坏当缺来源。presence 必须是 bool。present=True 时分数应为非 bool 的有限 int/float 且 0≤s≤1，**真实0允许**；present=False 时既有合法占位包括 bool False（public/Bioformer，包括 BioBERT 新 key 的旧来源）与数值0（BioBERT），可读作0，但不改原值/类型，非零或非有限占位拒绝。历史制备路径无 present=True 却没有分数的正常分支。

**固定集合。** 每个文档独立，先确认 exact key 唯一，保留原候选顺序。对候选 c 与来源 q，先按几何/identifier 建集合，再仅保留 `d[q_present] is True` 的 d：

- G1：d≠c，offset和length都相同，`d.identifier != c.identifier`。
- G2：d≠c，`d.identifier == c.identifier`，(offset,length)严格不同，且 `c.start < d.end and d.start < c.end`。因此包含/被包含及交叉都算，端点相邻不算。

identifier 全程原 literal 字符串，不 split/sort/trim、不 canonicalize、不重写 compound/NIL；比如 `A;B` 和 `B;A` 在本特征仍是不同 literal，符合原池 exact key。**这不是改变原 nested 的冲突定义**：后者仍按原代码对 compound 拆分集合、允许异ID proper containment。新特征不是排斥规则，也不删除任何候选。

**每组×每来源三值（2×3×3=18）。** 非空 G：`exists=True; max=max(d[q_score]); delta=(c[q_score] if c[q_present] else 0)-max`。空 G：`exists=False; max=0.; delta=0.`；尤其不能在空 G 时输出自身分数。self 缺来源而竞争项存在时 delta=−max，原103列的 self-presence 与 self-score 原样保留，足以区分缺来源/实际0。来源有观测但max=0仍 exists=True；不借 labels、selector 概率、cut 或“原先入选”筛选竞争项。

**增量确实不同。** 冻结 `calibrate.features` 的 n_overlap/covers_base/contained_base/same_base_id 和频次仅访问 `base['entities']`；`fit_control.feature` 只追加18个逐候选公共来源字段，不遍历完整候选池。因此本18列不是已有同 base 竞争特征的别名。它也不是改学习目标或再跑已关闭的全局解码。仍不能补缺失的正确 span/ID，不保证质量改善。

精确只读源码（SHA绑定本次核查版本）：

- `pub_prepare`: `/workspace/patientphex-worktrees/round2-sequence-crf/solutions/sequence_crf/prepare_public_selector.py`；SHA `e8187e464bef2c628298dd3e567a40c7d809b91ef1a261e93fb00d0169872bbd`。
- `bioformer_prepare`: `/workspace/patientphex-worktrees/round2-sequence-crf/solutions/sequence_crf/prepare_public_bioformer.py`；SHA `bda5b36ee0b43cae4c9537cc3b8170f7d5f43137873cdabfca7a7c2471869ed9`。
- `pub_merge`: `/workspace/patientphex-worktrees/round2-ontology-rules/solutions/ontology_rules/public_phenotagger_biobert_selector/dependencies/legacy/select_public_candidates.py`；SHA `3378df59be8c13c45d07f9c17575523d58906837bba9c6b3f3cc814d3a2f7bd6`。
- `bioformer_merge`: `/workspace/patientphex-worktrees/round2-ontology-rules/solutions/ontology_rules/public_phenotagger_biobert_selector/dependencies/legacy/select_public_bioformer.py`；SHA `7cf106a6503cb515dd389b0664cee2bbf2591a61bbbe686d53ab1dccae73cb25`。
- `biobert_prepare`: `/workspace/patientphex-worktrees/round2-ontology-rules/solutions/ontology_rules/public_phenotagger_biobert_selector/prepare_pool.py`；SHA `41872bde405defa212775ade97e394c3f4a097ececd110faa53e9d4f2cb2be37`。
- `base_features`: `/workspace/patientphex-worktrees/round2-ontology-rules/solutions/ontology_rules/public_phenotagger_biobert_selector/dependencies/solutions/ontology_rules/calibrate.py`；SHA `b7e18879673c889eec297c86a840349b23f8be5e6b3053c12ba58754091288d6`。
- `feature_assembly`: `/workspace/patientphex-worktrees/round2-ontology-rules/solutions/ontology_rules/public_phenotagger_biobert_selector/fit_control.py`；SHA `ae870b823abaae2cc50809e633f9b4f5b0ecf13a5515a846e022b284d4309253`。
- `nested`: `/workspace/patientphex-worktrees/round2-ontology-rules/solutions/ontology_rules/public_phenotagger_biobert_selector/dependencies/solutions/sequence_crf/decode_nested_selector.py`；SHA `7f4484fc1db132746f674c8c268832705cbeba7d05bbe65268c1ef0e275a7c7f`。
