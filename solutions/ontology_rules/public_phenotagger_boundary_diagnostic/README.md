# 训练侧边界供给诊断

目的：解释已冻结 Bioformer `.3` 的 384 个跨度缺口（269 仅重叠、115 无重叠），判断主要缺原始边界供给还是原生筛选。它是错误条件诊断，不是新训练方案或质量输出。

`prepare_queries.py` 沿冻结 evaluator v2/FN `unmatched` 恢复 384 个原子、333 个独立跨度；337 linked/47 NIL。gold ID 保持 literal，NIL 只在 exact linked 消耗后使用剩余供给。

`analyze_boundaries.py` 仅重放查询所在段落的既有 native tokenizer、offset restore 和修正过的逐句 2–8gram 生成器。两模型全量 raw/unique 缓存流式核 SHA，生成器坐标与对应 raw 集合逐项核对。三类为：准确坐标已生成但最终 native 消失；没有坐标、仅相同 lower-token tuple 缓存；坐标和词串都没有。无法与原段落 token 边界对齐时，独立切片 tokenization 的词串命中单列，不能冒充原位置可解码候选。

只统计第一类的两模型 top1/top3 真实列、固定 current 映射后的 literal 供给，保留 NIL 剩余供给与多 ID 原子分母。上界没有精度保证，也不能假设非重叠解码能同时输出多个相同跨度 ID。32-token 截断只调用原 tokenizer，无模型/权重加载；不改变阈值、候选或正式预测。

执行：先冻结外部 `plan.json`，再由 `remote_io.py deploy-start --root <外部目录> --client-dir <主库scripts>` 部署；`run_cpu.py` 为单次 CPU 120 秒＋TERM 5 秒硬限，拒绝覆盖。`status` 查看，`collect` 只回收小证据，约 625 MB 原始缓存留原远端。所有大输入、384 查询、完整逐查询证据留外部 `ontology_rules/public-phenotagger-boundary-diagnostic/`。
