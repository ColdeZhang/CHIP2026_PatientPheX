# NCR 八池准备源码

入口 `prepare_ncr_pools.py --plan PLAN`；计划需 `sources`（绝对源码路径→SHA，含本脚本）、`old_prepared_manifest`、`canonical`、`cache_manifest` 三个文件引用以及新的 `output_dir`。引用为 `{path,bytes,sha256}`（旧谱系中的历史 ref 可沿用仅 path/SHA）。输出目录须不存在；每个文件独占创建，异常留下的部分目录不会被覆盖。没有读取 gold、导入模型、fit、预测、评分或 MCP 的代码。

只接受原八池 manifest SHA `87264eb85a1801e249ac6f052faeec5ddeb03fa3044ecb014ca3b9c8d6d220bf`、四折64文档/244794旧候选行。新 cache manifest 的 status 必须是 `complete_NCR_current_canonical_from_verified_raw_cache_no_quality`，`canonical` 整项与计划一致，`raw_spans=5341`，`refs` 保存 verified finalization、原失败GPU收据、NN/source、input/query/head/current-map/provenance 等上游文件引用；本阶段复制这些已绑定引用，不下载远端资产或重跑模型。

canonical 每行必须正好是 `pmc_id/offset/length/identifier/text/ncr_present/ncr_score`，fullkey唯一、doc属于冻结64篇、present必须为 true、score必须为有限float且在[0,1]。这里不新增>.8或其他阈值，不重新映射、取max、round或去重；这些属于已完成上游 canonical 的冻结语义。保留 canonical 原顺序。NCR字段只有 `ncr_present/ncr_score`。

`merge(previous,supplied,doc)` 可供合成调用，不触发 I/O。旧候选按原顺序留作前缀，仅附两个缺源默认值false/0.0；共享键只填实际NCR值。新键在前缀后按来源顺序追加，`probability=.5, similarity=0., exact=False, is_base=False, crf=False, semantic_source=False`，旧12个public/bioformer字段均False，`crf_span`只由旧池相同跨度确定。原semantic `probability`保留，不是旧HGB posterior；任何 `biobert_*` 或 `selector_probability` 都拒绝。旧字段的值、精确类型、dict键顺序和候选顺序递归核验；原input/base/origin_index不变。

每个新键核唯一原passage及字符切片；共享键也核同原文。四折继承 `train_ids/input_ids/nested_input_ids/old_fit_manifest` 和32→16谱系锚点，另存 `old_pool_refs`，不凭OOF名称推广到新留出划分（H40）。每个doc在四个外折各出现一次，所以八池来源出现次数=4×canonical唯一键数；`added+matching=source`逐角色/全量同时检查，新增总行数≤4×5341只作重复/覆盖sanity，不硬套旧1.5预算，更不为过关删候选。

成功 manifest.status=`prepared_input_only_no_fit_or_quality`，顶层有 `old_prepared_manifest/canonical/cache_manifest/cache_source_refs/folds/new_features/old_rows/new_rows/added_rows/matching_rows/source_rows_across_eight_pools/maximum_train_row_ratio`。fold保留旧字段并新增 `old_pool_refs`；每个role有 `new_{role}_rows/added_{role}_rows/matching_{role}_rows/source_{role}_rows/{role}_row_ratio`；`train/input`指新压缩池的完整refs。新F/树数值状态及质量必须由后继真实fresh fit补录，这里不产出模型或后验。

本次只准备源码、内存编译与合成merge检查；未执行真实八池或canonical输入。
