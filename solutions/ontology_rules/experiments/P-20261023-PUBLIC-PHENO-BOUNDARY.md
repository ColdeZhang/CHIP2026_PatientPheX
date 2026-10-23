# P-20261023-PUBLIC-PHENO-BOUNDARY

目标：已关闭同跨度 top3/Sap 配方后，核清剩余 384 个边界 FN 是否能由现有公开 Pheno 原生候选供给解决。只做 train64 错误机制诊断，沿原冻结 `.3`、v2 all/text，不读原 dev，不新 NN/GPU/fit/质量选择。

固定分母为 269 overlap-only 与 115 no-overlap，337 linked＋47 NIL、333 独立跨度。查询沿冻结 FN `unmatched` 恢复，四折 98/92/91/103，原逐篇/pooled 计数完全一致。NIL 不等同“同位置存在任何 HPO”，compound 保留原子分母，不 canonicalize gold。

先做资源盘点：远端三个坐标 raw 和三个词串 unique 缓存共约 625 MB，均可读；每 1,000 行约 0.012–0.016 秒。预计加实际 target-passage generator/tokenizer 重放可在 120 秒内；外层单次 120 秒＋TERM 5 秒。只回收小聚合与审计，不下载缓存进 Git。

预定分类是准确坐标已生成但 native 最终筛掉、只有相同 lower-token 缓存但无准确坐标、两者都缺。第一类报告两模型 top1/top3 条件供给与词长/32-token 截断；词串缓存命中必须区分原段落精确 token 对齐与独立片段 tokenization。不会生成 gold 选出的正式候选或 oracle 预测。

实际完成：程序24.692秒、外层25.220秒，rc0；241段/43,395 native token/60,989候选位置与两模型raw坐标逐项相等，六个大缓存全部流式SHA通过。终点summary `60ca3b09`，详细审计外存，增量独审待回。入口和恢复见[源码说明](../public_phenotagger_boundary_diagnostic/README.md)，[终点证据](../evidence/round2/public-phenotagger-boundary-diagnostic.json)。

|供给状态|FN原子|独立跨度|
|---|---:|---:|
|准确坐标已生成，native最终缺|217|199|
|只有同词串缓存，没有该坐标|0|0|
|坐标和词串均缺|167|134|

无供给167原子分为unigram89（88跨度）、>8词53（27跨度）、2–8词被原过滤20（14跨度）、未对齐5（5跨度）。不要把compound展开原子数当独立边界数。

217已生成中，PubMed/Bioformer/并集top1条件供给72/66/95，top3为145/134/162。并集top3拆为135 linked＋27 NIL；190 linked中的9个literal目标已不在固定current head可达范围。该217全部没有32-token截断；另外3个缺供给长片段只是若编码会截断，并未实际进入神经缓存。

38原子（36跨度）至少一个模型原top1已过原>.95且current有效却仍未进入canonical，其中literal正确27原子/27跨度，说明不能把所有早期损失归因门槛。尚未把hybrid/abbr各阶段单独定位，下一只做机械全输入候选普查与实际阶段分层，仍不fit或看新质量。

这些是错误条件供给上界，没有精度保证；多ID同跨度也不保证能被当前greedy同时选中。结论是已有边界内继续扩ID无法触达此384缺口；先分别研究原坐标的native筛选损失与134个真正缺ngram位置，再决定是否值得新的边界模型。
