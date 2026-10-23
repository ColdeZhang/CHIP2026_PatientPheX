# P-20261023-PUBLIC-PHENO-BOUNDARY

目标：已关闭同跨度 top3/Sap 配方后，核清剩余 384 个边界 FN 是否能由现有公开 Pheno 原生候选供给解决。只做 train64 错误机制诊断，沿原冻结 `.3`、v2 all/text，不读原 dev，不新 NN/GPU/fit/质量选择。

固定分母为 269 overlap-only 与 115 no-overlap，337 linked＋47 NIL、333 独立跨度。查询沿冻结 FN `unmatched` 恢复，四折 98/92/91/103，原逐篇/pooled 计数完全一致。NIL 不等同“同位置存在任何 HPO”，compound 保留原子分母，不 canonicalize gold。

先做资源盘点：远端三个坐标 raw 和三个词串 unique 缓存共约 625 MB，均可读；每 1,000 行约 0.012–0.016 秒。预计加实际 target-passage generator/tokenizer 重放可在 120 秒内；外层单次 120 秒＋TERM 5 秒。只回收小聚合与审计，不下载缓存进 Git。

预定分类是准确坐标已生成但 native 最终筛掉、只有相同 lower-token 缓存但无准确坐标、两者都缺。第一类报告两模型 top1/top3 条件供给与词长/32-token 截断；词串缓存命中必须区分原段落精确 token 对齐与独立片段 tokenization。不会生成 gold 选出的正式候选或 oracle 预测。

当前为准备阶段；入口和恢复见[源码说明](../public_phenotagger_boundary_diagnostic/README.md)。实际终点与建议完成后补在本文，不能把条件上界当新验证分数。
