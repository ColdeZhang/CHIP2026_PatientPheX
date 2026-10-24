# 固定逆序原dev迁移：当前best关系并集

本次沿F1/F0已冻结逆序/CID双向映射，只运行原dev旧pool64条提示、原full64 epoch002 adapter。当前基线是三公开HGB+propernested及旧SFT+support的baac预测（1072实体/368关系，Score .738493753）。仅与新逆序关系取literal-v2并集；保留所有旧关系，包括35 publicnew关系、全部实体及非关系字段，不重算support、不加筛选。

原候选来源仍955实体/466文档概念，解析器保留原候选fallback，使用已存在的`--final-entities`输出当前1072实体；union强制非association全对象一致、原association元数据/顺序和原phenotype前缀不变。只新union做一次v2，reverse-only冻结但不评分。成功判据沿前两控制：Score严格上升且macro不降；原dev是历来开发集，不是盲测。无新训练、cut、组合选择或官方操作。

输入准备64条全部双逆字节恢复，6合成边界通过；202742输入tokens/max5194，无截断，旧同64为202743。旧full64推断A30实际341.386秒仅作预算参照，新运行A800，版本相同但硬件不同。600秒+TERM30；只173MB adapter/config从.100持久副本复制到自有SHM，新目录uid/余量和实际完整SHA核验后复用Qwen SHM。不会向约193MB磁盘写权重。

入口`reverse_json_order_dev/{prepare,transfer_adapter,operations,launch,finish}.py`，外存`research/reverse-json-candidate-order-dev/`。formal/plan.json SHA f47b408a8920b362882f9988b5b2711cab2d0c9be657b37e7177b73a76dab817；执行operations deploy→start→collect，finish assemble→evaluate。运行前已读main共享记录；本检查点仍未GPU/未新质量。
