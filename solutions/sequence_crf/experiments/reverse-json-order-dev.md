# 固定逆序原dev迁移：当前best关系并集

本次沿F1/F0已冻结逆序/CID双向映射，只运行原dev旧pool64条提示、原full64 epoch002 adapter。当前基线是三公开HGB+propernested及旧SFT+support的baac预测（1072实体/368关系，Score .738493753）。仅与新逆序关系取literal-v2并集；保留所有旧关系，包括35 publicnew关系、全部实体及非关系字段，不重算support、不加筛选。

原候选来源仍955实体/466文档概念，解析器保留原候选fallback，使用已存在的`--final-entities`输出当前1072实体；union强制非association全对象一致、原association元数据/顺序和原phenotype前缀不变。只新union做一次v2，reverse-only冻结但不评分。成功判据沿前两控制：Score严格上升且macro不降；原dev是历来开发集，不是盲测。无新训练、cut、组合选择或官方操作。

输入准备64条全部双逆字节恢复，6合成边界通过；202742输入tokens/max5194，无截断，旧同64为202743。旧full64推断A30实际341.386秒仅作预算参照，新运行A800，版本相同但硬件不同。600秒+TERM30；只173MB adapter/config从.100持久副本复制到自有SHM，新目录uid/余量和实际完整SHA核验后复用Qwen SHM。不会向约193MB磁盘写权重。

入口`reverse_json_order_dev/{prepare,transfer_adapter,operations,launch,finish}.py`，外存`research/reverse-json-candidate-order-dev/`。formal/plan.json SHA f47b408a8920b362882f9988b5b2711cab2d0c9be657b37e7177b73a76dab817；执行operations deploy→start→collect，finish assemble→evaluate。运行前已读main共享记录；本检查点仍未GPU/未新质量。

## 唯一终点

原job64条完整/466候选有效，无fallback；native230.897秒、wrapper239.176秒，峰19,627,511,296B，00:40:47 fresh GPU空。00:40:54完整reverse/union先freeze，00:41:05唯一union评完。Score **.738493752650→.746081964986**（+.007588212336）；mention .786960514、document .824512535原样；micro .708891596→.725118483，macro .633610366→.647736327。新增15TP8FP、无移除；1072实体、全部368旧关系/35publicnew支持关系及其他对象完全保留，新391关系、0错警，符合事前keep。预测a179a5de5b7271979a0fba851afc269d63125248a56681674e7388d0c6728c39。

原dev开发及A30→A800差异不变，不单独归因顺序、不重跑旧基线；仍未.8，不合并/不官方。入口增量独审ece9db59已过，完整终审PASS 5a106306（实际谱系0b9a71f3）；当前无活动GPU，完整结果与恢复见evidence/round2/reverse-json-order-dev-results.json，实际源未改。

恢复小清单 formal/backup-allowlist-final.json 明确列文件/大小/SHA，继承baac已实际恢复的HGB/关系依赖链；root负责远端上传及回下载验证，本清单不冒称已完成备份。
