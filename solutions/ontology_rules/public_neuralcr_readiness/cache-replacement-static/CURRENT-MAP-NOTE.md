# NCR 当前 HPO 列映射源码准备

入口：`build_ncr_current_map.py --columns C --hpo H --output M.json`。完整结果 `M.json` 包含全部 `mapping` 行；小摘要为 `M.json.summary.json`，只保存原列数/顺序、None/root、各映射原因计数、来源和完整产物 SHA。两输出及各自 `.tmp` 都拒绝覆盖。源码导入不读取文件、不执行映射，没有 NumPy/TF/fastText 或其他模型依赖。

固定输入是已完成 CPU preparation 的 NCR `columns.json`：185423B、SHA `cb8f61996d9f155b7e2c033cd5443d9cdb854bbed625d1bbb3b5d5ac60a84fc9`；原顺序 SHA `bb2edeb6af490408a31580d73f8959a34a2738884b1f79ab6328fdf4cd68e030`。原始14260个 HPO 概念加末列 literal `None`（index14260），root=`HP:0000118`。顺序 SHA 对 `'\n'.join(labels)` 的 UTF-8 字节计算，不含末尾 LF；不得混同完整 columns 文件 SHA。HPO 固定11222341B、SHA `a5092cbdf605f568403cf7380d9173014015692433b2cc631bc5c1b053876b1b`。

解析、图遍历和映射优先级复制已冻结 `research/public-phenotagger-reuse/build_label_map.py`（2740B、SHA `1b4a60a8a965a4704a3a78e52fa56e628fc855cf0781ec7b9c32b5cb3cd57e03`）的语义：逐 `[Term]` 读 `id/is_a/alt_id/is_obsolete/replaced_by`；沿 HP:0000118 子图取非 obsolete 的19120类；依次处理同有效 ID、官方 alt_id、恰好一个且直接指向 allowed 类的 official replaced_by，其余标为 unresolved/outside。无递归替代、多候选猜测或 consider 推断；alt 表同原代码只由非 obsolete term 建立。唯一负类差异是 NCR literal `None`，不套用 PhenoTagger `HP:None`。

映射保留每一原列的 `column/original_label/current_label/reason`，包括不映射列；不删除、重命名或重排原 head，也不改变 top1 的输入概率或归一化。原 root 在列映射中仍按有效 ID 映射，实际 native 解码早已按其原规则排除 root/None；不得把列映射偷换成新的解码器。合并多个 raw ID 到同一规范 HPO 的后继候选聚合不在本脚本执行。

`upstream_policy_reference` 是源码准备阶段已核的公共实现出处，不冒称运行时重新执行了旧 mapper；运行时实际读取的本脚本、columns、HPO 都记录完整 refs。后继应将本源码 SHA 绑定到自己的计划。此次只内存编译、导入和六个合成局部映射检查；**没有执行真实 columns/HPO 全映射，没有产出真实覆盖率或候选数量**。未来真实输入不符时在任何输出发布前失败；发布时 I/O 失败可能留下一个已经完整写成的文件或 `.tmp`，原样保留，不能无凭据覆盖重跑。
