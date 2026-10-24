# 强 BioBERT 候选池：全局兼容解码的只读可行性核查

2026-10-24；目标是确认固定候选和概率下是否还有未测试的冲突选择变量。本轮只读源码、有限历史和已冻结四fold无标签候选/概率；无gold、拟合、GPU、优化求解、评分或新预测。原dev当前配置不变。

**最终结论：root后继授权的唯一固定控制已完成，pooled Score小降，按事前规则关闭。** 下文先保留只读准备，再记录唯一质量终点。当前强池仍保留概率贪心加嵌套追加。旧WIS负结果使用较早Bioformer池且禁止所有重叠，不能直接覆盖当前允许异ID包含的候选图。实际统计只用了3.194秒；66个有冲突组件最大仅3节点，全部子集配置上界合计280。

## 实际现行语义与历史边界

强B `execute_control.py`（`10a74d25`）先调用 `greedy`：p降序、长度降序、offset升序，cut=.3，排斥所有半开区间重叠。随后 `decode_nested_selector.py`（`7f4484fc`）保留全部原greedy锚点，按同排序尝试追加候选。每个新增候选须合法HPO、非否定，且与已选项的每一处重叠都为proper containment、原子HPO ID集合不相交。proper containment允许共享一个边界，但排除完全同界；跨越、完全同界、包含且ID相交均冲突。不能把所有重叠都算冲突。

原greedy锚点本身未经过新增项的合法ID/否定门。此次候选宇宙保留这一不对称：原锚点 ∪ 其他达到.3且通过原新增项单项门的候选。若未来统一给锚点加门，会引入额外变量。全局选择若允许替换锚点，则将“必须保留原锚点”的顺序政策改成集合优化；不能声称逐条追加行为仍完全相同，保持的是候选概率、单项资格和两项兼容规则。

已核历史 `weighted-interval-control.md`：旧Bioformer四fold，精确十进制p−.3效用、严格非重叠WIS，2026-10-23 22:08完成，64篇新旧byteexact，pooled .725655389完全不变。随后 `selector-nested-control.md` 是保留锚点的嵌套追加，非全局替换。有限核查未发现强BioBERT＋允许proper-nesting的统一全局兼容优化已运行；不宣称穷尽全部历史。强B实际四fold特征维数为102/101/103/103，不把原dev 103列推广为全部四fold。

## 冻结无标签池的实际统计

| 项目 | 四fold共64篇 |
|---|---:|
| 全候选 / p≥.3 / exact .3 | 62,281 / 4,642 / 0 |
| 原greedy / 当前nested | 4,345 / 4,566 |
| 保留原资格后的候选 | 4,633（9个非锚点因否定排除） |
| 全重叠对 / 允许异ID包含 | 311 / 241 |
| 冲突边 | 70：跨越40、同界3、包含且ID相交27 |
| 冲突连通组件 | 4,497个孤点、62个2节点、4个3节点 |
| 最大冲突组件 / 最大重叠组件 | 3 / 5 |
| 非平凡组件所有子集数上界 | 62×4＋4×8＝280 |

重新在内存执行原decoder，只识别既有锚点和已选集合，并确认旧选择在该图中没有冲突；没有求新最优集合，也未写任何预测。输入候选对象与原prepared input逐项一致，四fold分数及无标签input的完整SHA均按原manifest核对；未打开train/gold/model文件。

只读阶段提出：后继可固定唯一效用sum(p−.3)，每组件精确枚举，效用并列优先保留当前nested实体、再按实体键确定性打破并列；无cut/效用扫描。由于分数不保证校准，效用提升可能导致F1下降；组件小只证明可算，不能证明有可替换项或实际质量改善。后继实际按root新授权执行，见下。

源和完整定位见[census.py](census.py)、[计划](plan.json)、[小摘要与四fold来源](summary.json)。完整逐文计数在外部 `research/global-compatible-decoder-census/census.json`（`ba64508e`，完整SHA见summary）；不复制候选正文、概率池或大日志入Git。

## 唯一固定全局兼容控制：小幅负结果

root事前授权≤60秒CPU：同4,633候选、70冲突边/66非平凡组件，孤点原样保留。效用严格为 `Fraction.from_float(p) − Fraction.from_float(0.3)`，读取原有限float的精确二进制有理数，不引入epsilon或小数四舍五入；并列先最大化旧nested键交集，再取排序后的 `(offset,length,HPO_identifier,original_candidate_index)` 字典序最小。7项有区分力合同包括三节点全局优于贪心、精确并列、一个ULP差异、原字节/转义嵌套键/CRLF及重复键拒绝，实际通过。

实际2026-10-24 04:32:30完成，**3.488秒、rc0**。四份完整预测和各组件精确目标/增删在04:32:29.921共冻，首次gold marker为29.923；确有变化后才各fold一次原v2 all/text及pooled一次，共5次，旧指标全部复用。没有新fit、GPU、cut扫描、原dev或官方操作。

| 指标 | 旧nested | 固定全局兼容 | 差值 |
|---|---:|---:|---:|
| pooled Score | .730993616492 | .730822309437 | −.000171307055 |
| F0 Score | .739700183730 | 相同 | 0 |
| F1 Score | .728056418176 | 相同 | 0 |
| F2 Score | .753373106038 | 相同 | 0 |
| F3 Score | .706780475185 | .706048694708 | −.000731780477 |

只有F3一篇的一个三节点组件改变：删除1实体、增加2实体；其余63篇相同，前三fold fullpred byteexact。精确效用增加 `259073311955265/4503599627370496`，但mention与document F1都下降。全部非entity原始UTF8片段（含关系）完全保留，关系micro/macro不变，四fold格式0错0警。说明该固定概率效用的局部增益没有转成任务收益；不泛化为所有全局解码无效。

事前保留条件为pooled Score严格升且无任何fold下降，实际两项均不满足，**keep=false；保留旧nested，关闭该固定控制，不改效用或cut、不迁原dev。** 完整源[control.py](control.py)、[固定计划](control-plan.json)、[合同](control-contracts.json)、[结果](control-results/result.json)、[freeze](control-results/complete-freeze.json)、[终点清单](control-results/terminal-manifest.json)均已保存。全四fold预测与82KB组件细节留外部 `research/global-compatible-decoder-control/formal/`；Git只保存小结果/源与SHA指针。统一终审待回收后补链接。
