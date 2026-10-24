# Mention boundary selection

完整结果：强基线并集 **.721846056324**、new-only **.714176156405**，未胜强基线 **.728697536469**，固定配方负结案。完整记录见 [result.json](result.json) 与 [WORKLOG](WORKLOG.md)。

在完整 Top8 多选负终点之后，固定补充539个训练侧 OOF 边界拒绝样本，保原2931目标、候选选项、用户上下文与顺序。新增独立 `BOUNDARY` 状态，仅拒绝该 exact span，不解释为临床否定；原 `NO`/`NIL` 定义不变。共享 system 提示和输出语法同步扩展，因此不是单纯数据消融。

固定6224非exact重叠候选→严格crossing与歧义排除后809→每文ceil(HPO+NIL/4)上限选539；不足不补。旧9B slot实际训练曾含454个重叠0标签且完整负，不能称从未试边界负监督。新的具体假设仅为Top8集合选择任务补边界状态。

所有大产物：`/workspace/patientphex-runs/round2-20261023/research/mention-boundary-selection-v1`。先核全3470训练/1416纯held native模板、完整监督mask和514合法状态；随后按已授权单链运行真实probe、fresh两轮最终epoch、完整推断及两预测共冻后各一次v2。强门固定 .728697536469；原Top8 .7210256/.7131837仅进展参照。目标未达到不得称global best。
