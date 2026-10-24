# 独立患者响应编码：固定完整 F1 控制

目标是判断在保留 clinical PID 与全部证据的同时，独立输出 PT 编码能否提高关系总 Score。它改变提示和输出表示，也改变监督 token 分布；不声称只是等价字符串置换。

固定数据为 F2+F3 的原短 normal 124 行（32 文献、87 患者），held 为原追加全文 F1 的完整 80 行。clinical ID、patient mentions、CID-HPO、原文、markers、完整报告后缀及原次序均保留；系统输出指令与追加映射表单独标出。旧 renderer 不重跑。PT→原 PID 严格可逆；未知、重复、缺 CID 或非严格 JSON 整行弃答，原始 IDs 和文本先保存，不修补。

CPU 实际 18.607 秒通过：训练 427560 tokens、max5910，监督 7170→8138；held 685372、max14570。训练工程 cap6144，held cap14848，generation1600，均不裁切。完整数据与逐行原生 IDs/标签哈希外存，仓库只保源码、统计和引用。

唯一真实 A100 GPU2 工程探针 `proc_6e7979ca…` 于 11:29:55.022 UTC 结束，153.633742 秒、rc0。先8微批一次实际 Adam 更新，再3累积（含最大128监督）及最长5910；12次前后向、496优化器状态、有限梯度、冻结base和三族LoRA更新均通过。峰 allocated 24.775GB / reserved26.724GB。探针状态丢弃。它没有新增 head A/B 数值比较；旧引用 e35dba61 的运行范围见 NOTES 的独立修正。

正式配置事前固定：Qwen3.5-9B+r16，原标准 indexed CE、seed20261023、LR1e-4、batch1/累积8、尾4，fresh2epochs/32updates/warmup2。两轮及两次完整 state 到 .100 的 SHA/fsync/reopen ACK 共用3600+TERM30；第二轮恢复完整 Adam/scheduler/RNG/global step。唯一最终 epoch2 进入全80 fresh F1（1200+30），没有中途选 epoch。

两完整预测共同冻结后各一次 v2，旧基线不重评：

| 身份 | 固定端点 | 保留规则 |
|---|---|---|
| primary | coded-alone | 总 Score 严格超过 N .676546354662594 可为候选 |
| diagnostic | N∪coded | 同强 N 门；两臂均过时取较高 Score，平局 primary |

全部1189实体、39患者及非关系字段保持。macro 完整披露，不额外作否决门。另报历史 fullsupport .6863389812716092 差值，不把胜 N 称为全流程最高；不自动扩 F0/原dev/官网。

成本是估计：实测压力82.430726秒，按行数/总token两种投影较高者加15%，另两次加载开销及600秒合计导出/控制余量，共2698.935秒。原3600硬帽不因传输/恢复重置。A100反复UNKNOWN/LUNA目前影响准备传输，未开始正式训练时钟；同460078B部署包用固定SHA补齐，不重复探针。实际状态见 [台账](../WORKLOG.md) 与 [恢复入口](../ready.json)。

唯一正式 managed 入口于11:45:06.757009 UTC被接收，原共享截止12:45:04.713781；模型实际进度待控制连接恢复后核实。held质量尚未开始。后续只填真实终点，不以训练loss或格式通过替代评分成功。
