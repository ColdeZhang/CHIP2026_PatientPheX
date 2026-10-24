# Mention hard negatives

已完整训练、推断和共冻双评，固定配方负结案：强基线 .728697536469，全部新增并集 .682735038528，新头替换 .617921909195。新增399实体仅14TP/385FP，未迁原dev或官网。

本方案在原2353三态名称生成监督上加入578严格OOF、与全部注释零重叠的annotation-negative，固定两轮。相比原头少输出115实体并提高局部精度，但不足以改善强系统；Torch跨主机差异也保留为归因限制。

当前摘要和恢复入口见[WORKLOG.md](WORKLOG.md)，完整终态小指针见[evidence/terminal.json](evidence/terminal.json)。大原始记录与模型状态按精确外存locator保存。
