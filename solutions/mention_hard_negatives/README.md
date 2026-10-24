# 固定 mention 负例监督

独立检验追加训练侧OOF拒绝监督的价值。保留原2353目标，按每文正HPO/NIL数量四分之一向上取整，增加578条与所有注释零重叠的候选，目标空数组。全2931行，仅CPU准备已授权，尚未新GPU训练或质量。当前原mention2353行run继续。

见[台账](WORKLOG.md)、[固定规则](plan.json)及[实验](experiments/E-20261024-MENTION-HARDNEG-001.md)。
