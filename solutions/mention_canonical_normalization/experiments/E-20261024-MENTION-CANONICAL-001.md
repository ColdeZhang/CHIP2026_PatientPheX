# E-20261024-MENTION-CANONICAL-001

假设：局部occurrence语义与完整公开名称生成，可减少固定跨度的ID错误或接受旧规范化未覆盖的真实表型。目标是实际完整强基线增量，不能用gold跨度上的名称准确率替代Score。

数据、三状态schema、继承祖先、无gold固定1416跨度池和预锁两端点见plan.json。训练2353，全部32文献；字符context最多581，不是已证明token上限。CPU数据准备0.364秒完整，尚无native shape或GPU结果。

后继拟复用现成native.train，fresh Qwen3-8B LoRA2epochs/epoch2唯一，590更新、尾1、warmup30，不holdout/earlystop。实际GPU预算尚未批，完整CPU合同后再据最大输入/监督形状给probe与成本。已有public2/strongbase祖先及GLiNER训练/strict/Sap均排F1，公开预训练全量谱系未知如实披露。
