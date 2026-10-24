# 当前任务

2026-10-24；base main0878cbdf，独立explore/mention-candidate-selection，仅本目录。目标：固定公开Top8多选+NO/NIL/OTHER是否可改善强Task1，当前只授权CPU原生输入/目标/成本准备与必要一次1034补Sap查询，不LLM训练或质量。

保全部2931(原2353+578难NO)，同原1416纯跨度/±256context；原1284 public查询逐SHA复用，新增1034词面（train428/held610、交4）。候选不插gold/exact/训练别名。输出selected数组+missing bool、NIL:true、NO:true互斥；missing不补ID或转NIL。新鲜Qwen3-8B/r16/lr1e-4/2epochs/seed20261023拟定，原native loop复用，不载旧2353adapter。

长期2931原proc92c的.3训练独立不改。原2353自由名称配方及旧BiomedTop8 CE负结论保留；本新输入/目标+578监督共同变化，不称纯prompt因果消融。外存R/research/mention-candidate-selection-v1；现先CPU实际长度，尚未Sap/GPU/LLM。
