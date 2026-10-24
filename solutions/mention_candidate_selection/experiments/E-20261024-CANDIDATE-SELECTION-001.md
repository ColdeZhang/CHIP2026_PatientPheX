# E-20261024-CANDIDATE-SELECTION-001

固定全部2931训练/1416纯held，公开Top8候选多选+missing/NIL/NO，数据/目标/祖先与未来双端点见plan.json。只到CPU准备，LLM训练尚未授权；原2931直接canonical run继续独立。

先补1034 missingquote（actualCPUmax15），已有1284公开同资产query精确复用；sequence G36 Sap→span PCL F0→本query顺序，任何启动前fresh核，只用独立namespace。输入不含旧ID/gold候选插入；所有8Ccodes附公开ID/name，无cosine提示。目标selected按候选序，包含部分compound+missing和emptyOTHER；emptyfalse非法。

复用已核8B模板/native编码、doc原训练器，不另造训练loop。新有限grammar恰好513合法输出，CPU逐个全原生token路径与实际HF processor核；完整4347原模板与监督mask/causalEOS逐行核，不拿旧形状当新接口已证。实际数量/耗时/上限待完整执行，不能报CPU准备为训练收益。
