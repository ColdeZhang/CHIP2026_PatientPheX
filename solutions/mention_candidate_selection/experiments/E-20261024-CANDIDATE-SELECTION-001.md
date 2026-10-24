# E-20261024-CANDIDATE-SELECTION-001

固定全部2931训练/1416纯held，公开Top8候选多选+missing/NIL/NO，数据/目标/祖先与未来双端点见plan.json。只到CPU准备，LLM训练尚未授权；原2931直接canonical run继续独立。

先补1034 missingquote（actualCPUmax15），已有1284公开同资产query精确复用；sequence G36 Sap→span PCL F0→本query顺序，任何启动前fresh核，只用独立namespace。输入不含旧ID/gold候选插入；所有8Ccodes附公开ID/name，无cosine提示。目标selected按候选序，包含部分compound+missing和emptyOTHER；emptyfalse非法。

复用已核8B模板/native编码、doc原训练器，不另造训练loop。新有限grammar恰好513合法输出，CPU逐个全原生token路径与实际HF processor核；完整4347原模板与监督mask/causalEOS逐行核，不拿旧形状当新接口已证。实际数量/耗时/上限待完整执行，不能报CPU准备为训练收益。

## 完整CPU与条件GPU链

全nativeCPU27.214870秒rc0；4347逐行独立渲染/IDs与2931完整mask/EOS，513完整合法输出native/HF及12293prefix，15真实processor和13拒绝通过；max训练684/17、held621、完整grammar max40含EOS。原8.248MB/51成员包连接UNKNOWN/LUNA时保已取分块，最终同SHA2ef7d8d3全回收，没有重跑CPU。

root新授权唯一240+30的13真实FB/1Adam+三模式和OTHER训练输入必要约束generate；固定成本公式见gpu-plan.json，warm12步max秒/token×3033240全tokens+734实际Adam，25%余量与120秒双ACK/I/O，冷FB只算一次。公式事前冻结，同时保存原native含冷启动估计和最坏逐行诊断，不据结果改公式。若≤7200且全部数值/内存通过则丢probe状态、fresh全两轮734更新7200+30，保存两fullstateACK；完整1416fresh1800+30后两完整共冻→2v2/旧0。门失败不裁样/量化/缩context/减epoch，资源问题与方法负区别。
