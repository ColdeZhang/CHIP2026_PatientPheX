# E-20261024-MENTION-HARDNEG-001

假设：当前2353仅正式gold span和87NO缺少提取器误报拒绝监督；追加纯OOF高混淆负跨度可以改善固定mention生成头。单变量是训练支持集追加578空数组目标；保持原公开模型/三状态语法/上下文±256/原LoRA训练器。

运行前固定：每文K=ceil(HPO+NIL目标行/4)，在与所有注释零重叠池按prob desc、sim desc、offset asc、length asc取K；不选比例、不扫阈值。原2353对象精确保留，完整训练2931；2epoch/734updates/尾3/warmup37为规模派生值。

现阶段仅CPU实际native输入/目标/掩码/预算，未授权新训练；旧run不改。未来若获授权则fresh最终epoch2完整F1，强B union全接受实体为主、同固定1416池new-only替换为诊断；关系对象不改，两完整预测共冻后2v2/旧0，任一严格胜强B才候选。

数据准备0.353s已完成；原targets和排序/逐文配额/原始候选分数证据外存prepared-v1。准确native形状待实际CPU，不提前声称继承GPU压力已通过。

CPU增量真实完成：新增578个原生/HF输入、全mask/1156答案prefix一致；全2931的Pareto仍原(333,28)/(336,27)/(383,23)，新增max369/3被已完成压力覆盖。完整1416 held编码未重做，原SHA不变。实际7.410s，rc0；30成员小包全SHA回收。成本按同预写公式替换rows/updates，保守估4521.224秒，尚无新GPU授权、训练或评分。

root后续正式授权：空闲.3GPU0，固定4800+30两轮/734/尾3/warmup37，两完整ACK→1416fresh1200+30→两个同端点jointfreeze再2v2。无需等原2353质量调整配方，578选择及配比均保持运行前冻结。新主机Torch版本差异如NOTES披露，原native.train/common/loss逐字沿用；首正式资产复用同机Doc完整SHA与当前stat一致守卫，不做重复压力或新下载。
