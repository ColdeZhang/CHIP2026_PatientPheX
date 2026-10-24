# E-20261024-MENTION-HARDNEG-001

假设：当前2353仅正式gold span和87NO缺少提取器误报拒绝监督；追加纯OOF高混淆负跨度可以改善固定mention生成头。单变量是训练支持集追加578空数组目标；保持原公开模型/三状态语法/上下文±256/原LoRA训练器。

运行前固定：每文K=ceil(HPO+NIL目标行/4)，在与所有注释零重叠池按prob desc、sim desc、offset asc、length asc取K；不选比例、不扫阈值。原2353对象精确保留，完整训练2931；2epoch/734updates/尾3/warmup37为规模派生值。

现阶段仅CPU实际native输入/目标/掩码/预算，未授权新训练；旧run不改。未来若获授权则fresh最终epoch2完整F1，强B union全接受实体为主、同固定1416池new-only替换为诊断；关系对象不改，两完整预测共冻后2v2/旧0，任一严格胜强B才候选。

数据准备0.353s已完成；原targets和排序/逐文配额/原始候选分数证据外存prepared-v1。准确native形状待实际CPU，不提前声称继承GPU压力已通过。

CPU增量真实完成：新增578个原生/HF输入、全mask/1156答案prefix一致；全2931的Pareto仍原(333,28)/(336,27)/(383,23)，新增max369/3被已完成压力覆盖。完整1416 held编码未重做，原SHA不变。实际7.410s，rc0；30成员小包全SHA回收。成本按同预写公式替换rows/updates，保守估4521.224秒，尚无新GPU授权、训练或评分。

root后续正式授权：空闲.3GPU0，固定4800+30两轮/734/尾3/warmup37，两完整ACK→1416fresh1200+30→两个同端点jointfreeze再2v2。无需等原2353质量调整配方，578选择及配比均保持运行前冻结。新主机Torch版本差异如NOTES披露，原native.train/common/loss逐字沿用；首正式资产复用同机Doc完整SHA与当前stat一致守卫，不做重复压力或新下载。

## 第一轮持久检查点

原proc_92c814d2于13:14:22.272530完成367updates，持久ACK31ee14d9/1764B；504adapter、504Adam/1512状态张量和scheduler/RNG原生重开验证通过。adapter174655536B/d00b85a4，state349612141B/9c37cc5d；完整远端locator与SHA见evidence/epoch1-ack-observed.json。此处是原收据经RPC解析的观察，终态再回收精确原件；不冒称大状态已独立下载或自动resume。固定第二轮继续，无epoch择优/held质量。

## 两轮正式完成

13:43:16.763233第二持久ACKa1e0bb45（1763B）完整734updates，全部5862行、1,646,466输入tokens、51,594监督，native3410.521524秒/完整3420.725578秒，原formal child rc0。两完整状态原生重开exact；epoch2 adapter2a1a5a09/state390ed7b6精确locator与全SHA见evidence/formal-complete-observed.json。原controller13:43:18.070760转全1416fresh，child1213791/1200+30；不按epoch质量选择，尚无held评测。服务器连接抖动是只读回执失败，后续原状态证明训练继续，不重启、不当方法负。

## 完整推断与回收

13:56:43全1416/16文献39患者完成，1273HPO/143NO，无NIL/cap/拒绝；804.881s，wrapper806.054s rc0。13:56:44.124964 .3GPU0真实4MiB/无该卡进程释放。原proc92 managed已running=false，但工具未给wrapper退出码，不能把只读command rc当作wrapper rc。服务器唯一终包5,587,542B/SHA965d4825；MCP间歇UNKNOWN/LUNA造成分块未齐，仅恢复同包缺块，不重训练/推断/打包。quality尚未启动，所有输出完整后再原两端点共冻2v2。

## 完整双端点结果：负，关闭

原唯一包43/43块恢复，2910成员全SHA通过；未重任何GPU作业。完整两预测共同freeze b3e3f313后各一次公共v2：强B .728697536469，union .682735038528（−.045962497941），new-only .617921909195（−.110775627274）。新增399=14TP385FP；new-only整体755TP518FP、precision .593087、recall .558019。关系对象与两关系分数全保持，原dev/官方0。

旧2353对应union .674197659658/new-only .613794585011，本次略改善但未达到强B门；输出NO28→143，HPO1388→1273，不能把这点改善称新best或已达到目标。原固定配方关闭，不按held错误改负例比例或阈值。Torch迁移使其不是逐位纯数据消融。`comparison-original2353.json`只读取既有完整metrics，没有第三质量调用。

唯一终态外存terminal.json 11641B/SHA99ca974bacda20a0c34c65bd5a31218a48ba691a5b9adb3f960b7dfbe899991a；完整pred ffd68142433c9d999c7392607765e0a74e2cb355956e131aaec6ceb30f061a5a / 4a2323d4346b391696a1923a5911ef0bcc11708dda4008595ab87bd443bf37aa。root独立NO-veto在14:03:04预锁、早于本次quality，是另一实验；本线源与原两个冻结预测不改。
