# 工作台账

2026-10-24：同文同概念患者MLCCE固定控制完整负结案。主 **.675529387933** < 强B **.686338981272**；89新增14TP75FP。独立new-only **.564954220539**，虽比geometry .507641提高.057313，仍低B；按原门关闭，不扫分组/权重/epoch/cut，不迁原dev/官方。

从main20fa611b独立建分支，实际源a7535827、作业指针099e59a8均核推。原geometry模型/参数/RNG/baglogits/NER完整继承，只换患者loss与自然尺度。1174概念组/93多正组，原3194袋813正1masked正不变，无新tokenize或NNprobe。

完整24×32/768、16篇39患者推断+固定规范化264.241秒，13:23:37.510430实际释放.100GPU0。109187763参数，206末权重CPU逐值重开，文件+两指定父目录fsync，非optimizer/RNG full-resume。

主相比geometry−.001265（micro降/macro升），比无几何24+.006344；弱端点改善不能替代强目标。原B471关系/1189实体及非关联对象全保，两预测共冻→唯一2v2/old0。62原包成员313600B/7a0560c4已全SHA回收，质量43内容118714B/ba6b01c8亦远端文件+父目录fsync及全回读；终态manifest636e43ca已单件fsync/readback。集中增量审入口已交付。

恢复入口：仓库外 `/workspace/patientphex-runs/round2-20261023/research/patient-competition-loss-v1/` 的quality-v1/result.json、assembled-v1/joint-freeze.json、collection-receipt.json；无在途GPU/新质量意图。

后继只读可行性已完成：同PCL raw可固定NER>0、严格patient>t(t≥0)纯CPU max重解码；原train32 NN/aliases排F0/F1，无B0/CID依赖。未拟t/新gold/预测/评分/NN；未来F0纯输入CPU与held-only入口尚未实现/授权启动，详见readiness/REPORT.md。
