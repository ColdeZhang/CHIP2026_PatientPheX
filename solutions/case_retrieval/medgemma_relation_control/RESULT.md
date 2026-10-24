# MedGemma 原生诊断结果与正式对照修订

2026-10-24，E-20261024-001。工程检查通过，尚无模型质量分数。

- 固定 MedGemma1.5-4B，19件8,639,802,098B资产；完整bundle4,300,079,472参数＋语言七投影LoRA29,802,496，合计4,329,881,968。每独立模型≤10B，合规计数不限制全链合计。
- 初次服务器复制因本地监控HTTP503/finally停源失败，已保留；唯一恢复窗口启动前据实测改900秒，实际722.052秒，10个原final继承校验、9个本轮fullSHA，19件精确完整。SHM是临时副本，.100固定原件持久保留。
- .100完整CPU10.453秒；A800新增差异核6.791秒、CUDA未初始化，原124条/382000tokens/最长5380、原80纯输入/587CID保持。
- 唯一GPU诊断实际21.588秒（加载3.752秒），最长5380行更新3.02278秒。原生loss .639429271，手算同logits .639429212，差5.96e-8；梯度有限且attention/MLP B真实更新，基础/视觉/head/embed冻结。原生权重缺/多/错键均0。
- 峰值allocated29,518,475,264B、reserved30,838,620,160B。真实2token forward为5274/past0→1/past5274，native DynamicCache复用；无完整JSON质量主张。
- 原probe进程结束，03:16:08.441Z GPU2已实际核空。诊断权重/优化器未保存，正式训练必须fresh。统一终审1dc8c37b通过，参见review/review.json。

## 训练前固定修订

在任何Med新质量前，root授权固定fresh strict32旧124条×2epoch、238语言投影r16、原生assistant CE、32次optimizer calls/2warmup；共用1200秒＋TERM30，涵盖训练、重载和两次完整epoch持久导出。每份adapter＋Adam/scheduler/全RNG实际.100 ACK和全SHA回读后才继续；OOM/非有限/截止即停止，无自动加预算。保持现trainer和推断数学。

最终epoch2唯一原80 greedy推断另600秒＋TERM30。主endpoint为Med standalone，预先唯一secondary为旧SFT32关联∪Med关联；实体和所有非关联对象原样。两完整预测共同freeze后各唯一原v2，复用旧baseline指标，禁止按新结果再试组合/cut/阈值。secondary只有相对旧SFT32的关系micro＋macro严格增加且macro不降才有益；再迁F0仍由root固定决策，本轮不访问原dev或官方。

当前尚未启动正式训练；正式源/plan增量准备中。
