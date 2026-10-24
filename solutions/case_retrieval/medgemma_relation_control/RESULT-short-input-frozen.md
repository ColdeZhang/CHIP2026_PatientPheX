# MedGemma 原生诊断结果与正式对照修订

2026-10-24，E-20261024-001。工程检查通过；后继固定正式对照已完整结束，两个预定质量终点均为负。

- 固定 MedGemma1.5-4B，19件8,639,802,098B资产；完整bundle4,300,079,472参数＋语言七投影LoRA29,802,496，合计4,329,881,968。每独立模型≤10B，合规计数不限制全链合计。
- 初次服务器复制因本地监控HTTP503/finally停源失败，已保留；唯一恢复窗口启动前据实测改900秒，实际722.052秒，10个原final继承校验、9个本轮fullSHA，19件精确完整。SHM是临时副本，.100固定原件持久保留。
- .100完整CPU10.453秒；A800新增差异核6.791秒、CUDA未初始化，原124条/382000tokens/最长5380、原80纯输入/587CID保持。
- 唯一GPU诊断实际21.588秒（加载3.752秒），最长5380行更新3.02278秒。原生loss .639429271，手算同logits .639429212，差5.96e-8；梯度有限且attention/MLP B真实更新，基础/视觉/head/embed冻结。原生权重缺/多/错键均0。
- 峰值allocated29,518,475,264B、reserved30,838,620,160B。真实2token forward为5274/past0→1/past5274，native DynamicCache复用；无完整JSON质量主张。
- 原probe进程结束，03:16:08.441Z GPU2已实际核空。诊断权重/优化器未保存，正式训练必须fresh。统一终审1dc8c37b通过，参见review/review.json。

## 训练前固定修订

在任何Med新质量前，root授权固定fresh strict32旧124条×2epoch、238语言投影r16、原生assistant CE、32次optimizer calls/2warmup；共用1200秒＋TERM30，涵盖训练、重载和两次完整epoch持久导出。每份adapter＋Adam/scheduler/全RNG实际.100 ACK和全SHA回读后才继续；OOM/非有限/截止即停止，无自动加预算。保持现trainer和推断数学。

最终epoch2唯一原80 greedy推断另600秒＋TERM30。主endpoint为Med standalone，预先唯一secondary为旧SFT32关联∪Med关联；实体和所有非关联对象原样。两完整预测共同freeze后各唯一原v2，复用旧baseline指标，禁止按新结果再试组合/cut/阈值。secondary只有相对旧SFT32的关系micro＋macro严格增加且macro不降才有益；再迁F0仍由root固定决策，本轮不访问原dev或官方。

该修订已在正式训练前独立冻结，原件见formal-preparation/pretraining-amendment.md。


## 固定正式终点（2026-10-24 03:39 UTC）

| 配置 | Score | 关系micro | 关系macro | TP/FP/FN |
|---|---:|---:|---:|---|
| 旧SFT32缓存基线 | .660942003 | .686600221 | .542228853 | 310/73/210 |
| Med standalone | .603701549 | .550000000 | .449867260 | 253/147/267 |
| 预定旧SFT32∪Med | .644663430 | .637964775 | .525750011 | 326/176/194 |

主终点较旧Score下降.057240454，并集下降.016278572；均不达预定留用条件。并集新增119关系中16TP/103FP，虽增召回但新增错误更多；这是一种固定Med训练/输入配方的负结果，不泛化为任何医学模型或任何全文SFT无效。

原80/587CID全部有效、0未决、0触顶，1189实体及所有非关联原字节不变。双fullpred共同freeze d7a52409在首次gold访问前，各调用一次原v2；旧分复用，无格式补跑/额外组合/阈值选择。范围是已反复观察的F1开发折，不是独立原dev/官方。

完整fresh两轮248曝光/32calls＋两份358,430,727B全状态持久导出/独立回读总657.090秒<1200；epoch loops共316.568秒。第二段native177.663秒仅该invocation，不能当两轮成本。多长度正式reserved峰58.391GB，allocated27.056GB；不以单行probe峰替代正式峰。原80 native226.016秒、campaign257.325秒<600，03:38:06.634Z GPU2实际空释放，所有原job结束，勿重启。

最终adapter9b46f756与完整Adam/scheduler/RNG/.100源码CPU依据保存于research/medgemma-relation-control/formal2ep/checkpoints/epoch-002；epoch1同级保存。raw/双fullpred留仓库外，Git保留source/计划/小证据。唯一summary2c8b202b，参见quality-result.json和quality-F1/。不迁F0/原dev，不提交官方，不按结果追加搜索。终点独立审核另补记录。

## 全文后继仅准备

严格保留CID→PID任务、原答案/候选/患者/消息的全文追加已单独CPU核：124训练965436tokens/ep、最长12824，旧监督7150不变；原80输入645000、最长13805。相对原382000为2.527319倍；51行超过当前训练8192，当前正式源未改。原epoch-loop乘token比仅粗估400.255秒/ep，不是实测、未计导出和注意力非线性。

全文GPU尚未执行。最长更新诊断源码仅显式单独14000容量，原native更新/cache数值段逐字节复用；当前Med双负之后是否值得新GPU由root决定。历史上已有开放HPO/quote全文LLM SFT负结果，不能说所有全文SFT都未测；本次未测的是同CID闭集目标的train+infer全文SFT。
