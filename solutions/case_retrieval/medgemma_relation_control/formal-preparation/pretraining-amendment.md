# MedGemma 正式对照：训练前固定修订


在任何Med新质量前，root授权固定fresh strict32旧124条×2epoch、238语言投影r16、原生assistant CE、32次optimizer calls/2warmup；共用1200秒＋TERM30，涵盖训练、重载和两次完整epoch持久导出。每份adapter＋Adam/scheduler/全RNG实际.100 ACK和全SHA回读后才继续；OOM/非有限/截止即停止，无自动加预算。保持现trainer和推断数学。

最终epoch2唯一原80 greedy推断另600秒＋TERM30。主endpoint为Med standalone，预先唯一secondary为旧SFT32关联∪Med关联；实体和所有非关联对象原样。两完整预测共同freeze后各唯一原v2，复用旧baseline指标，禁止按新结果再试组合/cut/阈值。secondary只有相对旧SFT32的关系micro＋macro严格增加且macro不降才有益；再迁F0仍由root固定决策，本轮不访问原dev或官方。

本修订在任何正式训练或新质量前冻结；后继结果另记，不覆盖该原件。
