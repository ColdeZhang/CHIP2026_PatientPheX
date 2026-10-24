# 当前状态

2026-10-24 14:04 UTC：固定 F1 两完整端点均负，配方关闭；原dev/官网均未调用。独立 branch explore/mention-hard-negatives，实际 GPU 源4fe445717d821c0ba6207597c4f139bff11c8690。原训练、推断完整成功，.3 GPU0 于13:56:44.124964真实释放。原proc92已running=false；managed工具未提供wrapper退出码，formal/infer child各rc0另有原件。

| 完整端点 | Score | menF1 | docF1 | 实体数 |
|---|---:|---:|---:|---:|
| 强B固定基线 | .728697536469 | .774943735934 | .821052631579 | 1313 |
| B并集全部新实体 | .682735038528 | .683197389886 | .728948985864 | 1712 |
| 新头全部实体替换 | .617921909195 | .575019040366 | .577874818049 | 1273 |

两端点关系对象和micF1 .659553831232/macF1 .659239947131均保持。新增399原子仅14TP/385FP；整个new-only是755TP/518FP，precision .593087、recall .558019，两种分母必须区分。相比旧2353模型，union .674198→.682735、new-only .613795→.617922有改善，仍不胜强基线；NO从28增至143，HPO从1388减至1273。不据此扫描比例、阈值或重跑held。

固定监督：保留原2353所有对象和顺序，追加578同train32纯OOF零注释重叠跨度，按每文ceil((HPO+NIL)/4)及prob/sim/offset/length排序；2931行=2222HPO/44NIL/665NO。负标签仅annotation-negative，有漏标风险。F3→F2/F2→F3候选祖先排F0/F1，head全F2F3；F1跨度proposer训练48含F0，整个管线不是pure train32。

实际完整：native新增578 CPU合同7.410s，30原件全SHA；2epochs/734updates/尾3/warmup37，全5862行、1646466tokens、51594监督，native3410.522s/含load3420.726s；两个完整504adapter/504Adam1512状态+scheduler/RNG ACK。全1416推断804.881s/wrapper806.054s，1273HPO/143NO/0NIL/0cap/0拒绝，最终adapter504逐值载入exact。正式无holdout/早停/挑epoch，探针旧形状证据复用，未新跑FB。

唯一终态：evidence/terminal.json（外存R/research/mention-hard-negatives-v1/terminal.json，SHA99ca974b，11641B）。完整回收2910原件ZIP965d4825/5587542B，collection9b84ec0d；MCP断连只补缺块，原始失败收据保留。共同freeze b3e3f313早于exact2v2/旧0；两个完整pred ffd68142、4a2323d4。原件在外存gpu-v1-collected；大adapter/fullstate仅远端精确locator及原owner全张量重开证据，不声称独立下载或实际resume测试。

入口：README.md、experiments/中的本实验、plan.json/gpu-plan.json、prepare.py/cpu_delta.py、reuse-sources.json。独立集中增量审已PASS：review9b6e2c3e/12946B，manifest98c41f89；20source/16reuse、2910成员、两ACK/734、1416原始与严格解析、完整双pred与两共冻评分均核。独立NO-veto同份报告亦负闭；完整入口evidence/final-review.json。77件/331088B小包于14:05:57.936676完成服务器逐成员核验及完整回读，SHA1c961562；独立receipt evidence/terminal-backup.json（42ed52a7），绑定结果commit9e67e472。原UNKNOWN write实际成功，恢复只读确认同SHA存在后0补写/0重打包；唯一terminal99ca974b不变。root另预锁明确NO删除strongB的独立控制也已负闭：.728697536469→.722573801595，删133实体=60TP/73FP，143NO中10在B池外；这是另实验、另1v2，不计入或修改本线原2v2终态。候选Top8新任务在另一分支独立进行。

运行边界：.3 Torch2.7.0+cu128与旧2353的.100 Torch2.7.1+cu126不同，TF4.51.3/PEFT.15.2/tokenizers.21.2及native数学源相同；不能声称逐位纯数据消融。所有非entity对象逐项exact，重新序列化JSON不承诺原始字节exact。
