# Mention canonical 工作台账

2026-10-24 UTC。explore/mention-canonical-normalization，base bc939eeb。本配方完整 F1 负结案；不扩展原dev/F0，不上传官方。主线程原dev≥.8目标仍未完成。

- **结果**：强B .728697536469 → 固定全部并集 **.674197659658**（−.054499876811）；新头替换 **.613794585011**。新增478=14TP/464FP。new-only共1388原子=771TP/617FP，precision .555476，recall .569845；不混淆总模型与增量精度。关系micro/macro逐值不变。
- **实际完成**：原2353标注目标全部两轮590更新，4706行/1,328,556输入tokens/48,126监督；native2548.302s/完整2554.713s，两个完整adapter+Adam+scheduler+RNG检查点原生重开exact。只用epoch2，不按质量择优。完整1416推断743.145s/wrapper743.982s，1388HPO/28NO/0NIL/0reject/0cap，1416raw和1416parsed全保留。
- **完整性**：16文献/39患者，两端点共冻c869ccf9后恰好2次公共v2/旧0；强实体原对象全保留、非实体对象exact，JSON重新序列化不声称字节exact。union预测68cbb45b/234728B，new-only b66b2cee/183252B。没有originaldev评测或官方提交。
- **解释边界**：预测侧381个新增是原基线跨度的新ID、97个是新跨度，未用gold再挑选；输入/格式完整，语义供给未胜强对照。仅标注2353行中的87显式NO，未包含proposer未匹配负例。独立2931负例实验已经预先固定并运行，本负结论不改变其数据/预算。
- **来源**：新头train32=F2F3，所有资源排F1；强F1 proposer48含F0，因此整个管线不能称仅train32，后继F0需要真正排F0的跨度池。完整公开预训练成员未知。CPU/source/方案见plan.json、formal-plan.json与实验记录；原2400成本门关闭是工程预算修正，不是质量失败。
- **恢复入口**：外存D=/workspace/patientphex-runs/round2-20261023/research/mention-canonical-normalization-v1，terminal.json SHA fcb6e0bf；formal-recovery-v1-collection.json完整2888件逐SHA，ZIP7b080c97/1602882B。四大state文件保服务器精确locator和owner逐tensor重开收据，未冒称独立下载或自动resume。
- **资源终态**：原proc_d77f2343/PID2737730同控制器完整完成；formal/infer child均rc0，managed原job读回running=false但未返回wrapper rc。13:20:29.893985实释放.100 GPU1 UUIDfc5d0525=0MiB/noapps，已交root，不再占卡。
- **备份/下一步**：79成员小包317915B/5a14e627已服务端逐件核验且完整回下载同SHA（13:23:46，evidence/backup.json），集中独审2d54b6fb已PASS（evidence/review.json）；本线不调阈值或再跑质量。原方法/完整过程见experiments/E-20261024-MENTION-CANONICAL-001.md；精确指标与终态索引见evidence/final-result.json、evidence/terminal.json。

源码正式入口3ac73adc已事前push；训练终态检查点d5691a67已push核验。当前结案提交随后核验，避免为记录自身hash反复提交。

独立后继readiness_top8/已完成全部2353训练目标公共Top8：2246原子cover@1 .818344→@8 .939893，264可竞争纠正行；1284查询23.374s，已释放GPU并备份9b1b63ff。只证train供给，无LLM训练/held质量，不改变本线负结论；原2931独立训练继续。
