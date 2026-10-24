# 当前状态

E-20261024-SEQ2SEQ-READY-001：LongT5-TGlobal-base247,587,456，原强G完整87条quote答案语义。分支research/seq2seq-patient-set，起点main e968d5e7；原7751b210 direct-ID保持未选。当前工程执行冻结源9eec22c349be6662bad4428798ca2f7687300fec，push/lsremote一致。

- 原始HF/MS身份、Apache2.0、C4-only无监督预训练；单base990446387B实际完整SHA/落盘重读成功。large静态缺head不一致保待核，未下载large。
- root先授权单base资产及600+30有界probe；通过后另固定10ep/220Adam与1800+30训练预算，仍等GPU资源协调未start。87/39完整供给，train输入max9907、held纯输入max10961，decoder max658；801weakquote/814canonical、5空、4原incomplete患者全部保留，不能声称818全监督。
- 标准JSON反斜杠也UNK，两个目标小于号用已有sentinel32099可逆传输；无新token/参数，87答案原字节与JSON顺序完整恢复。输入原文及其UNK不修。原/传输/IDs各SHA保留。
- CPU单base准备155.727s，CUDA未初始化。实测Python3.10.12/Torch2.7.1+cu126/TF4.50.0/tokenizers.21.4/hub.29.3；隔离环境已只读共享给sequence owner。
- 原probe-v1 17:02:39.307→17:02:45.594 UTC，在GPU前参数guard失败、0FB，freshGPU释放。零missing/mismatch；载入产生三个相同storage但不同Parameter的输入权重，原checkpoint独立247587456。CPU原生set_input_embeddings恢复设计共享，全部state字节与其他Parameter身份不变，LM head独立。原件保留，不称坏模型。
- root准同一工程恢复且不重置原钟。probe-v2 17:04:51.602→17:05:05.631 UTC，原TERM17:12:39.307/kill17:13:09.307未动，driver14.028s、工程13.155s，4FB/2Adam/短解码全部完成，exit0。GPU0 fresh 0MiB/24164free且无compute app；GPU1他人任务不受影响。
- 5571×658 FB1.097s+Adam.0572s，allocated4.918GiB；8192×213 FB.746s+Adam.0267s，allocated6.696GiB；9907×3 FB1.142s、7.319GiB；16384×3 FB1.763s、10.828GiB/reserved17.129GiB。后两例含已分配Adam状态，295梯度tensor全部有限。
- [完整报告](REPORT.md)、[供给](evidence/quote-supply.json)、[工程原件/释放refs](evidence/probe-results.json)。没有正式训练、held调用、quality或checkpoint恢复；工程状态已弃掉。4形状成本不能冒充全训练/解码总成本或质量。

## 唯一正式版本准备

- root原1200提案经完整87 token与四形状成本核后改1800+30：线性样本max-rate×1.5+Adam+120s ACK=1447.734s，最慢样本常数×870+Adam+ACK=1666.287s；均非已证上界。10epoch×87，batch1/accum4尾3均值，220Adam，lr1e-4/wd.01/warmup22/seed20261024，无epoch选择。
- core源码/计划3adf4884已push/lsremote一致。F1唯一`B4∪[T5∩(G_old∪S∪Q∪C)]`，严格Score正才F0；每fold最多1new-v2/old0，byteexact0。无pure-T5评分/阈值/种子扫。原parser/public b59规范化不变。
- generated transport保原nativeSP UNK文本再itemwise校验；87目标解码仍逐字节同training codec。混合坏quote只拒坏item的增量合同通过。未定义extra/padded-vocab ID保raw并拒transport。训练labels原件不动。
- .3已收到原关系GPUrelease，但工具随后LUNA_OFFLINE/UNKNOWN；同四资产私网copy源180s服务到期closed、delivered=[]，无GPU/无新增训练。保原client与未知stderr；停止.3追加copy。按root协调预备回.100缓存资产，不占SCST必要Sap窗口。
- .100正式CPU合同3.624s完整PASS：base247587456/126SP IDs/87目标字节；tiny原生LongT5 fullstate 120张量、29唯一optimizer参数/两state.step保存fsync重开恢复及fresh model load准确、input共享/head独立、0CUDA。原件/evidence见[formal-native-CPU](evidence/formal-native-CPU.json)。这不是全base训练checkpoint的实际ACK。
- 当前唯一正式GPU已在root最终协调后启动：2026-10-24 17:24:16.702 UTC，.100GPU0 UUID126293b9，proc_664e32e1-f0eb-47d4-bfbb-bb348930f780/PID2758035；TERM17:54:16.702、kill17:54:46.702，不延时。fresh门GPU0空闲24164MiB/无进程，GPU1任务不动。仅训练+最终完整stateACK，39greedy1600另锁时钟。实际[start/恢复定位](evidence/formal-start.json)。入口[formal/README](formal/README.md)。
