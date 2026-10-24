# 当前状态

2026-10-24 UTC；branch explore/mention-hard-negatives，base23b053e6。只改solutions/mention_hard_negatives。

- 真实目标：补足生成头拒绝误提取跨度的监督，判断是否带来强Task1增益；不把训练成功当质量改善。
- 约束：训练/候选祖先仅F2F3，公开Qwen3-8B，沿既有三状态prompt/trie/原native训练器，完成共冻前不读heldgold；新增GPU只按root最新固定链授权。旧2353 run proc_d77保持独立进行。
- 固定样本：2353全部对象/顺序保留，按每文ceil(HPO+NIL/4)追加578；2931总行=2222HPO/44NIL/665NO，无不足配额。每个新增跨度重新核与全部正式注释零重叠、原quote/offset一致。
- 两轮固定734updates/367每轮/尾3/warmup37；实际每epoch823233tokens/25797监督；新增max369/3，全数据仍383/28，Pareto与原三形状完全相同。完整F1仍固定1416跨度、强B并集/新头替换两端点；达到任一严格胜.728697536469才候选。
- 源复用：reuse/mention全部原字节，manifest记录精确origin/SHA；只有prepare与CPU新增578行检查是新代码。已有2353编码、1416纯输入及processor/真实FB证据只引用，不重复验证。
- 外存R/research/mention-hard-negatives-v1/prepared-v1/manifest.json；CPU proc_b044e900已12:35:48.589 rc0：实际7.410s、wrapper7.965s，CUDA未初始化；只新578行native/HF/掩码和1156答案prefix，旧编码/实际processor/FB证据复用。30原件ZIP40901579/1169122B全SHA回收。按原实测保守公式新总预算估4521.224秒；这是CPU准备历史估计，后续实际状态如下。

当前CPU已完整ready，源码722dbbcb已push核。root已分配实际空闲.3GPU0：fresh全2931两轮4800+30、两state ACK后全1416推断1200+30、两完整端点共冻→2v2。gpu-plan.json固定全部源/数据/预算。实际12:46:15.919277唯一proc_92c814d2/PID1209971已启动，源4fe44571 push/lsremote一致且freshGPU0/磁盘门通过；完整链4800+30→1200+30运行中，不重启。第一轮已13:14:22.272530完整367updates、ACK31ee14d9/1764B，504adapter/504Adam1512状态与RNG重开exact，第二轮继续；尚未质量。D/formal-latest-observation.json与evidence/formal-start.json为恢复入口。

主机差异明确：.100 Torch2.7.1+cu126→.3 Torch2.7.0+cu128，TF4.51.3/tokenizers.21.2/PEFT.15.2及native/common/loss源码逐字一致。该对照含运行环境迁移，不能称纯数据逐位消融；.3同8B完整大形状6155/113、4992/116和68updates/64fresh已有真实证据，且新样本仍被原mention三形状覆盖，因此不另跑FB探针。CPU编码文件跨机复制逐SHA原字节，.3模型复用同机完整13asset SHA及fresh size/mtime/inode身份，不重复下载或整权重hash。
