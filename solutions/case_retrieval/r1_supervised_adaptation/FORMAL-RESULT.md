# 原124 × 2正式训练已完成，尚无新R1质量

2026-10-24 06:31:12 UTC，fresh原基座两轮完成32次optimizer调用、248行、802582输入token／14092监督token。持久result记718.379秒，末尾stdout718.427秒，外层721.077秒，均在1800+TERM30内；rc0／无timeout。各步finite、首次非零更新与base冻结检查通过，末态504 LoRA梯度与504 Adam状态，峰allocated21,061,661,184 B／reserved24,893,194,240 B。没有截断、量化或更换recipe。

两轮分别保存完整adapter、Adam／scheduler及Python／NumPy／Torch／CUDA RNG；每轮重新打开504 adapter及1512 optimizer tensor逐位比较、RNG exact。最终checkpoint `2fbd4517…`、adapter `20f15b26…`、training-state `f49f32af…`、config `902e86fb…`。57小件已从服务器逐大小/SHA回收，4大张量文件留`.100`持久盘；精确定位见[结果指针](formal-result-pointer.json)。

训练结束后，独立CPU脚本再次核全部文件SHA/fsync，并补两ACK、actual及到project-root的父目录链fsync，receipt `b6723d59…`。06:36:12 UTC实际GPU1 UUID `fc5d0525…` 为0 MiB且无compute进程，已交sequence先运行原GLiNER48 B，再供旧Qwen normal B使用。

原wrapper terminal的`probe_state_discarded:true`为探针launcher继承字段，不能解释成正式checkpoint删除。保持原冻结terminal bytes，独立解释sidecar列出实际两份完整持久状态，不重写历史。

这只证明工程训练完成。先前原probe失败／native重复自身不稳定／同冻结logits CE exact的证据与限制保留。新R1的index73探针未绑定、未运行，按root最新排期等待唯一901官方接收；本次官方文件仅使用旧冻结模型。本次完整增量终审82b68957通过，范围仅工程训练与持久状态，不代表质量。

在尚未任何新R1质量时，root按用户唯一Score目标追加[事前判据变更](quality-policy-addendum.json)：保留两个arm／同强基线与两折复验，候选保留只要求Score严格升；macro仍必报，不再额外否决。原7699合同保持原字节，不追溯重开旧负实验；模型／输入／生成／预算不变，官方接收前后继仍暂停。
