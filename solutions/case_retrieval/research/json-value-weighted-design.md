# 原 JSON 患者列表值加权：CPU 准备

这不是二元新目标或后置融合：保留严格 train32 的旧 124 条完整输入、完整 JSON、候选、seed/r16/2epoch32calls，只把真实患者列表值所覆盖 token 的 CE 权重固定为 2。每行先 `sum(w*CE)/sum(w)`，再沿原实际组大小取行均值；不做类别重平衡，不删空列表或 EOS，不改变推断。未来唯一比较为旧原80 F1与缓存 JSON SFT32 .660942003，当前没有训练或质量操作。

CPU 实际 2.810 秒：402594 tokens/max5523、7170 监督 token 与旧完整编码一致；880 个列表值中396为空，3220 个 value-overlap token 占44.91%，按合并token计数得到的权重系数比例为61.98%（非实际NLL贡献，也非各行归一后总体loss系数）。空列表覆盖734 token。**这只是优化权重，不是质量证据。** 1702 个不可拆 BPE token 同时覆盖值和邻接标点，外侧只有 `":`、`,"`、`}`，没有 CID/患者字母数字；按完整串 offset 的非空重叠赋2，不能独立编码子串或假装标点没有连带加权。其他 assistant suffix/EOS 赋1，prompt/padding 为0。

拟用原生 `logits_to_keep` 选每个监督位置 j 的前驱 j−1，保持全部词表列；原生 labels 不再传入，显式计算相同 causal CE。124行只需7170个位置的词表 logits，约为全序列位置的1.78%；backbone/attention/LoRA梯度仍保留，不能据此断言整体加速。源码 `operations/json_value_weighted/loss.py` 尚未做真实模型数值验证。CPU 合成位移/加权梯度检查不能代替该验证。

拟在已释放 A100 GPU2 做一次180秒+TERM15的 fresh、丢弃状态 probe：真实最短591token/9监督位比较 native full CE、w=1选位CE、w=2手算目标与梯度，并验证masked padding；真实最长5523/106监督位做加权 forward/backward/非零lr step，记录三family LoRA梯度/B更新、lm_head冻结、时间及allocated/reserved。probe runner须先固定容差和实际来源，当前尚未启动。原32同配方训练实际1149.460秒只作成本参照；正式预算需新的实测且等待当前212结果。没有原dev/F0访问或新参数搜索。

完整 offset/权重逐行产物保存在外存 `sft-meta1-fusion/json-value-weighted-cpu-preparation/`；小证据见 [CPU索引](../evidence/round2/json-value-weighted-CPU-preparation.json)。当前212训练与F1单臂协议不改。
