# 原 JSON SFT 的患者子集条件分数

目标是检验现有 SFT 的连续决策信号能否补充贪心输出，先验证语义、数值和成本。不是再次把硬输出四事实送入 HGB；已有 SFT 四事实融合虽改善同弱基线，但仍低强对照。此前 monoT5 式独立 yes/no 的 4.145M prefix 预算、患者 CE/MIL、thinking 失败以及未执行的可选 NLL 都保留，当前没有更强的证据支持重新训练或改变 verbalizer。

固定 SFT32（训练 F2+F3）对 F1 的原80响应全部包含完整 CID 映射。按 `sft-json-subset-score-v1` 的 pmc、row_id、CID SHA 顺序，先选12不同文献各1prompt/1CID，再读取旧预测；不按答案、错误、长度筛选。总26个患者候选、98个合法患者子集，包括空集和多患者集。原提示不变，前序 CID 用旧贪心预测；12/12规范化前序都是实际 raw 的精确文本前缀，没有使用gold上下文。

对一个 CID，按输入患者顺序序列化其全部合法子集，续写直到该值后面的逗号或最终 `}`。将完整 prompt+预测历史+候选值整体 tokenize，再找所有分支的共同 token 前缀。设完整 IDs 长 L、共同前缀长 s：输入 `ids[:-1]`，只取最后 `K=L-s` 个 logits，第一个位置 s−1 预测 token s；对 `ids[s:]` 做 float32 log-softmax 并求和。右侧 delimiter 计入，EOS 不计入：这是完整值的部分续写事件分数，不是完整 assistant 回答的似然。

实际 CPU 结果揭示重要边界：**95/98 个分支独立 encode(prefix)+encode(value) 不等于整体 tokenize**，因为 BPE 会跨冒号、括号和分隔符合并。简单按字符串边界切标签可能漏掉首个决策 token。当前全部合法分支的原prompt IDs、完整文本回读、共同前缀、因果位移、右 delimiter 均核验通过；真实 Qwen forward 支持 `logits_to_keep` 和 `use_cache`。纯 CPU 5.50秒，无权重加载/模型前向/CUDA；首版仅在报告写出时把字符串当 Path，原源码/异常已存外部 attempt1，修复后实际报告完整通过。

子集分数用 logsumexp 在有限合法集合内归一，再对包含某患者的子集求和得到条件边际。它**依赖当前提示及旧贪心历史、合法子集限定、唯一患者顺序和 JSON 编码**，不覆盖其他排列、格式或非法输出，不能称真实无条件 posterior 或已校准关系概率。原始总 logprob 保留长列表的乘积偏好；不临时长度归一、温度缩放或用这12例找阈值。旧前序错误依然会影响后续分数。

98分支完整重算249,997 token、最长4161、评分suffix共849 token；若在相同12提示上做全部80CID，会增到530分支/1,725,321 token，所以本次只做预选12CID。为避免Qwen recurrent/conv缓存分支复制问题，选择 no-cache 独立前向；不宣称KV共享等价。既有80条F1生成的prefix250,696/282.5秒只作量级参考，不能当新forward实测。

root已授权一次≤300秒+TERM30的98分支GPU诊断，不训练、不重新生成、不读原dev。最长分支排第一，它的数值/内存检查属于这98次之内。全部实际分数和受限边际先freeze，再只在本地读取既有F1 raw-HPO标签：预先固定 .5 边际决策、Brier、TP/FP/FN/F1及双类存在时的ROC-AUC，旧贪心作为0/1对照；MAP仅报告与贪心是否不同，不作为另一质量选型。12CID/26pair是机制pilot，不是公共Score、泛化或校准可靠性的结论。超时保留partial，不能改样本后报完整。

[计划及来源](../evidence/round2/conditional-subset-likelihood-plan.json)。全部token IDs、分支和CPU原始证据在仓库外 `qwen35-preparation/sft-meta1-fusion/conditional-subset-likelihood-cpu/`。
