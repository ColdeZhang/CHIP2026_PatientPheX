# 方法笔记

H59查重依据main机制地图与实际grounded训练、推断源码：已有loss/时长控制和输出集合投票，但未找到两完整checkpoint的有效LoRA增量平均实际执行。缺少关键词不是普遍未试证明；本次明确限定原Gtrain32轨迹epoch11/22、相同base及固定输出保护。

Model soups（arXiv2203.05482）真实标题为 *Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time*，平均不同超参fine-tune权重。SWA（1803.05407）为 *Averaging Weights Leads to Wider Optima and Better Generalization*，对SGD轨迹多个点平均，通常配合周期或常数LR。我们使用原AdamW/decay轨迹仅两epoch，不声称复现任一论文的完整配方，也不由论文推断本任务必涨。

精确等式：原各层Δi=2BiAi；等权Δmean=B1A1+B2A2。令A′=concat(A1,A2,axis0)，B′=concat(B1,B2,axis1)，新r32/alpha32 scaling1即得目标。直接平均两个A与两个B会生成交叉项，不能替代有效delta平均。eval模式关闭dropout；实数等式不代表BF16/GPU逐位相同。

已有PEFT0.21.2固定tag eaa35c77缓存源码的`add_weighted_adapter(..., weights=[.5,.5], combination_type="cat")`采用同一拼接公式，并令新alpha=新rank。只复用成熟数学与schema，不自动调用其需要加载基座的API。准备阶段只做NumPy数学与源码核查。随后经根授权，在原服务器CPU实际完成PEFT0.21.2的12类tiny cat/save/load合同（无大基座/GPU），加载张量与输出exact；独立凭据见evidence/CPU-synthesis.json，不将tiny合同扩大为完整大模型GPU成功。
