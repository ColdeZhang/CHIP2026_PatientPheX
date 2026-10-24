# 工作台账

分支 explore/quote-set-completion，从 main 08a12023 创建，仅维护 solutions/quote_set_completion。当前仅成熟方法查重与 train-only CPU 供给/原生token准备，无GPU、fit、held输入或质量。

目标：检验给定已报告表型时的条件补全，是否比独立重新生成更能回收遗漏患者关系。当前原dev最佳 .769235669942；本线不读其gold、不宣称预计达标。先复用原train32的87患者弱quote目标，构造两个互补known/remaining视图；原真实关系标签不变，词面代表弱锚点而非人工患者证据。原双采样F1正/F0负已在旧分支关闭，不能重复其采样共识。

公开来源已实际HTTP200缓存Self-Refine、Levenshtein Transformer、BART及内在自纠错的反证论文；只借成熟条件去噪/插入目标，不宣称复现其模型或公开收益可外推本赛。CPU计划plan.json先固定，完整供给/长度核后再给具体成本与单一验证方案。训练与GPU尚未获授权，不占A800。

CPU唯一准备已完成 3.613秒：87患者→174互补视图，每个原801quote恰监督一次，25空known/25空remaining；全部920222tokens、最长9307、最大监督307。输出仅仓库外，manifest5210e0f3。见evidence/readiness.json与proposal.json。该结果证明可构建，不证明补全有效；完整HF runtime重放/新warm-start训练接口/真实压力与held输入仍未做，无GPU授权。元数据汇总首次只有Python保留字SyntaxError、解析前0写入，改字段名后复用原CPU终点，未重CPU/tokenizer。

## v2 完整停止视图与原生合同

Root在首次GPU前授权固定第三视图：每原87患者加入full-known→[]，261行单轮/freshAdam、warm G32 d52、LR1e-5、33updates/warmup2/尾5。v1的174提案原件保留但不执行。v2首次loader误插入列表assert、before-tokenizer TypeError，修复另源后唯一localCPU5.105秒，261行1380333tokens/max9307/监督12406/max307；112空target中87为新增停止视图，不隐藏目标分布变化。

唯一远端CPU15:07:32→15:07:51，19.062秒rc0；15原件SHA回收c2ff0d45，261全部HF/native IDs/labels/rendering/EOS、known/target集合隔离及33-update toy完整状态接口通过。未载模型或质量。见evidence/native-v2.json。

下一唯一240+30工程：warm-d52/freshAdam，固定8次反向非零更新→3次累积→最长9307/187与最多监督5773/307，13backward，保原indexedCE/native/BF16；完整adapter/Adam/scheduler/所有RNG临时SHM序列化并重开校验后弃掉。正式训练草稿尚未运行；不能沿probe权重继续。源码/plan先推后fresh门启动。

原13步工程路径修正后15:13:27→15:15:28按原绝对截止TERM，120.847秒rc−15；已真实group8 Adam/三族参数变，未完成全部极端shape/保存，不能记PASS。首次beforemodel5.225秒失败及恢复partial全部保留；旧进程/卡已实际释放。70秒剩余门过于乐观，已纠正成本依据，不再复用。Root另授权独立180+30最小工程：仅冻结两极形状各FB合并真实Adam，并全state保存/重开验证，fresh父d52不沿partial；不重复13步或261CPU。
