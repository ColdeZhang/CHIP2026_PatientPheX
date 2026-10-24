# 方法和风险

新增监督变量是OOF候选拒绝标签，不换生成架构。现成纯F3→F2/F2→F3缓存同时排F0/F1，参照HANDOFF H40。原四fold候选不能默认复用。

20756可用池只是标注意义负例，仍有漏标风险；全部边界重叠、正span错概念、NIL/outbranch不进入新增NO。固定1:4是成本受限的保守选择，不称最优。原候选prob来自class-balanced词面LR，并非上下文/概念正确概率；只做文内排序，不输入模型。

完整来源分析R/research/mention-hard-negative-readiness-20261024/REPORT.md SHA7e8fba82；census81d46d98。F1输入proposer仍含训练F0的强48模型；F0后继必须替换排F0强池，不能把head纯train32误称全pipeline纯train32。

运行环境迁移：本线.3 Torch2.7.0cu128，原2353为.100 Torch2.7.1cu126；其余TF/tokenizer/PEFT及核心数值源全等，不能把跨机比较宣称数据唯一数值变量。同机13资产此前已全SHA，复用实际size/mtime/inode守卫；之前成功压力/完整训练只证明工程边界，不证明本线质量。

- 新增拒绝监督使NO28→143并减少部分假阳性，仍未解决强基线之外自由名称供给的低精度：新增399只有14TP。不要把整个new-only精度.593与新增精度.035混为一谈。
- 同一MCP连接断续会让完整结果回收失败；此次原GPU训练推断均rc0且已释放，只恢复冻结ZIP缺块，禁止据下载失败重复训练。
- 旧9B entity-slot二元p<.5删除强B已有负终点（afcdca25）；后继NO-veto应承认该近邻，仅把新8B/OOF拒绝监督视为新供给。

- root独立明确NO删除控制也负：删133=60TP73FP，强B .728698→.722574。既有B计数TP1033/FP280/FN320意味着若只删除，提升mention F1需删除FP/TP比>1633/1033≈1.5808；实际73/60≈1.2167不够。此为既有metrics的代数解释，不是新阈值选择或质量调用。
