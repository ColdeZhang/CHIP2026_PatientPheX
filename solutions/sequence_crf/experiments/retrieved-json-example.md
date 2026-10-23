# 共享9B关系模型：一个完整检索JSON示例

该控制由sequence维护，只引用检索分支冻结的SFT32权重和runner，不修改检索入口。此前9B同80提示实际只有system/user，未跑过同候选train-only单例ICL；旧8B抽取及公开base全文追加不是等价控制。

严格train32(F2+F3)原124条完整训练messages/target。只按example user可逆重编号后的实际token≤2048取短例池，保留22条/19篇；不按target筛选，7条全空target中6条仍合格，80个query最终13次检索到全空target。TFIDF仅用名称、quote、上下文与anchor文本，word1–2gram/English stopwords/L2，IDF只拟合合格训练输入；相似度并列按row_id。输入超14000时沿同排名取下一个完整可容纳示例，无截断；本次80条全部top1，0fallback。

消息为原system、example user、example assistant、原query user；所有query元数据/完整文本保持。示例C/患者ID用EX命名空间，仅替换结构化code、真实PATIENT标记和anchor行首，逆映射后原example user/target逐字节相同；不改普通词、HPO、P1/P10子串。示例ID与当前query全不相交。原parser只处理query代码，缺失或foreign患者仍原baseline fallback；该隔离防止编号静默碰撞，不保证不会发生语义复制。

实际固定tokenizer/native Jinja核124旧prefix SHA/token及80旧实际input token全相同。新输入389316（原250696），median4872/p956420/max7581，预留1600输出后最大9181。CPU3.704秒，无GPU。旧A800同80推断282.502秒；按输入+旧输出的全wall线性代理435.868秒只是粗估，不能叫实测吞吐或保证5分钟。

root已授权唯一600s+TERM30推断：原SFT32 epoch2 adapter `4f4815b6…` / config `96564441…`，原runner `33a5a497…` / parser `817fdc44…`；A100 GPU2等待检索probe明确释放后fresh核空。基座只读复用，不再复制19GB；仅两文件allowlist临时服务传173MB adapter+config，完整SHA回读后关闭。A100 Torch2.6与历史A800 Torch2.7.1区别保留，不能将新旧运行称数值环境相同。

新80原生token/渲染SHA在实际TF5.20.2/tokenizers0.23.1再核，模型完整15文件由旧runner实核。完整响应→原解析/候选/实体→fullpred先freeze，再唯一F1 v2；基线已评Score .6609420026531936复用。保留准则为micro+macro之和严格升、且macro不降，无例数/提示/阈值扫描、不原dev。当前只有输入准备，不宣称推断或质量已完成。

准备与真实source恢复见 [CPU证据](../evidence/round2/retrieved-json-example-preparation.json)、[正式计划](../evidence/round2/retrieved-json-example-formal-plan.json)。外部 `research/retrieved-json-example/` 保存唯一query、完整选择记录、实际token报告；`formal/` 保存运行/回收/评分入口。源码在本方案 `retrieved_json_example/`，包括逐字节冻结的必要legacy依赖。
