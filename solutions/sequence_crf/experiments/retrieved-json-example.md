# 共享9B关系模型：一个完整检索JSON示例

该控制由sequence维护，只引用检索分支冻结的SFT32权重和runner，不修改检索入口。此前9B同80提示实际只有system/user，未跑过同候选train-only单例ICL；旧8B抽取及公开base全文追加不是等价控制。

严格train32(F2+F3)原124条完整训练messages/target。只按example user可逆重编号后的实际token≤2048取短例池，保留22条/19篇；不按target筛选，7条全空target中6条仍合格，80个query最终13次检索到全空target。TFIDF仅用名称、quote、上下文与anchor文本，word1–2gram/English stopwords/L2，IDF只拟合合格训练输入；相似度并列按row_id。输入超14000时沿同排名取下一个完整可容纳示例，无截断；本次80条全部top1，0fallback。

消息为原system、example user、example assistant、原query user；所有query元数据/完整文本保持。示例C/患者ID用EX命名空间，仅替换结构化code、真实PATIENT标记和anchor行首，逆映射后原example user/target逐字节相同；不改普通词、HPO、P1/P10子串。示例ID与当前query全不相交。原parser只处理query代码，缺失或foreign患者仍原baseline fallback；该隔离防止编号静默碰撞，不保证不会发生语义复制。

实际固定tokenizer/native Jinja核124旧prefix SHA/token及80旧实际input token全相同。新输入389316（原250696），median4872/p956420/max7581，预留1600输出后最大9181。CPU3.704秒，无GPU。旧A800同80推断282.502秒；按输入+旧输出的全wall线性代理435.868秒只是粗估，不能叫实测吞吐或保证5分钟。

root已授权唯一600s+TERM30推断：原SFT32 epoch2 adapter `4f4815b6…` / config `96564441…`，原runner `33a5a497…` / parser `817fdc44…`；A100 GPU2等待检索probe明确释放后fresh核空。基座只读复用，不再复制19GB；仅两文件allowlist临时服务传173MB adapter+config，完整SHA回读后关闭。A100 Torch2.6与历史A800 Torch2.7.1区别保留，不能将新旧运行称数值环境相同。

新80原生token/渲染SHA在实际TF5.20.2/tokenizers0.23.1再核，模型完整15文件由旧runner实核。完整响应→原解析/候选/实体→fullpred先freeze，再唯一F1 v2；基线已评Score .6609420026531936复用。保留准则为micro+macro之和严格升、且macro不降，无例数/提示/阈值扫描、不原dev。当前只有输入准备，不宣称推断或质量已完成。

准备与真实source恢复见 [CPU证据](../evidence/round2/retrieved-json-example-preparation.json)、[正式计划](../evidence/round2/retrieved-json-example-formal-plan.json)。外部 `research/retrieved-json-example/` 保存唯一query、完整选择记录、实际token报告；`formal/` 保存运行/回收/评分入口。源码在本方案 `retrieved_json_example/`，包括逐字节冻结的必要legacy依赖。

Adapter-only transfer recovery: the first transport attempt stopped with `RemoteDisconnected`; the source default interpreter is Python 3.8 and the handler used the Python 3.9 `str.removeprefix` API. The source was stopped and the empty destination preserved. The transfer helper now uses a prefix check plus slicing, accepts only that empty own destination, and writes a valid JSON receipt. Both old source/plan/process records remain in external `formal/attempt2-python38-transfer/`; no GPU or inference had started. Model, prompts, runner and budget are unchanged.

实际两文件传输18.432秒成功，目标逐文件完整SHA回读通过，临时服务stop后端口连接返回111。部署全部source/input SHA完成；最终ready入口为 `formal/readiness.json`。根在传输修复期间重新排短任务，当前明确暂停GPU启动，等待交回；此时没有推断/质量结果。

根随后明确调度到已释放A800 GPU2（原80基线同卡），原Qwen SHM和旧SFT32 adapter原位复用。新的唯一执行目录 `research/retrieved-json-example/formal-a800/`，A100版无GPU启动且仍归档。仅机器/路径、资源receipt及磁盘guard改变；实际可用197492736B，约2MB输入源码、无权重保存，启动要求≥64MiB。80输入/方法/600+30预算保持；实际运行版本仍由preflight记录。

唯一终点完成：80响应/587候选全部合法，无fallback与token上限；同A800 runner304.593秒、全wrapper313.157秒，峰20,302,075,392B，00:12:04.547Z fresh核空。预测a83abaa6先00:12:31.570freeze，再00:12:34唯一新v2：Score .660942002653→.645783102066，micro−.044148462、macro−.016487140，净−27TP/+5FP。实体全相同，格式0错警，预设keep失败，关闭这一个固定配方、不继续示例/提示扫描，不迁原dev。第一次本地装配仅默认Python缺sklearn_crfsuite，换已有day1 CRF venv运行同source即成功；失败命令/恢复环境保留且未读gold/未另推断。独立终点审pending，数据/迁移审已PASS54c/e3ff/547e。

新增完整终点独审b1a178ca PASS：80 strict/587CID、旧1189实体全不变、完整pred/v2逐字节复现，负结案口径不变。
