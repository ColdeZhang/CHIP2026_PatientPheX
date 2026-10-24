# 仅冻结901最高方法的官方B应用

用户已授权只提交当前最高`9016863c…`（原dev Score .7543613981990362）一次，由root独占网站登录/上传/接收确认；不查询或轮询官方成绩。此目录实现B应用闭环，不将正在训练的新R1或其他未出结果方法混入。公共输入合同为`R/official-highest-901-B-20261024/input-contract.json`（2c169066…），100篇/244患者/3802段，纯4字段且与train80文献隔离。

Task1使用现803b LightGBM+原103列vec、160轮、.3/proper-nested；ontology从共同train64规则/Sap/CRF底座追加原PubMed/Bioformer/BioBERT三公开来源。`apply_task1.py`只加载/transform/predict，无fit或标签/评分。特征与literal helper AST不变，参数化文献/候选数、B未知输出SHA及空候选自然行为；新入口先对原16缓存做唯一byteexact0e2c工程回归，不读取gold或重评分。B池完成并绑定SHA后才能实际B推断，不能把day1 train80 learned缓存混入。

Task2旧full64 Qwen1a703b正序/逆序提示来自独立CRF/Sap/LR64+GLiNER48 full-lr池；公共新概念只取原PubMed/Bioformer>.95 union，排除全部oldpool概念。旧normal/reverse的baseline association失败回退保留，不能只交实体；raw ID/EOS需先持久再原codec解析。具体入口由relation_preparation准备，不改变原greedy14000/1600或候选策略。

901的部署表达为newGate∪oldSFT∪reverse∪[patient−a179_oldGate]。只有严格源及真实B每doc canonical/alt处理后的patient合法概念空间与public-new空间互斥，才可数学消去old gate差集并全量unionpatient；否则必须复建旧HGB97 gate的counterfactual关系。不能用原dev特定6条作B规则，也不能擅自删除或恢复B患者结果。

资源由root调度：R1原formal完成并持久后暂缓其index73新模型探针，.100GPU1让旧normalB；A800安排reverse/public-new；sequence负责patient f787与GLiNER48和共同oldpool。所有B模型、候选、预测、原始响应与manifest留外部`R/official-b-901`，Git只源/合同/摘要。
