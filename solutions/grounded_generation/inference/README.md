# 固定完整F1推断与复制词面

唯一fresh39条、完整16文献/39患者；原生CPU全部IDs已逐对象等同发布tokenizer，`bind.py`只从该实际产物提取held IDs，不重编码。原native后端、BF16/SDPA、greedy1600、无输入截断。必须在正式两完整epoch及每轮model/optimizer/scheduler/RNG持久ACK已确认后才绑定实际epoch2；不得从probe续训或借旧模型输出。

`run.py`先保存生成IDs/end状态，再用发布原生decoder与实际HF decoder核对并保存raw，之后才允许本地`extract_quotes.py`解释。cap/无EOS整行弃答，合法JSON中无效/不可定位项逐项弃答；其余exact `(unit,quote,occurrence)`沿原生成顺序。quote是概念词面，不是人工患者归属证据。无gold、归一化或评分在GPU作业内执行。

`infer_driver.py`固定独立1200+TERM30，新鲜卡/磁盘门，绝对截止包含child启动开销。`remote.py`沿已验分片SHA、同服务器精确资产复用、唯一intent与UNKNOWN只恢复；`collect.py`完整原件归档/原字节续读，无重生成。

独立规范化政策`normalization/policy_v1.json`SHA b59eb3f8已在任何held输出前冻结，严格同train32/public优先，只有eligible unmatched才走冻结公共Sap银行。训练词面624查询与held未知词面的必要新查询分开记录，不把训练缓存声称完整held覆盖。不得按held标签改阈值或字典。

质量固定两端点见`policy.json`与`../quality/`：strongN∪new、new-only，保留strongN全部Task1与患者/非关系字节。两个完整预测共同冻结后各唯一v2/旧0；可选其中最高且严格超过strongN .676546354662594，tie主臂。主假设成功与单臂结果分开、macro完整披露。原dev/官方没有自动扩展授权。
