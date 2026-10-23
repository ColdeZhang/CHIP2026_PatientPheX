# 公开Bioformer原dev16无标签缓存源码

本目录为helper外部任务的8个原字节小源/配置快照；方案owner只保存Git，不重新推断或查看gold。原dev输入仅c0fc四字段；只复用Bioformer自身train64的相同lower-token tuple输出，明确不复用PubMed模型概率。模型、lower+cased表示、17266原head、5dp后>.95、词典/缩写/currentHPO政策保持。

真实执行由helper使用既有私有runtime与H5：fresh A100 GPU2、GPU300秒+TERM30、CPU后处理120秒，保留first实际128batch gate。CPU137变量恢复证据沿用，不称GPU完整数组独立重放。当前源/计划已冻结，实际终点状态见[source-index](source-index.json)及外部终点；此处不产生质量结论。输入、完整预测、raw、权重和日志不入Git。

`run_dev16.py`为推断，`make_provenance.py`为来源，`launch_cache.py`/`deploy.py`为已限定部署运行，`collect_and_verify.py`为回收/契约检查。它们依赖原远端目录，不能直接在本归档目录重跑或覆盖实验；恢复位置/校验值见索引与plan。MIT许可见[原许可证](../PHENOTAGGER-LICENSE)。
