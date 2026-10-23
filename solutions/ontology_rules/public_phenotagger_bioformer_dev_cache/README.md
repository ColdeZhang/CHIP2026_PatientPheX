# 公开Bioformer原dev16无标签缓存源码

本目录为helper外部任务的原字节小源/配置及本地启动修复快照；方案owner只保存Git，不重新推断或查看gold。原dev输入仅c0fc四字段；只复用Bioformer自身train64的相同lower-token tuple输出，明确不复用PubMed模型概率。模型、lower+cased表示、17266原head、5dp后>.95、词典/缩写/currentHPO政策保持。

真实执行由helper使用既有私有runtime与H5：fresh A100 GPU2、GPU300秒+TERM30、CPU后处理120秒，保留first实际128batch gate。CPU137变量恢复证据沿用，不称GPU完整数组独立重放。实际终点完成：16篇1353个canonical键，4655同模型旧词串的完整top3/native_line/parentindex逐项等值，50548新词串与真实forward文件相等。native70.274秒、CPU来源0.969秒、复用检查7.685秒；21:38:11Z GPU2核空释放。全部26项completion产物实际SHA/大小已由owner核验，[source-index](source-index.json)保留恢复指针；此处不产生质量结论。输入、完整预测、raw、权重和日志不入Git。

`run_dev16.py`为推断，`make_provenance.py`为来源，`launch_cache.py`/`deploy.py`为已限定部署运行，`collect_and_verify.py`为回收/契约检查。它们依赖原远端目录，不能直接在本归档目录重跑或覆盖实验；恢复位置/校验值见索引与plan。MIT许可见[原许可证](../PHENOTAGGER-LICENSE)。

本地启动helper曾在CPU文件已写完但managed进程仍未退出时停在门槛，随后处理MCP UTF8/JSON传输编码差异，始终要求解码后完整文件SHA相等；当时尚未native/GPU启动，模型源/计划未改。原helper快照与transport note一并保留，实际native仅一次。
