# 首日数据与共同评测

数据于 2026-10-23 从已报名账号的官方赛题与数据页重新下载，未采用旧实现、增强词典或历史预测。下载来源及文件 SHA-256 见 [OFFICIAL_DOWNLOADS.json](OFFICIAL_DOWNLOADS.json)。官方 A 包包含训练80篇、A集20篇及本体；B包包含无答案的100篇测试文献。

本体版本为 `hp/releases/2026-06-23`，官方包内文件与 GitHub 指定 release 字节一致，详见 [HPO_SOURCE.json](HPO_SOURCE.json)。当前文件放在仓库外 `/workspace/patientphex-data/`。

[day1-split.json](day1-split.json) 冻结 64/16 文献划分及文件校验值：先按 pmc_id 排序，以 seed=20261023 打乱，前16篇为验证。训练173位患者、验证36位患者。训练折之外的标注不能进入别名、模型、检索记忆或提示示例；最终B版可单独使用全部80篇重拟合。

生成脚本 [prepare_day1_data.py](../../scripts/prepare_day1_data.py) 将验证、A/B输入中的 entities/association 删除。数据重建命令和生成文件名可用脚本 `--help` 与 split manifest 查询。完整数据和运行预测不写 Git。

共同格式校验：[validate_predictions.py](../../scripts/validate_predictions.py)，验证文献与患者覆盖、原文全局坐标、标识/schema和指定HPO分支。共同评分：[evaluate.py](../../scripts/evaluate.py)，输出四项指标、计数、逐文档分数及不确定口径敏感性；默认全部输入段落、无ID文档概念按原文表示。此为公开规则近似，不能称作官方分数。

关键行为由 [test_evaluation.py](../../tests/test_evaluation.py) 检查：漏文档、错ID/坐标、NO、复合ID、无ID、空患者、重复ID和HPO范围等。原来的评测器不参与新实验。
