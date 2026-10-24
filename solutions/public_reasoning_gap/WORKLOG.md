# 公开思考机制缺口台账

2026-10-24 UTC；分支 `research/public-reasoning-gap`，起点 `main@d648efef`。

- **状态**：有界只读核查完成，关闭“公开权重reasoning＋最终关系解析尚未试”的泛化提案，不生成GPU任务。
- **已核事实**：准确公开R1无本任务adapter，完整F1 80响应实际80个closed think段，106142输出token；独立最终JSON codec，70合法/10弃答。旧alone .605216、union .634403，均负于同框架旧.660942。
- **区别**：公开Qwen9B完整thinking F1未核到；已有12份base/adapter实际manifest均false，1600/4096及人工边界都是任务SFT adapter短诊断，不冒充公开基座质量。
- **边界**：只读旧源/模板/已存响应计数和既有质量摘要，0新model/tokenizer/parser/fit/gold/质量；不重新裁决答案，不改原配置或运行其它路线。
- **入口**：[结论](REPORT.md)、[可复现审计](audit.py)、[准确来源](evidence/audit.json)。没有待执行作业；交root按已试机制登记即可。
