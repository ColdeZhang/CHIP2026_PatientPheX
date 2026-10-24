# F0 全文 FLA 固定验证

目标：在第二个训练内开发折确认 F1 的独立 FLA 推断增益。使用原 SFT48（8e32），全部训练文献排除 F0；不新训练。旧 native 全文 58 已完成，Score .6966039747439822，直接复用预测和指标。

新唯一运行：.100 A30 GPU1，58 条全 fresh、505090 输入 tokens，原 BF16／SDPA／native conv＋已核 FLA 两个 GDN 算子、greedy1600。最长 idx30／16245 首先执行；raw、格式及显存门持久后通过才继续57条。cap16384 无截断；总900＋TERM30，软870，从本次 start 计时。旧 native 使用 A800，存在硬件混杂，不能解释成纯 backend 因果。

主端点 FLA-alone，诊断 N∪FLA；两个完整16篇预测共冻后各一次v2，旧指标不重评。按 Score 严格升择优，macro完整报告；另列同947实体骨架强 triple .7095567430848545，低于强组合不称全局best。无原dev/官方/FLA训练 B/C 授权。F1和旧数值门记录保持原样。

运行源、固定计划先提交推送。完整raw/预测在 R/research/fla-fulltext-F0-20261024；GPU source 只读纯输入，不读gold。
