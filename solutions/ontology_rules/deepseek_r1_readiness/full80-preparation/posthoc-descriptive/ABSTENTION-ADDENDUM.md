# 已保存弃答类型补充

在主描述完成后，root 额外授权只读已冻结响应 gzip 和原 result 状态。逐条 canonical SHA 与 result.rows 相符；没有重新 parse 最终文本、生成、修复、评分或读取 gold。

**10 条整行弃答 = 8 条 `unknown_patient_string` + 2 条 `truncated_or_no_native_EOS`。** 这里的真实保存字段名是 `normalized_invalid_reason`。前 8 条均已正常 EOS、未 cap；后 2 条为原 index 48 / 50，均 4096 tokens、无 EOS。2 条 cap 完全包含于 10 条无效行，不是额外 2 条；80 行中没有 wall-budget 结束。

旧 strict / sparse parser 在同一批 8 条未知患者行中另存 5 条 JSON 起始错误、3 条 `invalid_patient_list`；它们是相同行的其他 parser 诊断，不能再累加，也不替代 normalized 终点的错误分类。全部 10 行按原冻结政策整体弃答，覆盖 80 个 CID 槽；有效行另有 29 个缺失 CID 弃答，无 fallback 或补负答案。本描述未读最终文本推断具体患者字符串的含义。

`abstention-types.json` 保留 10 条实际字段、SHA 引用、分类依据和 cap 交集。它仅补全 `description.json` / `README.md` 原先限于 summary 范围时明确保留的错误类型不确定性；并集新增 21 TP / 140 FP 的描述和负结论不变。
