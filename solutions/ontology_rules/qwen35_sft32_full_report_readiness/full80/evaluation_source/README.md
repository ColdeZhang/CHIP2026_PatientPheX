# SFT32 追加全文：完整 80 条后的固定评测入口

这是准备源码，不是推断结果或质量结论。`evaluation-template.json` 固定 `execution_ready=false`，全部实际结果引用为 `null`。仅本体负责人在唯一正式运行达到完整终态后绑定与执行；本委托未加载模型、操作 GPU/远端、读取 gold 或运行质量评分。

目标是区分「仅追加全部允许文章是否改善原 SFT32 关系归属」与工程错误。保持原 80 条 F1 输入身份、587 CID、39 PID、原基座/适配器及 greedy；输入上限 14400、生成上限 1600。完整运行固定原 index 19 精确缓存一条 + 79 fresh、1200 秒 + TERM 30 秒。实际响应按输入长度排序，评测按 row_id 恢复原身份，不依赖错误的原索引输出顺序。

预定两个完整端点：

- `diagnostic-fulltext-alone`：只消费推断器已保存的严格 canonical 映射。非法/缺键/无自然 EOS/截断整行弃答，没有局部补答案或旧模型回填。
- `primary-oldSFT32-union-fulltext`：旧 SFT32 与上述关系的逐患者字面集合并集。只用此主端点对缓存旧指标作保留判断：micro F1 + macro F1 严格增加且 macro F1 不下降。

两个预测都以旧 `baseline.pred.jsonl`（SHA `ffb2c54a…`）为模板，保留其 1189 个实体、全部患者以及每行顶层 `association` 值以外的原始 UTF-8 字节。旧候选池不参与重新筛选。概念字符串保留字面值，包括 `NIL` 和 compound HPO；固定公共 evaluator v2 按准确的 `HP:[0-9]{7}(;HP:[0-9]{7})+` 规则计算评分单元。

## 完整终点接口

需要同一个 `terminal/` 中的 `responses.jsonl`、`canonical.jsonl`、`summary.json`、`complete.json`、`wrapper-complete.json`、`gpu-release.json`、`actual-input-and-assets.json`，以及三组原证据 `generated-ids/000.json` … `079.json`、`raw-responses/`、`canonical-records/`。必须是原正式运行产生的完整结果，不补跑、不重解码、不重解析响应。

在完整终点实际回收并核验后，由负责人运行：

```bash
python evaluation-source/bind_completed.py \
  --template evaluation-source/evaluation-template.json \
  --terminal terminal \
  --output-plan evaluation-bound.json \
  --output-dir quality
python evaluation-source/evaluate_full80.py --plan evaluation-bound.json
```

命令以 `full80-v1/` 为工作目录；所有路径参数也可使用绝对路径。binder 的行为只有读原证据、检查 SHA/上下文/预算/完整 80 条/模型清单/真实 EOS/缓存身份/rc0/GPU 释放，然后独占新写 enabled plan；不读 gold、不评分，也不覆盖准备模板。

评测入口再次验证实际终点，保存两个完整预测和共同 freeze，再写全 terminal 唯一 `evaluation-started-full-report-v1.json` 标记。之后先以 canonical 模块名注册已核 SHA 的 `validate_predictions`，再加载公共 `evaluate.py`（H50）；首次 gold SHA/读取在共同 freeze 后。每个新端点仅调用一次固定 v2 `all/text`，旧指标直接复用，旧评测调用为零。该标记或输出目录已存在时不得重新启动评分。没有自动 F0、原 dev 或官方迁移。

## 源码与检查

`literal_helpers.py` 的五个通用函数逐字复制自已冻结 R1 evaluator SHA `4fbc58db78de352a4a8c473b7df384d4ce1a0324b15a2467e4c2dfe51a9ee522`，只复用字节保留、字面并集与 retain 函数。没有带入 R1 parser、completion gate、模型加载或 F0 控制；沿用的局部变量 `med` 无模型含义。

`self_test.py` 只用合成数据检查 Unicode/空白/CRLF 字节保留、compound/NIL 字面值、未知与重复 PID、重复 CID、缺失患者/重复或缺失行、整行弃答、共同 freeze、disabled plan 和 H50/首次 gold 的顺序。首轮 fixture 缺少 synthetic baseline ref，修正 fixture 后实际自检通过；未触及真实数据评测。结果见 `self-test-receipt.json`，约 0.01 秒、预算 60 秒。准备模板只核固定非 gold 资源身份，实际输入/质量仍须正式终点检查。

`manifest.json` 保存本包文件身份；同字节副本位于本体方案 `qwen35_sft32_full_report_readiness/full80/evaluation_source/`。源码与验证收据可入 Git，真实 80 条输出、gold、HPO 和模型留仓库外。生成中的 OOM/TERM 可能没有返回部分 IDs，这一上游限制不因本入口而消失；缺完整结果时 binder 拒绝，不评分部分子集。
