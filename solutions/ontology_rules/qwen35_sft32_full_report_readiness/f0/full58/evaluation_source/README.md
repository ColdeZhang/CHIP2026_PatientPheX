# F0 原 SFT48 全文完整 58 条：固定评测后继

本目录准备源码；未读 gold、运行模型、调用质量评测或访问远端。F1 原文件不修改。沿用已审 F1 evaluator `ef28ec27…` 的完整终态、原字节保留、两预测共同 freeze 与 H50 导入顺序，单独固定 F0 身份和保留标准。

固定 F0 范围：原 58 prompts（`d387f5b8…`）、16 文献、47 患者、415 CID、1166 CID×PID 对；原 baseline `63aefe68…` 的 947 个实体和所有非 association 字节保持。旧缓存指标 `e16ff5b0…`，Score `0.6954439475814587`，旧评测调用为 0。适配器只认原 SFT48 model `8e32daee…` / config `165dbc59…`。

实际 CPU 已明确：总输入 505090、最长 16245 且唯一原 index 30，cap 向上取 512 后 16384，generation 1600，实际 context 262144。CPU result `ccdcd461…`、row-audit `2fa5c495…`、probe plan `b21fb7f9…` 均钉入源码。完整运行只接受最长 probe 精确缓存 1 条 + 57 fresh，1200+TERM30；完整 58 条及真实 rc0/GPU 释放后才可绑定质量。

两个端点事前固定：

- `diagnostic-fulltext-alone`：消费原 runner 已保存的严格 canonical 映射；缺键/未知 PID/无自然 EOS/到 cap 等整行弃答，不解析 raw 重新找答案、不做填空或旧模型回退。
- `primary-oldSFT48-union-fulltext`：旧 SFT48 与全文端点逐患者字面集合并集。主保留条件明确为 **Score 严格上升且 macro F1 不下降**。诊断端点也报告相对旧的指标/规则判断，但不据结果变成主选择端点。

只复用冻结 helper 的四个通用字面/字节装配函数；没有携带 F1 micro+macro retain 函数，F0 自己定义 Score+macro 比较，并有合成边界检查。概念输出保留 compound HPO/NIL 等原字面值，公共 v2 自己按原规则算评分单元。

## 分阶段接口

`evaluation-template.pending.json` 当前 `execution_ready=false`，五份 actual probe 文件、正式 full58 source/launcher/plan 及正式结果均待绑定。gold 路径/SHA 只摘自已冻结 R1 F0 `evaluation-template.json` 元数据：`fold-0/validation16.gold.jsonl`、`630c9e39…`；准备时不读取 gold。

从 `full58-v1/` 运行以下命令，均由负责人执行。正式 source/plan 与真实 probe 回收齐全后，先冻结一个仍禁用的模板：

```bash
python evaluation-source/freeze_template.py \
  --pending-template evaluation-source/evaluation-template.pending.json \
  --full-plan plan.json \
  --inference-source infer_full58.py \
  --inference-launcher launch.py \
  --probe-root ../probe-v1 \
  --output-template evaluation-source/evaluation-template.json
```

该 builder 只核原 CPU/probe/full-plan 身份、cached bytes 与原 SFT48 runtime，独占新写模板，不执行推断或质量，也不绑定正式响应。只有原完整 58 条真实结束后，再执行：

```bash
python evaluation-source/bind_completed.py \
  --template evaluation-source/evaluation-template.json \
  --terminal terminal \
  --output-plan evaluation-bound.json \
  --output-dir quality
python evaluation-source/evaluate_full58.py --plan evaluation-bound.json
```

terminal 必须具备 `responses.jsonl`、`canonical.jsonl`、summary/complete、wrapper/gpu-release、actual-input-and-assets，以及 `generated-ids/`、`raw-responses/`、`canonical-records/` 三组各 58 个原文件（共 174）。响应按 `(input_tokens,row_id)` 排序，原 index 通过 row_id/CPU 审计对应；不能因排序误认缓存或丢行。

binder 对完整 58 条、来源、真实生成 IDs/EOS/cap、每个原 shard、缓存原字节、固定 budget/runtime 与 GPU 释放做纯 CPU 检查，通过后才新写 enabled plan，不读 gold。评测入口再次核对，先完成两个全量预测、共同 freeze 和 terminal 唯一 marker，随后以 canonical 名注册 `validate_predictions` 再导入固定 evaluator（H50），才允许首次 gold SHA/读取及两个端点各一次 v2 `all/text`。同一 terminal 已有 marker 或输出目录已有时，不重复评分。没有 F0→原 dev 或官方自动迁移。

合成自检约 0.007 秒通过：literal/compound 字段、UTF-8/CRLF 非关系字节、患者/行覆盖、整行弃答、双预测共同 freeze、H50 顺序、F0 Score 严格升/macF1 不降（专门拒绝只有 micro+macro 改善而 Score 不变）。`self-test-receipt.json` 是实际合成结果；原 complete58、模型/质量、独立终审仍由负责人按真实状态记录。

`manifest.json` 列本准备包身份。仓库副本只保留源码、禁用模板、说明和收据；完整真实输出与长证据留仓库外。继承上游限制：生成中 OOM/TERM 不保证有返回部分 IDs；本入口在不完整结果上拒绝评分，不补跑或假称恢复。
