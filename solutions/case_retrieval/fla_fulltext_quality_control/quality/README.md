# FLA 正序全文的固定双终点评测准备

固定 primary = FLA-alone，diagnostic = 已缓存 N_full ∪ FLA。强对照为原 native 正序全文 N：预测 `f33a30f1…`、指标 `875292c3…`、Score `0.676546354662594`。两个新终点任一 Score 严格提升才候选保留；均提升时取较高 Score，完全相同时选 primary。macro 及全部分项完整披露，不另加 macro 门。

本目录只完成源码准备，`template.json` 保持 disabled，实际 endpoint/source refs 尚未绑定。没有执行 binder、模型、响应解析、gold 读取或评测。GPU 所有者是父代理，本目录不启动 GPU。

输入和证据保持原正序全文 80 行：`2a95bad1…`、audit `35f9a00e…`、CPU 结果 `f0297790…`，670122 输入 tokens，最长 14390；模型输入 cap 14848、greedy max-new 1600、旧 SFT32 adapter `4f4815b6…`。新 FLA 必须完整 80 fresh、cache 0，没有旧 probe19 或 native 响应复用。保持 16 篇、39 患者、1189 实体及全部非 association 原始字节。

`inference_gate.py` 核验已完成原始包的终态、所有成员 SHA、已发布计划/源码、逐行身份、原生输入证据、raw-first 标记、EOS/cap、canonical 与整行弃答记录。额外核实际两个 GDN 的 FLA 绑定、FLA/Einops 依赖证据、原生卷积和版本，不把它称为数值等价替换。只使用 GPU 已保存的 canonical 字段，不重新解码或解析响应正文。

`evaluate.py` 只调用原 `ef28ec27…` 中已审的 `canonical_gate/make_primary/assemble_two/delta` 和 `0fa4ef7b…` literal helpers；不调用旧 `load_inputs`、缓存加载、模型解析、`verify_inference` 或旧 main。`assemble_two` 返回 `(alone, union, counts)`，明确映射到 `(primary, diagnostic)`，不倒置身份。其 14400 输入检查只验证本批已证的最大 14390，不修改 GPU 的 14848 cap。

两份完整预测先写入、fsync 并共同冻结，然后才导入 evaluator 和首次读取 gold；各唯一 v2 all/text，旧 N 不重评。额外 exclusive marker 防止换输出目录重复评测。CPU60、绝对 wall65+TERM5，失败保留 stdout/stderr/terminal，不自动重试。

父代理在完整 GPU 包回收后，使用已发布的冻结 GPU plan 绑定动态来源：

```bash
python bind_actual.py --template template.json \
  --inference-dir /absolute/complete-collected \
  --published-plan /absolute/published-gpu-plan.json \
  --source evaluate.py --output /absolute/bound-plan.json
```

绑定器要求回收 plan 与 published plan 字节/SHA 相同，再绑定其已冻结 sources 和实际成员；它不读取 gold。质量源码、bound plan 保存后，由父代理执行：

```bash
python run_quality_once.py --plan /absolute/bound-plan.json \
  --source evaluate.py --output-dir /absolute/unique-quality-execution
```

这是已反复使用的 F1 开发对照；旧 N 在 A800，新 FLA 在 A30，硬件与后端数值差异如实披露。不自动扩展 F0、原 dev 或官方提交。
