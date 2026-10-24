# 原 SFT48 / F0 58 条追加全文：输入准备

目标是在 F1 的同一追加全文机制下检验原 F0 的稳健性。这里完成来源定位、完整输入构造与实际 tokenizer CPU 入口准备；本阶段没有加载 tokenizer/模型/adapter、读取 gold/训练内容、调用评测或访问远端。实际 native CPU、最长工程探针及后继完整 58 条由本体负责人执行。

固定原输入 `/workspace/patientphex-runs/round2-20261023/sequence_crf/qwen-pair-fold0/prompts.jsonl`（SHA `d387f5b8…`），其 58 行覆盖 16 文献、47 患者、415 CID、1166 个 CID×PID 对。全文来自原剥离标签 F0 `validation16.input.jsonl`（`ec38c135…`）。比较对象为原 `adapter58.pred.jsonl`（`63aefe68…`），947 实体、已有 Score `0.6954439475814587`；缓存指标 SHA `e16ff5b0…`，不重新评分旧端点。两预定完整端点为 `fulltext-alone` 诊断和 `oldSFT48∪fulltext` 主端点；主保留条件是 Score 严格上升且 macro F1 不下降。最终预测必须保留旧 baseline 全部患者及所有非关系原字节，候选池不重新筛选。

适配器固定原 `relation-training/formal2ep/epoch-002`：model SHA `8e32daeea3c078af25a7d01e9d9cebf1fe8d99bbfaf2ca5e2c172e643d346513`（173181072 字节）、config SHA `165dbc59c340d68b9756443331cdc1a5f3e81c1834b1237a99d004c33e14cbb9`（1287 字节）。原 checkpoint、实际历史 hash 收据、旧 response manifest 三方相同；原 48 个训练文献 ID 与当前 16 文献无交叉，训练 204 行/2 epoch/52 steps 仅核原元数据，没有打开训练内容。这里没有选 reverse prompts `49ae…`、F1 的 SFT32 或新的 patient-first F0 adapter。

原 A800 易失副本候选 `/dev/shm/patientphex-sequence-sft48-reverse-f0-1045`；.100 持久原件在 `case-retrieval/qwen35-preparation/dev-relation-transfer/adapter-epoch002/`。新鲜远端身份由负责人在本目录 `asset-readiness/` 独立核验，本构造程序不读取远端权重。完整引用与这些定位见 `source-plan.json`。

## 已完成的本地构造

- `input-cpu/construct.py` 与 `legacy_full_context.py` 分别逐字复用 F1 已审 SHA `b201fb7f…` 与 `ae7f5853…`。同一患者 marker、原段落顺序/offset/section/type、同一全文 suffix；不改原 schema、CID/PID、局部证据或其他字段。
- `prepare_inputs.py` 先注册冻结 sibling 模块再按文件加载（H50），检查全部 58 行和来源隔离，再独占写新产物。完整 609 passages、158 given mentions 均保留，插入 158 markers、0 overlap skip；全部 47 患者被表示。逐段去标记文本分解可恢复原文，逐行移除 suffix 可恢复原输入对象及原 user UTF-8；原 JSON 行序列化方式也核为相同。
- 实际纯 CPU 构造约 0.18 秒。`expanded58.jsonl` 为 2021497 字节，SHA `d633225256dbf1cd5b64e94c74d7ce7f220b834cd748158082b306f12191fab2`。逐文献/逐行审计保留在仓库外，摘要与清单可入 Git。

构造入口（已执行，禁止覆盖已有成果）：

```bash
python prepare_inputs.py --source-plan source-plan.json --output-dir .
```

## 待负责人执行的真实 native CPU

将 `input-cpu/plan.json` 的 `files` 六项及 plan 原字节部署到独立 CPU 目录，从该目录用已有运行库执行：

```bash
python inspect_inputs.py
```

`files` 包含两个冻结构造源、新 CPU 检查源、原 58 行 gzip、原 16 篇纯输入 gzip、构造后 58 行 gzip；不需要传模型或 adapter。载入 tokenizer 前核六个本机小资产的完整 SHA，仅复用现有 `/dev/shm/patientphex-qwen35-1045-c20223623576`；CUDA 隐藏。固定实际 TF 5.20.2、Torch 2.7.1+cu126、tokenizers 0.23.1。

对全部原/新 58 行，比较 Auto encode、Auto template tokenize 和发布原生 `tokenizer.json` 的实际 IDs；两种完整 decode 都须等于 rendered 输入。CPU 同时重新构造并核 `expanded58` 完整字节，保存各行 input/rendered/row/suffix SHA。没有截断、重写、补 marker 或输出格式修复。

输出 `row-audit.json` 与 `result.json`，独占写入。输入 cap 只从真实 observed maximum 向上取 512 的倍数，generation 保持 1600；若 observed max >32768 或 rounded cap+1600 不适合实际 context，明确输出 `probe_input_gate_passed=false` 并报告，不通过删文、截断或偷偷改 cap 继续。完成输入 census 不表示显存/时延/质量通过。最长 probe 的 240+TERM30 与条件完整 58 的预算、实际启动由负责人管理，本准备不启用 GPU 或评测。

`manifest.json` 只列本委托的固定文件，排除负责人并行维护的 `asset-readiness/`。仓库副本保留源码、plan、摘要和引用，不复制 2 MB 全文输入、压缩输入、长逐段审计或任何模型文件。F1 原资料保持原位。
