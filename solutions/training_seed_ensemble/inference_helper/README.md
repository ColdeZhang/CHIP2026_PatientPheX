# 新训练种子 F1 后继入口

状态：仅本地源码、来源 SHA 与合成 CPU 合同通过。没有实际新 adapter binding、推断、Sap、gold 读取或评分；`infer-plan.json` 尚未生成。由 owner 审阅、统一 Git 与实际执行。

唯一端点为 `B4 ∪ [G_new ∩ (G_old ∪ S ∪ Q ∪ C)]`。B4 固定为 proposal 的 F1 SHA `5ad5cfe3…`、Score `.701325377212172`；保留 16 文献、39 患者、1189 实体及全部 524 旧关系。完整预测先冻结，再最多 1 新 v2／0 旧；零新增保持 B4 原字节，0 v2。没有新模型单独评分或子集评分。

## 绑定与运行接口

`bind.py --formal-plan PLAN --formal-directory DIRECTORY` 在两份真实 11／22-step durability ACK、完整两轮终态和实际新 trainer SHA 全通过后才生成 `infer-plan.json`。恢复版传 owner 的 `training/fresh_v1/formal-plan.json` 与外存 `formal-fresh-v1`；checkpoint/adapter 路径由传入 plan 的 `output_dir` 派生。默认值仅指原 `formal-v1`，不会自动把已失败原目录当恢复成功。

绑定时固定两个独立时钟：300 秒启动有效期、从绑定时刻起的 1800 秒推断绝对截止及 30 秒 TERM 宽限。二者在任何 `start-intent` 前锁定；本地 start、远端 claim、driver 入口、创建 child 与载模型前核启动有效期。过期不能重绑覆盖原 namespace；UNKNOWN 沿原 intent 观察及封锁证据处理，不自动重启。

运行入口保持原 G `remote.py` 的 `pack/deploy/start/observe --directory DIR` 和 `collect.py --directory DIR` 接口。`pack` 另需 `--plan inference_helper/infer-plan.json`。推断外存目录为 `research/training-seed-ensemble-v1/infer-F1-v1`，远端同名 namespace。A100 mount/UUID 来自原已核资源及所选 formal plan，fresh 空卡门仍在 driver 内。

`run.py` 复用原 39 行 inputs/IDs、原生 tokenizer、BF16、SDPA、非 thinking、普通 greedy、max1600 和 raw-first 写入器。合成合同核 `model.generate` 的 AST 与原普通 G 完全相同；没有 trie 或 prefix constraint。`whole_row.py` 调用原 quote parser，cap、非 EOS 或整体 JSON 结构错误整行弃答；合法数组内 invalid quote/duplicate anchor 按父版逐 item 拒绝并保留其余，保留原始响应且不再生成。

## 完整后处理顺序

以下由 owner 在真实完整收集后执行；不是本次已完成步骤。运行 extractor 时设置 `PYTHONPATH=/workspace/patientphex-worktrees/explore-grounded-generation`，让原 schema/parser 的真实源可导入。

1. `postprocess.py extract-bind` → `extract_quotes.py --plan inference_helper/extraction-plan.json`。
2. 原 G `normalization/held_prepare.py --extraction-manifest B/F1-postprocess-v1/quotes-v1/manifest.json --output-dir B/F1-postprocess-v1/normalization-preparation`。
3. 有必要查询时，`sap/bind.py` 生成真实查询计划；`sap/held_remote.py pack/deploy/start/observe` 与 `sap/held_collect.py` 收集到 `B/F1-Sap-v1`。无查询直接进入下一步。
4. `postprocess.py normalize-bind` → 原 G `normalization/held_finalize.py --plan inference_helper/normalization-finalize-plan.json`。
5. `postprocess.py assemble` 写完整唯一 `B/F1-postprocess-v1/quality-v1/prediction.jsonl` 与 `freeze.json`，不读取 gold；`postprocess.py score` 是唯一显式质量入口，先以排他 `evaluation-intent.json` 防重复，再按固定 all/text 调一次 v2。完整 result 披露 macro、Score 差异和新增 TP/FP；byteexact 分支直接继承旧指标。

这里 `B=/workspace/patientphex-runs/round2-20261023/research/training-seed-ensemble-v1`，原 `G=/workspace/patientphex-worktrees/explore-grounded-generation/solutions/grounded_generation`。

Sap 使用原已核 public41099 bank/labels/manifest、train32 alias、`b59eb3f8…` 政策、原 Sap 环境及独立目标 mount `1593f159…`。仅查询本次必要完整集合；不重编码 bank、不拟合。它另有 300 秒启动有效期和 driver 开始后的 180+30 秒运行预算，不能将其与推断 1800 秒时钟混称；原资源空闲门保留。

## 已做验证与交接

`source-manifest.json` 固定原 G、Copy F1 字节装配参考及新 helper 源，binder/后处理/Sap binder 启动前逐件核 SHA。`evidence/cpu-contract.json` 记录：普通 G generate AST 一致、父版混合有效/无效项保持、整行framing弃答、Unicode 原文位置、合法空数组、缺 ACK／错误 source／21-step 拒绝、两个时钟、集合交集与原始非 phenotype 字节保护。全部 Python 与内嵌远端片段已通过语法检查；没有模型库导入。

待 owner 完成：核真实恢复 formal 双 ACK 与必要本地文件、审阅 actual plan 和新资源状态、执行完整 F1 与必要 Sap、冻结唯一预测后质量。F1 严格正才讨论后继 F0；本目录没有 F0 实跑入口。CPU 合同不代表实际模型链或质量已经通过。

Owner启动推断前修正：初始helper把父版itemwise拒绝改为任一无效项整行弃答，已恢复直接alias原interpret并用混合反例验证；原交付a37fcd3d及全部文件外存`inference-helper-initial-delivery/`保留，未产生任何新推断/质量。
