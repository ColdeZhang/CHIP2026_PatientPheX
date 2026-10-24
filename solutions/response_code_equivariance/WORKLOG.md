# 工作台账

更新时间：2026-10-24 UTC；分支 `explore/response-code-equivariance`，任务给定起点 `main@0878cbdf`。

## 当前状态

- 目标：用唯一 cyclic1 PT 视图检验原 strict C 的关系稳定性，主端点 `B4 ∪ (原 strict C ∩ 新 strict C)` 必须严格超过 B4 Score `.701325377212172`，才支持讨论 F0。诊断端点为新 strict C 单臂，不能替代主门。
- 本地准备已完成：80/16doc/39PID/587CID/1557pairs，53 行多患者新 NN、27 行单患者原字节缓存；结构合同 17 项 PASS，实际 0.290318 秒。未读取 gold、原模型答案或训练 target。
- [结构结果](evidence/structural-result.json) `9d181088…`；新全量输入 `1c372b6e…` /2,893,388 B，留仓库外。原27行三类文件的81件 SHA 已核，只有引用，没有新解析/生成。
- [native CPU 模板](evidence/native-plan.template.json) 绑定历史 .100 Torch2.7.1/TF5.20.2/tokenizers.23.1/PEFT.21.2，120+5；当前未运行，新 token 总量、max、输入 cap 均为 null。
- 禁止额外排列/seed/训练/组合扫描；旧 strict parser 原字节复用。没有新模型、质量或官方提交。
- 该分支现由本路线 owner 独占维护 Git；运行前及时提交推送，核远端后记录。

## 下一步

1. 主线程完成源检查点后，唯一实际 native CPU；沿模板逐字节部署并核实际环境，不重复本地结构生成。
2. 依据实际 token/max/context 定推断输入 cap 与工程运行；53 fresh＋27 原 raw 缓存覆盖80。任何新生成先落 IDs/raw，再 strict decode；异常整行弃答，不补答案。
3. 两完整预测共冻后评分：主若变化1 v2、若与B4完全相同继承旧指标0 v2；诊断1 v2，旧0。未获新结论前不自动迁 F0/dev。

详见 [实验](experiments/E-20261024-EQUIVARIANCE-001.md) 与 [研究笔记](NOTES.md)。

## 2026-10-24 13:30 UTC：实际完整native CPU通过

唯一.100proc156e646f/PID2742802，120+5，实际11.7197s rc0/native10.9888s，8件全SHA。53新/27原cache、685372total/max14570/guard14848，原证据/IDs及逆映射完整。实际runtime2.7与原A1002.6差异明确。见CPU-REPORT.md及evidence/native-CPU-result.json；GPU/quality未启动，等待root排卡，原full64优先。

## 2026-10-24：获准唯一 GPU / 两固定质量，源已冻结

- 原 actual native 完整80已过；新增本地0.142047秒缓存/严格逆映射/生成AST合同通过，见 evidence/GPU-source-CPU-contract.json。没有重tokenizer、模型或质量。
- 固定 .100 GPU0 UUID `GPU-126293b9-6677-8464-cee7-01e693efbf29`，首次实际只读24164MiB空闲；启动器再核fresh无进程，GPU1不使用。原C32 adapter `2c777e88`，53 fresh/27历史三类文件原字节cache，14848/1600，唯一1200+TERM30。
- 入口见 evidence/GPU-ready.json，外部 `research/response-code-equivariance-v1/gpu-v1`；plan `a20c972a`，quality plan `e140395e`。本段为事前准备，尚无GPU job/新质量。
- 两完整端点共冻后主变化1v2（若B4字节exact则继承0）、诊断1v2、旧0。仍为mapping+host/runtime开发实验，无自动F0/dev/官网；诊断单臂若独胜B4也上报root。

## 13:36:35 UTC：唯一模型作业已启动

`proc_d8c112f4-496a-43cd-874b-7c738d093e7d` / PID2743313，91部署成员逐SHA/fsync/compile完成；源检查点 `876b02b8` 已推送且ls-remote核验。模型终点尚未到，检查原namespace见 `evidence/GPU-job.json`，禁止unknownstart重复。额外合成固定交集/非关系字节合同0.000642秒通过，无真实gold/评测。
