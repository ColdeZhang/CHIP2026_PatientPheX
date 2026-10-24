# 工作台账

更新时间：2026-10-24 UTC；分支 `explore/response-code-equivariance`，任务给定起点 `main@0878cbdf`。

## 当前状态

- 目标：用唯一 cyclic1 PT 视图检验原 strict C 的关系稳定性，主端点 `B4 ∪ (原 strict C ∩ 新 strict C)` 必须严格超过 B4 Score `.701325377212172`，才支持讨论 F0。诊断端点为新 strict C 单臂，不能替代主门。
- 本地准备已完成：80/16doc/39PID/587CID/1557pairs，53 行多患者新 NN、27 行单患者原字节缓存；结构合同 17 项 PASS，实际 0.290318 秒。未读取 gold、原模型答案或训练 target。
- [结构结果](evidence/structural-result.json) `9d181088…`；新全量输入 `1c372b6e…` /2,893,388 B，留仓库外。原27行三类文件的81件 SHA 已核，只有引用，没有新解析/生成。
- [native CPU 模板](evidence/native-plan.template.json) 绑定历史 .100 Torch2.7.1/TF5.20.2/tokenizers.23.1/PEFT.21.2，120+5；当前未运行，新 token 总量、max、输入 cap 均为 null。
- 禁止额外排列/seed/训练/组合扫描；旧 strict parser 原字节复用。没有新模型、质量或官方提交。
- 本子任务未 Git；主线程统一审阅、commit/push。不得把当前工作树文件存在称为远端已备份。

## 下一步

1. 主线程完成源检查点后，唯一实际 native CPU；沿模板逐字节部署并核实际环境，不重复本地结构生成。
2. 依据实际 token/max/context 定推断输入 cap 与工程运行；53 fresh＋27 原 raw 缓存覆盖80。任何新生成先落 IDs/raw，再 strict decode；异常整行弃答，不补答案。
3. 两完整预测共冻后评分：主若变化1 v2、若与B4完全相同继承旧指标0 v2；诊断1 v2，旧0。未获新结论前不自动迁 F0/dev。

详见 [实验](experiments/E-20261024-EQUIVARIANCE-001.md) 与 [研究笔记](NOTES.md)。
