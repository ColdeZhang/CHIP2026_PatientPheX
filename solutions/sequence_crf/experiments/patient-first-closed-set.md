# 闭集患者优先生成：CPU 准备完成

真实目标是改变同一关系矩阵的自回归因子化，检验能否改善患者集合判断。约束是仅复用旧 strict train32 标签与 F1 纯输入；本阶段成功标准为全部关系双逆精确、来源隔离、实际分词/掩码和严格 schema 成立。**本阶段无训练、GPU、权重加载、F1 targets、原 dev gold 或新质量分数。**

## 固定改动

原 `CID -> [patient_id]` 改为 `patient_id -> [CID]`，每个给定患者都有键，空列表保留。只替换 system 前两句任务/输出 schema；后续临床判断规则、整个 user 文本、全局患者锚点、候选/HPO 映射、患者顺序、候选顺序、row ID 和行顺序完全保持。禁止输出未列 CID 或 HPO 编号。

训练标签使用原 raw-HPO 口径，不作 alt-ID 规范化。`transform`/`inverse` 对全部124行的完整对象及 compact answer 字节双逆一致；594个正 pair、1912个负 pair 不变。新输出总346个患者键，106个空值；旧880个CID键中396个空值，7条整行全空保持。该统计是序列化结构，不代表标签或质量变化。

## 实际结果

2026-10-24 实际 CPU **4.070265秒**，外层退出码0，固定 tokenizers 0.23.1 / Jinja2 3.1.6，未导入 torch/transformers/peft，CUDA隐藏。

| 项目 | 原方向 | 患者优先 |
|---|---:|---:|
| train 行/文献 | 124 / 32 | 124 / 32 |
| train 全部 token | 402,594 | 400,614 |
| train prompt token | 395,424 | 395,920 |
| train supervised token（含 EOS） | 7,170 | 4,694 |
| train 全长 min/median/max | 591 / 3311 / 5523 | 593 / 3298 / 5556 |
| train label token min/median/max | 9 / 59 / 116 | 7 / 31 / 135 |
| F1 纯输入总 token | 250,696 | 251,016 |
| F1 输入最大 token | 5691 | 5695 |

所有 train/F1 prompt 都恰好多4个 token。训练最长行仍是 `11235872:relations:8`，F1最长仍是 `6063015:relations:0`；没有截断。新 target 更短来自输出因子化，不能声称监督 token 数相同，也不能把token下降直接解释为更快或更有效。

复用旧实际 helper 的 `encode` 函数：对完整聊天字符串分词，验证 prefix IDs 精确前缀、assistant-only mask、唯一受监督 EOS 和 causal label 存在。**旧124行所有已保存 native 审计字段**（full/prefix/label长度、render SHA、input-ID SHA、label SHA）均精确；F1旧80行的 render SHA/长度与既有逆序准备审计匹配。新方向在真实 Transformers 模型运行环境上的 GPU 前后向/数值/耗时尚未测量，不宣称已过。

F1文件只有 `row_id/pmc_id/code_map/patient_ids/messages`，80行/16文献/39患者/587候选/1557 pair，没有 answer。只变换输入 schema，不读取任何 F1 targets 或响应。

## 隔离与来源

旧训练文件 `cb7f67e1…`、准备清单 `356109e9…`、native token审计 `46b005ef…`、实际旧formal2ep run manifest `5535d33a…` 均在运行前核完整SHA。两条候选生成链 F2→F3 / F3→F2 的 generator/target各16文献、互斥并合为训练32；F0/F1排除，训练与F1输入文献交集0。沿用已审候选来源而不重新生成或从label补候选；没有读取祖先manifest中指向的F1 gold。

tokenizer `5f9e4d49…`、template `a4aee8af…` 来自既有固定 Qwen3.5-9B verified-small。未下载模型。全部实际文件路径、大小、SHA在 `plan.json` 与 `cpu-v1/summary.json`；旧资源仍原位只读。

## 有区分力的检查

5类有效合同：P1/P10和C001/C0010精确区分；完整转置/逆转；全空值；空候选集；JSON键顺序不改变语义。

20类拒绝合同：重复/缺失/未知患者键、未知或重复CID值、null/数字/bool/string/nested错误值、markdown、尾随第二对象、NaN、把旧schema误当新schema、重复患者或CID白名单、空患者白名单、旧schema重复CID键/未知或重复患者值。新 parser 只接受一个严格 JSON 对象；实际字符串控制未靠宽松字符串搜索恢复。

## 可执行复现与交接

```bash
env CUDA_VISIBLE_DEVICES= TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=/workspace/patientphex-runs/round2-20261023/research/thinking-budget-forcing-readiness/cpu-libs \
  timeout --signal=TERM --kill-after=15s 195s python \
  /workspace/patientphex-runs/round2-20261023/research/patient-first-closed-set/prepare.py \
  --plan /workspace/patientphex-runs/round2-20261023/research/patient-first-closed-set/plan.json \
  --output /workspace/patientphex-runs/round2-20261023/research/patient-first-closed-set/cpu-replay-UNIQUE
```

脚本要求新 output 目录且用独占写入；已完成 `cpu-v1` 不可覆盖。保留的全量新数据：`cpu-v1/train124.patient-first.jsonl` 和 `cpu-v1/F1.input80.patient-first.jsonl`。源码为 `patient_first.py`/`prepare.py`，小证据与全量数据的独立保存边界见 `handoff-manifest.json`，供 sequence owner 选择性纳入其分支。

后续 trainer **不能直接绕过旧SHA guard**：应以新数据manifest绑定旧来源与本次转置，沿用原2epoch/32调用/每行CE归一/AdamW设置和原encode；只按明确的新数据schema调整来源校验。推理须用新患者键parser，再反投影成旧association，不得使用旧CID-key parser。当前源码已经提供这两个方向的严格纯函数，尚无GPU runner。

本阶段不改变loss权重、不开始训练、不自动扩大到F0或原dev。正式训练预算仍待真实新最长行成本；原124×2约1149秒仅是旧运行参照。若最终生产采用新adapter替换旧同形状adapter，参数不增加；若与NCR一起保留两个adapter并集，总量会超过10B（见先前 score-gap-review 报告）。

## sequence owner接管与唯一最长行探针

已逐文件核验handoff全部13件实际SHA，将两源码原字节接管到`solutions/sequence_crf/patient_first/`；大训练/F1输入与逐行audits保持外部。独立CPU数据审 d50c850e 已通过，25合成不重复。

主线程已授权一次A100 GPU2最长5556/135监督真实更新，180秒+TERM30，fresh public9B+全hybrid LoRA r16/alpha32/dropout.05，原生BF16/SDPA/nonreentrant checkpoint/native完整assistant CE。仅adapter进入AdamW，lr1e-4/wd.01/clip1；这是明确非零学习率的单步诊断，未执行正式2轮的warmup。初始LoRA B为0时A首步梯度可为0；检查有限loss/全部梯度、三family正梯度和各一个小B矩阵真实变化，不复制9B base。状态退出即丢弃，不保存checkpoint、不推断、不质量评分。

新data_contract独立绑定新train SHA、原来源与逐124行实际tokens，严格124/max5556/total400614/supervised4694；只复用旧底层encode。原旧trainer的7170/402594/5523 guard保持原样，未绕过。虽然仍用有效assistant token均值CE，输出顺序和自回归分解改变会改变token级优化目标，不能称原数值目标等同。

参考已审A100旧5523 selected-weighted一步12.1805秒/峰23.8591GB；更接近算子的旧A800 native2ep峰36.6334GB、reserved81.1095GB（多形状训练池）。两者都不保证本单5556在A10040GB的实际峰/耗时，OOM/数值/预算首失败就停，不重试更换attention/loss/截断。

**当前阻塞**：A100 runtime只读核验立即返回LUNA_OFFLINE，未取得新runtime执行结果；ontology NCR成本probe亦尚未启动，root统一恢复连接。当前仅本地source/manifest/AST与静态增量审3275b461完成；真实nativeCPU绑定与GPU更新均未执行。等待ontology成功的实际只读roundtrip后恢复同一次CPU检查，先让其NCR短探针完成释放，再fresh查GPU空闲启动本单步；不凭mount online认定恢复。

外部`probe-preparation/`有resource/data-plan/source proposal/payload及MCP故障收据。恢复入口：`patient_first/deploy_cpu.py deploy --directory <probe-preparation>`，完成后`recover`回收三CPU文件；若工具返回unknown，先检查已落盘文件，禁止重复盲发同一部署/CPU动作。源/计划推送、实际CPU收据与最终GPU run-plan冻结后再使用`operations.py start`的单次intent入口。GPU启动前无需新增用户确认，但具体阻断必须先处理。

### 连接恢复后的部署中断（仍无CPU/GPU模型执行）

ontology于01:52:11实际run_command恢复成功后，本方案仅在自有目录部署。目录创建被controller观察为rc0；唯一254466B `upload.part000` 的write_file返回UNKNOWN_OUTCOME，随后只做一次该part实际SHA读取也返回UNKNOWN_OUTCOME。**不能记成part不存在或传输成功。** 没有解包、native runtime/data CPU命令、GPU start-intent或GPUjob；已停新MCP，等待root统一恢复。ontology的NCR也尚无GPU作业，不存在本方案占卡等待。

源码53d485c已推；本地zip、18文件清单、intent及两次未知收据完整，见[精确分段状态](../evidence/round2/patient-first-probe-deployment-state.json)。恢复后只核已有part实际SHA；若与zip相同，解包既有字节并逐成员核验/compile，再执行尚未执行的CPU步骤，不重新上传或调用`deploy`从头创建目录。`deploy_cpu.py recover`仅用于三个CPU结果已经落盘后的回收，不能替代本次upload-part库存检查。
