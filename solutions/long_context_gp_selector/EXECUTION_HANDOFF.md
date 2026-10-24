# 唯一执行/恢复入口（2026-10-24 17:12 UTC）

**状态：全部前置完成；没有GPU启动意图或模型作业。** root要求本阶段final释放槽，在schema的GPU1真实release后followup恢复本代理。不要轮询或停止schema；它原F0作业17:03:03启动、预计约17:16释放，硬钟17:33:34。GPU0与其他进程不属于本任务。

## 固定身份与授权

- worktree `/workspace/patientphex-worktrees/research-long-context-gp-selector`；branch `research/long-context-gp-selector`；只改 `solutions/long_context_gp_selector/`。
- `S=.../solutions/long_context_gp_selector`；`D=/workspace/patientphex-runs/round2-20261023/research/long-context-gp-selector-readiness-v1/formal-v1`。
- mount `.100=5a9955da-5240-41cc-9e52-05d3d8da8a3c`；唯一GPU1 `GPU-fc5d0525-3ec1-074a-9fa3-35d92a9da539`。
- remote `P=/home/zhangyuheng/projects/CHIP2026_PatientPheX`，run `P/research/long-context-gp-selector-v1/formal-v1`。
- **实际源/运行计划 `S/runtime/formal-plan.json`：5143B / `53f716b38f2be3747290035caaaa2909a1d915e44d0ed2a023b381e16cd60db2`。** 已部署并原CPU实际读取，`D/package.json`绑定全部5源SHA，`D/deployment.json`为真实部署证据。不要再次build/deploy或重新编码。
- 唯一GPU总 **900+TERM30** 包含3fresh公共父fit（GP03/GP02/GP023，256/256/384=896更新）、4完整score passes以及3完整model/optimizer/scheduler/RNG持久ACK。GP23只读已训model-only权重01215671供F0，不续训。
- 固定CPU端点 **60秒**，原101→103，160trees/.3/nested，一次F1 v2/旧0，严格胜 `.7286975364690524` 才保留，不扫cut/alpha、不自动F0/originaldev/官网。

## 已完成，不重复

完整数据：`D/data.json` 33589956B / d80a0e7df9c5a8eb384fd638d275f94b558e433291f71f4a55e3f163f6177805；local-preparation.json四组全候选61282边界，5个训练OOF不可对齐false/0；F1全部16417可表示。mapping-contract.json独立对照原1313全部矩阵坐标exact。

实际CPU：`D/cpu-tool.json` rc0，原receipt在remote根`cpu-complete.json`，已由 `D/observation-1792861803790532935.json`完整解析回收，**7.8846917秒、64全文native IDs/offsets/special exact、真实CPU tiny encoder/GP反向、GP23 model-only严格加载PASS/CUDA未初始化**。不是GPU/长序列显存或完整checkpoint重算证明；旧完整GP工程复用。其8371>8192 warning来自未截断的全文token合同；模型窗口仍沿旧8192/512覆盖，无裁文。无需重复CPU。

实际runtime：ScratchVL Python3.10.12，Torch2.7.1cu126，TF4.50.0/tokenizers.21.4来自`P/research/seq2seq-patient-set-v1/runtime-t450`，hub.29.3、numpy2.2.6、safetensors.5.3。source旧Torch2.7.0cu128/hub.36.2/numpy1.26.4/safe.7不同已披露，CPU实际模块SHA写receipt。GPU进程显式沿计划的完整PYTHONPATH，不改共享包。

资产已完整本地盘持久：`P/research/long-context-gp-selector-v1/transfer/assets/`，公共5文件以及GP23-model.safetensors(596465072B/012156714032c83363902cf8ea9e258537f7796e7f50bae9c2de3091d96e23cb)、GP23-model-manifest.json(d455796a/5307B)。原GP23完整1.789GB state15798e36和ACK0cc2保留.3原目录，model-only逐136tensor导出重开，不是可续训state。

`D.parent/asset-copy-v1/copy-complete.json`记录11.778秒/全SHA/fsync/reopen，源HTTP PID1227574按start_ticks/UID/argv精确关闭，`source-closure.json`实核/proc不存在；不得因MCP抖动重开/重导/重传。两次source UNKNOWN/LUNA原件保留；恢复只发生CPU资产阶段，无模型训练。

## 后继唯一动作

收到schema**实际release**后，确认本地没有`D/start.intent.json`。从worktree执行：

```bash
python solutions/long_context_gp_selector/operations.py start
```

它创建唯一ticket（admission300s）、核源/CPU、fresh UUID/noapps/空闲显存/≥8GB盘后，独立900+30运行。记返回managed ID，并一次`operations.py observe`取得真正child PID、start/TERM/KILL时点；不要把managed请求发出当模型已经开始。当前**没有watcher**；启动后仅单owner≥120秒节奏只读observe，root需读本地observation无需另查服务器。

若任一启动工具UNKNOWN：先恢复原`D/transport/`意图、remote `job/{exclusive-launch,started,ended,expired-before-entry}.json`与原admission时钟。**不得直接再执行start**；空卡/暂未见进程不证明旧请求失效（H69）。不自动改预算、模型、精度、数据或重启fit。

终态后马上将真实release发root和schema。完整成功必须三ACK＋四个target-scores；失败/timeout保完整partial/日志、停止不评分。仅CPU收原件：

```bash
python solutions/long_context_gp_selector/operations.py pack
python solutions/long_context_gp_selector/operations.py collect
```

pack只纳源/metadata/分数/终态/stdout、stderr，不搬5.37GB新states；UNKNOWN恢复原terminal.zip及index，不重包。collect按文件现长度续相同ZIP，核wholeSHA/每成员/embeddedmanifest，并fsync/reopen。若pack已建或collect已有标记，不盲重运行；读原意图。

完整成功后，先确认`D/collected/job/ended.json` rc0、output complete为3fits/4passes、三个ACK；然后**唯一**：

```bash
PYTHONPATH=/workspace/patientphex-runs/round2-20261023/research/lightgbm-selector-readiness/deps/runtime OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout --signal=TERM --kill-after=1 60 python solutions/long_context_gp_selector/fit_control.py --plan solutions/long_context_gp_selector/quality-plan.json --actual /workspace/patientphex-runs/round2-20261023/research/long-context-gp-selector-readiness-v1/formal-v1/collected --output /workspace/patientphex-runs/round2-20261023/research/long-context-gp-selector-readiness-v1/formal-v1/quality --hpo /workspace/patientphex-data/hp.obo --gold /workspace/patientphex-data/round2-inner4/fold-1/validation16.gold.jsonl
```

该source先完整旧概率与解码复现，再新fit、完整prediction freeze，最后1v2。**不要提前读F1gold或只评部分输出**。有fit/evaluation-intent时禁止重fit/重评；恢复已持久模型/指标并说明实际阶段。

质量/所有分项/时间和真实三state locator一次报告root，由root安排集中增量审与最小包。不自行重复开review；严守当前强48祖先，正也不自行迁F0。源计划checkpoint及时push并核remote SHA。
