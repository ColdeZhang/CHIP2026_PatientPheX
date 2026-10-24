# 严格32文献共同旧权重起点：新两轮顺序对照

状态：入口与实际 CPU 已准备，尚未启动 GPU 训练。2026-10-24 的 MCP 连接反复离线曾暂停新远程调用；01:20后主线程实际读写恢复，允许源push后fresh核资源并按既定预算启动。此前成功的源部署、存储和小传输证据仍有效，不能代替未来启动时的空卡核验。

## 目标、变量与预算

检验新增加的等量训练中，可逆候选排列是否改善原80条 F1 关系预测。两臂从同一 oldSFT32 epoch2 adapter（权重 `4f4815b6`、checkpoint `5cecc79c`）开始，新建相同 AdamW、seed 和两轮 LR 阶段：控制 O/O，增强 R/O。每轮124条、16次 optimizer call；每臂248曝光、32次更新。只重排候选及对应 CID/target，不改病例、患者、概念、标签或样本顺序。

这不是 fresh 四轮训练，也不是旧优化器的精确继续。保留旧四逻辑轮 CPU 方案作为历史，当前选择通过复用共同旧权重省去重复两轮训练。两臂都有额外训练量，不能把相对旧2ep的差异全部归因于重排。

两个完整新阶段、四次 epoch 导出及两次原80推断共享 **3900秒＋TERM30秒**，单张 A800 GPU2 顺序执行；无 FLA，无新推断模板、阈值或额外臂。预算依据旧两轮训练1149.460秒/臂、旧80推断282.502秒/臂和已完成173MB HTTP传输18.573秒估计，不能保证实际时长。

## 已完成证据

- R124及32-call schedule由独立代理核验：语义双逆变 exact，O/O与R/O均14340监督tokens，完整tokens805188与805190。训练数据只来自原严格32文献，F1与原dev均排除。
- 新 trainer 在 A800原native runtime以CUDA隐藏执行CPU模式，2026-10-24 01:17:19收齐，7.006587秒、`cuda_initialized=false`；124行/32文献、max5523tokens，LoRA43,278,336参数。实际源 `93301f60`、manifest `f8e48487`，不是GPU训练结果。
- 原checkpoint实际权重/config SHA与旧记录一致；旧optimizer只stat，不读取。新阶段epoch1→2从自己的完整checkpoint恢复优化器、scheduler、RNG，且要求完整持久导出ack。
- 01:07:22实查A800项目盘仅187,875,328字节可用，/dev/shm250,649,587,712字节；.100持久盘445,938,155,520字节。新checkpoint必须写自有SHM并逐轮导出，不能写A800项目盘。
- 01:09:11已完成一次1MiB A800→.100真实HTTP传输、持久fsync和SHA验证，0.263864秒；已停止server。该小测不替代173MB实测，也不重复进行。每epoch全权重、优化器/RNG、checkpoint与run manifest按逐文件size/SHA核对后才继续。
- 传输CPU合同包括真实断流后从123456字节续传、短源EOF立即失败、错误hash/缺文件/重复ack拒绝；不只检查happy path。较早fixture的不足及修订快照留外部，不污染最终合同。
- endpoint只在双臂完整80响应齐全后组装两份完整预测，先共同freeze再首次打开F1 gold；每臂唯一格式检查与v2评测。保留1189实体和非关系字段，比较两个新臂及旧2ep固定基线 `.660942003`。这是F1开发对照，不能当原dev达标证据。

## 恢复入口

外部根：`/workspace/patientphex-runs/round2-20261023/research/candidate-order-warmstart-entry`。冻结总入口为 `formal/run-plan.json`，tiny source清单 `entry-source-manifest.json`，实际CPU在 `actual-cpu/`；候选全量数据和模型均在仓库外。

网络恢复后先沿用统一入口审核结论，完成 `deploy_fetcher.py` 的小源部署及源SHA核验，再由 `run_paired.py --directory <外部根>/formal` 进行fresh资源检查并开始同一3900秒预算。该controller尚未运行，`formal/execution-state.json` 尚不存在。已经部署完成的CPU入口不能重新执行；它会拒绝已有job。

controller在每个start_process之前先fsync保存intent，再保存返回的已知job；模糊失败没有job ID时按intent查原进程，不能自动补launch。运行后若会话/连接中断，读取 `formal/**/job.json`、monitor游标和已有 `.100` export ack，先核原进程，禁止盲目重跑controller或复制训练。所有新launch受同一绝对deadline约束，连接故障不延长预算。完整结果之后用 `freeze_evaluation_plan.py` 固化动态元数据，再用既定 `evaluate_paired.py` 执行两臂唯一评测；不能提前读取gold。

当前 `.100` fetcher 尚未部署，新训练checkpoint尚不存在；“入口已准备”不代表长训完成或checkpoint已持久备份。用户原dev≥.8目标仍未达到，不合并本路线实现，不上传官方。
