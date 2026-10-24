# 工作台账

2026-10-24 UTC，分支 `research/training-seed-ensemble`，起点 `main@ff7ad16777e3907ed3b4b4231794439017aa5bf8`。仅修改本方案目录。共享规则已读，后继启动尤其遵循 H49 原始响应先存、H68 按实际成本做最小工程检查、H69 启动有效期与运行预算分开。

**当前：本地准备已43f8fe1c推送核验；根线程已授权正式链，独立控制器已冻结待部署；尚无作业。** 不更改任何其他 owner 的运行。15:42 根线程资源观察为 .7 GPU2 quote、.100 GPU0 Copy/GPU1 boundary、.12 teacher pilot、.3 Sap；本线没有另查卡或占用外部进程所在卡。观察仅为历史，不代表下一次可用。随后根线程15:44:04只读核到 .12 GPU2=0MiB/noapps，取消R1大模型迁移，允许本线提出在原缓存机排程；正式具体预算仍待根线程确认，本线未据此启动。

- 查重：实际旧 G32 的 trainer 与 run manifest 对齐；仅 seed20261023。full64 同 seed 但不同数据；双采样的20261024/25是推断种子；平均源 `fresh_training=false`；quote261改目标并 warm-start。查重范围是已存在的来源/计划，不声称覆盖未记录实验。
- CPU：初版 0.205 秒核 seed-only 来源、87行两轮分组和旧实际分组；G-only 提案未执行。root 在任何新模型/质量前改为四源固定支持并集。V2 0.184 秒重新核唯一公式与全部 refs，133/141 纯预测供给，不读 held gold，不重新分词/fit/v2。
- 产物：`/workspace/patientphex-runs/round2-20261023/research/training-seed-ensemble-readiness-20261024/prepared-v2/`；`result.json` SHA `3b01db60fc8200dc0a05af05672a3ab6591affb97c0d57ac783169083fb7925b`；新 trainer SHA `43a902c49151c4f232c3402bf8b6ee5e05b55fe7ddc21cce62538b25da7cbc94`。原 prepared-v1 保持。
- 固定协议：[proposal.json](proposal.json)，SHA `e7d310902b6948c4d88d329878b0f7f59f5b45db8736698e01766e38634af184`。同文件在外存保留原件。来源、论文和实际成本集中于 [evidence/readiness.json](evidence/readiness.json)，不再新建大索引。
- 一次元数据整理曾误把采样计划中的 adapter 路径字符串当成 ref 字典，整理脚本在输出前退出；已改读真实 copy manifest 的 d52 SHA。不是模型或 CPU 合同失败，无重算预测。

下一步：根线程确认原 .12 上3000+30两轮预算及后继1800+30 F1后，再准备独立实际控制器。teacher已改排 .100，不再以旧teacher排队描述阻塞本候选。若执行，先为新 seed 绑定实际原模型/编码/runtime 与新 namespace、校验 fresh 卡/≥2GiB 持久空间；同 host/shape/API 不机械重做13/14步压力或全 tokenizer。保留真实 fresh-load、有限梯度、参数更新和两轮完整 state ACK 门。初始化影响由 Python/NumPy/Torch CPU/CUDA 同源 seed 控制，但本轮未加载新模型，不能提前写“新初始化已实测”。

时间提案及条件 F0 在[实验记录](experiments/E-20261024-TRAIN-SEED-001.md)。资源变更需明确 runtime/成本差异；未获 GPU 执行授权。此阶段提交后推送本分支并核远端，最终 commit 以 Git/交接消息为准，不为记录自身 SHA 追加提交。

正式授权追加：原A100唯一fresh seed两轮共享3000+TERM30（含两份完整state持久ACK）、fresh持久盘≥2GiB；同host不新probe。新训练plan `training/formal-plan.json` SHA `5b5b27290bfe259e00867f3438cfca84517a9615f6ec2c6baa40d99ea59d0525`，CPU来源合同e1e639ba；新trainer仍43a902c4，仅三seed改动。启动intent先固定issued+300秒TTL与issued+3000秒全程deadline，driver在入口/Popen前核TTL，trainer载模型前及训练/保存前沿原absolute deadline，未知请求不重start。后继新adapter完整F1 39/1800+30和必要Sap已授权，唯一固定公式；F1严格正才F0 47/2100+30，开前据实际F1核成本。纯准备proposal.json保留原历史GPU=false，不冒充当前正式plan。

实际操作入口：`python solutions/training_seed_ensemble/training/formal_remote.py {deploy|start|observe} --directory /workspace/patientphex-runs/round2-20261023/research/training-seed-ensemble-v1/formal-v1`。start只能调用一次；已有local start-intent后仅observe恢复。pack已完成，archive及完整成员为外存`package.json`（8a1addeda）。训练唯一只读watcher为`training/watch_formal.py`，只有原任务成功start后才以独立session/文件日志启动，owner.lock防双监控；不发新NN。终态`training/formal_collect.py`只收元数据、重开fullstate SHA，保大权重在原持久位置。
