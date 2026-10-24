# 工作台账

2026-10-24 UTC，分支 `research/training-seed-ensemble`，起点 `main@ff7ad16777e3907ed3b4b4231794439017aa5bf8`。仅修改本方案目录。共享规则已读，后继启动尤其遵循 H49 原始响应先存、H68 按实际成本做最小工程检查、H69 启动有效期与运行预算分开。

**当前：原fresh训练16:30:57完整成功，87×2/22更新及两完整持久ACK已在原watch终态核到；恢复收件后接39条F1，尚无新质量。**

**历史释槽交接（现已恢复）**：根线程安排本 owner idle，原训练与只读 watcher 保持。两轮及逐轮完整 state 持久 ACK 由同一远端 trainer 自动完成，无手动跨轮/导出依赖。唯一恢复指针 [evidence/active-idle-handoff.json](evidence/active-idle-handoff.json) 记录已核活 PID/锁、原硬钟与后继命令。root 看到外存 `formal-fresh-v1/watch/terminal-observed.json` 后唤醒本 owner；先收原终态/两 ACK，成功才绑定39条新推断。不能使用历史失败入口的 start，也不凭暂时 idle 判定模型已结束。

**唯一活跃入口**：`training/formal-entry-fresh-v1.json`，外存`training-seed-ensemble-v1/formal-fresh-v1`，远端proc89bc/child36557。只读watcher195513（start_ticks11011112）负责观察；不要再运行下方历史formal-v1/recovery_v2/v3/v4的start。新共享TERM16:47:38.444861。16:04:12已实见epoch1 step3，step2三LoRA家族B更新及native源SHA通过。 不更改任何其他 owner 的运行。15:42 根线程资源观察为 .7 GPU2 quote、.100 GPU0 Copy/GPU1 boundary、.12 teacher pilot、.3 Sap；本线没有另查卡或占用外部进程所在卡。观察仅为历史，不代表下一次可用。随后根线程15:44:04只读核到 .12 GPU2=0MiB/noapps，取消R1大模型迁移，允许本线提出在原缓存机排程；正式具体预算仍待根线程确认，本线未据此启动。

- 查重：实际旧 G32 的 trainer 与 run manifest 对齐；仅 seed20261023。full64 同 seed 但不同数据；双采样的20261024/25是推断种子；平均源 `fresh_training=false`；quote261改目标并 warm-start。查重范围是已存在的来源/计划，不声称覆盖未记录实验。
- CPU：初版 0.205 秒核 seed-only 来源、87行两轮分组和旧实际分组；G-only 提案未执行。root 在任何新模型/质量前改为四源固定支持并集。V2 0.184 秒重新核唯一公式与全部 refs，133/141 纯预测供给，不读 held gold，不重新分词/fit/v2。
- 产物：`/workspace/patientphex-runs/round2-20261023/research/training-seed-ensemble-readiness-20261024/prepared-v2/`；`result.json` SHA `3b01db60fc8200dc0a05af05672a3ab6591affb97c0d57ac783169083fb7925b`；新 trainer SHA `43a902c49151c4f232c3402bf8b6ee5e05b55fe7ddc21cce62538b25da7cbc94`。原 prepared-v1 保持。
- 固定协议：[proposal.json](proposal.json)，SHA `e7d310902b6948c4d88d329878b0f7f59f5b45db8736698e01766e38634af184`。同文件在外存保留原件。来源、论文和实际成本集中于 [evidence/readiness.json](evidence/readiness.json)，不再新建大索引。
- 一次元数据整理曾误把采样计划中的 adapter 路径字符串当成 ref 字典，整理脚本在输出前退出；已改读真实 copy manifest 的 d52 SHA。不是模型或 CPU 合同失败，无重算预测。

历史准备阶段下一步（已被下方实际授权与运行取代）：根线程确认原 .12 上3000+30两轮预算及后继1800+30 F1后，再准备独立实际控制器。teacher已改排 .100，不再以旧teacher排队描述阻塞本候选。若执行，先为新 seed 绑定实际原模型/编码/runtime 与新 namespace、校验 fresh 卡/≥2GiB 持久空间；同 host/shape/API 不机械重做13/14步压力或全 tokenizer。保留真实 fresh-load、有限梯度、参数更新和两轮完整 state ACK 门。初始化影响由 Python/NumPy/Torch CPU/CUDA 同源 seed 控制，但本轮未加载新模型，不能提前写“新初始化已实测”。

时间提案及条件 F0 在[实验记录](experiments/E-20261024-TRAIN-SEED-001.md)。资源变更需明确 runtime/成本差异；未获 GPU 执行授权。此阶段提交后推送本分支并核远端，最终 commit 以 Git/交接消息为准，不为记录自身 SHA 追加提交。

正式授权追加：原A100唯一fresh seed两轮共享3000+TERM30（含两份完整state持久ACK）、fresh持久盘≥2GiB；同host不新probe。新训练plan `training/formal-plan.json` SHA `5b5b27290bfe259e00867f3438cfca84517a9615f6ec2c6baa40d99ea59d0525`，CPU来源合同e1e639ba；新trainer仍43a902c4，仅三seed改动。启动intent先固定issued+300秒TTL与issued+3000秒全程deadline，driver在入口/Popen前核TTL，trainer载模型前及训练/保存前沿原absolute deadline，未知请求不重start。后继新adapter完整F1 39/1800+30和必要Sap已授权，唯一固定公式；F1严格正才F0 47/2100+30，开前据实际F1核成本。纯准备proposal.json保留原历史GPU=false，不冒充当前正式plan。

实际操作入口：`python solutions/training_seed_ensemble/training/formal_remote.py {deploy|start|observe} --directory /workspace/patientphex-runs/round2-20261023/research/training-seed-ensemble-v1/formal-v1`。start只能调用一次；已有local start-intent后仅observe恢复。pack已完成，archive及完整成员为外存`package.json`（8a1addeda）。训练唯一只读watcher为`training/watch_formal.py`，只有原任务成功start后才以独立session/文件日志启动，owner.lock防双监控；不发新NN。终态`training/formal_collect.py`只收元数据、重开fullstate SHA，保大权重在原持久位置。

首run实证失败：15:49:07.248，0.121秒在verify_inputs的schema guard退出，未import Torch/载模型/训练，GPU2已0/noapps。新plan错误由本线引入，原错误/source/plan/claim不改。恢复入口改用 `training/formal-entry.json`（95f455ce）指向 `training/recovery_v2/` / 外存 `formal-recovery-v2`；schema恢复原契约、route_id另加；train源43a902c4不改。新driver先要求目标解释器CUDA隐藏的原verify_inputs实际PASS；CPU30+TERM5，不新tokenization/模型。共享原TERM16:39:05.690703保持，开前剩余门2570.350358秒=1.2×实际2041.958632+120；不足即不启动。

V2 CPU调用返回UNKNOWN，30秒纯读及后继库存均没有结果/进程；不能确认是否已执行或稍后到达。root明确该检查只读固定源/数据，唯一小回执写入，无GPU/共享checkpoint，允许独立第二次CPU检查，不套GPU fence。新V3用不同namespace与`cpu-verify-attempt2.json`，原未知请求仍只会写V2；不声称仅一次CPU调用。新CPU外层python3保存真实child/stdout/stderr/terminal并30+5，避免未知返回丢掉错误。原formal共享截止/2570.350秒门不变，新单入口`training/formal-entry-v3.json`。

第二次CPU实际0.418秒：原verify_inputs完整返回87/32，随后新加distribution metadata版本guard失败。此guard不能反映PYTHONPATH overlay真实import版本，先前未在断言前存版本值，不能凭失败冒称差异细节。独立V4/attempt3改为实际import版本与distribution metadata先存、再核；仅CPU CUDA隐藏，不加载权重/forward，trainer/目标/预算不变。单入口training/formal-entry-v4.json，所有首失败和UNKNOWN/V3终态保留。

V4 CPU真实4.761801秒PASS，同87/32输入及实际import Torch2.6.0+cu124/TF5.20.2/PEFT.21.2/tokenizers.23.1。V4 formal仅启动前只读库存UNKNOWN，尚无start-intent/claim/start_process；原剩余已不足2570.350门，原schema campaign明确0模型/0fit关闭，不再start，保16:39旧钟。root预授权此条件下独立fresh3000+30，新命名空间formal-fresh-v1；复用V4真实CPU回执/17资产/同训练源码，不再CPU或probe。当前唯一入口training/formal-entry-fresh-v1.json，新硬钟将在其唯一start-intent时固定，非旧job延期。

实际fresh作业proc_89bc0c0d-a45a-423a-a371-d40eb5226ff1，driver36536/child36557；新TERM16:47:38.444861、KILL16:48:08.444861，3000共享两轮与ACK。实际入口training/formal-entry-fresh-v1.json；bf35498c已启动前推送。唯一只读daemon195513/start_ticks11011112已跨exec实核仍活，180秒原observations，路径外存formal-fresh-v1/watch-launch/receipt.json；不会start任何NN。未来恢复先核此PID/start_ticks/锁和现有observation，不另起并发watcher。小Git入口evidence/formal-running.json。

后继源已接收helper并在输出前修正：初始helper把合法数组内无效quote/duplicate变整行拒绝，偏离原G；已直接alias原interpret，保父版逐item拒绝，只有cap/nonEOS/整体JSON错整行弃答。混合有效/无效反例实际CPU6507f58d通过；原helper整份a37fcd3d快照外存inference-helper-initial-delivery保留。新manifest指明修正来源，无新推断/质量。绑定推断时必须显式传fresh_v1正式plan/目录；不能用README中的任何旧失败默认目录。必要Sap原.3当前别线占用，真正推断后再协调原cache/空卡，未声明已可用。

已回收的具体版本差异更正：distribution Torch2.6.0、实际import Torch2.6.0+cu124，TF/PEFT/tokenizers完全相同；此前overlay只是环境背景，不能写成已证TF overlay冲突。原CPU第三次观测/首次真实三族更新/native来源小原件在evidence/first-update-runtime.json；0新质量。

16:40原watch自行恢复并收到实际终态，两轮持久ACK分别16:14:52/16:30:56，controller2000.189秒，实际16:31:00 .12GPU2释放；非.100卡释放。训练源码与原3000时钟未变。完整终态观察见[evidence/training-complete-observed.json](evidence/training-complete-observed.json)。首collector inventory成功，freeze返回UNKNOWN，保原请求并只读原包恢复，不重训/不把缺收件称质量失败。SCST已直接通知仍需等待本39推断实际释放。

原UNKNOWN freeze已实际完成；原25件metadata包全部逐SHA收回，collection196b22f6/17826B，两完整state仍远端持久且重新核SHA。已通过binder绑定真实finaladapter c3dc7de5及双ACK/22步；F1唯一39全fresh计划66bf2e69，1800+30与300秒launch TTL在绑定时锁，旧普通G generation/parser不变。原概率/标签未用于选择，无新质量。
