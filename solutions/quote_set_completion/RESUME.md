# 本线恢复入口

更新：原formal已完整ACK4b486327/collection3c799b44；现在唯一39推断proc_4ac60fcc/driver36169/child36205，原TERM16:38:30.114 UTC，KILL16:39:00.114。本地只读watch PID/SID197018、ready4a4c49c7，每180秒原observe；实际入口见evidence/live-inference.json。下述formal步骤已完成，不再start或重训。必要Sap已由root预排.100GPU0，仍须fresh空卡门；不碰GPU1/.3。

当前分支 `explore/quote-set-completion`，工作树 `/workspace/patientphex-worktrees/explore-quote-set-completion`，只改 `solutions/quote_set_completion/`。先读当前 main 的 AGENTS/HANDOFF/WORKLOG，核本工作树状态。实际作业与 source SHA 见 [live-formal.json](evidence/live-formal.json)。

令 Q 为本方案目录；N 为 `/workspace/patientphex-runs/round2-20261023/research/quote-set-completion-v1`。当前唯一原正式作业 `proc_ad68dbeb-c7e6-453a-821a-b0fac0d57ab9`，A800 GPU2、driver13674。模型 TERM **16:28:43.644688 UTC**，KILL16:29:13.644688；持久ACK最迟 **16:30:43.644688**。原共享4200秒包含120秒ACK，不能重置时钟或重启模型。

本地 detached/flock watcher **PID/SID190459** 在 `N/formal-v1` 维护 `local-watch-events.jsonl`、`inspect-*.json`、`local-watch-failure.json` 或 `local-watch-complete.json`。正常每180秒只读原远端状态，epoch成功后自动执行同一有界导出和 .100 完整ACK。已实际核独立会话与锁；它不是普通exec未完成会话。root在本阶段释放代理槽后只读这些已有本地文件；不要增加远端轮询或停止 watcher/正式作业。

1. watcher健康时只看现有本地记录。若停止或出现failure，先核 PID/starttime/命令、lock、最后intent/receipt；实际 UNKNOWN 只恢复原状态。唯一只读 operations 入口是 `python Q/training/controller/operations.py inspect --directory N/formal-v1`，不能与健康watcher并发。任何原start-intent存在都禁止再start。
2. `campaign-complete.json` 出现表示完整一epoch/33update和 .100 ACK已成。需等原export HTTP wrapper/child实际退出后用 `python Q/training/collect_complete.py --directory N/formal-v1` 一次回收完整元数据；大adapter/Adam/RNG保留原持久locator，不回传权重。若collector尚报告原HTTP未退出，仅恢复原终态，不重导出。
3. 完整collection成功才运行 `python Q/inference/bind.py`，绑定真实新权重和261/33终点，生成 `Q/inference/infer-plan.json`。后继源码已写但实际绑定、执行尚未发生。push源/plan后，沿 `inference/remote.py pack/deploy/start --directory N/inference-F1-v1`（pack显式 `--plan Q/inference/infer-plan.json`）唯一启动原授权39fresh/1800+30。继续A800GPU2，fresh UUID/process/disk门；原全部39 native/HF IDs在加载模型前核，native greedy1600，任何cap/非EOS/strict失败整行弃答、不重生成。
4. 原39完整终态后 `Q/inference/collect.py --directory N/inference-F1-v1` 回收全部raw/IDs。`Q/postprocess/bind.py extract` 生成实际绑定计划并打印原已审 extractor 与 strict preparation 的准确命令。复用原G32 source，不复制改写parser/normalizer。
5. `N/F1-normalization-preparation-v1` 只含同train32/public41099/b59的必要queries。若非零，先与root/.100 owner协调真实空闲卡；`Q/normalization_runtime/bind.py --preparation-manifest ... --gpu-uuid ...` 绑定该机已核公共bank/Sap/TF4.51.3，必要180+30，原GPU0或GPU1必须明确实际owner/fresh门，禁止并驻其他模型。没有queries则不启动GPU。正常化源政策不变，不重新编码bank/fit。
6. 规范化完整后 `Q/postprocess/bind.py finalize` 打印原固定finalizer命令；完成后 `bind.py quality` 只绑定唯一 `B4 ∪ 全部合法补全关系`。`Q/postprocess/quality.py assemble --plan ...` 先完整freeze，再 `evaluate` 唯一v2 30+1/旧0；零新增byteexact则0v2。B4 .701325377212、524关系/1189实体/39患者均固定。无单臂、阈值、过滤或第二pass。
7. `Q/reporting/f1_entry.py` 汇总已成终点，无模型/评分调用；支持正、负、零新增。随后一次集中独审/最小持久包、更新台账并push。严格F1正才由root协调同固定F0；本授权不含originaldev/新官方动作。

当前后继源通过语法、固定来源SHA/import和baseline lineage本地检查，未运行39推断、Sap或质量，不把草稿入口当实际终点。原两个工程坑已由root写入H67/H68。历史174/3600和所有失败原件保留，不能混成当前261/4200配置。
