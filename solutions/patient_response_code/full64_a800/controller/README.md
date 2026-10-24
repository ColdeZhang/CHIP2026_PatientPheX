# A800 Full64 正式链入口

唯一阶段入口 `entry.py`。新 controller 默认 `R/research/patient-response-code-full64-a800-v1/controller`，由主 owner 在真实新探针完成后创建；这里没有 actual GPU-AUTHORIZATION/campaign/start。

1. `formal_cost.py --readiness R/research/patient-response-code-full64-a800-v1` 读取新 `probe-v1/package-runtime-v2/terminal`，保留12行52201 tokens实测外推以及旧 A100 时长作为明确标记的保守 proxy，加入两完整导出1200秒和保存/更新120秒余量。总估计必须≤5400。A800 实测末余量≥10GiB、峰保留<70GiB；成本只是估计，不保证完整265行。
2. `entry.py prepare --controller <new-C> --probe-terminal <actual-wrapper> --cost-decision <actual-cost>` 在创建目录前核新探针 plan/source、A800 UUID/runtime/base/CPU账、12F/B+1Adam、两真实极值、finite、释放、probe state弃用与成本。旧 A100 terminal不能激活。
3. owner选择原唯一部署/启动路径：正常 `deploy → preflight → start → watch`；或已验证精确包桥接 `full64_source_bridge.py {stage,serve,ready,fetch} --controller <C>` 后 `bootstrap-target → bootstrap-prepare → bootstrap-start → watch`。两条启动路径不能重复使用。
4. `entry.py collect-training --controller <C>` 完整回收。两ACK确认后 `entry.py bind-inference --controller <C> --inference-directory <new-D>`，默认后继目录 `R/research/patient-response-code-full64-a800-v1/originaldev64-inference-v1`。

source/persistent 各机器独立根均为 project/`patient-response-code/full64-a800-v1`。source GPU2≥70000MiB free/≥80000MiB total且无app，source盘≥512MiB，SHM≥4GiB；.100目标盘仍≥2GiB。模型/状态不落 A800 项目盘。A800 mount `69b991ee-183f-4add-af1a-579af92479c9`，私网10.5.4.7；.100持久 mount 保持 `5a9955da-5240-41cc-9e52-05d3d8da8a3c`。

`source_bridge.py` 原机制逐字保留；只能调用相邻 `full64_source_bridge.py` wrapper，它覆盖原 MA 和 REL 为新 A800/source-bridge namespace。不得直接执行机制参考模块，以免去旧 C32/A100 路径。

原 detached watcher、只读 LUNA/UNKNOWN 退避和完整 state/ACK恢复逻辑保留。未知 mutation不能重试启动；先恢复原 intent/job。共享5400+30截止不会被阶段恢复重置，两个 epoch 的状态持久导出仍各受 min600/整体剩余时间限制。HTTP子进程终止由owner实际 PID/原截止核，不把 driver退出当子进程证据。

无训练配方、推断提示或阈值改动，无gold/质量/官方提交入口。源码 helper 不执行远端或Git，所有实际阶段由主 owner 统筹。
