# A800 全文最长诊断部署

root 已明确授权一次最长 12,824-token 的 fresh-base 原生更新/cache 诊断：A800 GPU2 `GPU-f516a193-de9c-8552-7ac5-4725f24cc0fc`，180 秒上限，加 TERM 30 秒；第一 OOM、非有限或截止即停止，不自动重试、不换短行、不进行正式训练或质量评分。

本目录仅准备和部署，**没有调用 `start_process` 或运行 GPU**。主线程是唯一启动者，须在运行前即时确认空卡，保存唯一 intent/job 与绝对截止。`run-plan.json` 提供 `probe_argv`、`timeout_argv`、`timeout_command`、`env` 与新输出目录，明确不再运行旧 CPU delta。后续实际终点由 owner 单独记录。

已将 9 件输入/源码以 799,850B 的压缩包部署到 A800 `research/medgemma-relation-control/fullcontext-probe-a800-v1`。压缩包 SHA `b50ba27c4b1e7e70d4f71dd141fcf0715c0544e26cfa215bd60bdfce40b0d399`；包内全部文件经过实际远端落盘、fsync 和回读 SHA/大小核验。新 namespace 与输出均要求不存在；不覆盖其他实验。既有 `a800-preparation` 的 8 件源码/原训练/隔离输入/manifest 小文件核验通过；原模型目录和 native runtime 复用，未打开或搬运模型权重。

计划 SHA `ef56dddce2693cfaee146f85fd17b1b0bc9fb450185a86441f4cde769f2434d1`；部署终点 SHA `7b1adbb72e958e9df3d69922eaadd48da9530c38be97301de44f1f5d5908004b`。数值 entry 保持 `9c0e87a2…`，合同保持 `9b4d7505…`；本次未改已冻结源码、CPU 结果或原 trainer。显式新诊断 cap 14,000 与原正式 cap 8,192 分开记录。

压缩包、上传分片和完整传输 receipt 留仓库外；小源码、计划、清单和最终落盘摘要按 `backup-filelist.json` 由 owner 统一备份。当前没有 GPU 显存、运行时间或质量结论。
