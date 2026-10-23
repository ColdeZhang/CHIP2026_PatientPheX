# 提交凭据与产物恢复

首日清单：[day1-20261023.json](day1-20261023.json)。以平台文件名、提交时间、预测 SHA 和代码版本唯一对应一次提交；本地近似分与官方分开记录。今天只核验接收，不回收官方成绩。

## 备份边界

源码、摘要、分支提交及时推送 GitHub；最终上传的 JSONL 与接收回执额外保存在用户提供的 MCP 服务器，模型权重、数据集和完整日志不进入 Git。认证资料不在这些备份中。

服务器：`10.5.2.100`，mount `5a9955da-5240-41cc-9e52-05d3d8da8a3c`，mount 根为 `/home/zhangyuheng/projects/CHIP2026_PatientPheX`。

- 最终预测与回执：`day1/coordinator-backup/submissions/`。
- Git bundle：`day1/coordinator-backup/git/`，包含 main 和五个 day1 分支及完整可达历史；不包含未提交文件。
- 连接参数只从环境变量 `MCP_URL/MCP_TOKEN` 读取，不打印值。

五份最终预测及五份接收回执已全部备份并核对远端 SHA，实际清单见 [预测备份 manifest](day1-20261023-backup-manifest.json)。外部原始 manifest 位于 `/workspace/patientphex-runs/day1-20261023/`。备份工具以 SHA 核验：同名同内容跳过，同名不同内容拒绝覆盖。512 KiB 分片传输、合并后再次检查完整哈希。

## 备份与恢复命令

在仓库根执行（`--manifest` 指向保存的小型备份清单）：

```bash
python scripts/backup_submissions.py backup
python scripts/backup_submissions.py restore --manifest MANIFEST.json --destination RECOVERY_DIR
python scripts/backup_git_checkpoint.py --repo . --output-dir EXTERNAL_BUNDLE_DIR
```

若本地 manifest 丢失，用 MCP `read_file` 从对应远端备份目录读取 manifest；它会列明文件名、大小、SHA、所有分支和恢复命令。先恢复 bundle 到新目录，再执行 `git clone -b main BUNDLE_PATH NEW_CHECKOUT`；恢复五个实验分支使用 `git -C NEW_CHECKOUT fetch BUNDLE_PATH 'refs/heads/day1/*:refs/heads/day1/*'`。不要覆盖现有工作树。

2026-10-23 已对前三份预测及三份回执做实际恢复并逐文件验证 SHA；重复备份确认全部跳过同内容文件。初始 Git bundle 已从服务器下载并独立克隆，六个 refs 一致。最终 bundle 已覆盖 `main@be0cf2e` 及五个最终实验分支，完成自包含和远端 SHA 校验，见 [Git 备份 manifest](day1-20261023-git-backup-manifest.json)。主分支随后仅追加这份备份指针和台账状态，不为记录自身 hash 无限生成备份。
