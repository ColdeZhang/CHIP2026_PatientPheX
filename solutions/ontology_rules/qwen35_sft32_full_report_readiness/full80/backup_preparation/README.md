# SFT32 全文完整终点：本地持久包准备

本目录只准备打包源码与固定清单，尚未制作真实终点包、上传服务器或验证远端持久性。最终统一 review 仍待负责人提供实际文件及 SHA；不能以源码审查替代真实终审，也不生成空白占位 review。

目标是在恢复会话时保留本次 `probe-v1` 与 `full80-v1` 的必要终点，避免复制模型、全文语料和工具回执。`selection-policy.json` 是 SHA 固定的非递归白名单：18 项 probe 文件、39 项 full80 文件、三个目录各 80 项原行证据、1 项实际 review、5 项本打包准备文件，共 303 项；压缩包另含 `members.json`。每项均保存原路径、归档相对路径、原字节数和 SHA-256。

保留内容包括原 probe 和正式推断 loader/runner/launcher/plan/模型资产 manifest；输入 CPU row-audit；全部 240 个 generated IDs/raw/canonical 原证据；完整响应汇总、真实 rc0/释放记录；评测源码与绑定 plan；两份完整预测、metrics、格式校验、共同 freeze 及唯一评测标记；负责人指定的最终 review 原字节。

不包括 gold、训练标签、模型/适配器/tokenizer 资产、完整 `expanded80.jsonl`、部署/下载压缩包、MCP/client/operations 源与原始工具/登录/上传回执、普通 stdout/stderr 与重复 inspect、其他实验和 F0/dev/官方产物。所含模型 manifest 仅为资产身份，不含权重。已有 terminal-manifest 中仍可能引用被明确排除的日志；本包只声称恢复自身 `members.json` 列出的成员，不声称恢复整个外部实验目录或全部复现资源。

实际运行由负责人在完整推断、双端点评测和统一 review 均完成后执行一次：

```bash
python persistent-backup-preparation/package.py \
  --probe-root /workspace/patientphex-runs/round2-20261023/research/qwen35-sft32-full-report-feasibility/probe-v1 \
  --full-root /workspace/patientphex-runs/round2-20261023/research/qwen35-sft32-full-report-feasibility/full80-v1 \
  --review /absolute/path/to/actual/final-review.json \
  --review-sha256 ACTUAL_FULL_SHA256 \
  --output-dir /workspace/patientphex-runs/round2-20261023/research/qwen35-sft32-full-report-feasibility/full80-v1/persistent-terminal-backup
```

命令从 `full80-v1/` 执行，或将脚本也写为绝对路径。review 必须是真实 JSON 对象，SHA 与显式参数一致；packager 不猜测审查格式或改写审查结论。所有白名单文件、240 个原证据、review 缺失时直接拒绝；在检查通过前不创建输出目录。检查包括固定 plan SHA、正式完整 80 条和 rc0/GPU 释放、两端点质量终态、共同 freeze 与两预测/metrics 的身份、共同 freeze 中全部 240 条原证据 SHA。只读取已经完成的结果元数据，不导入模型、评测器或 MCP，不打开 gold，不调用质量评分。

输出目录须不存在；成功后包含：

- `qwen35-sft32-full-report-terminal-v1.tar.gz`：仅普通文件、固定相对成员名、文件权限 0600、确定性 tar/gzip 编码；不使用 `extractall`。
- `members.json`：包内清单的同字节外部副本。
- `local-archive.json`：整个压缩包与清单 SHA、成员数、逐成员实际解压读取核验结果，明确 `NO_REMOTE_WRITE`。

每成员至多 1 MiB、未压缩总量至多 16 MiB、最终压缩包至多 1 MiB；预期很小，但实际大小只在执行时记录。上限超出应检查清单，不静默删证据。输出使用独占写入、`fsync`，然后直接逐 tar 成员读取并复核全部 SHA；不执行成员内容。已有输出时不重复打包，先检查原包/回执。父执行者之后按既有服务器渠道上传一次，核全包和逐成员 SHA，再保存独立远端 receipt；此本地源码不提供上传功能，不把本地归档冒称持久备份成功。

合成自检已通过（约 0.03 秒、60 秒预算）：二进制/Unicode 字节往返、重复成员/路径穿越/符号链接/篡改 SHA 拒绝、固定 240 项选择、review/终点缺失不产生输出。仅使用临时合成文件；未打真实包、未读真实终点/gold、未远端操作。准备包身份见 `manifest.json`。
