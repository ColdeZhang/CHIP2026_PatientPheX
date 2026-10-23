# 前任仓库归档说明

归档日期：2026-10-23。用户决定完整归档旧工作，从干净的 `main` 开始新实验。

- 原始来源：`919d7a2b2e18f9b0e0d3a569f16b1db97cd37d27`。
- 归档前工作版本：`da2739551fb0d74a7b531fa371193fbb35a9f85e`。
- [manifest.json](manifest.json) 列出原始 31 个文件的原路径、新路径、大小、SHA-256 和 Git blob ID；已逐字节比对原始提交。
- [README.md](README.md) 是从原始提交恢复的旧说明；其中路线和分数声明不代表当前结论。接手期间修订过的 README 保留在 Git 历史中。

## 内容与边界

| 路径 | 内容 |
|---|---|
| `src/` | 14 个旧实现脚本，未修改 |
| `data/` | 旧训练集、B 集和 HPO；HPO 版本不符合比赛指定版本 |
| `dict/` | 两份旧词典；增强词典与训练标注重合，不能直接用于新验证 |
| `outputs/` | 11 份历史预测，完整保留 |
| [docs/REPOSITORY_REVIEW.md](docs/REPOSITORY_REVIEW.md) | 接手审计结论，属于历史分析 |
| [docs/review/2026-10-23/](docs/review/2026-10-23/README.md) | 诊断脚本、结果和原始校验清单 |
| [docs/experiments/E-20261023-001-legacy-audit.md](docs/experiments/E-20261023-001-legacy-audit.md) | 接手诊断实验记录 |
| [docs/competition/LOCAL_NOTES.md](docs/competition/LOCAL_NOTES.md) | 旧仓库与官方要求的差异 |

新项目的官方资料、执行规则、台账及实验模板仍在仓库根目录和顶层 `docs/`。旧资料中的“当前”“建议”“未修复”均指审计当时的旧实现；用户选择从头开始后，不再按旧路线继续迭代。

## 路径与复核

旧文件维持内部相对布局。审计 JSON 中 `source_inputs.path` 相对本归档目录，`artifacts.path` 相对审计 manifest 所在目录；历史命令和结果中的旧路径也按这一上下文解释。诊断脚本字节未变，其相对定位仍指向本归档。

仅在需要复核历史时，从仓库根目录先执行 `cd archive/legacy-20261023`，再参考审计说明中的命令。输出写到归档之外；旧绝对路径和缺失权重等问题仍存在。本次只搬迁并核验，没有修复旧代码或重新运行训练。
