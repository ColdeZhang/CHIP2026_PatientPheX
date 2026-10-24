# SFT48 F0 完整58持久包准备

本目录复用已核F1通用归档函数，固定F0 probe/plan与完整58。仅读完成结果，不导入模型、评测器或MCP，不读gold、不评分。实际统一review尚待；缺完整58、双质量、174原行证据或精确review即拒绝。

固定内容240项：18 probe + 42 full58 + 174逐行证据 + 1实际review + 5归档准备文件，另含members.json，实际tar为241成员。保留两完整pred、metrics、共同freeze、模板来源，排除全文语料/模型/日志/工具回执。包上限1MiB，普通安全相对路径、逐成员SHA、独占写/fsync；不重复打包，已有输出先查原回执。

终审后唯一执行（从full58-v1）：

```bash
python persistent-backup-preparation/package.py --probe-root ../probe-v1 --full-root . --review /absolute/final-review.json --review-sha256 EXACT_SHA --output-dir persistent-terminal-backup
```

输出local-archive.json只代表本地归档；服务器完整SHA/成员核/fsync回执另记。原F1 archive及审查均不改。self_test仅合成字节与安全/缺终点反例，未打真实包。
