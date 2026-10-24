# F0 终点证据汇集准备

目标是把已完成的单次患者优先 F0 实验交给同一 validator 终审。这里只准备汇集源码，**没有执行终点、打开 gold、重新评分或检查运行中的目录**。源码与当前冻结控制器/计划的小快照见 `source-bindings.json` 和 `source-snapshots/`；当前阶段仅做 AST/语法静态检查，不重复旧检查。

`collect_terminal.py` 仅使用 Python 标准库。owner 完成既定训练回收、最终 58 条推断回收、双预测共同 freeze 和两臂既定评分后，才执行一次：

```bash
python /workspace/patientphex-runs/round2-20261023/research/patient-first-closed-set-F0/formal-preparation/terminal-preparation/collect_terminal.py \
  --directory /workspace/patientphex-runs/round2-20261023/research/patient-first-closed-set-F0/formal-preparation \
  --output /workspace/patientphex-runs/round2-20261023/research/patient-first-closed-set-F0/formal-preparation/owner-terminal-manifest.json
```

它要求以下证据已全部存在且相互一致，任一缺失/失败/哈希冲突/时序冲突均报错，不生成成功清单，也不创建生产子目录：

- `training-collection.json`、`training-terminal/`：原完整两轮/52 calls、两份 `durability-epoch-00N.json` 与 checkpoint metadata、实际退出与 GPU release。
- `inference-collection.json`、`inference-terminal/`：同一 durable epoch2、完整 58 条严格 EOS/PID/CID、0 fallback、原推断 plan 与实际退出/release。
- `assembled/freeze.json` → `secondary-union/common-freeze.json` → `before-both-quality.json` → primary `before-gold/results` → secondary `before-gold/results`。核 SHA 引用与已有 UTC 字段的顺序，不用 mtime 猜先后。
- 两份格式结果、两份公共 v2 all/text 指标、汇总 results；16 文档/47 患者/947 实体/415 CID/1166 对，固定 primary 与 oldSFT48 literal union secondary。旧 baseline metrics 只读复用，不调用 evaluator。

脚本不导入 `finish` 或评测器，不重新解析预测、不重跑 tokenizer/schema/训练检查，不调用 MCP/GPU。全量 raw/fullpred/日志仅原地核 SHA 后保留外部引用，不复制内容；不递归遍历 JSON 内的任意资源指针。计划里的 gold locator 不打开或哈希。权重/optimizer/RNG 不读取；清单中的远端路径、大小和 SHA 明确来自已有实际 fsync+reopen durability receipt，再与已回收 checkpoint metadata 对照，不能称作本次重新核权重。

所有条件通过后才在内存中生成单个 `owner-terminal-manifest.json`，用排他临时文件 + fsync + 原子 hard link 发布并 fsync 目录，拒绝覆盖。若文件写入阶段异常，保留 `.pending` 供 owner 判断，不应据此重跑训练或评分。汇集 JSON 是 owner 索引，仍标记 `pending_independent_terminal_review`；其时序证据来自已有记录与源码绑定，不能独立证明历史上从未访问过 gold。

建议仅纳 Git：脚本、README、准备小 manifest、最终 owner manifest、关键 results/metrics 与 validator 终审；原训练/推断日志、完整 raw/fullpred、模型全状态继续留仓库外。不要把整个 `source-snapshots` 重复入 Git：生产源和已有独审已持久化，这份快照只用于当前交接的 H34 字节恢复。
