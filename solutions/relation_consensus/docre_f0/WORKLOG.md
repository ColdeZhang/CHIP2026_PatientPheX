# DocRE30 → F0 固定推断

2026-10-24：为 root 已获 F1 正增益的 `B ∪ (DocRE30 logit>0 ∩ schema)` 做 F0 复验。这里只生成完整 DocRE logits；不训练、不读 F0 gold、不做单臂 v2。root 决定共识质量终点。

- 完全复用已审 final epoch30/global_step240 checkpoint `67317bdd…`（205 张量，110,277,377 参数）。祖先报告 `446e470f…` 证实训练 F2+F3 与 F0 无交集。
- 复用原三个冻结公共候选源；F0 纯输入 16 文献、47 患者、2,195 患者×HPO 对、1,904 个物理提及。无候选截取。
- `vendor/` 四个文件从 DocRE 原源逐字节复制，SHA 在 plan；新代码只包装 F0-only 数据和 checkpoint 推断。原 prepare/encode/marker CLI 与 train.main 不调用。
- CPU 完整 token/marker/owner 合同随后在原 tokenizer/runtime 执行。GPU 原预算唯一 180 秒 + TERM 后 30 秒；无 probe、优化器、早停或阈值搜索。
- 外部产物：`/workspace/patientphex-runs/round2-20261023/stage8-relation-consensus/docre-f0/`。源码/计划先推送，CPU 门通过后补冻结实际 encoded SHA，再启动唯一 GPU。

入口：本地 `prepare_inputs.py --mode public`；原远端 runtime 执行 `--mode encode`；`driver.py --plan <frozen-plan>` 唯一窗口内执行 `infer.py`。完整实际命令、原输入与资产 SHA、进程和截止时间保存外部。
