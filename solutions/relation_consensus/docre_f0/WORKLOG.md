# DocRE30 → F0 固定推断

2026-10-24：为 root 已获 F1 正增益的 `B ∪ (DocRE30 logit>0 ∩ schema)` 做 F0 复验。这里只生成完整 DocRE logits；不训练、不读 F0 gold、不做单臂 v2。root 决定共识质量终点。

- 完全复用已审 final epoch30/global_step240 checkpoint `67317bdd…`（205 张量，110,277,377 参数）。祖先报告 `446e470f…` 证实训练 F2+F3 与 F0 无交集。
- 复用原三个冻结公共候选源；F0 纯输入 16 文献、47 患者、2,195 患者×HPO 对、1,904 个物理提及。无候选截取。
- `vendor/` 四个文件从 DocRE 原源逐字节复制，SHA 在 plan；新代码只包装 F0-only 数据和 checkpoint 推断。原 prepare/encode/marker CLI 与 train.main 不调用。
- CPU 完整 token/marker/owner 合同随后在原 tokenizer/runtime 执行。GPU 原预算唯一 180 秒 + TERM 后 30 秒；无 probe、优化器、早停或阈值搜索。
- 外部产物：`/workspace/patientphex-runs/round2-20261023/stage8-relation-consensus/docre-f0/`。源码/计划先推送，CPU 门通过后补冻结实际 encoded SHA，再启动唯一 GPU。

入口：本地 `prepare_inputs.py --mode public`；原远端 runtime 执行 `--mode encode`；`driver.py --plan <frozen-plan>` 唯一窗口内执行 `infer.py`。完整实际命令、原输入与资产 SHA、进程和截止时间保存外部。

CPU 门已全部通过：原 BertTokenizerFast/transformers4.50.0，64,854 tokens、172 个510/384完整窗口（单篇最多19），所有提及唯一 owner、剥 marker 原 token 精确还原、相邻/共享/嵌套边界合同通过。编码 a3fda0a2（160,911B），保留全部2,195对。原 F1 164窗推断1.598秒；F0 180+30 包含读取/校验1.3GB检查点，预算足够。

已结案：唯一 `proc_9715fc3b-4626-4e3c-815a-0ef1af4c0c8c`，实际11:01:44.410178 UTC开始，180+30，10.322515秒rc0成功；11:01:54.885448实际释放GPU0（4MiB/noapps）。全部2195有序keys逐项与完整编码精确一致，logits有限/probability精确，原vendor逐字节不变；checkpoint严格205张量完整加载。raw scores `969b31af…`（214954B），外部21件ZIP `8def940f…`（333035B）已本地完整逐SHA回收，服务器副本可恢复。零上下文对保留，未训练、未读gold、未调用v2。详见[result.json](result.json)；root接手固定共识，本站无GPU后继。
