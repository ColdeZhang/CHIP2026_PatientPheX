# 标准化子任务台账

目标：原文定位quote独立映射HPO/NIL/abstain，给训练别名外的词面保守语义回退，突破原候选词表限制；不改变已冻结target/schema/prompt/runtime，不读held/dev gold，不使用旧train48/64 detector、概念计数或label映射。

2026-10-24 当前完成：`strict.py` 源31d380f5不变。固定policy `b59eb3f8…` 在任何Sap查询分数/held模型输出前冻结；cosine≥.90、不同概念margin≥.03，仅strict unmatched。实现完整批量/坐标API及CLI，12项新增CPU反例合同通过。87训练行801目标/624unique的model-independent公共与逐文排除诊断已实际完成，见README；尚无Sap查询分数、held预测或质量结论。

外部根目录：`/workspace/patientphex-runs/round2-20261023/research/grounded-generation-v1/normalization-v1`。`prepared/` 保存完整624query（SHA `e1836f1158a5c5849c27b04658f89783a30ffffa0943834b6a95a4e846067ccf`）、801诊断case及CPU摘要。queries只含train32目标原文，不含held输入、人工外部标签或旧训练统计。

## 已批准、未开始的唯一运行

- root授权624 query encode，最多120秒含加载/验证，TERM后15秒KILL；不fit、不新编码bank、不自动重试。队列为 **DocRE实际释放 → Sap查询 → span控制**。DocRE owner `/root/document_relations` 已被通知把release发父 `/root/sequence_recover_0518` 和root，父负责接管，本子代理不空等卡。
- `.3` mount `1593f159-19bc-41ca-8b4c-2e7455f5db99`；GPU UUID `GPU-07726469-bf7a-b832-9dee-019dc276dc1d`。
- 远端ROOT：`/home/zhangyuheng/projects/CHIP2026_PatientPheX/grounded-generation-v1/normalization-v1/query-v1`。
- 原生解释器：`/home/zhangyuheng/projects/CHIP2026_PatientPheX/semantic-neural/venv/bin/python`；`PYTHONPATH=ROOT`，cwd也为ROOT。同目录部署strict/public_bank/semantic，避免导入父runtime或旧detector。公开模型目录为项目 `semantic-neural/model`，各文件SHA写在冻结plan。
- 只读取精确旧bank项目 `ontology-round2/dev16-sapbert/embeddings/bc3b715246322f5190fdaa5564ac161a8ef701617cdc268836fbbe94366910bb.npy`，SHA `4fc0e062…`。源42575行中只取已证明的41099公共行，所有ID重新来自HPO；缺失/不符直接终止保留失败，不盲重算。

## 部署、运行与回收

已生成 `deployment/deployment.zip`：448246 bytes，SHA `9e55f77750286295b09ab722914d77e23135939445206e1877f71e3f2c84b1d4`。`deployment/package.json` SHA `05d5adb6…` 绑定10成员；源码/输入来自 [query-plan.json](evidence/query-plan.json)，SHA `afbd1e17031d810f59b54960f6a7c990191691eee420804959f46b0dcde3df6b`。本次没有调用remote工具，无在途请求、job或未知启动状态。

1. 用已有MCP `write_file`（base64）把唯一ZIP上传 `ROOT/deployment.zip`，先保存本地调用intent。`deployment/deploy-command.sh` 是准确的已生成命令：校ZIP/member SHA、安全名字、源码编译和fsync，再生成`deployment-complete.json`；同内容已有文件可核对，不覆盖不同字节。
2. **等DocRE实际release后**，保存唯一start intent，再用MCP `start_process`执行 `deployment/run-command.sh` 内容。命令已设置GPU UUID、三种BLAS线程1、完整PYTHONPATH和原生解释器，启动 `query_driver.py --plan ROOT/query-plan.json`。driver独占`driver-started.json`，查目标GPU无计算进程及实际盘余量，再在绝对截止内运行 `query_once.py`。进程状态未知先查原job/markers，禁止再次启动。
3. 回收`wrapper-complete.json`、`driver-started.json`、`child-started.json`、`resource-check.json`、stdout/stderr与`actual/complete.json`。成功状态须为`COMPLETE_FIXED_PUBLIC_TRAIN_QUERIES`、query_count624、public_bank_rows41099、no_fit/no_bank_reencoding真；按complete.assets逐SHA回收 `query-scores.json`（<1MiB）、`query-token-counts.json`、`encoder-runtime.json`、`public-bank-manifest.json`。`query-vectors.npy`约1.9MB、`public-bank.npy`约126MB、public labels留持久远端并记录locator/逐SHA，不假称已本地回收。
4. driver在子进程退出后保存 `target_gpu_processes_after_child`；实际无进程后立即通知root和 `/root/span_normalization` release，再做CPU诊断与备份，勿占槽执行其他GPU任务。
5. 对已回收的完整scores只执行固定policy一次train-only诊断：

```bash
python solutions/grounded_generation/normalization/diagnose.py evaluate \
  --hpo /workspace/patientphex-data/hp.obo \
  --train /workspace/patientphex-runs/round2-20261023/ontology_rules/nested-outer0-inner1/train32.jsonl \
  --policy solutions/grounded_generation/normalization/policy_v1.json \
  --prepared /workspace/patientphex-runs/round2-20261023/research/grounded-generation-v1/normalization-v1/prepared/cpu-summary.json \
  --scores /path/to/collected/actual/query-scores.json \
  --output-dir /workspace/patientphex-runs/round2-20261023/research/grounded-generation-v1/normalization-v1/train-semantic-diagnostic-v1
```

报告strict重代入、public-only、LODO各自接受正确/错误/弃答及语义增量；不扫阈值，不把训练诊断称held质量。首次实际加载/查询仍可能发现工程差异，按原失败+原预算保存，不改变政策掩盖问题。

实际唯一运行完成：proc8ccd4671 / driver1197856，09:57:49.906495Z起，wrapper16.717259秒rc0；624query/41099公共bank/0overflow。完整14小件+外部3大件SHA/fsync/reopen已收ce208cd1，scoresad265ecc；actual bankbc4ea231/126256256B、publiclabels60a27430/10807562B、queryvectors2171fe40/1917056B保留.3同ROOT/actual。postchild target GPU processes为空，已通知root/span释放。唯一固定诊断summary0f999a0d：LODO语义128接受=117正确+11错误，完整603正确/13错误/185弃答；只是train32逐文排除标准化诊断，不是F1模型质量，不扫阈值。

后继held入口（尚未运行）：`held_prepare.py`在同32/public strict上确定完整必要unknown集合，`held_query.py`仅复用已冻41099公共银行计算这些真实词面，`held_finalize.py`调用原decide，不改b59eb阈值。root已授权单≤180+TERM30，排DocRE30真实释放后；实际query数由全39fresh输出确定，不用heldgold。原624train-only诊断和查询不重做。
