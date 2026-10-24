# SCST 八条响应的训练奖励接口

本子目录只实现固定 `[76,44,83,19]` 四患者的八条响应标准化。原 grounding、严格别名、b59 政策、公共 Sap 实现逐字节复制于 `reuse/`，出处在 `reuse-manifest.json`；不改旧部署 parser。任何坏 grounding item 都使**整条训练奖励**为0，保留旧 parser 产生的全部 anchors/rejected 证据。非法行已经确定为0，故其部分 anchors 不属于必要 Sap 查询。

`assets.json` 锁原 train32、train87、624条原FP16查询缓存和公共HPO。路径可经 `--assets` 换到相同 SHA 的副本，不可换训练集合。gold 原子化与冻结v2相同：只有完整 `HP:ddddddd;HP:ddddddd` 字符串才拆分，818原子全部保留，包括原G目标无法构造的4个literal；不规范化 gold。预测按原 normalizer 映射、复合概念去重、NIL 保留 quote 原文。

## 主控制器接口

所有输出目录必须全新，完整响应先持久化再解释。命令以系统 `/usr/bin/python3` 运行 CPU 部分（已核本地 NumPy 环境）：

```bash
/usr/bin/python3 normalization/prepare.py --responses raw8.json --output prep
/usr/bin/python3 normalization/finalize.py --preparation prep/preparation.json --output reward
```

`raw8.json` 为完整列表，每条至少含 `row_id, arm, response, EOS, cap, generated_ids`。`arm` 为 `greedy|sample`。词元IDs到实际文本及EOS状态的一致性由主生成运行器核，接口保原IDs及其SHA，不伪称从文本重建过IDs。完整原JSON bytes不改，reward同时绑定其SHA、每行UTF8文本SHA、IDs紧凑JSON SHA、长度、EOS/cap。

`prep/queries.json` 非空时，不可直接 finalize。先在本地主控制器读到主体原绝对 TERM/KILL 后绑定：

```bash
/usr/bin/python3 normalization/bind_query.py --preparation prep/preparation.json \
  --remote-root /EXCLUSIVE/NEW/SCST/SAP/ROOT --shared-term ACTUAL_MAIN_TERM_UNIX \
  --shared-kill ACTUAL_MAIN_KILL_UNIX --output bound
```

这只写 `bound/deployment.zip`、`query-plan.json`、`package.json`，**不传输、不启动远端**。package 提供完整成员、SHA、argv、environment、mount 和完整收件名单。owner 应将12成员复制至该新根、逐SHA/fsync/reopen，再按 package 单次启动，UNKNOWN 恢复原 intent，不重新绑定延长时钟。共享 deadline 必须来自已运行主任务，不能用新的 `now+1800`。

正常化源固定 `.100 GPU0` 的同一 public Sap/41099 bank；当前卡先由 sequence F0 占用，必须等其实际 release，再 fresh UUID/无进程/空卡/盘门，不能因为该源码已就绪就启动。`query_driver.py` prebound 5分钟 admission 与计算截止独立；TERM取自身start+180和主体原TERM较早者，KILL取局部TERM+30和主体原KILL较早者。入口和载模型前都核 admission，`driver-started.json` 排他创建，失败/UNKNOWN 不覆盖。动态代理20+.02秒/quote沿旧624实际15.347秒，是成本估算而非保证；剩余窗口不足直接停。

必须收齐 `package.collect_all` 所列文件，并核 `wrapper-complete.success`、真实 child absent、GPU释放。query产物保完整向量、token counts、scores、runtime、complete。随后：

```bash
/usr/bin/python3 normalization/finalize.py --preparation prep/preparation.json \
  --query-output collected/output --query-plan bound/query-plan.json --output reward
```

finalize核完整新查询覆盖、plan/source/policy关联和所有输出成员SHA。原FP16 encoder 的新查询不得用 CPU FP32、截断、漏词或 strict fallback 替代。零必要查询时不得传query参数、不得启动模型。

`reward/rewards.json` 顶层有完整 `rows`8、`pairs`4、`raw_responses`、`original_raw_responses`、`supply_gate_pass`。pair含 `row_id,greedy_reward,sample_reward,advantage,greedy_valid,sample_valid,gold_count,semantic_difference`。优势是未经归一化的sample−greedy浮点常量，主运行器应停止其梯度。

修订后的固定供给门：至少2组非零advantage，至少1组gold非空且两条均合法、有不同概念F1；另一组允许真实格式有效性差异。原proposal保留，由parent activation记录事前修订。此门只检验训练供给，不是质量分数。

## 已做检查与边界

`contracts.py` 只在train32/原87目标及合成响应上做CPU合同；输出在仓库外 `research/patient-set-risk-v1/normalization-cpu-v1/`，短指针见 `cpu-evidence.json`。验证完整身份/缺条/重复、空gold、复合和NIL、818原子保留、不合法JSON/EOS/cap、部分坏item整条奖励0、格式+语义供给门、一个实际train原文新quote被完整请求及其缺证据拒收、动态12成员包的source/时钟绑定。

合成 `generated_ids=[1,2]` 只是接口测试，**不是模型输出**。没有GPU、模型、远端、fit、held标签或v2调用。远端 GPU runner 此时仅静态/编译检查，实际FP16新增query须由父主任务执行并核完整结果。所有临床原文/合成响应均在外部产物目录，不入Git。
