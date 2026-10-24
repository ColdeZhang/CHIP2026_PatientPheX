# 强三公开实体基底上的既有 GLiNER 供给核查

只做输入/来源普查，不生成新完整预测、不读取新金标、无 fit/NN/原dev。sequence 历史记录与 ontology 负责人交叉核查，尚未做“当前三公开 HGB + proper-nested，仅补旧固定 GLiNER 已选非重叠实体”。早期 CRF/Sap + GLiNER 或再加 Pheno 的组合不等同于该控制。

供给严格定义为旧 `augmented.jsonl` 减去原 CRF/Sap `baseline.jsonl` 的实体对象；不能把整个旧 augmented 当 GLiNER 新供给，否则会重新加入当前 HGB 已拒绝的 CRF 实体。两份旧固定筛选的 assembler SHA `6d154631…`、Sap 概率 ≥.65 / cosine ≥.92、否定及 GLiNER 分数排序政策完全一致，旧增量自身互不重叠。

| 目标折 | 旧 GLiNER 增量 | 当前强基底已有同 key | 其他重叠阻断 | 非重叠新 key |
|---|---:|---:|---:|---:|
| F0 | 26 | 21 | 2 | 3 |
| F1 | 69 | 44 | 5 | 20 |

F0 实际训练 48 篇（F1+F2+F3），1744 窗/4360 步；F1 实际训练 32 篇（F2+F3），1150 窗/2880 步。实际训练窗口 pmc 集、训练/链接 manifest 和目标输入 SHA 都核对一致，分别与目标 16 篇交集 0。两份 HGB 强基底各训练 48 篇，因此 F1 是训练资源规模不同的附加供给，不能说是同规模替换。模型、过滤与 F1 选择均为历史开发产物，未来增量也不能称新盲测。

只剩 3+20 个机械可追加对象，足以支持廉价 CPU 供给控制的可行性，但不证明质量，也不支持为此重训四折。当前没有生成/评分该控制；不补充被新基底碰撞阻断的 7 个对象。

源码 `research/check_gliner_strong_pool_supply.py`（标准库），命令：

```sh
python solutions/sequence_crf/research/check_gliner_strong_pool_supply.py --root /workspace/patientphex-runs/round2-20261023 --data /workspace/patientphex-data/round2-inner4 --output /workspace/patientphex-runs/round2-20261023/research/gliner-strong-pool-supply/report.json
```

外存报告保留每个旧增量与所有新基底碰撞 key；Git 小索引见 `evidence/round2/gliner-strong-pool-supply.json`。未核 F2/F3 现成模型，不声称四折供给齐全。
