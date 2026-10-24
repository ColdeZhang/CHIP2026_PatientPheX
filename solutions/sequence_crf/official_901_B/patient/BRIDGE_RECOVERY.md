# Patient B 原包传输恢复

2026-10-24：原 420 条 NN 成功后的 A100 collector 已冻结包；第三个 128 KiB `read_file` 与后继只读库存分别返回 UNKNOWN。原失败收据及前 262144 字节保留。按 root/parent 的单 owner 授权，仅桥接既有 ZIP，不重打包、重推断、修改 codec 或评分。

`D=/workspace/patientphex-runs/round2-20261023/research/official-901-B/patient`。

| 完整产物 | 字节 | SHA-256 |
|---|---:|---|
| `D/inference-collection/patient-complete.zip` | 897210 | `1df686ceaaf69e55913198482494533ede264dad2effa4561dd4ed4da9593e36` |
| `D/bridge-recovery/transport-receipt.json` | 3356 | `2614f79ecb63ccab42a68922e36b03bbb939ed148c15f03b0a1669044ab78d00` |

源 A100 mount `476f81cf-09e7-445b-a7f0-d0e12b12e50f` 当前私网 IP `10.5.3.12` 经实际 compact 库存确认；原包位于项目 `sequence-round2/official-901-B/patient/complete-collection/patient-complete.zip`。新自有 HTTP 仅绑定该私网 IP，限 `.100` peer 和临时认证，仅服务该不可变 ZIP，TTL 180 秒。唯一实际进程 `proc_f4827a2a-88ba-49ea-af55-42de6758f14d` / PID 251759 于 07:49:48.232Z 启动。

`.100` mount `5a9955da-5240-41cc-9e52-05d3d8da8a3c` 的实际 IP `10.5.2.100` 也已核；新副本为 `sequence-round2/official-901-B/patient-bridge-recovery/patient-complete.zip`。一次 fetch 2.06444 秒完成，全 SHA、文件及祖先 fsync、重新打开校验通过。随后停止唯一自有 HTTP，实际 `/proc/251759/stat` 已不存在。`ended.json` 未产生，不宣称优雅退出；实际停机证据为 `cleanup.json` 的原 PID 不再运行。临时 token 已删除，不在长期 refs 中。

从 `.100` 按 128 KiB 分块补齐余下 635066 字节；原两块逐 SHA 复用不改写，全部七块合并后 whole SHA 精确，文件与祖先目录 fsync。原 `archive-binding.json` 及 member-index SHA `643557e709cea3f11a54ce4b9efd7a4084099d57fb353788c0d5ecaae1cb49d1` 保持。收据绑定 source/target、原包、原 prefix、真实停机及所有脱敏阶段收据。

实际命令为新入口依次 `deploy → launch → fetch → stop → download → finish`，均传 `--run-dir D`。原 `collect.py` SHA `0539c9ad…`、infer `5b899375…`、driver `46abb835…`、export `caf48c8d…` 均未改。新源 AST 与 `git diff --check` 通过。原包已交 parent 继续其独立纯本地 `finish_local_collection.py --run-dir D --transport-receipt D/bridge-recovery/transport-receipt.json`；成员解包及 exporter 验证由该后继记录，不冒称传输入口已完成解包。

本次传输已结束，不重跑 `launch`。剩余 archive 均可通过既有 SHA 库存恢复；凭据不恢复、不重建，不再访问原 NN 或 GPU。
