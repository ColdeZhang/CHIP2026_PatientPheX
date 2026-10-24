# 患者交换偏好训练台账

目标：在固定四源融合中替换Qwen3关系组件，提高患者归属质量。原train64/dev16目标≥.8；本线先做train32→内部F1固定控制。

- 分支 `explore/patient-swap-dpo`，从main `979e9492` 建立；仅本目录。
- 已完成：74成对样本/17篇、1149合法swap供给；完整原生token CPU1.917s；原124chosen与缓存IDs/mask全部exact。14个选中样本为多阳性，其余患者正例保留。
- 已核原B4关系集合严格等于B471∪四pure中至少2票，非关系对象及B471列表前缀保持；重序列化对象列表顺序不同，不称byteexact。
- 固定：Qwen3 train32 epoch2适配器543c82a5；β.1/LR1e-5/1epoch74pairs/10updates，新Adam；单完整主终点，不读取heldgold/不评分。
- 当前：**固定DPO配方已负结案**。唯一F1 `.701325377212172 → .7009053889090149`（Δ−.000419988303157），80/80严格/0弃权、预测18c067c9。1newv2/0old；不进F0/full64/原dev/官方，GPU已释放。最终报告/备份进行收尾。
- 仓外 `R/research/patient-swap-dpo-readiness-20261024` 保存数据、完整public源码与CPU产物；候选协议见 `candidate-plan.json`。

- 实际.3 tiny CPU通过（0.245s、28个LoRA梯度max1.30e−8、真实Adam），两原件完整回读a52ea3d/563b93d0；首只读guard异常后恢复原件，没有重跑。root已授权固定ref900/probe600/formal1800/infer1200各+30，真实GPU尚未启动。执行模板 `runtime/` 保留原方案预算历史，13压力对/两Pareto/26FB计划固定。

- GPU前增量源码检查拦住共享checkpoint所需durability_root漏项；v2 be5ac193已显式绑定，v1 disabled原件未部署并保留。reference缓存plan/data/parent三项身份核补齐。下游24合成合同/strict codec与raw-first循环AST复用通过，单质量默认disabled。下一步control.py --directory R/research/patient-swap-dpo-v1/deployment-v2 deploy→cpu→collect_cpu；actualCPU后才activate→start。

- 目标.3 CPU实际14:16:14完成4.580s：148全HF4.51.3/tokenizers.21与raw、IDs、完整prefix/EOS双解码一致；519954tokens/9344监督/max5534，parent543c重核，13base仅stat继承cd37。原CPU全件6bd1019b已回收。execution-plan已冻结并上传，readback进行中；尚未发送GPU start。

- 唯一core已14:18:46.52103044接收：proc_561bb090 / driver1216102，plan965b322d。首次activation只读失败保留，同路径byteexact恢复后才start；没有重复上传/start。原ref900→probe600→formal1800链正在运行，已加任务子timeout保留父控制器退出后的限时。只读watch源和显式下游阶段CLI已准备；实际质量仍0。

- 14:21原core停止：reference133.745s成功，probe12.449s首checkpoint backward metadata错位；原chosen两replay差0，GPU已释放。109成员/ZIPfa6c081a回收。真实CPU BF16+checkpoint0.355s：cache=True梯度门失败（非同一CUDA异常），False两replay差0、28梯度/Adam通过。只禁autocast cast cache，保原阈值/配方；新application/run-recovery-v1，沿原TERM14:31:00.518905+30，新formal仍fresh543c。详见evidence/probe-failure-recovery.json。

- 恢复attempt已14:25:48.723319接收proc_4c647956/driver1216597，plan34cea8c6，6件全部SHA/fsync；7052a9a5已推送核远端。唯一watch182584记录deployment-recovery-v1/core-watch；原参考未重算。probe仍须13完整对/26FB/真实Adam/成本全部通过，才fresh正式。下游collector显式继承原reference并核原绝对deadline；尚未正式/质量。

- 14:27:04实际GPU工程门通过：26FB/26coefficient forward，504梯度/504Adam状态/1512tensor finite；峰allocated20.672GB/reserved21.987GB，成本估1040.034s≤1800。fresh正式14:27:05.278767，timeout父1216698，原probe状态丢弃。证据probe-pass-formal-start.json；后续完整ACK再绑定80fresh推断。

- 14:36:21正式终结并释放4MiB/no apps：74/148/519954tokens/9344监督，FP32 LoRA504、Adam504/1512 tensor及所有RNG回读完全一致；完整adapter75518d93(174655536B)、state d31283d6(349612077B)，无恢复旧Adam。完整小终态ZIP全成员SHA/安全路径核验后回收；原reference三件再次SHA绑定。inference-plan按实际epoch1/global10启用80fresh，1200+30，未发送start。

- 推断首次start前补H64：旧driver只由控制器限时，现加native自身GNU timeout1200+30；旧68b69118只到部分源部署，MCP只读失败已退出，实际未发送infer start。保旧intent/source/plan，切deployment-infer-v2及远端downstream-bounded-v2。新计划2e17afa5，模型/80输入/生成/codec/quality.py/主端点不变。

- 部署原07写接口失败后先读同路径取得NOT_FOUND，才进行一次有记录的同内容重试；456576B完整回读一致，原intent/失败/恢复receipt均保留。其余已核receipt由最终远端全SHA+fsync统一再次复核，避免恢复时重复所有读取。sole watcher已扩展infer模式，仍不触发任何start。

- 08写结果未知后严格只读恢复：原read_file返回UNKNOWN_OUTCOME，30s后同路径纯CPU读取全部415819B并核3ce16f96成功，原文件已经存在，未重写；该receipt明确SHA验证而非全量字节回传。全部10件最终远端bytes/SHA/fsync及父目录链通过，plan2e17afa5就绪，模型计算仍0次infer。

- 唯一80fresh14:46:00.677696接收，e6965a9d已推送核远端；原始字节/IDs先存再解析，driver+native独立1200+30。watch184607已核真实PPID1/SID/start_ticks和独占，无第二查询器/重启。下一步watch终态→stages.collect_infer→bind_quality→assemble冻结→唯一evaluate；相同语义继承原B4并0新v2。

- 14:56:39唯一完整主端点负：保护B471，四源两票最终524关系，+2为1TP1FP，−2为2TP0FP；micro−.0019157088，macro+.0002357556，Task1差0。严格80/0弃权，全269成员/ZIPf310dc23已SHA回收。新Q387关系(旧416)，不能据数量变化或单臂未评分推断Q自身精度。依预写strict positive门关闭固定74pair配方，不扩展验证；原dev未读取。
