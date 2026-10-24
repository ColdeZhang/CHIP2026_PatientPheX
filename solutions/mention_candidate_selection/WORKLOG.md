# 当前状态

2026-10-24 14:44 UTC：独立 explore/mention-candidate-selection，实际 GPU 源 f47a8bcb。唯一 controller proc_791d296d-367e-4dd6-9e63-209793d50c3e / PID2744284 在 .100 GPU1；真实压力探针全部通过，已于13:51:04.487677自动 fresh 开始正式两轮。两轮已14:43:07.960933完整734更新并保存两个ACK；原controller已14:43:08.911899接全1416推断1800+TERM30，child2747622。尚无完整验证预测或质量结果。

固定2931训练行（原2353+578 annotation-negative）、1416纯跨度、±256字符上下文；公开Sap Top8不插gold/exact/训练别名。生成互斥 selected+missing、NIL、NO；empty+missing=false非法，missing不发明HPO或变NIL。新Qwen3-8B/r16/lr1e-4/seed20261023，2epochs/734updates/尾3/warmup37；保原生训练循环，无holdout、早停或挑epoch。正式7200+TERM30，两个完整状态ACK；随后全1416推断1800+TERM30，两个完整端点共冻再2v2/旧0。任一严格胜强B .728697536469才是候选；不涉及原dev/官网。

完整训练：5862行参与、3033240tokens/67044监督，native3116.626s/含load3123.402s/wrapper3124.309s rc0，原complete322547B/ad7adea5。第二ACK1790B/883952ec，finaladapter732e99ba/174655536B、fullstatecdd8e4cf/349612141B，504adapter/504Adam1512张量与scheduler/RNG均重开exact。evidence/formal-complete-observed.json索引原件身份；推断完成前GPU1仍本线占用。

第一轮ACK原件1789B/da53e406，504adapter/504Adam1512张量及全部RNG重开exact；adapter174655536B/4a440385、fullstate349612141B/997d5f45，证据evidence/epoch1-ack-observed.json。仅恢复，不挑epoch；两轮预算和固定终点不改。

实际探针：13FB/1Adam，最大572/17与684/13；四类训练输入受限generate均EOS、未cap/拒绝，只证明运行合法，不代表语义正确。native34.164秒、wrapper34.947秒rc0，峰17.607GB。事前token成本门估4530.200秒≤7200，冷步旧估6885.573秒；不是耗时上界证明。探针状态丢弃，正式从公开权重重新加载。evidence/probe-and-fresh-formal.json保原始收据指纹。

数据与CPU：新增1034查询完整19.589秒；全部4347行native/HF输入与训练mask核验27.215秒，513合法grammar/12293prefix；每epoch1516620输入tokens、33522监督tokens，训练max684/17、held max621；全部51原件SHA回收。122HPO训练行missing（117empty OTHER+5partial），6多选、44NIL、665NO全保留。入口evidence/data-ready.json、evidence/native-cpu-complete.json及experiments/E-20261024-CANDIDATE-SELECTION-001.md。

恢复入口：R/research/mention-candidate-selection-v1/gpu-start-intent.json、gpu-started.json、gpu-latest-observation.json；唯一50秒只读watch与唯一finish_gpu_once在本地运行。先恢复原job，不因MCP UNKNOWN另起训练。完整大产物在外存，仓库仅源码、方案和小证据。

归因：原2353自由名称和旧Biomed Top8逐pair CE负结论保留。本路线候选输入/目标与578监督共同变化，不称纯prompt因果；F1 proposer训练48含F0、head训练F2F3，不能称整个管线pure train32。另一条原2931自由名称在.3按原job独立完成，不受本线结果影响。
