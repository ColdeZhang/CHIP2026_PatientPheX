# BioClinical ModernBERT 长上下文 Task1

当前：唯一工程23.737秒PASS/实际GPU释放，root已授权fresh8轮256Adam+完整17窗的共享1200+30；正式源已准备，尚未启动。分支 research/biomedical-long-context，起点main21dca5f3。

目标是检验全文语义能否改善Task1候选，与已有Biomed/BioLink/GP/GLiNER及短窗FLERT区别明确。原dev最佳.769235669942未达.8。只改本路线，0新gold/质量/模型调用；后续GPU以独立实际receipt为准。

- 唯一模型 BioClinical-ModernBERT-base@c3648aa8，149655232存储元素，MIT。ModelScope只找到不同NeuML嵌入模型，故原厂HF回退；原厂权重598635000B已69.771秒CDN下载并全SHA ea738868核验；首次origin config0B网络失败保留。
- 训练32全文122147原生tokens/max7492；F1纯输入16篇39患者62699/max8371，完整17长窗。
- 监督严格复用旧GP：2266正跨度；新tokenizer2265可表示/1词内边界；初census2274多的8项是旧规则排除的非phenotypic ID，v1保留并由v2纠正。
- [固定候选计划](candidate-plan.json)；源码train_census.py；外部原件与短报告见evidence/readiness.json。
- 下一步：原生TF4.50全部ID/offset/监督集合合同、精确资产核查、source push；fresh A30 GPU0门后唯一180+30工程。正式预算等待真实probe，不截文献。

实际资产：assets-cdn-v1/observation2.json已证明原厂完整权重。工程部署原包11成员2.54MB顺序写入，CPU/唯一GPU探针均等待完整部署收据。源码/计划最近已核推e8cf10aa，执行计划8bbdfa83。

工程实际：149112704参数、4FB/2Adam、第2非零lr1.2e-6，136参数全部更新；峰3.9855GB，547state tensors重开/回装通过，完整probe statecc522仅保存不复用。16:32:18.129445 GPU0=4/noapps。正式预算原保守983.559秒→授权1200+30，无重复probe/CPU。见evidence/probe-entry.json及formal/formal-plan.json。
