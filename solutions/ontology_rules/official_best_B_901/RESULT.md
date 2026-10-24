# 最高分901官方B三public进度

当前固定模型输入推断，无gold、拟合或本地质量评测。三源单GPU任务proc_f3f4ef06，根输入878f/合同2c16，batch plan6fc113a8，顺序总2400+30。原dev18ac候选池重建压缩字节完全一致，103列transform由retrieval保持原803b。

| 来源 | native秒 | 来源归因秒 | 文献 | canonical | 状态 |
|---|---:|---:|---:|---:|---|
| PubMedBERT | 338.456953 | 5.185938 | 100 | 8929 | rc0，完整回收 |
| Bioformer | 271.310108 | 5.384167 | 100 | 8726 | rc0，完整回收 |
| BioBERT | 334.466832 | 5.384242 | 100 | 8723 | rc0，完整回收 |

PubMed小包7df8a846/383539B本地逐成员SHA通过；全部raw预测仍在A100持久project/native-B901-cache，远端重开核验全manifest SHA。最终canonical fbcadc36、manifest5ef5e971已交retrieval逐源构造关系提示。只提交根线程最终冻结的最高方案，本文不代表官方已上传或得分。

Bioformer小包bb230a70/372908B全部成员核，final manifest4d1ea0d/canonicala38a4766已交关系提示线程；Task2两公共源已齐。

三源整体970.229908秒rc0，06:50:47.914198 GPU2实际0MiB/40339MiB free/noapp，已交sequence patient。BioBERT包e62052d9/374043B全成员SHA，manifest6e920d5f/canonicalf66e1f45。原runtime目录终点后只读metadata确认为TensorFlow2.12.0/Keras2.12.0/Transformers4.30.1，未导入模型。Sap原完整包续传后再构建Task1池；不把半包视为可用数据。
