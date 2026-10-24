# 当前 LightGBM 最佳结果：缓存复现与上游边界

本次只读源码与已有 manifest；未打开模型、预测、候选池、gold 或训练数据正文，未执行模型/评分/远端操作。精确来源与各文件覆盖在 `upstream-inventory.json`。当前 best 为 **0e2c / .748583439566**；旧 .732257765 资源表仅用于追溯未变上游，不能沿用其97列选择器描述当前模型。

## 已有58件包能够恢复什么

包168b0c39、归档288622b9（3,827,082B）已由主线程实际从服务器恢复58件并逐SHA核验。它**直接**包括当前803b模型/103列向量器、0e2c完整预测、a179旧关系模板、baac前序预测、LightGBM4.6固定wheel、feature/nested/literal源码及训练/推断/审查记录。它引用72件a179包的manifest，并不把整个旧包再次收入。

当前58件包不含输入候选18ac（754,726B）、纯input c0fc（294,884B）或官方HPO a509（11,222,341B）。但**c0fc已在上游72件包中**，不是整条恢复链缺失；a179模板又已直接在58件包中，不能重复报为缺项。18ac两个包均不含。

## 即将补齐的最小可执行边界

纯加载入口若实际合同通过，可以从18ac冻结候选池、c0fc纯原文及a179模板出发：加载803b/原103列vec → transform/predict → 固定.3/原proper-nested → literal替换entities → 与0e2c逐字节一致。全过程不需要重fit、dev gold、公共v2或新神经推断。

这仍是**从已生成候选缓存开始**的复现。原full64_dev.py含fit/eval，旧sequence薄入口是97列HGB+greedy，不能把任一原样宣称为当前LGB103/nested纯加载入口。新的owner入口和实际byteexact结果当前待完成；本清单不提前证明它们正确。

## 从新原文重建仍须的环节

| 环节 | 不可由当前模型pickle替代的依赖 |
|---|---|
| 基础候选 | train64本体/词面状态、CRF pipeline8753与crf.modelee0f及迁目录绑定、SapBERT/有序42,575别名bank和编码、Sap类型判别器adb8 |
| 三公开来源 | PubMed/Bioformer/BioBERT三头；原native分词/lemmatizer/POS/词典，旧17,266列到指定HPO的映射，canonical与来源flags、top3/严格>.95政策 |
| 关系候选 | 独立CRF+Sap64+GLiNER48旧池。GLiNER不在当前Task1公开LGB池；不能拿新实体集合替换关系候选 |
| 旧关系与逆序 | 固定Qwen3.5-9B、full64 LoRA1a703b24/configc4a478a7，同一adapter的两次提示/解码及literal并集 |
| 35条公共新关系 | 原97列HGB db2fc288的实体支持政策及新概念SFT/装配。当前被LGB替换的是Task1的103列42e，两者不能混淆 |

这些阶段有各自成功运行与来源记录，但本任务没有重查远端大权重存活，也没有跑一条从新原文到最终预测的全链。58件包内只有LGB wheel；完整Python/sklearn/Numpy/Scipy与神经runtime也不是包内自包含环境。

## HPO恢复及补包建议

`/workspace/patientphex-data/hpo-source.json`固定官方 `v2026-06-23/hp.obo` URL、a509 SHA与11,222,341B。`official-download-manifest.json`还证明本地官方A压缩包4a5b3347（2,859,663B）含同字节HPO；官方目录与单独OBO均有已登记位置。本次仅检查12份现有artifact/day1提交备份manifest，没有找到a509成员；这不是对全部服务器存储的否定，也没有远端确认。

最小外部补包应保存新入口/固定调用/真实合同及独审、缺失18ac；c0fc可以沿72件包恢复而不重复复制。如需避免未来外网下载，另收**HPO原件本身**并压缩，不能为它把含训练gold的整个官方A包加入。a179模板、803b模型、wheel和旧完整包不必重复打包。权重、候选数据与OBO继续留Git外。
