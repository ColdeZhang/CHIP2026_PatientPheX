# Span normalization 工作台账

- 目标：为强实体系统提供新的高质量精确跨度及HPO候选；最终仍按原dev Score≥.8验收，不把弱单路提高当完成。
- 当前：E-20261024-SPAN-001源码与事前计划完成，尚未运行CPU实际编码或GPU训练。主协调统一首个32(F2+F3)→F1，原48→F0草案在任何执行前替代。
- 方法：BiomedBERT+GlobalPointer显式start/end、RoPE64、multi-label categorical CE；保留嵌套。原成熟Sap/fold-local aliases负责规范化，暂不同时加联合概念loss。
- 固定：8epochs、3e-5、batch16、384/stride64、BF16 encoder/FP32head+loss；原始池logit≥−2，预测>0，不扫描cut。完整预定义池留外部，便于以后学习融合。
- 对照：强F1 .728697536469，训练规模不同，仅作部署基线；GP-only诊断和固定强基线追加臂共冻后各1v2。细节见[计划](plan.json)。
- 资源：.3 GPU0由root排在document_relations之前，≤600+TERM30含一次丢弃probe/8epoch/fullheld/linking；先CPU≤120秒实际编码/数学合同。真实start及释放随后记录，不以旧空卡快照替代新门。
- 数据/权重：仓库外R/stage8-span-normalization，.3现有Biomed/Sap公开权重只读；任何监督词库仅从新train32重建，排F0/F1。
- 最近验证：源码compile通过；成熟方法来源已实际联网获取。尚无本路线新分数。
