# 全文SFT：独立indexed-head正式版本

已获root8400+TERM30授权，原formal_train及部署原样保留。新源码仅将原full-logits loss调用替换为实际283秒/12步PASS的indexed_head原字节函数；原124×2/7170监督/32updates/warmup2/seed/优化器/顺序不变。新namespace/fingerprint/私有SHM及.100持久目录，逐epoch完整Adam/scheduler/RNG出口ACK后才能续第2轮。原F1两端点与Score-only强基线保持。尚未启动，不能称质量通过或显著提速。
