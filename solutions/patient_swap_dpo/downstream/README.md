# 固定 DPO 后继推断与单质量端点

此目录只提供准备好的源码，模板默认 `enabled:false`。准备阶段没有远端调用、模型、heldgold读取或评分。父线程负责实际完整训练、持久ACK、启动/回收和统一Git保存。

`infer.py` 复用旧原生Qwen3 .3 F0加载流程，改为新的 **DPO epoch1/global10** 身份；原strict codec和整个raw-first生成循环AST相同。固定完整F1 80行、666591 tokens、最长14322、拒绝上限14400、greedy/nonthinking/max_new1600，1200+TERM30。目标Torch2.7.0/TF4.51.3/PEFT0.15.2。基座引用父正式阶段的SHA/stat库存，不重新整包传输；新adapter、训练state和ACK在服务器逐件重开检查。实际80 Auto/native IDs在模型加载前检查。

额外部署输入仅两件，精确身份在 `inference-template.json/local_inputs`：原`held.jsonl.gz` 456576B/d4ec43b0；仓外新压缩原full80 prompts 415819B，解压后仍2a95bad1。不读取gold，不改变提示。

训练真正完成后绑定推断（全部参数为实际路径）：

```bash
python bind.py infer --template inference-template.json \
  --formal-complete LOCAL_FORMAL_COMPLETE --persistent-ack LOCAL_ACK \
  --verified-assets LOCAL_FORMAL_VERIFIED_ASSETS \
  --remote-verified-assets REMOTE_FORMAL_DIR/verified-assets.json \
  --held-encoded-remote REMOTE_HELD_GZIP --prompts-gzip-remote REMOTE_PROMPTS_GZIP \
  --run-root REMOTE_RUN_DIR/infer80 --output inference-plan.json
```

父线程将完整目录、两个小输入、已绑定plan按实际SHA部署，唯一启动 `python driver.py --plan inference-plan.json`；driver的run_root必须尚不存在。此目录不含自动重试或GPU调度器，未知状态恢复同intent。

完整回收后绑定唯一质量端点：

```bash
python bind.py quality --template quality-template.json \
  --formal-complete LOCAL_FORMAL_COMPLETE --persistent-ack LOCAL_ACK \
  --native-directory LOCAL_INFER_NATIVE --driver-result LOCAL_DRIVER_RESULT \
  --run-root LOCAL_NEW_QUALITY_DIR --output quality-plan.json
python quality.py --plan quality-plan.json assemble
python quality.py --plan quality-plan.json evaluate
```

装配先核完整80 raw/IDs/strict解析、完整训练及真实GPU释放，然后固定 **B471 ∪ 至少两票(G,S,Q_DPO,C)**。保B471列表前缀和所有Task1/patient/其他对象；旧四源关系重建须等于B4。只输出一个主预测，不单评Q。完整freeze后才允许1次v2 all/text，基线.701325377212172，记录新增/删除及micro/macro；语义关系集合不变时继承原B4精确文件并0次新评分。`evaluate` 中新intent使用排他创建，不能重复评分。

24项合成合同和全部Python编译通过：见 `ready.json`。内容包括80行587CID完整/弃答/cap/cache、合法多患者和重复键拒绝、替Q新增/删除、compound/NIL、B前缀/非关系字节。合同不替代新DPO实际推断或质量结果。
