# 固定 Sap 的 A30 排程绑定

root 为避免 .3 长关系 GPU 等待，将本次必要 Sap 移到 .100 GPU0（126293b9），仍只原新39完整响应产生的必要查询，180+TERM30、launch TTL300。原 .3 Sap 源保持未执行；新目录没有绑定实际 query plan，也没有 GPU 动作。

`host-contract.json` 复用 QSC F0 已实际成功的同 public 模型/41099 bank 资产。10个外部文件的大小/SHA与原 G 全相同；query/driver/admission 数值与执行源码字节不变，仅 remote/collector mount 改为 .100。真实运行版本从 Torch2.7.0+cu128/Transformers4.50.0 改为已验证的2.7.1+cu126/4.51.3，明确为host/runtime迁移，不宣称跨host逐位数值等价。原 FP32 CLS_L2/max32、train32 aliases、b59政策及cut不变，不复用 QSC 的27条查询结果。

全39收件与 preparation 完成后运行本目录 `bind.py`，生成 `held-query-plan.json`；提交推送后沿 `held_remote.py pack/deploy/start/observe --directory B/F1-Sap-v1`，pack传实际plan，完整后 `held_collect.py`。这里B为外存`research/training-seed-ensemble-v1`。后处理继续原helper `postprocess.py normalize-bind`，同一本次collection路径，无质量/source变更。fresh卡/磁盘/逐资产SHA门仍在真实driver内。源码准备不代表资源已经可用。
