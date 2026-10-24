# 严格 OOF 后继：有限资产盘点

这是只读可行性盘点；没有新 pipeline、fit、NN、predict 或质量调用，当前固定20步配方仍负结案。

- 已有严格 F3→F2 / F2→F3 SapBERT／规则候选：`ontology_rules/ce-oof-control/pool3-to2`（50489f7a，517593 B）和 `pool2-to3`（e38d03b2，528217 B）。原池逐字节 SHA 与实际16篇 ID 数量已核；原生三公共 train64 缓存也齐，均未用任务标签训练，可按训练折取固定来源。
- 对应 train16 CRF 模型／输出尚未在现有目录找到，sequence owner 亦无已知入口；常用 `nested-outer1/inner2/3` 是 train32，不能代替。已有 a465 F1 selector 用48篇拟合，现存 LightGBM fitted catalog四折均48篇，未找到匹配 train32 的旧公共基线。
- `train32_prepare/prepare_events.py` 与 `native_cpu.py` 可复用映射／分词／保护逻辑，但输入身份和32篇断言需按各 train16 独立绑定，重新建立正例、NO、固定负例、冲突与 unsupported 保护；不能直接切分既有1252行，因为跨折 query 冲突会改变保留集合。此次没有实现这些修改。
- 原 train32 CPU准备实际18.536317秒（native14.164822秒）；两个 train16 的数量、时间均未测。约37秒仅是两次旧全32准备的保守成本参照，不能代表完整 OOF 成本；新 CRF16、H5两方向和两个匹配 selector 仍须单独计入。旧train48 LGB约13秒同样不是新完整控制实测。

若后续获授权，先补齐严格训练谱系，再让新／旧公共来源 selector 使用相同 train32 OOF 训练资产；匹配弱基线的提高仍不足以替代当前强 a465。完整已核引用在外存 `research/public-h5-train32-formal-v1/strict-oof-feasibility/inventory-v1.json`（8c03ba6e，7423 B），不读取原 dev gold。
