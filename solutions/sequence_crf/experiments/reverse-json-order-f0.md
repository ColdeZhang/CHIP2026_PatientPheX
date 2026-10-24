# 固定逆序关系并集：唯一 F0 迁移

F1预设old∪reverse主臂小增且macro不降，经独审baf7aec6通过；root授权一次F0同方法，不重评reverse-alone，不使用public新concept，不原dev。

实际原58提示/415概念、旧SFT48 epoch2。56提示变化/2单概念不变，所有字段及user双逆byteexact；原195355→195356 tokens、max6040、输出1600预留后7640，无截断。旧同58 runner225.733秒仅成本参照，固定新推断600+TERM30。提示/方法/原parser不变，只逆序并同步CID/code_map；主臂完整保留原实体/关系/所有非association字段，literal-v2关系并集，不加intersection/支持筛选/阈值。

旧SFT48持久副本在.100 `case-retrieval/qwen35-preparation/dev-relation-transfer/adapter-epoch002`，weight8e32daee/config165dbc59；已实际完整SHA重读。仅两文件173182359B临时复制到A800 `/dev/shm/patientphex-sequence-sft48-reverse-f0-1045`，目标uid1045、独立0700目录，SHM实余250.996GB；要求size+64MiB、完整SHA回读后停临时服务。180s传输上限，沿H33兼容Python3.8字符串切片；保留持久原件，SHM为可再生临时副本。A800小输出盘193.798MB不用于权重，部署仅小源码/提示/输出。基座复用既有已核SHM，旧F0与新F0同A800/private runtime。

先source/plan push，再传输/部署并fresh核GPU f516空闲启动。原baseline .695443948指标直接复用；reverse原始响应与两完整pred保存并共同freeze，只有union做一次F0 v2。keep仍Score严格升且macro不降。原始训练48/旧204严格OOF来源只读复用，不重训。无额外quality臂。

恢复：外部 `research/reverse-json-candidate-order-f0/formal/` 的plan/job/transfer/terminal/assembled/evaluation；源码 `reverse_json_order_f0/`。计划 `54f05699c7848bf7d2aba196e8da2e4b08cd0b141b0f3f4a26d82f086c149c61`，小索引 `evidence/round2/reverse-json-order-f0-plan.json`。
