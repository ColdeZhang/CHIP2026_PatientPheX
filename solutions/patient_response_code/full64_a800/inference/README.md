# A800 Full64 两ACK后的纯64推断

`bind_after_training.py` 只在新 namespace 的原5400秒 campaign和两真实完整持久ACK/global34、68校验后生成包。入口为相邻 controller/entry.py 的 `bind-inference`；然后主 owner 依次执行 `operations.py deploy/start/inspect/collect --directory <D>`。未知结果先检查原intent，不重启模型。

沿实际 .100 native329 的 held64输入与SHA、16篇/36患者/466CID/1006对，523821输入tokens、最长13425，上限13824；64条全部fresh，gen1600，1200+TERM30。A800 Torch2.7.1+cu126/native TF5.20.2，公开base和本次epoch2 adapter完整核SHA，全部实际64 input IDs再匹配原CPU账后推断。

raw-first、原EOS `[248044,248046]`、整行strict弃答、无repair/重试/旧cache不变。`assemble_component.py` 生成纯关系组件；所有非关系对象保持原16/36底座，以原始完整canonical为解析出处。只交root已预锁组合，不单独调用originaldev evaluator，不读取originaldev gold。

远端独立目录 `patient-response-code/full64-a800-v1/originaldev64-inference-v1`，GPU2实际fresh gate≥70000MiB空闲/≥80000MiB总量、无app，项目盘≥512MiB。原A100推断目录不会使用。
