当前仅正式训练源码草稿；未运行或导入 trainer、Torch、tokenizer、模型，未部署、评分或读取 held gold。正式预算及本路线 GPU probe 仍由父线程绑定；实际 native CPU 完整结果已绑定；本目录不包含可启动的正式计划。

收尾时已把唯一成功 native CPU v2 的真实 result `629c2c16…`、encoded `4b503cd2…`、runtime、plan及修复源 refs 填入接口清单，并绑定完整 collection `237aaa65…`。这里只读既有元数据/核文件 SHA，未重复 CPU 或逐数组对照；父线程已完成的对象一致性结论沿用。旧 v1 的 generation-config lookup 失败原样保留。GPU probe 三项 actual refs 和正式预算仍未绑定。

目标是 87 个患者的完整报告输入到可定位词面 target，使用标准逐行监督 token 平均 CE。目标 quote 是概念规范化的弱词面锚点，不宣称人工患者归属证据。原 87 行顺序和完整 native IDs/labels 保留，包括空 `[]`、结构、EOS、换行；每轮 `Random(20261023 + epoch).shuffle`，10 个 group8 加尾 group7，完整两轮 22 次 Adam 调用、warmup2。固定 fresh text-only9B + r16/alpha32/dropout.05、AdamW1e-4/wd.01/clip1、BF16 SDPA、非重入 checkpointing。不加载 probe adapter/state，不提供自动 resume 或短样本训练入口。

索引头路径沿 ontology 已实际使用的 `formal_indexed/train.py`（SHA `3de5ed915a2b01a2abe36bd214ebffc6227774994b369abf4d850292bc45148f`）的模型完整输入、原生 CE、实际 group 平均、Adam/scheduler 和完整 epoch 状态保存顺序。正式训练直接导入与新 probe 相同的 `runtime/indexed_head.py`，强制 SHA `0d28201832721c800696321c6afa1ef7831d63b697620a02290d65317a4428bb`：`labels=None`，全骨干保持梯度，causal 监督位置 q−1 选择 BF16 head，FP32 `fixed_cross_entropy(..., num_items_in_batch=None, ignore_index=-100)`。冻结原 head，不 monkeypatch loss，不使用 alpha=.5 混合损失或任何患者/value 额外权重。每 row loss 外除实际 group 大小，尾7不会再除8。

fresh 模型加载沿新 probe 的 `slot_model.load_fresh`（`e8baf724…`）及原 `qwen35_runner`/`qwen35_relation_cost`/`qwen35_smoke` 依赖，不调用这些旧模块的数据、槽分类或推断入口。所有依赖以实际路径和 SHA 绑定并按 canonical module 名加载，避免 sibling/import 身份漂移。检查实际 Torch2.6.0+cu124、TF5.20.2、PEFT.21.2、native modeling b4b17fe0、loss83db16b2、无FLA与冻结 head；同 new probe，不把原 ontology A800 Torch2.7.1+cu126 当同一 runtime。

数据入口改为本路线 native CPU 的 `encoded.json.gz` 原 `train`87 列表；不再在 trainer 内 tokenize。真实 CPU PASS/result/runtime/plan/source/encoded SHA 链均须匹配，逐 row 对既有 audit 检查 IDs/labels/rendered SHA、prefix、监督位置、EOS和原顺序，核 451672 tokens /11884 supervised /max9210。gzip 内 label-free held input IDs 不用于训练。父计划还必须绑定本路线完整 probe step/complete/manifest；训练器只读既有证明，不重复 head 数学检查或压力 probe。

持久化三函数从已完整跑过的序列 trainer `3eab37084ad45a7e0acde7ce1d1c32fd3023825490f0d87aa76aa4187c1c965a` 原样复制，仅 epoch 步数检查 `16*completed`→`11*completed` 及错误消息 `original124`→`grounded87`。AST 已静态核对只有这两处变化。每轮保留 adapter safetensors/config/README、optimizer、scheduler、Python/NumPy/Torch/all-CUDA RNG、参数名、epoch/step及实际源码/数据 fingerprint；staging 写完后全部成员与深层目录 fsync，rename，沿明确持久根逐父目录 fsync，重新逐 SHA 开启所有成员后才写 exclusive `durability-epoch-00N.json`。终点需两 ACK；目录存在不代替 ACK。状态保存存在不意味着已经检验 GPU resume 数值等价，当前没有 resume 执行接口。

父 controller 负责正式预算、fresh GPU/磁盘门、单次启动/TERM/KILL、终态收件及后续质量。trainer 接口仅 `--plan` 和启动时的 `--absolute-deadline-unix`；完整字段见 `interface.json`。未知预算或任一实际证明未齐即阻断，草稿不替父线程假造正式授权。
