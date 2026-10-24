# 最长全文输入诊断源码（尚未运行 GPU）

目标仅是确认已完成 CPU 审计的最长训练样本 `12628480:relations:8`（完整 12,824 / 前缀 12,754 / 监督 70 tokens）能否执行原生更新与实际 cache 解码。当前只有源码与 CPU 输入合同；尚无显存、耗时或质量结果，也不构成新正式训练授权。

`probe_fullcontext_longest.py` 从原 `probe_longest.py`（`0a9828c4…`）适配。原生载入、同 logits 的 assistant CE 对照、反向传播、语言 attention/MLP LoRA 实际更新、视觉禁用、base/head 冻结、两 token 原生 cache 观测和终点的 **5,562 字节完整段保持原样**（SHA `aea45fdd…`）。未来诊断只加载固定 fresh base，新建 LoRA/AdamW；不加载当前 Med adapter，不保存诊断权重，不将两 token 输出当关系质量。

新增 `fullcontext_contract.py` 仅绑定输入：原 strict32/124 的训练和候选隔离合同不变；读取纯 train32 原文，按已审 constructor 重建全文追加，逐行还原原始消息、CID/PID、候选和答案。必须匹配 CPU summary `5d295ec5…`、行 token 审计 `80f5bbce…`、全文数据 `9e80e541…`。未来诊断原生编码后，再逐行核 token SHA、监督 SHA、完整 prompt SHA、掩码和原生结束符，按全部 124 条的实际最长选样；不选短样本、不截断、不读取 F1 输入或 gold。

**新诊断上限固定 14,000；原正式 trainer 上限 8,192 保持不变。** 新值只是长度拒绝阈值，不要求 padding 至 14,000，不代表 GPU 能装下。未来若诊断失败，不自动换短行或重试。卡、绝对截止与运行预算由主线程后续决定；本目录没有 MCP launcher。

`cpu-contract.json` 记录本地 13 项输入/源码合同通过：包含真实 124 行还原、目标/CPU 收据变更拒绝、小构造的 token/掩码/结束位置反例、原数值段字节一致。没有再次运行 native tokenizer、hash 权重资产或导入模型库。已有 native CPU 结果直接复用，原探针的资产载入保护保留在待运行源码中。

执行接口与依赖见 `source-plan.json`。小文件由 owner 统一备份；全文数据、逐行审计和现有 trainer/interface/constructor 使用已冻结外部引用。后续正式训练如获授权，需要独立明确数据与容量配置，本目录不提供新训练框架。
