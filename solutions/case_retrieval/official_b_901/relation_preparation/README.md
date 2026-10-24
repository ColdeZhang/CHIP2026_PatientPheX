# 最高 901 方案的官方 B 关系输入准备

目标是在同一 100 篇 B 输入上，保持已接受开发版本的旧正序、旧 reverseJSON 和 public-new 三臂。根输入合同 `2c169066…` 固定 100 篇、244 患者、3802 段落；本目录不训练、不选择阈值、不评分、不上传。

当前为源码和依赖准备。`plan.template.json` 的旧候选与两个公共缓存尚未绑定，因此不能执行；未产生实际 B 提示、token 统计或互斥证明。完整路径和 SHA 见 [inventory.json](inventory.json)。

## 必须保持的政策

- 旧闭集来自原 CRF/Sap/LR train64 + GLiNER span48 的 `full-lr`，开发文件为 `c1c2e508…`（955 实体、466 文档概念）。它不是 HGB/LGB 筛后的实体。B 对应物须包含原 LR fallback 关系；任何新增拟合都不在本任务范围。
- 正序沿原 `prepare_pair_rag.py`：排序概念、每批 8 个局部 CID、500 字符窗口、最多 4 证据、原患者 anchors/SYSTEM、无示例。逆序只在原批内反转 JSON 对象并重新分配顶层 CID，反转两次必须恢复所有字段和文本。
- public-new 只取固定 PubMed 与 Bioformer 的 `>.95` canonical 缓存，先按 PubMed→Bioformer 合并去重，再排除**全部旧 full-lr 文档概念**。HPO compound 拆分后按原 HPO alt_id 映射；`-1` 仍保留 typed text。BioBERT 仅在父任务 Task1 链使用。
- 推断使用旧 full64 adapter `1a703b24…` / config `c4a478a7…`，Qwen3.5-9B revision `c2022362…`；BF16、SDPA、greedy、batch 1、max input 14000（超限拒绝）、max new 1600、no thinking。不改为新的全文 SFT adapter。
- 正序/逆序沿原 `parse_json` 和逐 CID 合法 PID 检查；缺失、非法或 cap 的候选回退原 full-lr 对应关系。public-new 此类候选弃答，不补旧关系。新 LGB 仅由父任务用于 Task1 和 public-new 的支持筛选。

## 输入与输出

`prepare.py` 只载入 SHA 固定原源中需要的纯函数，避免导入旧训练类。绑定 plan 的所有 ref 均为 `path/bytes/sha256`，需要根输入合同、原 HPO、四份固定来源源码、精确 B old full-lr/来源清单，以及两个 canonical cache/清单。最终绑定时还需按 sequence 实际清单 schema 核实其 input/output SHA 关联；当前模板不假装该前置已完成。

```bash
python prepare.py --plan /absolute/bound-plan.json --output /absolute/new-preparation-directory
```

入口限时 60 秒，输出三个 `*.prompts.jsonl`、public-new 实体、逆序映射和 `candidate-space.json`。空间文件逐篇保存旧原始 identifier、原子 HPO、canonical HPO、观察到的 alt 映射、两个 public canonical 集合及三臂 code_map。它首先证明 old/public 差集；sequence 还须独立检查患者模型合法候选及实际正关系的交集，方可使用父任务的省略旧 HGB 等价表达式。没有实际 B 空间证据时不能据源码声称已互斥。

随后每臂只做原生 tokenizer CPU 清点：

```bash
python token_audit.py --prompts /absolute/normal.prompts.jsonl \
  --tokenizer /absolute/verified-small/tokenizer.json \
  --template /absolute/verified-small/chat_template.jinja \
  --output /absolute/normal.token-audit.json
```

固定 tokenizer `5f9e4d49…`、template `a4aee8af…`；记录每行 render/IDs SHA、往返、token 上限和总量。此原生清点不声称已实测 GPU runtime AutoTokenizer。后者由推断入口在模型加载之前逐行核对相同 IDs、render 与实际解码；不相同则停止，不静默换 tokenizer。

Qwen 发布 normalizer 为 NFC：分别记录逐字符往返和发布 normalizer 下的往返，后者作为合同；不因合法 NFC 归一化误判输入损坏，也不手工改提示。实际 Auto 解码必须与原生解码 SHA 相同。

`qwen35_runner_raw_first.py` 由旧 `33a5a497…` 作机械修补：固定旧 adapter、加入 native audit、先 fsync 生成 IDs/EOS 再解码、响应 fsync，并拒绝带孤立 IDs 的隐式重生成。原 `from_pretrained` 与 `generate` 调用经 AST 比较相同，decoded response 仍为原 `skip_special_tokens=True`。输入参数为 `--model --source-manifest --adapter --prompts --native-audit --output`；GPU 用 `CUDA_VISIBLE_DEVICES` 绑定，Python/runtime 路径由父任务计划给出。必须先有真实 B token 预算及父任务运行分配；本目录尚未启动模型。

旧开发运行只供成本参考：正序 64 行 A30 341.386 秒，逆序 64 行 A800 230.897 秒，public-new 42 行 139.727 秒。B 耗时须按实际提示与 token 清点安排，不直接按 100/16 宣称固定倍数。

静态检查见 [static-check.json](static-check.json)。它只证明源码准备与限定调用保持，不是实际 B 准备、运行或质量结果。最终支持筛选、患者组合、输出格式校验和一次官方提交由主线程统一处理。
