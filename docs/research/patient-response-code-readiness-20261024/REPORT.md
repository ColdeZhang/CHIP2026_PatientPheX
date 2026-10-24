# 独立患者响应编码：CPU准备完成，值得一次固定控制

**尚未训练或评分。** 唯一原生CPU检查18.607秒通过（wrapper19.274秒，CUDA未初始化），不是质量提升证明。建议保留为一个可证伪的表示控制，先做一次有界真实梯度压力，再决定完整训练；目前没有GPU/fit/v2授权。全部产物在本外部目录，原patient四轮分支未改。

## 机制与查重

原normal任务仍是CID→患者列表。训练使用旧SFT32短124行`cb7f67e1`（F2+F3，32篇/87患者），held使用既有全文80行`2a95bad1`（16篇/39患者），匹配强normal N `f33a30f1` / Score .676546354662594；不用已负的全文124训练，也不退回held短输入。

按原患者顺序添加双射PT001等输出handle，原O/P clinical ID及身份含义保留。**原user正文整段UTF8、所有已有marker/身份锚点、候选与全文suffix逐字保留，旧renderer零调用。** 只改system两处输出指令并加身份说明，在user尾追加明确的`response_code→original_patient_id`表；原patient_ids、CID-HPO和其他元数据/次序保持。target只改患者值，124条答案逆映射后原UTF8完全相同。这是新的prompt/输出表示；不是所有消息不变、纯语义置换、随机患者顺序实验或已证编号偏置。

旧patient-identity-bridge是BiomedBERT二元pair-CE在原384 token后追加128 token身份原句，完整质量已负，但没有响应code。旧ontology P001修复仅22窗中3个格式失败窗，且改inline/患者元数据/输出键；不是本次同源完整SFT控制。准确12项来源见`delegated/`，不以名称重复判定机制已试。

## 结构及原生分词事实

- 训练124/32篇/87患者/880CID/2506候选对，held80/16篇/39患者/587CID/1557对，全部保留，训练与held文献不交叉。训练目标仅取原冻结answer，无新gold、标签扩候选或held标签读取。
- 输入官方患者清单与每文档原patient_ids顺序完全相同。训练255条mentions中6种文本在不同患者间共名，held162条中2种；真实同名身份不合并。合成多患者编号、同名不同病人、重复mentions及逆映射通过；12项未知code/原PID误输出/重复/缺CID/多CID/类型/Markdown/别名碰撞等拒绝合同通过。
- 原生发布tokenizer、AutoTokenizer/chat-template和decoder逐行完整IDs/往返一致。训练prefix/full边界、assistant-only mask/EOS完整，held无target。未截断、未加载模型；实际runtime Torch2.7.1+cu126/TF5.20.2/tokenizers0.23.1。

| 原生实测 | 原表示 | 响应code表示 |
|---|---:|---:|
| 短训练124总token | 402594 | 427560 |
| 训练最大（index6） | 5523 | 5910 |
| 训练监督token / 单行最大 | 7170 / 116 | 8138 / 128（index94） |
| 全文held80总token | 670122 | 685372 |
| held最大（index19） | 14390 | 14570 |

PT001实际是4个token；不称新增单token或更短输出。标准CE公式/每row均值不改，但表示使token监督分布有意变化（+13.50%），不能称7170监督不变。训练沿6144工程拒绝上限足够；held1行超过旧14400，建议新拒绝上限14848、generation1600不变，实际max+1600=16170低于262144模型context。上限是容量保护，不是官方规则，不能因超旧限丢样本。

## 固定解析与后继最小控制提案（未执行）

严格JSON对象必须包含且只包含全部CID；列表只能用本行合法response code，不接受原PID、重复患者、重复键或补缺CID。严格通过后按双射还原原patient_id；不合并同名病例、不修改HPO literal值。EOS/cap或任何解析错误沿原normal全行弃答政策：先保存原始IDs/response及错误，整行不增加关系，不自动补答案、修格式或重生成；全部80输入仍记录并计入完整终点。

最小trainer以原normal SFT32配方为准：fresh Qwen3.5-9B/r16/alpha32/dropout.05，seed20261023，batch1/累积8，每轮15组8+尾4，2epochs/32calls/warmup2，AdamW1e-4/wd.01/clip1，原native SDPA、BF16及完整状态恢复。复用已实证`formal_indexed`的原函数及两次SHM→.100完整ACK链，重新绑定本次数据/encoded/mask SHA与cap6144；不复制旧全文训练SHA或7170断言。indexed head保持该新target的标准causal CE，旧局部数值/真实工程证明可复用，但不声称完整backbone逐位相同。

建议固定primary为新coded-normal-alone，复用旧强N质量作为对照；如root需融合机会，可在任何GPU前另明确唯一diagnostic N∪coded-normal，两完整预测共冻后各1v2、旧0，按Score严格胜强N判断，macro完整披露。没有本次质量调用、第三臂、阈值扫描、中途epoch选择、F0/dev/官方迁移；强N不是所有历史组合最高，public支持F1 .686338981另作上下文。

旧indexed工程最长13549大于新5910，但其12步压力最多70监督位，低于新128；不能称新形状已经实测。建议一次240+TERM30真实工程：fresh同base/r16，先完整8微批非零诊断更新建Adam，再保3条累积（含最大监督index94）后处理最长index6，保存finite/三LoRA族更新/base冻结/峰显存/用时，状态弃掉；正式顺序和warmup保持原配方。GPU、正式预算由root后续调度。

旧相同normal短SFT32 A800两轮1149.460秒，token比1.062×给1220.742秒只是代理；平方长度比1.118、目标位增长/硬件差异和checkpoint也影响时间。建议正式训练含两次约520MB完整持久ACK总3600+TERM30、独立全文80推断1200+30；均为待探针校准的上限提案，不是实测或执行许可。A10040GB需当时fresh资源及真实压力证明，不把旧A80035.123GB reserved直接当新卡保证。

`readiness-plan.json`与`sources.json`给完整实际SHA和边界；`native-terminal/`保存一次CPU原终点，训练/held大JSONL留外部，不进safe review正文。
