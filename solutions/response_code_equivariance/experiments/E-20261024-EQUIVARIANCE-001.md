# E-20261024-EQUIVARIANCE-001：唯一 PT cyclic1

**目标与假设。** 输出患者句柄可能影响关系生成。保持全部临床信息及已训练 C32 adapter，只换一个明确 PT 映射视图；原/新 strict C 共同支持的关系能否在 B4 上提供增量，必须由完整冻结预测验证。目前只是可行性，不预设涨分。

**固定来源。** 原编码 F1 80 `54c8cb8c…`；原临床完整报告80 `2a95bad1…`；原 codec `da6410ac…`；不训练的 C32 adapter `2c777e88f86ed6f1aa620755f30ec176668578fe6349009470c4a58941cc0316`。精确输入/输出、原 native 审计和原27行缓存引用由 evidence/source-manifest 定位。无原 dev 或 F1 gold 读取。

**变量。** n>1 的患者 i 对应输出 `PT((i+1)%n+1)`，PT 表按原代码顺序排列，并与逆映射同步；临床 user 前缀、system、marker、patient_ids、CID/HPO、行顺序均不变。n=1 整行直接使用原序列化字节。旧 `response_representation` schema 保留，其映射值合法；外部唯一 view ID `patient-response-code-cyclic1-v1` 标识本控制，decoder 严格要求此唯一映射。

**已执行结构检查。** 0.290318 秒，80/16doc/39PID/587CID/1557pairs 完整。53 条新视图、27 条恒等缓存；逐行确认完整临床前缀 UTF-8、元数据和 key 顺序。每行用合成完整矩阵覆盖全部患者及逆序列表；另17项具区分力的合同核 cyclic 方向、不同临床ID的相同mention、逆映射顺序、重复/未知/缺CID/错表等拒绝。新输入 2,893,388 B，SHA `1c372b6e51e954903dc1c4dcd937d5521bb2557452e1355ad6862b4bff260f54`。未打开旧响应 body；81件缓存内容仅校验字节SHA。

**后继 CPU，尚未执行。** `native_driver.py` 固定120+TERM5，CUDA隐藏；逐6资产SHA，实际版本/源码核验，原/新各80条 Auto/native/template IDs、双decoder与旧原80 native IDs比对。27恒等行必须IDs全等、53变化行IDs不同。新总量/max/cap在实际收据前保持未知；不加载模型或adapter，不执行评分。

**预锁质量。** primary = `B4 ∪ (原 strict C ∩ cyclic1 strict C)`；diagnostic = cyclic1 strict C 单臂。二者保全B4的实体/患者/非关系字段，完整80装配为16doc后共同freeze；主变化则1 v2，不变则继承B4且0 v2，诊断1 v2，旧0。主 Score 严格大于 `.701325377212172` 才支持讨论F0；报告micro/macro，不以诊断替主、不自动迁原dev或官方。

**限制与决策。** 没有新token成本、GPU或质量结果。原C与未来新视图如host/runtime不同，只能称联合开发控制；不声称纯因果。此次源和本地结果交主线程提交推送，原证明/模型/缓存不变。

## 唯一实际运行的事前冻结

actual native CPU10.9888秒通过；新本地结构增量0.142047秒，81缓存SHA与原generate AST一致。推断 source/plan/quality入口分别见 `../inference/`、`../evidence/GPU-ready.json`、`../quality/final.py`。已授权53fresh1200+30、固定两quality。GPU启动与完整结果待实际回执，不将许可写成已执行。缓存canonical中的fresh字段保留历史原字节；本次summary逐index单列 `exact_original_C_single_patient_cache`，不会将27条计入新生成。

## 实际终点与决定

唯一模型13:36:35–13:43:57，442.208813秒rc0，53fresh＋27原缓存。80strictEOS/无cap弃答，253件终点含240shards，zip3adf1ecc/94139B。实际allocated22.283GB、reserved25.030GB（十进制），A30显存余量小；GPU释放有原回执。

两完整预测先共冻251dfb1d，后各唯一v2、旧0，质量总0.754053秒。主Score .7012824986440518，比B4 .701325377212172低 .000042878568120；micro +.000242887771240、macro −.000414402043721。主新增关系2TP/3FP，关系集合的微小增量未改善总分。诊断alone .6217565062604539，比B4低 .079568870951718。

全部1189实体、39患者和非关系字段原字节保持。工程完整并不意味着质量收益；主/诊断都失败，本固定cyclic1关闭。没有换主次、另一排列、seed、阈值或第三融合，也没有F0/dev/官方调用。原C在不同host/runtime，不能将诊断下降纯归因输出代码置换；已有原响应编码路线的有效结果保持。

内容备份中13:47:07 offset622592单片write_file返回UNKNOWN，保持原e777a0e3包、实验与质量全部不动；恢复先读原namespace实际库存，再跳过已匹配片，只续缺片。最终持久ACK独立记录，不能以已上传多数片冒称完成。
