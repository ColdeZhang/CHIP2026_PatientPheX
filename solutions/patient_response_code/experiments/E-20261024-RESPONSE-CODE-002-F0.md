# 同32权重在完整 F0 上的固定复验

目标是确认F1正候选在另一个完整内部折是否仍超过现有强normal全文N。复用最终adapter `2c777e88…`，不重训、不修复F1弃答、不挑文献或epoch。

固定主臂 `coded-alone`，诊断 `N∪coded`。两完整预测先共冻，再各一次v2、旧0。最高Score严格超过现有N `.6966039747439822`才保留，平局主臂；macro完整报告。另披露现有strongB0 `.711507580027`与root2of3 `.724457355332`差距，避免将胜N说成全流程最高；没有原dev或官方动作。

祖先检查实际通过：原58全文输入来自F0 train48的CRF、Sap候选及correctedLR；48篇与held16文献集完全不交，全部415候选概念经原`association_keys`与实体池一致（包括一个真实`-1`用原词面关联）。全文追加和PT转换后clinical ID、原患者顺序、CID/概念、原文和markers保持。初始检查把`-1`原identifier直接当关联键而失败，已保留该源码/失败元数据并复用原函数修正检查；没有修改数据或重跑renderer。

coded关系头用F2+F3 train32；旧N关系头和上游候选使用排F0的train48，可包含F1。旧N运行在A800 native2.7，coded在A100 native2.6。故这是实际候选比较，并非统一train32流水线、仅编码变量或仅后端的因果实验。F0也是开发折，不称盲测。

同一PTcodec先在完整58纯输入上核字节保留、无alias碰撞、可逆身份映射，然后用现有原生tokenizer独立核全部IDs、模板与decode。CPU120+5，不加载模型或读取heldgold。若长度超过F1 guard14848，按完整实测max向上取整512固定新工程guard，不裁切；真正最长行作为正式58之一首先运行，没有额外probe。

正式唯一预算1200+TERM30；同原A100 UUID、9B/r16权重、native5.20.2、BF16/SDPA、greedy1600。原始IDs先落盘、raw先于解析；未知/重复/缺CID/不合法PID、无EOS及cap沿原strict整行弃答，无答案补齐或重生成。静态检查已确认F1与F0的loading/generation/decode调用及完整raw-first采样循环AST相同，PTcodec逐字相同。

当前恢复入口：`../f0/ready.json`。F1已另行完成持久备份；本F0质量尚未运行。

CPU实际13.125329秒rc0，native12.197954秒；58原生全IDs与decode通过。总token505090→516365，max16245→16542，最长唯一idx30。故明确提高工程guard14848→16896，不截断，模型context262144足够max+1600。正式源绑定同adapter2c777e88，最长idx30计入58先跑。原A100 start曾LUNA；确认原namespace完全空后同exclusive driver唯一重入成功，没有备用CPU或重复模型。
