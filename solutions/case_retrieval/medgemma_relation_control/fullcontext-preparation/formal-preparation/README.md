# 闭集全文 Med：固定正式运行入口

本次仅把旧 CID→PID 输入末尾追加允许全文，fresh 公共 base；不会继承短 Med adapter。完整配方、强对照与唯一两端点已在 `pretraining-amendment.md` 固定，来源及实际 CLI 见 `formal/run-plan.json`。

先做两完整 epoch，各16 optimizer calls。一次2700秒绝对窗口包括两load、完整checkpoint保存、A800→.100导出和逐文件重开SHA；epoch1持久ACK之后才同stage恢复Adam/scheduler/allRNG。两次完整ACK之后才单独900秒固定80推断。所有remote timeout以原绝对终点计算，加TERM30；监控未知保留原job/intent，不能重复launch或自动停配套server。GPU进程第一OOM/非有限/截止不继续下阶段。

A800模型继续复用原SHM8.64GB副本、.100有固定全SHA原件；新SHM output=`/dev/shm/patientphex-medgemma15-1045-20261024/fullcontext-formal2ep`。持久目录=`research/medgemma-relation-control/fullcontext-formal2ep/checkpoints/epoch-001,002`。exporter收完整adapter+Adam/RNG、9个源码与输入token审查/调度；`_cpu/reference/summary.json`、`_cpu/reference/row-token-audit.json`、`_cpu/preparation.json` 保留原实际依据。fetcher原Range/全SHA及完整文件/目录祖先fsync逻辑保持；仅自身namespace改变。

唯一入口：`python run_formal.py --directory formal`（在绝对外部路径执行）。启动前要求源推送、实际小部署全SHA、H48触发的tokenizer文本对照、fresh GPU2 UUIDf516空卡及SHM/持久空间。控制器首次运行写execution-state/每阶段intent/knownjob。若已存在execution-state，必须恢复检查原job，不重复execute。

准备已完成20项训练源/数据合同、16项双端点合同、4项控制合同；原训练数值段及自身checkpoint恢复保持，HTTP Handler/Range AST保持。新H48 CPU一次17.716秒：Med与Qwen各80固定文本Auto/native JSON IDs与decoder结果相同；序列化与未使用音频special差异独立如实保存，不宣称全tokenizer等价。原12,824-token实际工程probe30.681秒/更新8.469秒/allocated58.364GB；这里只是容量与单步证据，正式多形状峰与总时间待实际测量。

全部raw/fullpred冻结后才新unique v2两次；F1是重复观察的内部开发折。未授权F0、原dev、官方或其他组合。
