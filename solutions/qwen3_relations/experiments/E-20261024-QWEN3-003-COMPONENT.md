# E-20261024-QWEN3-003：三源固定融合所需full64组件

root预锁三源2of3在F1/F0均正，F0 .724457355332高于固定双采样 .7219695，并已审99e5a6d8。此新部署仅提供其中的Qwen组件。旧独立Qwen并集F0 .688186671128为负，结论保留，不复活独立方案。

## 数据、目标与成功标准

保留原normal监督265行/train64，公开Qwen/Qwen3-8B从fresh基座训练，不续训32adapter。所有候选与训练来源排originaldev；训练候选四个train48→held16，dev候选CRF/Sap/LR64+GLiNER48子集，资源差异保留。CPU native4.51.3已全ID/解码/mask核验：train866495 tokens/max6155/监督15410，dev64全文509609/max13175。详见full64_readiness/与实核416da6ef。

同BF16基座/FP32 LoRA r16 alpha32 dropout.05七线性、lr1e-4/wd.01/accum8/clip1/seed20261023、标准FP32 mean causalCE indexed监督前位。不截输入、不合并packing。两轮68updates，尾批1按实际组大小平均；warmup .05取ceil=4，与原normal full64 manifest一致。源合同13bb8914执行真实epoch_groups验证530行曝光/每轮各265完整，并继承原loss源码44201b60和strict parser AST；无新loss数学重跑。

成功是完整工程链和供root冻结融合的预测交付，不在本线评价原dev成绩或挑checkpoint。901实体壳1061实体/16篇36患者字节保持，Qwen只生成466概念对应候选关系。

## 固定执行

.3 GPU0/A30，原生TF4.51.3/PEFT.15.2/tokenizers.21.2，Torch2.7.0cu128（原32为2.7.1cu126）。每独立模型8,234,382,336参数<10B，复用同HF b968826基座缓存。

1. schema Sap实际释放后fresh卡/盘门，唯一240+TERM30 probe。13次indexed FB：8累积→真实Adam→3累积→Pareto(4992,116)/(6155,113)，有Adam及梯度常驻；不重原数学/旧5534/held-prefill。实测有限、完整，原成本式估计≤2400才继续，probe state丢弃。
2. fresh两轮2400+TERM30，两个完整adapter/Adam/scheduler/RNG重开校验和持久ACK。ACK创建后再fsync父链，仍不把逐值重开称为中断续训等价。
3. 64条完整全文fresh1200+TERM30，gen1600/greedy/strict whole-row abstention，raw IDs→raw text→parse。全部原行保留，无重试/格式修复。
4. 完整new-only预测固定在901壳上，只校格式，不读originaldev gold、不独立union/评分；交root预锁2of3和另一独立grounded端点共冻后评分。

入口full64_component/plan.json、driver.py、native.py和assembly-plan.json。运行大产物R/stage8-qwen3-relations/full64-component，远端P/research/stage8-qwen3-relations/full64-component-v1。未知状态只恢复原job，不重启；OOM或真实成本阻断保留失败并报root，不自动改方法。

## 实际状态

源f69d88d0已push。首次小ZIP上传LUNA_OFFLINE，三次间隔只读恢复后明确NOT_FOUND；没有旧GPU/start。v2源已实际解包/完整data与schedule合同通过，freshGPU0 4MiB/noapps/盘169.31GB。唯一控制器proc_e5d04680/PID1207543于12:08:35.986798Z启动；精确probe子进程和截止见evidence/full64-component-start.json。后续按两个完整ACK及全部64raw补记。

新shape已64.748844秒完整PASS，13FB/1Adam，全部504梯度及504Adam状态finite，峰allocated21.628GB；固定成本估2245.071238秒<2400，旧数学/held-prefill0次。已discard后fresh正式child1207643于12:09:42.129377启动，TERM12:49:42.129377/KILL12:50:12.129377。两个完整epoch ACK待实际终态。

Epoch1完整34updates已12:22:09.652993 UTC取得ACK：504adapter/504Adam1512tensor及scheduler/RNG重开exact，checkpoint6b681bb2；adapter174655536B/df4ef6ca，state349612141B/dc56ad04。receipt记录3.212588秒，不把它当包括最后ACK/父链同步的严格总成本。第二轮沿原窗口继续。

正式两轮已12:34:29.703457完整PASS：68updates，530行曝光/1,732,990tokens/30,820监督tokens；train1478.328467秒、phase1487.504494秒，峰allocated21,627,827,200B。两份完整ACK成立，epoch2 checkpoint a8a02d4e、adapter59fb484a/state45e5c7e4。原控制器12:34:31.050998接唯一infer child1209224，TERM12:54:31.050998/KILL12:55:01.050998；完整64fresh仍待终态，无质量调用。

64/64 fresh全文推断已12:41:03.507730完成，strict完整64、弃答0，infer392.385980秒/child393.468161秒；峰allocated21,105,800,704B。实际GPU0于12:41:04.692081回4MiB/noapps，全三阶段rc0、无timeout。终包1,076,977B/284件，SHA8d1283bf；当前只收原包并装配固定new-only，不读取originaldev gold。

全部284成员SHA回收通过后，12:43:53.079831冻结完整new-only预测3d2e4ca3（138474B）：16篇/36患者/1061实体/265关系，候选466概念、全部64行fresh strict完整，0cap/0弃答。原901全部非association字节精确保留，公共格式验证0error/0warning，质量调用0次、原dev gold读取0次。

该组件已交root预锁2of3及最终共冻流程；本线只判工程交付完成，不独立判原dev涨分，也不复活旧Qwen独立F0负方案。完整原始IDs、responses、canonical和checkpoint ACK路径见evidence/full64-component-terminal.json；两个大完整state留.3原位置，小source/assembly补包另有持久回执。
