# E-20261024-MENTION-CANONICAL-001

假设：局部occurrence语义与完整公开名称生成，可减少固定跨度的ID错误或接受旧规范化未覆盖的真实表型。目标是实际完整强基线增量，不能用gold跨度上的名称准确率替代Score。

数据、三状态schema、继承祖先、无gold固定1416跨度池和预锁两端点见plan.json。训练2353，全部32文献；字符context最多581，不是已证明token上限。CPU数据准备0.364秒完整，尚无native shape或GPU结果。

后继拟复用现成native.train，fresh Qwen3-8B LoRA2epochs/epoch2唯一，590更新、尾1、warmup30，不holdout/earlystop。实际GPU预算尚未批，完整CPU合同后再据最大输入/监督形状给probe与成本。已有public2/strongbase祖先及GLiNER训练/strict/Sap均排F1，公开预训练全量谱系未知如实披露。

真实CPU合同12:16:08唯一start proc818c7032，120+TERM5预算下18.695秒全部通过；TF4.51.3/tokenizers.21.2/Torch2.7.1cu126只读既有runtime，隐藏CUDA/线程1。2353训练664278tokens、24063监督，完整prefix/mask/EOS；1416held385719tokens，无labels。最大形状383/28与386，全部cap1024足够。21710目标prefix、所有held的精确NIL分支、三状态实际HF processor和8拒绝例通过；production model.generate未执行，不能冒称GPU解码正确或模型质量。原job actualrc0，CPU无GPU资源释放声明需求。

root已授权条件GPU链：.100GPU1唯一180+30，native initial8→Adam→3accum→3Pareto=14真实FB，加训练行三状态constraint generate，各原生128tokens，raw先存，cap/弃答不修。透明同步计时原Adam.step不替换其算术；预写load+cold首FB+1.25×(4706×其余13步最大FB+590×首次Adam)+120s双ACK/I/O≤2400才fresh正式两轮，保存两完整state ACK；成功后1416全推断1200+30。无条件触发的后台重试，正式不复用probe权重；两端点共冻后各一新v2。

## 真实探针与预算修正

原180+30探针实际34.423s、wrapper35.195s rc0；14FB/1Adam、三原生训练状态约束输出EOS。steady FB max0.592576s、Adam0.059550s，预写公式3654.233s>2400，完整成本门关闭，未开展训练或质量。69原件完整回收SHA986ab81d，原job及release见evidence/probe-complete.json。

root明确批准fresh正式预算4800+30，保持2353×2/590/warmup30及所有学习变量不变；新formal-plan.json只增加前次成功probe及asset绑定与formal-only入口。原gpu-v1及旧计划保历史。后继仍全部1416请求、固定双端点共冻后2v2，不因成本裁样。

全两轮真实完成：590updates/4706rows，完整2554.713秒，wrapper rc0；两epoch fullstate ACK逐504adapter/504Adam1512tensor/RNG重开精确。估3654秒是预算依据，实际训练仍未包含质量。第二轮唯一adapter932e4f12，13:08:05.921同控制器已转全1416 fresh推断，原1200+30不改。

## 完整终点与解释

13:20:29.893985原控制器完成全部1416 fresh请求，原生743.145s/wrapper743.982s，HPO1388/NO28/NIL0/拒绝0/cap0；实际GPU1=0MiB且本卡无进程。2888小原件ZIP7b080c97/1602882B完整SHA回收，四大checkpoint仅留owner重开和精确SHA/locator。原managed进程running=false，工具没有wrapper exit字段；两个真实阶段child rc0分别保留，不用run_command回执冒称wrapper退出码。

两份16/39完整预测共冻c869ccf9后唯一2v2：强union .674197659658（men .666030534351/doc .711966325917），new-only .613794585011（men .562568405691/doc .573816155989）；关系mic .659553831232/mac .659239947131逐值不变。强B Score .728697536469，两个端点均负。478新增14TP464FP；new-only771TP617FP/precision .555475504323/recall .569844789357。这是语义质量负结果，不是工程中断。

预测集合只读诊断：910原子与强B相同，381个新增ID位于强B已有跨度，97个新增位于强B之外；这里只表示分歧，不将每个分歧自动称gold错误，也不据此筛选held输出。完整公开canonical约束只保证合法名称，不保证上下文映射正确；少量NO监督与跨度池假阳性分布差异是事前独立2931实验的动机，不能因该并行实验尚未完成就宣称已解决。固定2353配方关闭，不修改阈值/模型/原预测，不扩F0或originaldev。

完整结果、两ACK、raw/parsed、评测时序、来源限制见evidence/terminal.json；集中独审和小备份只核原件，不重评。
