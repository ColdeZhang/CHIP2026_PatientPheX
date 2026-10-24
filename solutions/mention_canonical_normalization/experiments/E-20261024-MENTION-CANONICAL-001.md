# E-20261024-MENTION-CANONICAL-001

假设：局部occurrence语义与完整公开名称生成，可减少固定跨度的ID错误或接受旧规范化未覆盖的真实表型。目标是实际完整强基线增量，不能用gold跨度上的名称准确率替代Score。

数据、三状态schema、继承祖先、无gold固定1416跨度池和预锁两端点见plan.json。训练2353，全部32文献；字符context最多581，不是已证明token上限。CPU数据准备0.364秒完整，尚无native shape或GPU结果。

后继拟复用现成native.train，fresh Qwen3-8B LoRA2epochs/epoch2唯一，590更新、尾1、warmup30，不holdout/earlystop。实际GPU预算尚未批，完整CPU合同后再据最大输入/监督形状给probe与成本。已有public2/strongbase祖先及GLiNER训练/strict/Sap均排F1，公开预训练全量谱系未知如实披露。

真实CPU合同12:16:08唯一start proc818c7032，120+TERM5预算下18.695秒全部通过；TF4.51.3/tokenizers.21.2/Torch2.7.1cu126只读既有runtime，隐藏CUDA/线程1。2353训练664278tokens、24063监督，完整prefix/mask/EOS；1416held385719tokens，无labels。最大形状383/28与386，全部cap1024足够。21710目标prefix、所有held的精确NIL分支、三状态实际HF processor和8拒绝例通过；production model.generate未执行，不能冒称GPU解码正确或模型质量。原job actualrc0，CPU无GPU资源释放声明需求。

root已授权条件GPU链：.100GPU1唯一180+30，native initial8→Adam→3accum→3Pareto=14真实FB，加训练行三状态constraint generate，各原生128tokens，raw先存，cap/弃答不修。透明同步计时原Adam.step不替换其算术；预写load+cold首FB+1.25×(4706×其余13步最大FB+590×首次Adam)+120s双ACK/I/O≤2400才fresh正式两轮，保存两完整state ACK；成功后1416全推断1200+30。无条件触发的后台重试，正式不复用probe权重；两端点共冻后各一新v2。
