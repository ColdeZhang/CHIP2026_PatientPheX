# Mention canonical 工作台账

2026-10-24 UTC。独占分支explore/mention-canonical-normalization，base bc939eeb；主线程目标仍原dev公共v2≥.8，当前无本线质量。

- CPU数据完整：2353 train32目标=2222HPO/44NIL/87NO；16compound、8分支外正注释显式排除。拟每侧256原字符，保整个mention，不按标签选择上下文。
- 纯F1跨度池：16篇/39患者/1416行；强B独有530、共有783、GLiNER独有103。强B源8cb61ba9，GLiNER接受源6ac9aa51；不提示旧ID/候选，不读heldgold。
- schema：非空公开canonical数组→HPO；唯一{"NIL":"原quote"}→NIL原文；[]→NO空实体。strict parser拒未知/重复/错quote/不完整，不补猜。
- **实际CPU合同已PASS，18.695秒、wrapper19.242秒rc0。** 全3769原生/HF输入一致、2353完整mask/EOS、21710答案prefix；HPO/NIL/NO实际HF processor和batch2beam2分行NIL均通过。max train383/监督28，held386；cap1024/maxnew128无截断。33件完整原件已全SHA回收，ZIP f4f12fda/2615041B，无模型/权重/GPU。
- **真实规模：** 每epoch664278输入tokens/24063监督tokens，held385719tokens/1416请求。Pareto(333,28)/(336,27)/(383,23)；拟两轮590更新、尾1、warmup30。CPU速度不能当GPU预算；root已授权唯一180+30真实probe，按预写实测成本≤2400门直接fresh两轮，随后全1416推断1200+30。
- 复用原件在reuse-sources.json：canonical_names真实token/trie与doc原生train/probe/common/loss；不另造训练器。新target grammar和输入要自己实际验证，不能拿旧87患者合同冒充本线通过。
- 外存：R=/workspace/patientphex-runs/round2-20261023/research/mention-canonical-normalization-v1；prepared-v1/manifest.json是当前完整数据入口。源码48f1fae6已push/ls-remote一致；CPU proc818c7032已12:16:27.568结束，不重启动。

下一步：gpu-plan/入口/质量源push后，fresh .100 GPU1门唯一启动条件链；原native.train完全复用。保存实际Adam计时和14步FB，上界估计不通过则只留probe停止。全部完成后2端点共冻→2v2，任一严格胜才候选。
