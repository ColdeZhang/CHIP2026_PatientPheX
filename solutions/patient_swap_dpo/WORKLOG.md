# 患者交换偏好训练台账

目标：在固定四源融合中替换Qwen3关系组件，提高患者归属质量。原train64/dev16目标≥.8；本线先做train32→内部F1固定控制。

- 分支 `explore/patient-swap-dpo`，从main `979e9492` 建立；仅本目录。
- 已完成：74成对样本/17篇、1149合法swap供给；完整原生token CPU1.917s；原124chosen与缓存IDs/mask全部exact。14个选中样本为多阳性，其余患者正例保留。
- 已核原B4关系集合严格等于B471∪四pure中至少2票，非关系对象及B471列表前缀保持；重序列化对象列表顺序不同，不称byteexact。
- 固定：Qwen3 train32 epoch2适配器543c82a5；β.1/LR1e-5/1epoch74pairs/10updates，新Adam；单完整主终点，不读取heldgold/不评分。
- 当前准备实际tiny Torch/PEFT CPU合同，50+5秒，CUDA隐藏。真实大模型显存/成本门尚未执行，GPU未启动。
- 仓外 `R/research/patient-swap-dpo-readiness-20261024` 保存数据、完整public源码与CPU产物；候选协议见 `candidate-plan.json`。

- 实际.3 tiny CPU通过（0.245s、28个LoRA梯度max1.30e−8、真实Adam），两原件完整回读a52ea3d/563b93d0；首只读guard异常后恢复原件，没有重跑。root已授权固定ref900/probe600/formal1800/infer1200各+30，真实GPU尚未启动。执行模板 `runtime/` 保留原方案预算历史，13压力对/两Pareto/26FB计划固定。

- GPU前增量源码检查拦住共享checkpoint所需durability_root漏项；v2 be5ac193已显式绑定，v1 disabled原件未部署并保留。reference缓存plan/data/parent三项身份核补齐。下游24合成合同/strict codec与raw-first循环AST复用通过，单质量默认disabled。下一步control.py --directory R/research/patient-swap-dpo-v1/deployment-v2 deploy→cpu→collect_cpu；actualCPU后才activate→start。
