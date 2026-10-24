# 工作台账

状态：**固定 F0 校准 → F1 开发验证负结案**。F0预锁11点选择 t=0；F1主臂强B4 .701325377212 → .683634645392（79新增=6TP/73FP，micro/macro均降）。诊断new-only .564954220539；不保留、不改grid/epoch/模型、不追加GPU。原PCL负结果保持。

GPU源码8aac430f、唯一质量源码b3a0520d均已先push；实际运行不是最新文档commit。当前分支explore/patient-competition-calibration，起点d497848a；质量/终态见 evidence 与 experiments/E-20261024-PCL-CAL-001.md。

完整F0 CPU13.725s、16篇47患者61460tokens/164窗1741367候选，native IDs/offsets exact。原模型9ad43c/206 tensors strict CPU装载，唯一GPU1 proc_bf9e676a，13:35:57.372423开始，13:37:57.372423原截止+TERM30；实际21.745s rc0，13:36:19.116406释放并交schema，未重训或probe。

外部根：`/workspace/patientphex-runs/round2-20261023/research/patient-competition-calibration-v1`。57内容raw包416628B/ea17c20b已全SHA回收；15MB可重建几何仅远端持久。完整F0/F1正值support表及校准/质量留外部，不把候选明细写Git。50内容质量包94db98c7/145691B及单独terminal518e078e/12415B也已远端文件+父目录fsync、全量回读相等。当前仅root下批增量审，不自动开启新机制。
