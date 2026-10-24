# 支持gate最佳 + 患者六差集

根目标仍是原train64/dev16公共v2 Score≥.8并通过格式/来源/复现。此前两个独立端点已经知道结果：支持gate3049为.754216969282，患者a179并集为.746261042824（相对a179仅3TP3FP）。root明确提出并授权一次固定开发合成，不能当作未知数据的独立验证。

预先固定以3049完整预测为底，仅加入患者765cd5a0相对a179的6条关系差集；不转移a179实体或整份391关联，避免恢复新版gate删除的2public。源码b9ea4f50/plan edee73ab含20份实际SHA，引用已审literal-span拼接helper，保留1061实体/36患者/410现有关系/54public/非关联原字节和旧列表前缀；不预设六差集与已有关系是否重叠。e731663先push并核远端，再执行两CPU阶段各30+1，首失败停；无NN/拟合/解析模型回答/扫描。

06:18:59实际六差集与410/54均无交集，完整16pred9016863c（140578B）先freeze d3f058fd；06:19:04才标记before-gold并执行唯一v2，旧metrics复用，不评额外臂。全部6加入，最终416关联；0删除、2public仍缺席，全部实体/旧关系/public54/非关系原字节及前缀通过，格式0错0警告。assemble .02849秒、eval .17502秒。

Score .7542169692819372→**.7543613981990362（+.0001444289170991）**；men .7928011075219197、doc .8286778398510242完全不变；micro .7369640787949016→.7387802071346375（+.001816128339736），macro .6584248509599031→.6571864382885636（−.001238412671340）。新增3TP3FP。按固定Score门是候选全局best，无额外macro否决门，但增益微小、macro降低须如实报告。距.8尚差.045638601801；未达标，不合并实现或官方上传。

外部D=`/workspace/patientphex-runs/round2-20261023/research/support-plus-patient-six-delta`；全部pred/freeze/metrics留外部，Git仅[结果](../evidence/support-patient-synthesis/result.json)、[终点索引](../evidence/support-patient-synthesis/terminal-summary.json)及源计划。[恢复入口](../support_patient_synthesis/README.md)。当前等待统一终审及持久小包；不重复评测。
