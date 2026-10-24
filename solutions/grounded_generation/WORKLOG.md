# 工作台账

分支`explore/grounded-generation`，worktree `/workspace/patientphex-worktrees/explore-grounded-generation`，起点main `0c0672d`。只修改`solutions/grounded_generation/`。旧序列路线已f0785f7完整结案/推送，α=.5固定负结果不重开。

## 当前目标和状态

- 目标：9B原文证据复制＋独立HPO规范化，补充强全文N的患者关系；保持Task1骨架，固定总Score主门。原dev最佳.754361398，官方.6922为用户回传；本路线尚无新分数。
- 原32金标38c353ea/969963B和356109e9谱系已核：F2+F3、32文献87患者；F1输入16文献39患者，与训练不交。held/原dev gold没有读取或哈希。
- 历史边界已核：旧8B窗口抽取/SFT直接并集两折不稳；旧患者HP+quote自由编号严重失败。新机制改变为患者条件证据复制、禁止HP生成、独立规范化，不能只把换9B当创新。
- 真实schema已核：association只有patient_id→phenotype；实体独立含offset与identifier。目标quote只是按关系选出的弱证据，不能称人工归属证据。用原passage单位ID保全原始offset，避免按gold切句。
- 目标v2实际85979cdc/.484秒已完成，全87训练行/39held保留：814有效原子键全部有可定位词面，801去重目标，4关系无效/分支外明确列出；原文位置边界7项通过。local token实际c8f29baf/3.253秒：451672完整训练tokens，最长9210，监督11884/max610，held203588/max10080；三Pareto形状9210/399、6521/449、5676/610。原生TF CPU120+5已root授权，源码/计划冻结准备部署；规范化helper仅负责normalization目录的公开本体/同32资产。未开始GPU/训练/推断/评分。CPU回放之后已有单次420+30工程授权：首8真实Adam→3累积→三个Pareto形状14次反向，state丢弃；正式87×2预算待实测。
- 资源：A100 GPU2由root保留；实际新input/答案token census后提出工程压力和完整训练预算，不套旧32.693GB，不截最长样本。其他卡/作业由root调度。

## 执行与下一步

1. 固定每个目标患者的完整输入、可定位quote构造、不可构建与歧义统计；所有训练/held患者均在覆盖表。
2. 实际计算训练全文/答案/监督位置与held预算，原生tokenizer核对；纯CPU合同及源/计划先推送。
3. 与root协调一次有界真实最难工程检查，fresh正式训练与完整F1推断；不从工程状态续训。
4. 两完整预测共同冻结后各唯一v2、旧分数复用。主臂N∪new、诊断new-only，Task1/患者/非关系字节保持；另一内部折有益后再full64/原dev验收。

大文件放 `/workspace/patientphex-runs/round2-20261023/research/grounded-generation-v1`。当前native CPU v2已原proc8bd3ffbe完整rc0/16.244秒，18件全部SHA回收237aaa65；87/39全输入与6数值PASS。GPU14步已唯一10:00:29.145Z启动proc_7d1bc518-3e0c-4ed5-9bf4-5d3a369671e5 / driver66162，420+30，以driver实际TERM/KILL为准；B/probe-v1，原probe_remote.py observe/collect恢复；未知状态先查原intent，禁止重复启动。


2026-10-24 GPU前澄清：全文先判目标患者的概念，复制quote只作概念词面锚定，不把该处出现当患者归属证据；最近mention只是确定训练代表词面的弱选取规则。首CPU产物targets-v1与源c3b69eb保留，v2仅修正prompt语义，目标/身份/全文不变并逐项对照。170/814处其他患者更近已如实披露。

2026-10-24 CPU首尝试v1在token回放前asset lookup失败5.609秒，完整8件97e8f008已保留；v2只删不存在且不属于tokenizer的generation_config lookup，其他四发布资产及输入/数值不变。独立120+5 source6483fda已push exact。训练器与strict/Sap独立规范化并行准备，后者在DocRE释放.3后才使用已批准120+15 query encode。
