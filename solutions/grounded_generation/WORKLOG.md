# 工作台账

分支`explore/grounded-generation`，worktree `/workspace/patientphex-worktrees/explore-grounded-generation`，起点main `0c0672d`。只修改`solutions/grounded_generation/`。旧序列路线已f0785f7完整结案/推送，α=.5固定负结果不重开。

## 当前目标和状态

- 目标：9B原文证据复制＋独立HPO规范化，补充强全文N的患者关系；保持Task1骨架，固定总Score主门。原dev最佳.754361398，官方.6922为用户回传；本路线尚无新分数。
- 原32金标38c353ea/969963B和356109e9谱系已核：F2+F3、32文献87患者；F1输入16文献39患者，与训练不交。held/原dev gold没有读取或哈希。
- 历史边界已核：旧8B窗口抽取/SFT直接并集两折不稳；旧患者HP+quote自由编号严重失败。新机制改变为患者条件证据复制、禁止HP生成、独立规范化，不能只把换9B当创新。
- 真实schema已核：association只有patient_id→phenotype；实体独立含offset与identifier。目标quote只是按关系选出的弱证据，不能称人工归属证据。用原passage单位ID保全原始offset，避免按gold切句。
- 正在实现CPU目标/原文位置/原生token合同；规范化helper仅负责normalization目录的公开本体/同32资产。未开始GPU/训练/推断/评分。
- 资源：A100 GPU2由root保留；实际新input/答案token census后提出工程压力和完整训练预算，不套旧32.693GB，不截最长样本。其他卡/作业由root调度。

## 执行与下一步

1. 固定每个目标患者的完整输入、可定位quote构造、不可构建与歧义统计；所有训练/held患者均在覆盖表。
2. 实际计算训练全文/答案/监督位置与held预算，原生tokenizer核对；纯CPU合同及源/计划先推送。
3. 与root协调一次有界真实最难工程检查，fresh正式训练与完整F1推断；不从工程状态续训。
4. 两完整预测共同冻结后各唯一v2、旧分数复用。主臂N∪new、诊断new-only，Task1/患者/非关系字节保持；另一内部折有益后再full64/原dev验收。

大文件放 `/workspace/patientphex-runs/round2-20261023/research/grounded-generation-v1`。当前无运行job，无需要恢复的GPU进程；未知状态先查原intent，禁止重复启动。
