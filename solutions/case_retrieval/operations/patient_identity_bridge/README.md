# 患者身份上下文配对控制

`prepare.py` 只做训练48 CPU输入构建；`train_encoded_ce.py` 直接消费冻结数组，不重新分词，不读验证标签或评测。完整数据与模型在仓库外，入口路径均由plan配置。

- 固定4127 OOF patient-concept键与969 canonical正例。旧SFT的967只差两个alt-ID；旧gold-entity CE的5114行不能作此配对基线。
- 保留旧384编码全部active IDs、segments、位置及原末SEP；处理臂在其后加最多127个身份锚点原句tokens和1个SEP，对照尾padding，双方张量512。空bridge完全相同。
- 只选输入中真实坐标锚点。最近锚点可属于其他患者；仅同passage或紧邻前一同section paragraph，所有身份带offset，冲突/跨句并列歧义/超预算拒绝。句子可含临床事实，因此不称纯身份token因果控制。
- 两臂如获后续授权，均fresh同公共BiomedBERT、相同seed/4epochs/weightedCE/AdamW/1032steps/103warmup；原384前缀相同不等于浮点logit bitwise等价。实际forward显式传同arange512位置。
- 当前probe只准一个最长16×512标准首step。warmup初始lr=0，检查loss/梯度方向/Adam分配及耗时，权重首步不变是预期；不保存可续训状态。probe计划无法启用formal模式。

CPU准备：
```bash
python prepare.py --plan /EXTERNAL/cpu-plan-v2.json --output /NEW_OUTPUT
python train_encoded_ce.py --plan /EXTERNAL/probe-plan.json --mode cpu --output /NEW_OUTPUT
```

实际GPU入口须使用已核私有环境和模型6文件SHA，先核分配UUID无进程；外层timeout约束预算。不可重复执行存在输出或job的命令；先查实际进程恢复。当前没有正式训练授权或质量结果。小证据见 [CPU结果](../../evidence/round2/patient-identity-bridge-cpu-preparation.json)、[probe计划](../../evidence/round2/patient-identity-bridge-probe-plan.json)。
