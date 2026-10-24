# E-20261024-RELATIVE-CANDIDATE：固定池相对公共来源证据

当前：唯一四fold已完整完成，pooled下降且仅1/4fold不降，按事前条件负结案，不迁原dev。目标是检验同一候选的来源证据，相对同篇竞争候选是否能改善实体选择；不换候选、标签、阈值或解码。方法是在原dev已参与研发的固定误差分型之后提出，不能称盲选。历史查重见[前置短记录](../lightgbm_selector_control/research/next-entity-history.md)，未发现同强池该特征实跑。

## 固定变量与边界

基线是刚通过的LightGBM4.6四fold：pooled **.731545429370**，各fold .740572140440/.728697536469/.754425066951/.706374759627；不用较弱HGB作为新对照。复用强BioBERT八pool，原特征102/101/103/103及标签函数/来源隔离保持；每个模型仅新增18个数值列，预期120/119/121/121。160树/15叶/min35/lr.06/L2=5等全部配置原样，.3/原greedy+nested原样。

[features.py](features.py)固定两种竞争集合：同exact span且不同literal ID；相同literal ID但严格不同、正长度重叠span。排自身与邻接span，不拆或排序compound。分别对public/Bioformer/BioBERT计算该来源竞争项存在、最大score、self score减max，合18列。竞争者需该来源present=True；无竞争三值全0，有竞争而自身缺来源时self按0。presence/score字段必须存在，presence为bool，观测score有限数值[0,1]，缺来源接受历史False/0并拒绝非零；未观测不是模型真实posterior=0，而是有原presence位区分的特征占位。

三score是既有native-hybrid在exact literal key上的最大值，**不是raw posterior**。禁用fit后selector_probability，原单候选feature函数保持原字，仅合并上述输入侧字典。[来源语义](SOURCE-NOTE.md)、[冻结合同](feature-contract.json)和两个最小diff随源保存。不产生117个缺失正确候选，不保证把重叠FP转为正确边界。

## 实际最小检查与唯一正式计划

[input-census.json](input-census.json)：8个纯输入pool/246804候选完整检查，**5.475秒**；18列全部非零且非恒定。8项有区分力边界断言和5项拒绝案例通过，涵盖邻接/compound/自身缺来源/无竞争/非法字段/selector污染；原对象不修改。无gold/fit/predict/v2/GPU，未下载新依赖或模型。

[formal-plan.json](formal-plan.json)304c9e95、[entry](entry-manifest.json)035abcff：四个fresh相同配置模型fit+held predict+全量保存回载共享**180秒**，端点另**30秒**；首失败停，不自动改配置或预算。全部4份16篇预测共同freeze并保持全部非entity literal bytes，之后每fold和pooled各一次v2，新5次/旧0次。保留条件：pooled Score严格升且至少3/4fold Score不降；不扫描子集，不按原dev调参。本轮不迁full64/dev或官方。

运行：固定计划python执行本目录 `execute_control.py --plan <E>/formal-plan.json`，环境沿原private LightGBM。E=`/workspace/patientphex-runs/round2-20261023/research/relative-candidate-evidence-control`；恢复查看`formal-control/execution-state.json`，存在输出目录时禁止重复启动。模型、完整分数和预测在仓库外；最终实际结果与统一审查另补。


## 实际终点：固定配置负结案

source28cd1f9已push+lsremote exact后，唯一controller session57662于05:07:45.545859启动；四fit/predict/保存回载 **61.879秒**，finalize **2.647秒**，5子作业与controller均rc0。每模型实际160树，列数120/119/121/121，候选与原字段不动、正列/回载概率exact。05:08:49.694834全部四pred共冻，49.695834首次gold marker，随后仅5次新v2，旧0次。

| Score | 原固定LGB | 加18列 | 差值 |
|---|---:|---:|---:|
| pooled | .731545429370 | .730783429343 | −.000762000027 |
| F0 | .740572140440 | .739338684725 | −.001233455715 |
| F1 | .728697536469 | .729434531345 | +.000736994876 |
| F2 | .754425066951 | .752221408651 | −.002203658300 |
| F3 | .706374759627 | .705686363515 | −.000688396112 |

pooled menF1 .772867639229→.771291866029，docF1 .811077715758→.809605488851；关系两项完全不变。43篇54加40删，mention净−2TP/+16FP。**keep=false**，不按仅正F1扫描特征子集、调整cut或迁full64/dev；结论仅限本次固定18列native-hybrid证据配置，不能泛化为全部边界模型无效。

[结果295c2def](results/result.json)、[共同freeze](results/complete-freeze.json)、[terminal manifest6b67cbea](results/terminal-manifest.json)保留；当前外部execution-state为complete，无运行CPU/GPU。独立源审f0a2d9e9通过，[完整终审3f767632](results/terminal-review.json)已PASS/固定负，64篇全部非entity/关系literal exact，未重fit/评分。当前原dev best仍是旧固定LGB .748583439566；新负配置没有改它。
