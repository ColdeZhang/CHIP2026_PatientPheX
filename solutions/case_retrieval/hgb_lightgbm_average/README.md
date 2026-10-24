# E-20261024-HGB-LGB-AVERAGE：原强HGB与固定LightGBM概率平均

当前：唯一固定均值完整负结案，不扫描权重/cut或迁原dev。目标是用两种既有树实现可能互补的误差，检验固定1:1平均；不重新训练或调用模型推断，不保证集成提分。

有限三路线及本轮LGB历史未发现此精确控制。本轮首次强池LGB四fold04:48:51结束，原源码一直只解码单LGB正概率；旧弱池投票、全局兼容解码和刚关闭的18列控制均不同。本次**不含负18列概率**，也不扫描权重或阈值。

每fold固定两份已冻强BioBERT池概率。实际[输入对齐](alignment.json) **3.065秒**，4fold/64篇/62,281候选，候选逐键、字段与类型（包括False/0差异）、顺序完全匹配原prepared pool；两模型的训练来源、48→16隔离、labels函数、列序102/101/103/103、全部原生依赖与来源manifest相同。仅直接计算`0.5*p_HGB + 0.5*p_LGB`，不读gold、解码、fit或模型预测；逐fold浮点hex流SHA可供正式入口核对。

正式[control.py](control.py)在**共享30秒CPU/TERM5**内重核固定来源/对齐、执行原.3 greedy+nested、保留LGB旧预测的全部非entity原始UTF8片段，四完整pred共同freeze后才读取train64 gold，各fold和pooled各一次v2。对照固定原LGB pooled **.731545429370**及其四fold；保留条件pooled Score严格升且至少3/4fold不降。零fit/新NN/新模型预测；不迁原dev、无官方操作。

方法选择已受既有开发结果影响，train64四fold也曾用于研发，不称盲测。既有HGB/原LGB已经同严格来源审查，组合不新增训练资源；新source/plan及实际输入收据接受统一增量审。若失败或负结果，只关闭这个固定权重，不自动加臂/换cut。

执行：计划python运行本目录`control.py --plan <E>/formal-plan.json`。E=`/workspace/patientphex-runs/round2-20261023/research/hgb-lightgbm-average-control`，formal-control已complete，session15356已rc0；恢复查看该目录execution-state，禁止重复启动。平均概率无需再复制完整候选大文件，源概率与均值floathex SHA已绑定，完整新预测保持仓库外。


## 实际终点

source0cdda9d已push+lsremote exact后唯一运行，**4.622秒<30秒**，0fit/0新模型预测。05:12:48.196059四份完整pred共冻，48.197064首次gold marker；之后仅5次新v2，旧指标复用，格式全0错警。

| Score | 原LGB | 固定均值 | 差值 |
|---|---:|---:|---:|
| pooled | .731545429370 | .731417753008 | −.000127676362 |
| F0 | .740572140440 | .739905445172 | −.000666695268 |
| F1 | .728697536469 | .728634659671 | −.000062876798 |
| F2 | .754425066951 | .753917876202 | −.000507190749 |
| F3 | .706374759627 | .707096620066 | +.000721860439 |

仅1/4fold不降，**keep=false**。29篇22加22删，pooled mention计数/F1相同不代表预测相同，docF1 .811077715758→.810567010309；关系及全部非entity原始UTF8片段保持。[完整结果4c3b72b4](results/result.json)、[终点c368767d](results/terminal-manifest.json)、[唯一终审908fbc8c](results/terminal-review.json)均保留。终审通过来源/对齐/实际算术入口和freeze顺序，未重算概率或读取gold/模型。

按预定关闭这一固定均值，不能从仅正F3挑结果，不增权重搜索/新组合/原dev试验；当前最佳仍为原单LGB原dev .748583439566。
