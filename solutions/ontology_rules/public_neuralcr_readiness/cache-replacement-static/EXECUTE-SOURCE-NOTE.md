# NCR 四折唯一正式窗口：源码准备，未执行

入口 `execute_ncr_control.py --plan PLAN`。保持fit计划原键，另需：`python`、`formal_CPU_cap_seconds=900`、`TERM_grace_seconds=30`、`nested_source`、`evaluation_source`、`hpo`、`train64_gold`、`baseline_freeze`、`baseline_pooled_metrics`、`preB_pooled_metrics`。每fold在fit键外新增 `baseline_prediction/baseline_metrics/baseline_validation/preB_metrics`。FEATURES按list等值；内置exclusive JSON writer，不依赖helper是否导出write。

**baseline统一为强BioBERT**：`baseline_freeze`指已完成 `ontology_rules/public-biobert-selector-control/formal-control/complete-freeze.json`，SHA `cf8c1a2820ca94099d6129e9c121eb11c1fd049e4ec35d89de725ef169408f82`；每fold baseline_prediction必须精确等于该freeze的 `folds[f].new_prediction`，baseline_metrics/validation强制指同fold `biobert.metrics/validation.json`，pooled指该freeze同目录 `biobert.pooled.metrics.json`。preB预测直接取该freeze的 `folds[f].baseline_prediction`，指标强制对应其同目录 `nested.metrics.json`及上一级`nested.pooled.metrics.json`；preB_metrics/preB_pooled_metrics单列，仅用于次级归因，不能拿弱对照的提升判keep。

数据、NCR raw恢复、当前HPO映射与八池准备在正式窗口前完成。控制器前检仅核已完成产物、来源/HPO/旧指标ref；不读train64评测gold。随后独占创建新output_dir与window.json：一个monotonic绝对deadline覆盖四次fresh fit/held infer、固定.3 greedy/nested、四预测freeze和唯一v2评测。每子进程只得到这个deadline的剩余秒，没有每fold900、旧120+30分段预算或1.5行数护栏。

所有子进程 `start_new_session=True`、单线程、隐藏CUDA、显式工作目录，PID/PGID落execution-state；控制器自身还有同一deadline的SIGALRM，超时或中断只对本控制器新建的对应group发TERM，最多30秒后KILL。清理期间屏蔽重复TERM/INT/ALRM，避免第二次信号跳过清理。保留真实returncode、信号、结束/残留group状态；KILL后最多5秒只用于reap收据，不延长计算窗口。既有正式output_dir拒绝使用，已失败任务和partial文件均不覆盖；不存在自动重跑/续训/扫描。外部SIGKILL无法由Python保存收尾，H32恢复仍须核已记录原PID/PGID，不能凭断线另起任务。

最终子阶段读取新fit manifest/source/feature/runtime/tree refs，校真实160迭代、F及tree-state绑定；不加载旧model或posterior。每文从**新NCR held posterior**调用原冻结 `decoder.greedy(...,.3)` 与 `decoder.decode(...,cut=.3)`，decoder源SHA还须等于强BioBERT freeze的proper_nesting_source_sha256。强/弱旧entities均不传入解码。仅深拷贝强B非entities，逐对象核与preB相同，新entities全部覆盖。

全部4份 `fold{k}/ncr.nested.jsonl` 完成、共同complete-freeze写成、完整SHA重新核验后，才导入冻结v2/validator、写evaluation-start并打开train64评测gold。原HPO版本为2026-06-23、all/text口径；每fold和pooled各评价新预测一次，不重评旧基线。输出fold `ncr.metrics.json/ncr.validation.json`、pooled两文件、results.json；强/弱差分都含menF1/docF1/Score，非entity关系数量、micF1/macF1及患者数核同强弱基线。

keep预定为 pooled `(menF1+docF1)/2` **严格超过强BioBERT**，且至少3/4fold该实体均值非降；无容差变更、cut扫描或事后改规则。结果不自动触发原dev；不读a179原dev预测，它只属于owner后继生产依赖的单独manifest。`execution_ready`仍由root在实际新池/预算/授权就绪后设置；本次只写源和局部静态/合成检查，没有启动任何子进程、读取gold或真实池、fit、预测或评测。
