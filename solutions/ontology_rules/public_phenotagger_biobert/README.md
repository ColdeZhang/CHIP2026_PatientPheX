# 第三公开 BioBERT PhenoTagger head CPU准备

仅下载固定 `lingbionlp/PhenoTagger_v1.2@02ce3f529820dccca46bae168c1933ba25253125/biobert_PT_v1.2.h5`。元数据486,624,416B/SHA1c4d799c…，需完整实际核验。准确ModelScope镜像404；已有BioBERT config/vocab的MS/HF副本逐字相同，复用而不下载基座。

没有找到此前该head的实际任务运行；名称或预期结构差异不代表收益。使用现有A100原TF2.12私有环境、CUDA隐藏、CPU线程1；300秒有界分块下载及180秒原生CPU strict加载。所有变量先置NaN，topological load不skip，再逐H5数据集完整相等/finite核对、17266列及current映射保持；四固定合成短语只验证表示/finite/sum1，不是赛事推断/质量。预期201变量/121587826参数须以实际产物确认。

`manage.py download/status/cpu/collect` 是外存原件的部署入口，实际目录 `/workspace/patientphex-runs/round2-20261023/research/public-phenotagger-biobert/`，源码逐字镜像在这里。所需上游representation/label/config/vocab仅外存，plan列SHA。H5、完整stdout/stderr留外部；临时CDN URL私有且完成删除。无GPU、训练、task标签/原dev或质量。未来缓存规模/耗时仅估计，须另行授权执行。

首次CPU启动漏带既有 `FORCE_TF_AVAILABLE=1`，Transformers在构建encoder前拒绝；5.375秒失败/state/stderr独立保留，没有加载H5或任务分数。`attempt2` 只补回原环境标记，CPU硬限174秒使两次名义累计仍小于180；H5/model/source数学不变，不重新下载。后续状态以实际恢复终点为准。

实际准备已完成：59.266秒下载，全H5大小/SHA与固定元数据一致；原生201变量、121587826参数逐数组exact，完整17266列与原映射保持，4合成输出finite/sum1，GPU设备为空。恢复CPU20.054秒、含首失败总25.430秒<180。完整小终点外存 `completion.json`；质量未执行。详细边界见上级实验 `P-20261023-PUBLIC-PHENO-BIOBERT.md`。
