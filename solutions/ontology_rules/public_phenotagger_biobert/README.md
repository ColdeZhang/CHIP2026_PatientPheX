# 第三公开 BioBERT PhenoTagger head CPU准备

仅下载固定 `lingbionlp/PhenoTagger_v1.2@02ce3f529820dccca46bae168c1933ba25253125/biobert_PT_v1.2.h5`。元数据486,624,416B/SHA1c4d799c…，需完整实际核验。准确ModelScope镜像404；已有BioBERT config/vocab的MS/HF副本逐字相同，复用而不下载基座。

没有找到此前该head的实际任务运行；名称或预期结构差异不代表收益。使用现有A100原TF2.12私有环境、CUDA隐藏、CPU线程1；300秒有界分块下载及180秒原生CPU strict加载。所有变量先置NaN，topological load不skip，再逐H5数据集完整相等/finite核对、17266列及current映射保持；四固定合成短语只验证表示/finite/sum1，不是赛事推断/质量。预期201变量/121587826参数须以实际产物确认。

`manage.py download/status/cpu/collect` 是外存原件的部署入口，实际目录 `/workspace/patientphex-runs/round2-20261023/research/public-phenotagger-biobert/`，源码逐字镜像在这里。所需上游representation/label/config/vocab仅外存，plan列SHA。H5、完整stdout/stderr留外部；临时CDN URL私有且完成删除。无GPU、训练、task标签/原dev或质量。未来缓存规模/耗时仅估计，须另行授权执行。
