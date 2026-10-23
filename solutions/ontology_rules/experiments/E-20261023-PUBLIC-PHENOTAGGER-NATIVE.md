# E-20261023-PUBLIC-PHENOTAGGER-NATIVE：公开原生模型实体控制

## 目标、约束与固定判据

目标是检验已发布 PhenoTagger 原生权重是否能低成本补足 CRF/Sap 的漏实体。唯一对照为既有 nested32→held16 的完整 base `8acd182b…`，Score **.6447615937066596**；沿用公开 evaluator v2 all/text。只补实体，所有原实体和所有非 entities 字段（包括 association）保持；新实体按原否定和非重叠规则取舍。成功证据是同一冻结输入上的实体/文档及总分提升；这是反复使用的训练内开发集，不是原 train64/dev16 ≥.8 验收。

只做一个固定控制，无阈值/模型搜索，无原dev和官方提交。与前面本项目从头训练的 HPO100ep 弱监督 Pheno 模型完全分开；本次公开模型不使用本项目训练标签，不下载外部人工表型标注集。

## 来源、协议与可复现入口

- 公开权重 `lingbionlp/PhenoTagger_v1.2@02ce3f529820dccca46bae168c1933ba25253125`，`pubmedbert_PT_v1.2.h5` SHA `d58af55a…`；122,759,794 参数/201变量原生完整恢复，17,266列原分类头保持。模型元数据 Apache-2.0，工具代码 MIT；附 [上游许可](../PHENOTAGGER-LICENSE)。发布权重历史人工评估参与选模的具体谱系仍不明，来源研究如实披露；不能将比赛允许公开工具等同主办方对本文件SHA的最终认定。
- 上游 `ncbi-nlp/PhenoTagger@10f8ee3e1ea4c26fb44c47df28a555082748ebf2`。ModelScope准确路径/revision未找到后复用已下载公开原权重。来源及CPU/GPU201变量检查在外部 `research/public-phenotagger-reuse/`；[推断计划](../evidence/round2/public-phenotagger-native-plan.json)固定资源SHA和实际依赖。
- 仅修复事前无标签确认的跨句 ngram 上限递减错误，每句独立 min(8,len)，不改变其他过滤（H36）。候选70,971→75,790，52,185 token 坐标全过；65,406相同lower-token输入缓存复用。
- 原native hybrid：原top3顺序、float32概率先5位舍入再严格 >.95，原词典合并/缩写传播；`onlyLongest=False`、`abbr=True`、无祖先扩展。原head分类后才映射当前HPO，不重算softmax。此为明确的项目适配，不宣称上游CLI完全原样。
- Git [public_phenotagger_native.py](../public_phenotagger_native.py) 与远端已执行入口 e6dbdc72 字节相同；[候选修复函数](../public_phenotagger_candidate_function.py) 与1569e545字节相同，保留上游空白。按原部署目录运行，资源根为cwd，实际恢复命令/环境见计划所引full-inference-job；不重复执行已完成native输出。
- [实体装配](../assemble_public_phenotagger.py) 只读取 input/base/原生结果/HPO：score降序、length降序、offset和ID升序；原 is_negated；拒绝原实体及已接受新增实体的任意重叠。原生decoder已固定阈值，不给词典或缩写重新设门槛。坐标逐项核对后保留所有base对象，association逐项相等。
- [CPU终点](../run_public_phenotagger_endpoint.py) 先装配并写完整predfreeze，再格式检查和唯一v2，旧base指标绑定复用；[固定装配计划](../evidence/round2/public-phenotagger-assembly-plan.json) 保存全部路径/源SHA。七个合成差异检查覆盖否定、两类重叠、无新阈值、关系保留、空候选、坐标拒绝；没有读取gold。

## 实际准备与运行

main共享更新已同步 `4b92005`，221个solutions文件SHA集合不变。研究helper负责native模型/环境及单次推断；本方案收编其冻结源并负责装配/唯一质量终点，不重复下载、GPU或native推断。

2026-10-23 20:57:48Z，A100 GPU2 UUID `GPU-2c22a952-40e2-952f-854c-ff668b5c9dc1`，proc `proc_767916f8-f75b-4c54-bebd-7281de690c71`；统一授权1500秒+TERM30，实际外层82.047秒/程序75.315秒/神经45.882秒，峰664,721,152B，rc0。75,790神经候选+1,174词典候选，native合并1,756→缩写1,769→当前有效1,769（0drop）；结果1769只是预测数量，不是正确数。GPU已fresh核空释放。

原生完整小终点 `research/public-phenotagger-reuse/native-pipeline/terminal/`：canonical SHA `de7de49e…`，manifest `2fc80461…`，transfer `d86cec42…`。31.6MB unique raw和47.0MB全文候选仅留远端持久目录，完整SHA/大小见manifest/transfer；权重/依赖同机持久可恢复。Git仅源/小计划/摘要。

## 唯一完整质量终点与决定

21:01:47.903Z先冻结完整pred `24d18da2…`（freeze `ee7bfb29…`），21:01:48.234Z才启动唯一v2。1769原生候选中1374与原实体重叠、60否定、27与先接受新增实体重叠，接受308；所有原实体/所有非entities字段完全保留。格式0错/0警。

| 指标 | base | 固定原生实体补充 | 差值 |
|---|---:|---:|---:|
| menF1 | .667191188 | .689122807 | +.021931619 |
| docF1 | .747747748 | .780198020 | +.032450272 |
| micF1 | .616151545 | .616151545 | 0 |
| macF1 | .547955894 | .547955894 | 0 |
| Score | .644761594 | **.658357066** | **+.013595473** |

新增实体134TP/174FP（精度43.51%）；文档+93TP/90FP，关系TP/FP/FN全部不变。独审 `validation/public-phenotagger-terminal-review/review.json` SHA `7ed866e20c6a68522d40edaee8fe226d9e078cfade114cc2c1327c80010cdad6`：独立CPU重装字节一致、v2完整字段相等、来源/预算/GPU释放及先freeze后gold通过，复用原生201变量/候选证据，无重复NN。

**结论：本固定公开原生模型提供有用的互补实体召回，总分明显提高，但新增精度较低，单个反复使用的训练内留出不证明跨fold稳定。** 不扫描阈值、不改关系、不原dev。root授权下一独立控制：相同固定政策迁移outer0，直接复用helper正在构建的train64无标签原生缓存；本路线不重复NN，sequence另负责严格OOF可学习删换。[完整结果与恢复引用](../evidence/round2/public-phenotagger-native-results.json)。
