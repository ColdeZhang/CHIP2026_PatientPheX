# 任务背景

人类表型（Phenotype）是指由基因型与环境共同作用产生、能够在患者身上观察或测量到的临床特征（如身高、血压、症状组合、药物反应等）。表型不仅是疾病的外在表现描述，也是罕见病诊断、疾病机制研究、基因发现和精准医疗的重要依据。随着生物医学文献的快速增长，如何从海量非结构化文本中自动抽取、标准化和组织患者表型知识，已成为生物医学信息处理领域的重要研究方向。

表型概念识别（Phenotype Concept Recognition）目标从生物医学文本中，自动识别出表型实体并将其标准化至本体库中的标准概念，仍面临着诸多挑战。一方面，生物医学全文篇幅长，表型描述形式多样，存在缩写、省略表达、否定、复合实体等复杂现象，增加了识别难度；另一方面，人类表型本体（HPO）[1]概念规模庞大、层次结构复杂，单纯依赖字符串匹配难以准确完成表型标准化映射。此外，现有研究[2-4]多数关注从给定短文本中识别出所有表型概念，而缺少研究全文的特定患者表型概念信息抽取。在生物医学文献中，往往同时存在多个患者或家系成员，不同个体可能具有不同的临床表现，仅抽取表型实体难以满足患者画像构建、家系遗传分析及个体化医学研究的需求。

因此，本评测任务将聚焦“面向生物医学全文的特定患者表型概念识别”，旨在从文献中自动识别给定患者的表型概念信息，建立患者与表型之间的对应关系，为患者级知识库构建、基因型\-表型关联等下游研究以及精准医疗应用提供技术支撑。

# 任务内容

本评测任务面向PMC公开的生物医学文献全文，评估和比较各参赛模型在患者表型个体化信息抽取任务中的性能。任务以生物医学文献全文作为输入，模型目标自动识别特定患者的表型概念信息，共包含两个子任务：

- **子任务1-全文表型概念识别**：不区分患者，从文献全文中识别出所有表型概念，包括文中具体位置，并将其标准化映射至人类表型本体（HPO，统一使用v2026\-06\-23版本，Phenotypic abnormality （HP:0000118）分支）对应术语概念。

- **子任务2-特定患者表型概念识别**：给定特定患者，识别出与该患者相关的表型概念，给出相关表型的HPO标识符集合。

参赛队伍可采用传统机器学习方法、深度学习模型、预训练语言模型或大语言模型等技术方案完成任务（要求模型参数不超过10B），可结合提示学习、监督微调、检索增强生成（RAG）等方法，但需在技术报告中说明所采用的模型及外部知识资源，不允许使用外部人工标注表型数据集。

# 结果提交

## 1 提交方式

比赛设立A榜阶段与B榜阶段两个环节，具体安排如下：

- **A榜阶段（开发阶段）**：参赛队伍在成功报名后，可通过天池平台下载训练数据和A榜测试数据集（开发集）文件，在本地进行模型调试与训练，并在线提交评测结果。A榜阶段成绩仅用于模型开发调优，不计入比赛最终成绩。且A榜测试集答案不公开，不用于B榜阶段模型训练。

- **B榜阶段（最终评测）**：B榜阶段将开放B榜测试数据集（测试集）供下载，参赛队伍需根据该数据集生成预测结果并提交至天池平台，各队伍在B榜阶段的历史最优成绩作为最终排名与奖项评定的依据。

**提交说明：**

- A榜每支参赛队伍每日最多可向天池平台提交5次评测结果，B榜每日3次。

- 排行榜整点更新，按评测指标从高到低进行排名。

- 评测失败不扣除提交次数。

## 2 提交格式

A榜和B榜均要求提交对应测试集预测结果的Jsonl文件，文件编码为UTF\-8，格式和训练集中结果格式一致，可参考数据集中提交样例，具体格式如下：

```JSON
{
  "pmc_id": "...",
  "pmid": "...",
  "entities": [
    {"identifier": "HP:0000519",
      "type": "Phenotype",
      "offset": 206,
      "length": 30,
      "text": "bilateral congenital cataracts",
      "note": null},
      ...
   ],
   "association": [
    {
      "patient_id": "P1",
      "phenotype": ["HP:0011410",...],
    },
    ...
  ]
}
```

字段说明：

- entities：全文所有表型实体预测结果
- association：特定患者的表型实体预测结果

# 奖励设置

本任务为学术评测任务，获奖队伍将会获得获奖证书，进行大会汇报、发表英文评测论文（EI收录）。

# 参赛规则

- 每名选手仅限注册一个账号，一旦发现选手注册多个账号进行提交，将取消所有账号的参赛资格。
- 在A榜结束前，填写报名信息表[(点击此处)](https://docs.qq.com/form/page/DRE5IVEJJbnRvZkhK)
- 允许使用外部资源，包括但不限于代码、工具和知识库，但不允许使用外部人工标注表型数据集，要求所使用的资源是公开可获取，须注明来源。
- 获奖队伍需要注册并参加CHIP会议，在会议上进行口头汇报，撰写技术论文，否则取消参赛成绩，名次递补。（注：会议注册费、食宿交通费用由参赛团队自理）。
- 国内外在校生及企业人员均可参赛，个人须依托单位报名参赛，比赛组织方成员不得参赛。实名认证说明：参加比赛的队伍（包括队长及全体队伍成员）需要在报名截止前完成实名认证（认证入口：官网\-右上角个人中心\-认证），未完成认证的参赛队伍将被取消比赛资格。
- 组队人数限制1\-5人。
- 该任务数据仅用于本次评测，如需用于其他目的，请与任务组织者联系。



# 获奖队伍材料提交

获奖队伍需提交如下材料：

- 相关代码及说明。
- 评测论文，可参考往届CHIP会议评测论文（[http://www.cips-chip.org.cn/2024/accepted](http://www.cips-chip.org.cn/2024/accepted)）。
- 如果方法使用了额外公开的资源，要求说明并提供资源文件或地址，包括但不限于代码、工具和知识库。要求所使用的资源是公开可获取的。

选手需要将以上材料在评测论文提交截止日期前发送至邮箱，邮箱地址：dutir_bionlp@163.com，邮件的标题为：“CHIP2026Task1-评测材料-参赛队伍名称”。代码及文档需打包成一个文件（zip，rar等均可），作为邮件附件传送。

说明： 若没有提交上述材料，将视为自动放弃参与评奖。评奖资格将会自动顺延至排名紧随其后的队伍。

# 日程安排


| 事项 | 时间 |
|---|---|
|报名时间|8月20日-9月27日|
|A榜评测开始|8月20日|
|A榜评测截止|9月27日|
|B榜评测开始|9月28日|
|B榜评测截止|10月1日|
|评测论文提交截止|10月15日(暂定)|
|会议日期|10月30日\-11月1日|

# 评测任务组织者

**组织单位**\
大连理工大学       罗凌、王健、孙媛媛、林鸿飞

**联系人**\
朱旋律（xlzhu@mail\.dlut\.edu\.cn）

**相关网站链接**\
CHIP官网：http://cips-chip.org.cn/ \
GitHub地址：https://github.com/DUTIR-BioNLP/PatientPheX

**钉钉群**

大赛官方钉钉群请扫码加入，最新通知将会第一时间在群内同步。

<img src=https://static.tianchi.aliyun.com/tianchi/race/images/prod/532512/1242990779776242/otzc88vq6rj6nqcs.png width="300">

# 参考文献
[1] Gargano MA, Matentzoglu N, Coleman B, et al. The Human Phenotype Ontology in 2024: phenotypes around the world. Nucleic acids research. 2024;52(D1):D1333-46.\
[2] Luo L, Yan S, Lai PT, et al. PhenoTagger: a hybrid method for phenotype concept recognition using human phenotype ontology. Bioinformatics. 2021;37(13):1884-90.\
[3] Feng Y, Qi L, Tian W. PhenoBERT: a combined deep learning method for automated recognition of human phenotype ontology. IEEE/ACM Transactions on Computational Biology and Bioinformatics. 2022;20(2):1269-77.\
[4] Tao Y, Huang Y, Wang Y, et al. AutoPCR: automated phenotype concept recognition by prompting. Bioinformatics. 2026;42(Supplement_1):btag304.

# 引用

如果您使用到了本评测数据，请引用。

```
@article{PatientPheX,
  title="{PatientPheX: A Full-Text Biomedical Corpus for Patient-Specific Phenotype Information Extraction}",
  author={Xuanlyu Zhu, Ling Luo, Zhijun Wang, Chentao Huang, Yunxiang Li, Yang Zhong, Tengxiao Lv, Juntao Li, Chao Liu, Feng Tian, Yuanyuan Sun, Jian Wang, Hongfei Linn},
  journal={bioRxiv},
  year={2026},
  doi = {10.64898/2026.10.07.757319},
  url = {https://doi.org/10.64898/2026.10.07.757319},
}
```
