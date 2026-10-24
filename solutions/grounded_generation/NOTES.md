# 研究笔记与边界

## 已核旧结果

来源/实际版本见 evidence/initial-sources.json 中 old_8B_history，重点 r2-08 至 r2-13。

- 旧Qwen3-8B全局窗口quote：零样本跨度F1 .3983，SFT配对 .5392，唯一格式恢复后 .5600。新增跨度噪声仍大，不能裸并集。
- 固定Sap过滤后的实体消融：F0 .711354→.714061；另一折F1 .644796→.643815，收益未复现。其关系检索小涨不超过强规则，不能声称稳定有效。
- 旧完整患者自由HP+quote：SFT严格grounded仅3TP/32FP/376FN，Score .396343；exact别名检查没有一个生成HP编号语义一致。字面quote真实不保证HP含义或归属正确。
- 新9B闭集/患者优先能提升组合，但候选漏召回仍是上限。患者均衡α=.5主臂 .661497低原 .663842，真实负关闭，不扫描α。

## 本轮可检验差异

按目标患者给完整原文与全部患者mention，输出仅`[unit_id, exact_quote]`；独立规范化器不能接受模型臆造HP号。原文单位是原full_text passage，单位ID由输入位置决定，不依gold切句；全部offset可逆。主比较保持实体，只测新增关系。

训练的患者–HPO标签是原标注；与之同HPO的实体出现只是候选证据，最近患者marker选择是弱监督假设。记录多候选、其他患者更近、无可定位实体、compound/alt/NIL等情况。prepared空不等于gold空；不为了任务形式把不可构建正项写成确定负证据。

规范化的首选资产是公开SapBERT及公开HPO、仅train32别名。旧train48/64学习头、别名计数或阈值选择不能冒充同32资源。未匹配不自动当NIL；NIL沿官方原文短语语义处理。最终政策由train-only覆盖盘点先冻结，不读取F1标签挑阈值。

## 工程复用限制

已核indexed-head核心可避免对全部位置投影，旧最长13549压力峰32.693GB仅覆盖短候选答案。新quote监督可能更长，须计算全部目标位置并验证本次实际最难状态。保持原生tokenizer发布链/精确EOS、raw-first、完整durable状态、实际None kwargs等既有规则；旧probe不替代新目标的压力证据。

## 外部方法来源

后续只引用实际查阅的原论文/代码及固定版本，不把方法名当已实现证明；优先关注可逆结构化抽取、复制约束和独立实体链接。当前上述决定基于已完成项目证据，尚未冒称复现某篇论文。


2026-10-24 GPU前澄清：全文先判目标患者的概念，复制quote只作概念词面锚定，不把该处出现当患者归属证据；最近mention只是确定训练代表词面的弱选取规则。首CPU产物targets-v1与源c3b69eb保留，v2仅修正prompt语义，目标/身份/全文不变并逐项对照。170/814处其他患者更近已如实披露。

2026-10-24实读原论文摘要：https://arxiv.org/abs/2101.05779 ，Paolini等《Structured Prediction as Translation between Augmented Natural Languages》(TANL)。其核心是把联合实体/关系等结构化预测表述为可解析的增强自然语言翻译。本路线借鉴可解析文本输出这一任务表达；没有声称复现TANL架构、训练数据或论文benchmark，也未据论文替代当前病例关系/词面弱监督边界。

## 补充成熟结构化抽取依据（只读摘要，不改变当前端点）

实际读取[GenIE](https://arxiv.org/abs/2112.08340)与[UIE](https://arxiv.org/abs/2203.12277)原始摘要，元数据/摘要校验在`evidence/methodology-primaryrefs.json`与仓库外缓存。GenIE以生成式triples配合两级约束保证知识库schema合法，说明生成结构与规范化约束应分层；其closed-IE结果不能直接当本比赛患者归属效果。本轮是生成后精确定位检查，并没有实现论文的解码时约束，不能把两者混称。

UIE将异构抽取统一为带schema提示的结构生成，与本轮把患者归属和可复制词面明确分工相符；仅借任务表达思想，没有导入其模型或训练集，也不能据其13数据集结果预测只有32篇监督下本任务收益。若本轮后续真实输出显示格式/不可定位项主导，解码约束可以作为另一个明确单变量假设；现在不加新arm、不改greedy1600或解码策略，先由完整F1裁决。

### 已确认的接口坑：旧指标文件不一定内嵌身份字段

公开v2标准metrics可只有metrics/config/counts，pred/gold SHA在原pairedfreeze/evaluation-bound/summary。消费旧基线时先核实际schema，再沿冻结链绑定，不能猜prediction_sha256字段。GG首组装已在任何新pred/gold/v2之前发现并保留；独立恢复只改metadata绑定，未改变科研端点。对应根H39的真实返回对象/manifest合同问题，不凭字段缺失重评旧基线。
