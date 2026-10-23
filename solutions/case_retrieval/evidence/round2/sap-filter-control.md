# 固定 Sap 跨度筛选控制

状态：准备与真实合并/评测均已完成，原同构HPO负控制完整保留；随后只使用唯一工作点 `p >= 0.65 && cos >= 0.92`。这些是本体路线已有阈值，本次不扫描；原 fold0 曾参与该路线选型，不能称为全新未触碰测试集。

## 模型与隔离

- 冻结来源 `ontology_rules/frozen-fold0-sapbert/`：train48 SHA `d87bf189d2f861f9da759bfafd51e1384a246e087a695780a4a0cc4a335fab7d`；不是 train64/dev 模型。
- SapBERT revision `090663c3ae57bf35ffe4d0d468a2a88d03051a4d`，FP16 派生权重 SHA `dc5e5071ed867afd44cc2fdb21974f2c4ca2288442ac765be353eec80c2e4814`。没有比赛数据微调；实际模型文件均重新核对 SHA。
- 原 LR 未序列化。明确使用“冻结模型精确重建”：原 train48 字节及顺序、源码、seed、原负例逻辑和只读 embedding 缓存；cache miss 立即失败，禁止生成新训练向量。3600 正跨度、23353 负跨度、20 次迭代与原记录一致。
- 原 14,480 候选完整回放：概率最大差为 0；CPU 与原 CUDA 矩阵乘法的 cos 最大差 `1.430511474609375e-6`，小于预定 `2e-6`；两个单项阈值、联合阈值及 nearest ID 全部 0 差异。证据见 [重建报告](sap-filter-classifier-reconstruction.json)。
- 重建系数 SHA `1146433a8157b787f12125f34f2e54daf939a136e2528a21f820dd502ef7708e`。继续使用原 **42,236** 条 ontology/train48 bank，SHA `d8bb8709a238a562b45913c19878c78e45584eaf755678fdbe3429fdb76220b8`；不能替换成检索 HPO 链的 43,437 条 bank。

## 新跨度评分与合并边界

源码入口 [sap_span_control.py](../../sap_span_control.py) 的 `prepare/score/merge`；重建入口 [reconstruct_sap_detector.py](../../reconstruct_sap_detector.py)。只在 `score` 显式初始化编码器；沿用原逐文献 `sorted(unique(norm(surface)))`、batch256、max_length32、FP16 backbone、FP32 CLS 后 L2 归一化。分类特征仅为 span embedding，没有患者或段落上下文。

`repaired101/requests.jsonl` 包含 1,068 个原文精确跨度，368 个与 CRF 全部实体无字符重叠；原90窗口942/319请求另存，不覆盖。评分在 root 交接的 .3 GPU0 完成，stderr 空，全部16文献/1068跨度覆盖；结束复查目标 GPU 无进程后归还本体路线。评分 SHA `619f352e4bbf2420ac1bc70f5e9c915f4eaaf156f38e9de6a1068a09e38f8dc7`，见[评分来源](sap-filter-repaired101-scores.json)。未读取验证 gold 来筛选或设阈值。

`merge` 必须接收 HPO 链已完成的 **new-only** 预测，核对 request/score/reconstruction/source hashes 后只筛非交叠新增跨度。保留已有 CRF 实体及链接器的 HPO ID；不增加否定、词边界或优先级规则；不在 LLM 新跨度间再做去重叠。它暂时保留基线 association，**完整 Score 前须给未筛选控制组和筛选组使用同一个患者归属入口重算**。

外存根目录 `/workspace/patientphex-runs/round2-20261023/case_retrieval/sap-filter-control/` 包含请求、完整分数、模型系数、运行命令和远端进程凭据；大 bank 与原模型留在 .3 原目录只读。原始完整日志/向量不入 Git。

## 核验

6 项合成数据测试和 compile 通过：阈值端点、接触但不交叠的跨度、原实体/HPO/association 保留、错误训练折、未通过重建、缺失分数与未绑定分数拒绝。见[测试证据](sap-filter-control-tests.json)。上述测试不是比赛评分。后续真实内部评测保留18新增跨度，含HPO为12TP/6FP，固定原规则关联Score .711354→.714061；同k21/.35完整管线 .657639→.662397。完整结果与SHA见[sft-linked-filtered-control-summary.json](sft-linked-filtered-control-summary.json)，待独立复核和另一固定内折，不宣称已达标或可泛化。

独立审查补充：旧14,480池使用原embedding重建LR时p逐项相同；新1068池重新按其逐文献unique列表做FP16批编码，与旧池共享829个相同`(pmc, offset, length, text)`项的p最大差0.0012167349、cos最大差0.00036394596，但两个单阈值及联合阈值决定全相同。这两种核验不可混称bitwise复现；新批次形状可产生数值微差，实际分数文件保持冻结。
