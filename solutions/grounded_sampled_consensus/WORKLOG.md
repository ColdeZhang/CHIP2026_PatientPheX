# 工作台账

分支 `explore/grounded-sampled-consensus`，起点 main `504659ec`，仅修改 `solutions/grounded_sampled_consensus/`。外部产物 `research/grounded-sampled-consensus-v1`。

## 当前任务

**固定双采样已负结案、终审通过、最小补包已持久**：F1 Score .701325377212→.705226083109（14新9TP5FP），F0 .728195628911→.725132683325（13新2TP11FP），F0 micro/mac均降，故不满足预定双折都正，不迁full64/原dev、不扫seed。F0完整94原205件collectionab5781b7，1374.396秒；89EOS/5cap整行弃答，全部47患者保留。592合法anchors→195公共Sap8.4366秒/13原件b6c4→完整freeze b26048a8 后唯一1v2/旧0。统一入口1d23783b，结果0e3eb2ba，见evidence/F0-result.json。全部GPU14:52:39.579562真实释放并交copy_decode，当前仅本地结案/持久小包。F1 root实际restore721cd7d7已核。

原 train32 grounded 模型 d52、原 F1 16篇/39患者全文输入；固定 seed20261024/25、native nonthinking sampling T.7/p.8/k20/minp0/repetition1/max1600。两完整39经原严格 quote parser 与相同 train32 alias/public41099/b59政策后，唯一主端点 B4(524关系/.701325377212) UNION 两seed literal关系交集。新关系为零则原基底byteexact且0v2；否则完整freeze后1v2/旧0。严格Score正才F0；双折正才full64。

初提presence1.5在任何源冻结/模型/质量前由root撤回；最终presence0，无新customprocessor，避免惩罚合法重复quote。旧闭集CID双采样是候选限定的患者集合生成，本路线检验开放quote及独立规范化后的共识；不能按新名字重做旧任务。

## 两任务边界

旧 `explore/grounded-generation@5166c629` full64/36fresh供给已完整，统一review58141a6f通过，正在独立最小补包；原模型、输出与source freeze不改。新采样仅用旧train32 d52，不使用新full64 e129，不读originaldev gold、不查询或提交官网。

## 执行

先实际原生39输入/新78随机种子接口CPU120+5；源/CPU/成本checkpoint后，A800 GPU2 fresh空卡门唯一1800+30包含两seed/一次加载，不扩截止、不重训、不按格式挑重试。全78及必要Sap180+30齐才组装。未知MCP状态恢复原intent，不重复start。当前仅准备，没有GPU任务。

预算前置修订：root在任何GPU start/intent之前根据1779.411秒保守估计，明确将唯一正式78共享上限1800改2400+TERM30。原1800源f6a85817/decision保留，新decision-budget-v2记录；CPU120+5、全部39×2输入/1600/两seed/normalizer/单quality均不改，无GPU失败/试跑。

原CPU唯一proc1e022915/driver32173在13:44:17完整结束，6.779秒；12实际件collection3882542a全部SHA收齐。39全文native/HF/local IDs与模板/decoder全等，78种子无重复、同seedCPU draws一致，原d52四cache文件精确SHA通过；没有GPU/fit。实际GPU计划e3707f21绑定原d52及新2400+30，全部78 fresh，下一步先checkpoint再fresh卡门唯一启动。固定后处理SOURCE-HANDOFF c18ad4fa与合成4165bc40也已完整，不读heldgold。

迁机前置修订：原A800完整部署但从未GPU intent/start；卡交本体full64C。root改为.100 A30 GPU0，原native39 CPU receipt复用。actual249dad20核Torch2.7.1/TF5.20.2/PEFT.21.2/tokenizers.23.1及两runtime SHA同一，base15件stat仍同旧全SHA证书，tokenizer/config四件新SHA通过，两卡空闲/盘396GB；3个准确目录均无d52。仅四件173197471B私网同字节复制，原完整训练state不迁移。

本机真实同9B native116/128行分别805.858/851.003秒；按输出tokens代理新78约2369秒，1.45余量+150约3585秒，因此root在任何GPUstart前明确唯一共享3600+TERM30，decision-host-budget-v3记录原1800→2400→3600且旧源保留。每seed正式最长10080输入先行、原每行随机种子使排序独立，1600 cap和全部78不改；每行保存实际峰值显存，禁止临时截断/量化/KV变更。

13:51:33四件173197471B已完整私网复制20.214秒，source/destination SHA+fsync/reopen及祖先目录通过；原源保留，0 optimizer/NN。A30最终plan d5c60e9a、package0e8dbe24绑定19件和原CPU3882542a，不跨host假复用部署文件。独立源HTTP180+10原TTL待核关闭；准备源checkpoint后唯一正式GPU启动。

新78实际入口已经完整绑定；A30计划/源检查点f599620f已远端核。必要Sap按root新队列固定使用同GPU0在9B真实释放后一次180+30，复用原publicbank与train32政策，不等待GPU1、不共驻9B。

13:55原4文件HTTP已实际关闭：原managedPID不存在、准确namespace无进程、端口无监听，closure8f97e1dc，无kill/重开服务。必要Sap normalization_runtime已3f20aee6远端核，15metadata用例通过，仅准备不执行；完整78+真实release之后才绑定/查询。

F1收口：collection19b81037原173件，1411.159s，alloc21.012GB/reserved21.922GB；Sap6ccab164原13件/10.852s。两纯关系253/245、交集163；保护B4原524后538，全部1189实体/39患者/16文献及非关系字节保持。14:19:22完整freeze1527f730后唯一v2，predf9c33866/resultf9e268a2，0单seed/其他arm。汇总脚本首次仅SyntaxError，未写文件/模型/质量；机械修正只整理已有结果。

F1正式独审：c54828bd/12959B，41safe snapshots/manifestddd801e5稳定；入口9bd386cc保原pending时点不改，另evidence/F1-final-review.json追加接受与persistence指针。F0源机械迁移在首次GPU前静读修正指标名称F1不随fold改名、完整94字段，未产生预测或质量；同接口/阈值/模型不变。native47仅新输入身份与94seed去重，原sampler toy/processor复用。

root已唯一实际读回F1补包，3 read_file，128内容+1embedded全部SHA/fsync/reopen，manifest字节exact；恢复721cd7d7/17273B见evidence/F1-actual-restore.json。未改9bd386cc入口/1527f730质量freeze，未复制旧权重或重模型。

F0完整收口没有模型或质量重试；future f0_entry原写入器首次即通过，完整源/实际plan逐阶段预推。原94与195查询各唯一作业，严格弃答保持，不根据已知质量修正。末段独审followup曾因全局7槽满未启动，root协调后续；该资源约束不改变已经完整的负结论。

F0统一独审a8bd9a47/6741B、manifest17580d0f稳定通过完整性和负结案。唯一最小补包74355340/732980B、manifestf984460a/40509B，90内容+1embedded，.100实际逐成员SHA/fsync/reopen收据a4ef2d40/34911B，路径见persistence/F0-receipt.json；旧F1包和大权重不重复。root后续唯一下载恢复，当前不冒称已本地读回远端包。
