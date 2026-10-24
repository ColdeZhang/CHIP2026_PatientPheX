# 患者归属理由蒸馏：只读与 CPU 供给准备

状态：唯一15条公开R1教师供给正在.100 GPU0执行，driver2752489/child2752493；15:57:26 UTC开始，TERM16:42:26/KILL16:42:56。最长9400输入先行，固定3072；0学生训练/0held gold/0质量。恢复见 `evidence/pilot-start.json`，不可重复启动。

目标：判断训练标签约束的教师理由是否提供直接 JSON 标签、弱 quote 与 DPO 之外的新监督，先检验可定位证据和患者歧义的真实供给。

约束：原 train64/dev16 最终目标≥.8；当前最佳保持。无新大权重下载、无 GPU；不把 nearest/same-sentence 或答案正确等同人工确认的归属理由。

成功标准：准确查重；给出有来源的 train-only 覆盖与验证局限，决定是否值得一个小型供给探针。尚未提出或执行训练配方。

- 2026-10-24：从 main `ff7ad16777e3907ed3b4b4231794439017aa5bf8` 新建 `research/verified-rationale`；已读当前共享规则/HANDOFF/MAP。
- 已核旧 grounded 的目标是 nearest 选取的弱短语；R1 SFT 显式只监督 JSON+EOS，不制造思维链；RAG 目标保持旧标签。完整 public R1 thinking 是推断，不是教师理由蒸馏。
- CPU 全32文/87患者完成：814合法原子正关系均有原文词面，145关系的所有出现都更近别患者，横跨15篇。每篇SHA固定一条，共15条；全文输入和公共名称已构造，未分词/模型调用。详见 [报告](REPORT.md)。
- 决定：机制有新意且有训练供给动机，但没有可信教师理由产量，暂不提出全量蒸馏训练。仅建议后续有界15条训练集供给探针，由root调度；exactquote/金标答案一致不冒称人工归属证据。
- 记录范围已完成：0 held gold / 0 model / 0 fit / 0 quality；15条草案与1.04MB训练库存留仓外，Git只放源码与小摘要。新脚本真实CPU运行、链接/编译/diff/体积检查后提交并推送本分支。
- 原检查点 `e1c17c1cfac126bd215256b2df567c05d8781511` 已推送并核远端。root继续授权 CPU native15/盘和网络成本准备；学生训练仍禁止。
- .100 native15真实CPU全过，最长9400/总82445；已预锁单15pilot与证据验收，[计划](pilot-plan.json)。.12盘余量49.46GB可容一份16.39GB缓存；32MiB探针target上传遇离线，未启动任何服务/接收器。15:44之后只读在线退避，保留失败，无16GB复制/模型调用。
- `cbce430a87ce4dbb3f38c816ad91b11ca46fe088`已推送，包含native15/固定机械及语义验收草案。随后本地对照发现AddedToken字典使草案字符摘要错误；原v2保留，修正后15渲染SHA与native全等、messages输入字节不变，没重CPU。成本亦按78fresh同scope纠正为103199/3431.089269，pilot预案2700+30，尚未执行。
- 原native两件payload（100960B encoded+3871B audit）已实际全量回读逐SHA；无partial冒称完整。原16GB迁移仍未开始，网络小探针等15:44一次退避在线/盘/GPU库存。
- 网络原回执UNKNOWN已只读完整恢复：32MiB2.737733s、11.69MiB/s，server实际关闭；16.39GB纯网络1337.16s>900门，root取消迁移，不扩大复制预算。
- root已授权等待.100 Copy47+必要Sap全部释放后，原缓存同15/3072/2700+30唯一pilot。`host-revision.json`保原.12提案；36层真实KV/SDPA/logits_to_keep、旧19.50GB峰给出A30条件可行依据，freshfree必须实际比较，OOM首条即停。
- 当前源码 `pilot.py` / `pilot_driver.py` 已实现并通过编译/静核；H69实际3类拒绝均无child；正做source-bound execution/deploy，无GPUstart。Copy owner承诺整链实际release再交接，不能在G47原模型结束时抢卡。

- 12件小payload已在.100原模型host逐SHA核验部署，[入口](evidence/deployed.json)。当前未创建admission/启动意图；唯一启动器要求上游整链release凭据，先freshGPU门再生成5分钟SHA绑定admission；UNKNOWN只查原作业，不自动重发。
- root对实际pilot/driver/evidence_contract和execution2549842e的增量静核PASS，2780B/SHA df947aee，[独立指针](evidence/source-review.json)；未重CPU/GPU/质量。当前继续等copy47+必要Sap整链release，实际启动不再需要重复预算确认。

- 15:56:38 Copy47+126必要Sap整链实际release后，freshGPU0=0MiB/noapps/24164free；唯一proc_e0d16e85于15:57:26启动，原2700+30时钟/300秒admission均已绑定。先保存raw再解析，等完整15后逐条语义审查；不据部分行判供给通过。
