# 条件 train64 准备：尚未训练

仅在当前F0独立增益或主线程固定2of3通过后，才可能由主线程批准fresh full64模型。当前完整CPU：原train64 64篇173患者全保，1711有效原子关系都有可定位词面，1678去重quote；6无效/分支外标签保留说明。378个选择离其他患者更近，仍是弱词面锚定，不是假称人工归属span。原dev纯16篇36患者，未打开dev gold。

**隔离作用域**：原train64包含内部F0/F1监督。这批目标、alias与token文件仅在本独立准备目录，当前F0的d52模型、train32 alias、输入、解码和全部阈值早已冻结，不读取或使用这里的任何资源。此处held指原dev16，不应把“未打开单独foldgold文件”误写为train64不包含其标注。

目标构造完全复用原51d7 construct与460d schema；CPU首尝试只因StrictNormalizer默认train32 SHA guard正确拒绝train64而停，targets-v1只留intent/failure；第二次显式传原构造器的expected_train_sha256=cb45，不改算法/阈值，targets-v2完整1.515秒。源v1=60ae3ff，修复6a4a6bc，原计划和失败保留。

发布原生tokenizer完整census5.954秒：训练957793 tokens，监督24997，最大输入10637、最大监督646；Pareto形状10637/557、6346/646。dev输入163925/max8858。无任何截断，全部EOS监督与q−1位置核对；这是local发布tokenizer编码，仍需实际目标runtime209行回放和新Pareto压力，不能把旧9210/610压力当已覆盖。

同配方两轮拟173×2、44次Adam、每轮尾5，warmup4保持原2/22比例；配置尚未激活。基于原A100实耗2037.651秒、实际长度/监督比例给出约4300秒训练＋完整两ACK的代理估计及6000+30候选上界，**不是实测或授权**。A800速度不能直接由A100保证。双epoch完整model/optimizer/scheduler/RNG约1.040GB，拟至少1.5GiB启动空闲；当前A800约1.3GB，不称训练资源就绪、不清其他route资产。精确公式、祖先SHA和shape见summary.json。

规范化仅未来train-only aliases32→64变化，显式原train64 SHA；b59eb阈值、纯公共41099 bank与Sap encoder全部固定，无重bank编码或新fit。先等当前F0/固定共识裁决；不自动原dev质量/官方提交。
