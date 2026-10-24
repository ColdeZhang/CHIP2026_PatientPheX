# F0 最长输入格式与成本终点

唯一实际模型尝试通过。原index30／3563609:relations:0，6028→3427 tokens，78.938066秒／43.413782 token/s（含prefill），自然EOS、think闭合；strict、sparse、normalized均8/8，通过且无围栏或整数转换。完整IDs→raw→parsed字节绑定实际通过，没有质量评分。

wrapper89.506988秒、main88.690058秒、load5.074515秒，rc0；399 BF16变量／8,190,735,360参数、形状/有限数值、scale1与TF32 initial/end false通过。并未另做加载后全值与文件逐tensor相等比较；文件由九件完整SHA保证。峰allocated19,495,760,896B、reserved19,505,610,752B。05:01:02.963 GPU2 f516实际无目标进程，used1/free81156/total81920MiB。

原wrapper把free81156错作total，约.237秒于主程序spawn前失败、0NN；原源/plan/实际回执保持。独立修复只把total常量改81920，真实gate与拒绝用例CPU检查通过；同模型/输入/采样/codec/240+30。一次部署校验字符串SyntaxError与恢复也保留，没有重复模型推断。

九资产复制由原2400+30窗口完成：fetch1797.001798秒，04:55:02.246完整16,388,603,517B/fullSHA/fsync，ACK87353c49；.100持久源保持，A800 SHM副本易失，原source HTTP已停。监控RPC失败仅恢复原observer，未重启传输。

[summary99bff8ec](probe-attempt2/summary.json)、[统一产物清单58bf3f06](probe-attempt2/terminal-manifest.json)。原始生成和IDs在外部及A800自有project，不放Git。

新增后继决策（在F1质量前由root单独指定，不能回写为F1原计划）：冻结F0 6000+TERM30，idx30精确缓存1＋57fresh，逐row重置seed；只有原F1完整80与两个终点核验，且oldSFT32∪R1满足原retain（mic+mac严格升且mac不降），才自动启动。F1原primary/secondary名称和automatic_migration=false原件不改。F0完整58及两fullpred共冻后才各唯一v2。当前尚无F1完整质量、无F0全量启动或质量。
