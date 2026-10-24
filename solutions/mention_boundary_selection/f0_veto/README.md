# 固定F0 BOUNDARY veto

同final epoch2头作用于强F0全部1085跨度，只删除明确BOUNDARY。缺373个公共Top8词面，所有来源/固定目标见plan.json。先完整nativeCPU，再必要Sap180+30、完整推断1800+30，唯一完整veto冻结后至多1v2。旧F1两个负端点保持，不增原dev/新headunion。当前源与CPU准备中，尚无GPU启动意图。

373 query原生CPU已完整通过，最大12 tokens、over32/512均0，CUDA隐藏，无截断。沿相同publicSap/max32/Top8规则；启动有效期120秒并以独占admission.lock防重复，实际query另有180+TERM30。
