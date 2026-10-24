# 固定Top8边界拒绝监督

2026-10-24：root预锁独立BOUNDARY状态及严格539候选。原2931 all-target保持；训练分布HPO2222/NIL44/NO665/BOUNDARY539。固定fresh Qwen3-8B LoRA r16、lr1e-4、seed20261023、两epoch、868更新/每epoch434/尾6、warmup44，无holdout/eval/earlystop/epoch选择。

授权完整链：509必要Sap query≤180+TERM30；CUDA隐藏CPU完整native合同；GPU probe240+30；actualcost≤7200才fresh正式7200+30；两持久ACK后1416fresh1800+30；strongB并集/new-only两完整预测先共冻再2v2/old0。资源仅.100 GPU1，fresh核UUID/空闲/磁盘，不抢其他卡。当前未启动GPU。

原生CPU实际35.22秒rc0，train3470/2,002,420tokens/37,834监督，held1416/800,912tokens；Pareto632/17与744/13。514状态/12,300prefix、20真实HFprocessor和17拒绝均过。原9.4MB包回收遇MCP LUNA，185s后只取小终态/原成员SHA，全native数组仍在服务器原SHA绑定，不重CPU。旧同hostTop8 token线性成本代理约4115秒，实际唯一probe再决定7200门；不据此缩数据。

15:21:37.533682Z：源2b4c8553 / 部署f17fe898 / freshgate通过，唯一proc_98b3a99e-9316-409c-8e71-feaf7bf88516、PID2750059。13真实FB+Adam及5训练状态原生generate工程；probe状态丢弃，成本门固定7200，无质量观测。

实际probe20.4655秒/wrapper21.2574rc0，13FB/1Adam、峰17,648,261,632B，5类训练输入均生成合法HPO（仅工程，不称识别正确）。预锁warm-token成本5642.17≤7200；原含cold诊断9099.33与worst-row5764.30完整保留。15:21:59.012453 fresh正式已启动868steps，probe state丢弃。

原CPU ZIP de55a04b / 9,438,522B已完整73块重组，56成员全部SHA核验；仅恢复原缺失range，没有重pack、重CPU或改模型。此前selected证明和传输错误记录保留。

## 第一轮完整状态

2026-10-24 15:55:26.766964 UTC，原formal在434更新后完成epoch1持久ACK；adapter174655536 B/SHA59886a2e，training_state349612141 B/SHA52bc9694。504 adapter张量、504 optimizer states/1512 tensor entries与RNG回读exact；ACK1785 B/SHA874860639d890bd9d6e4290ed3f0418c1341321a2d33a439e33a328c26e8acd9。记录耗时3.674663秒不包含最后ACK追加与父链同步尾操作的严格上界。第二轮同一任务继续，未做中途质量或epoch选择。小证据为原ACK对象+字节身份观察，完整原件随终包回收。

## 两轮训练完整与固定推断启动

2026-10-24 16:28:54.633374 UTC完整fresh训练成功：3470×2=6940行、868更新、4,004,840输入tokens、75,668监督tokens。native training_seconds4009.018807，含模型load及最终保存的phase elapsed4015.542905，wrapper4016.458222秒rc0，均在7200预算内。epoch2 ACK16:28:54.585388，1784 B/SHAd87852b017accb585158040284b58fea4b2e2d5cae1d63e6555a14bbc9882120，504 adapter与504 Adam/1512 tensorentries、RNG回读exact。最终adapter174655536 B/SHAbeecfb84824764826ab8f207f4979db09da161dc6395e55b032b77a413bb0579，training_state349612141 B/SHA360544ce073c0b6008c97152ff01df48e621a6b8bf7bf2bf36b4a96a53535a28。保存记录seconds2.972009不作为最终ACK/父链收尾严格上界。

16:28:55.624190原controller唯一启动完整1416推断，timeout PID2754394，独立1800+TERM30预算/deadline16:58:55，使用第二轮最终adapter；首7条仅吞吐观察，未看质量或调参。原formal/complete380452 B/SHA840f132de7b2469608c5bb2a4b27cc8b7c5f94bd2e0d67a4085e4586d249344a，当前小观察摘要移除868updates数组，仅保count，完整原件随终包回收。16:27只读状态一次UNKNOWN后按30秒退避恢复原任务，无重启或预算重置。

## 全量推断与两端点结案

1416/1416在16:45:42.935412 UTC完整成功，native1007.237241秒、wrapper1007.995771秒rc0；16:45:43.619895 GPU1=0MiB/noapps真实释放，原release SHA7f8a1a22。原managed proc98b3在16:48:03实际running=false，但API不返回wrapper退出码，不能把只读工具rc冒充原wrapper rc。完整原ZIP1726973 B/SHA6bfec309首次14块中5个UNKNOWN；保9块，30秒退避后仅恢复原缺块，未重pack/模型。最终2946小成员全部size/SHA核验，4大状态只通过原owner真实tensor回读与SHA证据，不声称重新下载。

解码HPO1290（包含1个empty+missing）、NO123、BOUNDARY3、NIL0、reject0/cap0，1289接受实体原子。missing为解码器自声明，不能当held gold候选覆盖；3个BOUNDARY也不是边界召回率。

两完整预测先共冻f79f1d83d8c06a5c8992f9f07779de42d388f70637391a8de014dab200b14440，再各一次固定publicv2/all/text（新2、旧0、原dev0、官方0）：

|端点|Score|menF1|docF1|相对强B Score|
|---|---:|---:|---:|---:|
|强B|.728697536469|.774943735934|.821052631579|0|
|strong-union|.721846056324|.757706093190|.810884353741|−.006851480145|
|new-only|.714176156405|.745647236942|.792263610315|−.014521380064|

关系mic/mac保持强B对象及分项完全相同。union保1313原实体、增加124=24TP/100FP（增量精度.193548），共1437；new-only1289=985TP/304FP，FN368，整体precision.764158/recall.728012，不能把增量低精度写成整体精度。较旧Top8 union净TP数不变、净FP减少12，Score+.000820473；new-only净TP+2/FP−12，Score+.000992478。仅有小幅进展，两个强门均未过，配方关闭无F0/dev扩展或cut调整。新增539监督、共享SYSTEM与第四语法状态共同变更；只输出3个BOUNDARY不足以把净变化归因为这一状态的直接删除，root独立预锁veto另行检验，本文不代跑、不改原输出。

稳定外部terminal.json13143 B/SHA15896d47d89a41523e2e490c23488b39acd2af7ef25b0eecf13d89d097ddb615，之后只加独立备份/独审指针，不覆写该原件。全部既有head报告比较无新gold/evaluator调用，comparison SHA62e5cc19。
