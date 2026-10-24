# R1 H51 stress v2：真实Adam与累积最长通过

独立新attempt唯一完成，程序65.959秒／外层66.892秒rc0，其中12次前后向诊断39.382秒。先完整8条累积、lr1e−4 Adam更新，再带504个已建立Adam状态及3条累积梯度处理5519-token最长样本；全部完成。7类projection各1个LoRA B探针实际改变，adapter全finite，504份梯度finite、冻结基座均无梯度。原生/selected跨完整backward门仍保留原失败结论，不是本次硬门；CE门沿已审同actual logits精确证明及CPU合同，不重复native模型调用。完整[指针](stress-result-pointer.json)。

峰allocated **21,063,563,776B**，峰reserved **24,882,708,480B**。最大reserved接近A30容量，空闲边界不能只看allocated；本次没有OOM、量化、截断、allocator设置或手动empty_cache。原12行总45142tokens，在已有Adam和3条累积梯度情况下完成最长后向；它仍不是完整两轮稳定性的证明。

每步当前idx/loss/finite/stat先保存，再断言；CPU合成NaN检查确认null+finite=false及idx证据先落盘。所有实际行均finite。首次更新后的gradient174,587,904B、Adam tensor349,177,824B，状态只属该诊断，未保存adapter/checkpoint，进程结束后丢弃；不得接着这个权重开始正式训练。

49份server小件全部回收逐大小/SHA校验。GPU1实际释放0MiB/free24164MiB，查询时间和完整receipt保存外部；GPU0其他任务未操作。原数值失败、独立数值诊断与本stress三段分别保留。独立终审pending；完整训练未启动，也没有新质量。

完整train32建议仍为fresh base/2epoch、124条×2、32次更新、2warmup、按每组实际大小平均。802582输入tokens按本12行39.382秒作线性量级估计约**700秒**，不包含额外optimizer次数、完整epoch保存、加载及I/O变化，且序列长度对代价非严格线性。建议完整训练硬上限**1800秒+TERM30**，包含两份完整adapter/Adam/scheduler/RNG状态及fsync；先由root确定正式合同。原R1完整80推断3445秒仅成本参照，适配后是否缩短未知，训练与质量仍应单独限时。
