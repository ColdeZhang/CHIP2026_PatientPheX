# 首次 GPU 探针及最小采样器修复

20秒GPU可见import gate实际9.369929秒通过：两个GDN为FLA、两个conv为native；进程退出后卡1 0MiB且无进程。唯一numeric于2026-10-24 00:57:50 UTC启动，原绝对deadline为01:07:50.116 UTC，kill grace到01:08:20.116 UTC。

attempt1单GDN native forward/backward完成（0.434534秒），随后缓存快照因真实`dict[int,Tensor|None]`被当序列遍历，整数键上`.detach()`失败。进程自然退出，卡1已0MiB；没有fused数值结果，没有完整9B/LoRA结果，不能当FLA数值负结果或速度对照。失败源码/plan不改，stdout/stderr、actual-bindings、timing与进程终点保留attempt1/。

root新增授权仅修采样器后CPU合同/owner push，再在原绝对deadline内运行独立attempt2；不重置600秒、不改数值阈值。新增wrapper只替换state capture及结构校验，原probe/plan所有SHA不变。真实TF5.20.2 DynamicCache32层、混合dict/list/tuple/tensor/None、非连续键及None槽位均保留元数据；要求至少一个真实Tensor，不以跳过int得到空快照。未知值、漏键/层、dtype变动和空快照必须失败。

CPU合同已通过且CUDA未初始化，见attempt2-cache-cpu-contract.json。GPU attempt2仍须owner push确认和fresh空卡之后才启动；本文件不是attempt2成功声明。
