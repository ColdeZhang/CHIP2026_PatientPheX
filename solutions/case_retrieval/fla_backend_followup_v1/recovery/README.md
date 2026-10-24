# H39 元数据序列化机械恢复

原A在模型加载后的base-loading.json保存处失败（probe136/evidence44），30.448秒、0 forward/0 raw，GPU释放，原21件包2ec994c2保留。只逐字复用旧legacy json_default：set/frozenset按repr排序，未知类型仍硬错；原probe、输入、权重、精度、输出与统计政策保持。独立evidence.py c486dfb3，CPU唯一.005805秒检验空/非空/frozen/nested/mixed/unknown/NaN拒绝，结果6e1b1d61。

独立attempt2沿原绝对硬截止2026-10-24T08:54:12.927586Z（TERM30），软截止提前30秒，启动时只消耗原余量，不重置600秒。源/plan先push，新鲜卡核过才实际启动；原失败不能被后继成功改写。原CPU8b5379ee输入和数学证明沿用，不重跑；本合同只测元数据writer。实际进程由父台账记录。
