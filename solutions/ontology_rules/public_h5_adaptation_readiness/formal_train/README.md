# PubMed 原生H5固定一轮适配

已获root唯一600+TERM30授权；当前源码准备，未启动。1252训练query严格沿原顺序一次，19×64+36共20步，fresh原d58a权重/Adam5e−6。保存201模型变量/399 Adam状态/iteration20及可序列化RNG，完整H5和checkpoint均fsync重开、SHA与新对象逐变量恢复；TF legacy Dropout计数限制明确，不改原随机语义。

随后完整F1原16纯输入，65406原生unique全部新forward，绝不复用原公共模型概率；原native2–8/POS/字典/缩写/5dp>.95保持。当前主要比较沿原a465实际vec101（旧摘要103已独立更正），同8cb/.728697536469，diagnostic旧overlay/.658357066464；只替PubMed，均完整共冻后各1v2，旧0。总Score严格升为primary门，macro完整披露。未自动F0/dev，未改任何B/旧native文件。
