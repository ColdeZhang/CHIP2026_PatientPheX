# 方法与边界

研究入口：`plan.json:research`，原REPORT SHA050e6345，train-only census4fefcd39。成熟启发为TACRED位置注意力（作者commit18221efc，Apache2.0）及DocRED距离embedding（64622e60，MIT）；未复制外部训练数据/权重。准确URL/版本/许可在引用manifest中。

旧患者语义均值与span RoPE没有显式anchor到span距离，但并非完全没有上下文位置信息。旧项目LR已用几何，不能称项目首次位置特征。139/560多患者正袋全部实例有另一更近患者，故不使用最近患者伪标签或硬同段过滤。

`model.py`只缓存token段号与锚点列表。训练只对原全部bag实例union计算残差，推断沿原8192chunk。候选两端含边界、anchor右端不含；重叠/紧邻gap0；全部锚点排序去重均值，无远距截断。新增零Parameter不消耗RNG；旧参数列表从新模型显式排除geometry，保持顺序。

旧8轮metadata有过balanced BCE文字，实际GPU已是GP。这里继承实际24的effective-preparation，不改旧文件。最终checkpoint只部署模型，不声称optimizer/RNG full-resume。
