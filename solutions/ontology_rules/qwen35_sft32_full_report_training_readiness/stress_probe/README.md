# Qwen3.5 SFT32 全文训练工程探针

唯一授权420+TERM30，A800 GPU2 fresh base/r16；不是正式训练，不产生质量结果或可复用checkpoint。原124/F2+F3/7170监督、105332 token的12压力样本逐项匹配既有native audit；cap13824，不截断。

先seed20261023 epoch0原首8微批/8→AdamW非零LR1e-4更新，然后原shuffle位置53..55三个样本累积→位置56的最长idx98/13549。此处刻意跨原group边界形成压力；遵H51不重放formal warmup/scheduler，全部probe状态退出丢弃。模型原12目标/43,278,336 LoRA/8,997,081,600总参数、BF16/SDPA/原生GDN、冻结base和head；只将有效监督位置在FP32 CE前选出，同Torch CPU合同通过但不是fullmodel GPU等价证明。

入口 operations.py --directory 本目录，依次deploy、preflight、start一次，随后inspect原job、终态collect一次。双层exclusive intent与freshUUID guard，实际15件base SHA/空卡/空间ACK后再start；远端launcher再次空卡核验。未知状态先inspect不重复start。原GPU正在忙则停止，不占其他卡。

观察实际finite、三族LoRA B更新、Adam/3累积存在、base无grad、逐row/全程耗时、allocated及reserved峰、GPU释放；完整训练预算还需按真实时长并包含每epoch持久导出估计。首版本准备仅缺helper sibling import，CPU编码前退出；已修正。preflight manifest list/name/size适配在任何远端执行前修正并保留未执行初版。
