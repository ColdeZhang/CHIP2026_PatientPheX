# 公开 BioBERT 第三来源固定选择器控制

root已授权唯一四fold控制；当前处于完整无标签缓存后的准备，尚未生成新质量。保留旧PubMed/Bioformer候选对象的全部已有字段、类型和顺序，机械并入BioBERT native canonical键，并增加六个独立实际来源字段。旧特征家族实际每fold维数不同，按真实vectorizer记录，不造固定103列。

固定HGB160、cut .3与已保留proper-nested解码，比较 `selector-nested-control` 的四fold完整预测（pooled Score .727975655663）。非entities对象原样，重新拟合可改变实体选择。先无标签核≤1.5倍行数与4fit/infer≤120秒成本，再冻结源/计划/执行；四个新完整预测共同freeze后只评一次。保留标准为pooled两实体F1均值严格增且至少3/4fold不降，无cut扫描、原dev或官方操作。

实际候选/依赖/每fold数据及运行恢复入口将在正式计划列明；[事前授权政策](authorized-proposal.json)保留假设、比较和边界，不把准备当质量。

模型库存边界：若未来质量满足keep并获准迁移，现共享pipeline 9,475,880,164加本独立head121,587,826＝9,597,467,990参数，再按实际采用adapter计；同一Qwen共享仅计一次，不把已关闭entity/binary加入生产链。当前没有改变最佳配置。
