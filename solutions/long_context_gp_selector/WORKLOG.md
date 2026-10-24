# BioClinical GP selector 工作台账

仅readiness，分支research/long-context-gp-selector，起点main5333e7f8；只改solutions/long_context_gp_selector。

Root已确认忠实strong48协议，不把它误称全链train32。一次CPU1.149秒查完整候选/祖先/全F1映射、F0纯输入新几何：3个新GP fit+4个score pass+1个LGB fit才完整，当前都未执行。复用既有GP23只用于排F0的OOF评分，现1313分数不能冒充16595候选或GP48输出。

固定新信息是BioClinical全文上下文raw span score，两列为valid/logit，不新增规范化/候选/threshold。原BiomedGP特征F1正/F0负/pooled负，原新GP直接并集/veto也负，全部作为历史而非省略。详见REPORT.md；预计GPU约11分钟、建议若激活900+30，尚无新GPU授权。

下一步仅等待root选择是否激活已描述的单F1控制；禁止自行fit、读F1gold、扫cut或按partial选择模型。
