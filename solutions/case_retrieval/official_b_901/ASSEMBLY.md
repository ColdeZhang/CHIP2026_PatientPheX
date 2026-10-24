# 冻结901的B组合合同

`assemble.py`只读取实际完整响应和冻结实体，不调用模型、gold、评分或网站。正／逆沿原parse_json及候选缺失／非法患者／截断时full-lr回退；public-new同parser但未解析CID弃答；patient由sequence原limited codec交纯合法positive keys，零旧关系／零fallback。7个合成边界case与原CLI两种fallback策略逐关系集合一致，实际2.073秒，见assembly-readiness.json。真实B尚未执行。

设O为normal∪reverse，Gold/Gnew为旧／新支持gate后public-new关系，P为纯patient。原901在B部署为(O∪Gnew)∪[P−(O∪Gold)]。只有实际100文献的patient code_map完整等于oldpool原concept空间，且按HPO alt-ID canonical和原字符串均与public-new不交时，才得到O∪Gnew∪P。入口自行核空间与全部positive，含NIL文本碰巧等于HP字串的保守冲突检查；若发现任何交集即停，不能丢patient或私改为全并。具体源／实际空间证据再由统一审复核。

Task1固定103列LGB输出空关系模板，最终只填association；保留全部实体及其他literal字节。新输出按concept排序，历史中间append顺序不作为模型变量；只主张关系集合与既定表达式相同，不冒称未知B字节等于dev16缓存。

三个GPU臂使用`launch_arm.py`唯一新目录、精确source/prompt/nativeaudit绑定、fresh UUID／空卡／盘空间检查和独立硬期限。`run_arm_once.py`把只含源／提示／合同的ZIP经MCP小块部署，终态将每条raw IDs及响应打小包实际恢复逐SHA，未知传输状态不自动重启。预算须按实际提示数和max tokens先冻结。A80006:38实际旧SHM adapter仍在且1a703b/c4a SHA通过，无需再次迁移；project剩118.9MB，只放小产物，持久完整副本最终沿.100保存。
