# 901 原配方 FLA 部署控制

唯一目标：当前原dev最佳901（Score .7543613981990362）完整配方仅把四路Qwen推断后端改为已核FLA，其余Task1、候选、提示、解码和支持门固定。原dev已多次开发使用，不作盲测；前置F1/F0全文控制与本次原短提示不同。

实际四流：normal64/reverse64/public-new42使用原full64关系1a703b，patient64使用distinct full64 f787；共234行，旧实际715548输入token/11948输出token，最大5198。均原长度/row_id升序、greedy1600/inputcap14000、不加全文。所有新流须FLA全部fresh；旧raw仅用于CPU复装901和原token身份核，不能冒充新响应。

先冻结CPU四流published-native审查（60秒/0GPU），并由独立装配入口用旧四raw复装901逐字节exact。后继完整GPU源另冻；预计1450–1550秒，允许1800+TERM30共享绝对截止，实际fresh空卡/磁盘门，流切换不重置时钟。超时/失败保原始证据。

必须保留patient−a179_oldGate原差集，以及原CID fallback/患者limited codec；不能简单全并或硬编码旧六条。新完整pipeline预测先freeze，再唯一公共v2对901，旧0；Score不升则关闭这一配置，不扫任意FLA组合。无官方上传/查询。
