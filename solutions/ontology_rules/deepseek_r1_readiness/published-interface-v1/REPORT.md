# 发布 tokenizer 接口：格式与成本

正确发布输入及显式vLLM scale1语义的首轮pilot已经结束，不能与旧TF5重建编码混为一谈。新CPU7.9316秒核80条raw/generic/template IDs、完整backend组件、special与往返exact；first2014、最长5689，原config文件不改。两项接口修正同时生效，不作单变量因果归因。CPU报告3131是upper-middle，常规median3129。

事前原政策要求自然EOS、完整可选think、全段严格JSON、许可CID/PID及无重复；分别报告EveryCID与合法稀疏（缺键弃答）。实际首条生成746token/24.3150秒=30.6806token/s，自然EOS/think闭合、非cap/wall；加载5.3318秒，wrapper36.8119秒/rc0，03:55:44.577真实释放0MiB。全399/8,190,735,360 BF16参数实际形状/finite/加载检查保持，actual rotary scale1.0。

最终文本为Markdown JSON围栏，原两个parser首先在JSON char0拒绝；文本另可见整数PID[1]，但当次未执行到PID类型检查。未剥围栏/转换/填键，按事前first-invalid停止最长条。它是格式合同负结果，无gold或质量结论。原[合一清单](terminal-manifest.json) SHA9201f194、[终点摘要](gpu/terminal-summary.json) SHA335049a7；独审6c8e9293通过，原字节不改。

## 独立有限codec后继

主线程随后明确授权新事前codec：只去除包住整个final的一个```json/空语言围栏，拒重复key；CID必须许可，PID许可字符串保持，type(x)is int且非负时仅映射唯一原许可P<规范十进制>。bool/float/负数/未知/补零别名以及归一后重复均拒绝。缺CID仍弃答，不能填[]；自然EOS/完整reasoning仍必需，不提取任意正文JSON或语义修补。

实际Python3.10纯标准库CPU0.000987秒/rc0，5正23负通过；仅复用已冻first746生成文本，正常化后8个原有CID、2个整数→P1别名，6个[]原本已经明确给出，没有新填。原strict=false及raw SHA42ea06fa保持；新normalized-valid另记于[CPU收据](codec-longest-v1/actual-cpu-result.json) SHA9251c886。无tokenizer/模型/新预测/质量执行。

后继仅longest5689/index73，fresh固定seed从原提示生成，不续旧RNG/KV、first不重NN。单4096/240+TERM30/225soft，原发布tokenizer/scale1/采样参数保持；遇截断/非法归一结果停止，不继续宽化codec或预算。源a737f47e、plan d6f6bc5e，actual5源SHA/Python3.10compile通过，尚未启动。新政策及作业与已终止原pilot分开记录。
