# F0 full58 推断源码静态交接

只准备源码/template，**完整58 GPU预算尚未授权、没有启动或真实生成**。`execution_ready=false`；full_inference_authorization、process_budget_seconds、TERM_grace_seconds、cached_indices、new_indices、cached_rows、cached_provenance、target_copy_ACK均null。入口在输出目录建立和GPU查询／native import前拒绝这些未知项。没有新增自动binder：owner须在root后继授权与实际copy／probe终点后冻结独立实际plan，不能把模板当执行计划。

复用已完成F0 CPU e97b2ba4、audit7649cfc3、encoded gzip9a132bed，195157总输入tokens、最长原index30/6028。原58消息／CID／患者字符串与415概念/1166对/47患者保持，非HP literal不清洗。准备阶段只读取/hash无标签产物metadata，没有重跑native tokenizer或load_inputs真实路径。

`infer_full58.py`保留同发布JSON generic、原生BF16 SDPA、scale1、399参数变量／8190735360和实际finite／同设备rotary核验；加载至rotary块与原F0 probe逐字节相同。A800 GPU2／UID1045／完整f516UUID、Python/runtime路径与SHM位置沿已核probe，九资产source/.100与target-copy ACK分开验证的代码块原样。

每个fresh行单独reset torch及CUDA seed20261024，temperature0.6/top_p0.95/top_k0/max_new_tokens4096不变。缓存只能由未来明确冻结的`[]`或`[30]`决定，现没有预定57+1。可选缓存必须绑定原row、plan、result、wrapper、source、launcher、raw、IDs和实际完整模型／采样／资源释放；同一原始生成逐字段保留。正常cap/invalid缓存若要复用，只能保持同一弃答，仍须后继明确选定；wall或工程失败不能作为完成缓存。

`raw_first.py`先独占原子写完整返回IDs及结束统计并fsync，再解码并独立原子写raw文本／IDs，最后才三parser、独立parsed输出。decode未知异常仍有完整IDs，parser未知异常仍有完整IDs+raw；不会再因解析异常丢失正常返回末尾不足32 tokens。生成内部OOM/硬杀若尚未返回完整序列，只能保留真实32token检查点partial，不伪称恢复未落盘尾部。正常单行cap/expected invalid整行弃答继续；unknown/OOM/globaldeadline立即停止且complete_response_contract=false。新generation calls started/returned及成功终行分别计数，不把工程失败前已生成调用隐去。

评测接口：complete_status=`COMPLETE_FIXED58_RESPONSES_NO_QUALITY_EVALUATION`；result_target_count_field=`original58_target`；wrapper_coverage_flag=`fixed58_whole_response_contract`；source／launcher名为`infer_full58.py`／`launch_full58.py`。完整58严格原序、每row canonical SHA+实际rowref、gzip及uncompressed SHA、new/reused覆盖、正常EOS/cap/wall、实际wrapper rc0/释放符合evaluation-source的独立门禁。预算／TERM／cache_indices仍须未来填入其inference_contract；任何不完整／工程终点不能评测。

三parser以及parse_item与原F1已审源字节一致。`check_source.py`仅std库静态+纯合成：source/plan SHA、成熟asset/load/rotary块逐字节、unknown模板拒绝、literal稀疏不填补、invalid整行无fallback、decode异常保留IDs、parse异常保留完整raw。source-check记录PASS，0.032秒。没有调用任何真实tokenizer/model/load_inputs/predict/gold/evaluator/GPU/MCP；F1、probe、owner/Git未修改。
