# F0 R1 双终点静态入口

仅 source/template 准备，未执行真实预测组装、模型、tokenizer、金标读／hash 或评测；`execution_ready=false`。完整58 GPU预算仍未授权，本目录不新增该授权。F1活源保持4fbc58db原字节。

原F1入口的schema、Gold、80/587/1557/1189/39、cache0/73及2+78、10800预算和终态有硬编码，不能只换模板。新增F0入口按SHA import原4fbc源码，复用20个通用函数：文件/ref/JSON辅助，canonical item SHA，module/固定normalizer加载，以及association_span、nonrelation_bytes、relation_map、assemble_two、retain、parse_evidence、provided_relations、make_primary、delta。没有猴补F1全局、没有调用F1 main/reference_gate/verify_inference。

F0独立门禁固定58行/16文献/47患者/947实体/415文献概念/1166候选对；原提示d387、输入ec38、旧SFT48预测63aefe68及指标e16ff5原refs。已完成发布CPU e97b2ba4／audit7649cfc3固定195157 tokens、最长原index30/6028；不沿用F1长度。literal概念和患者原串由原code_map精确映射，不过滤非HP值。

唯一两臂：R1 normalized-only standalone，以及oldSFT48正关系字面集合并集。固定9c7 codec；整行invalid无fragment/fallback，缺CID弃答，不伪造`[]`为原模型答案。完整submission中的空正关系集合明确表示无提供证据。实体及每行association值以外原字节保持。两完整预测先独占写入并共同freeze+SHA核对，再canonical validator导入、evaluator导入、首个gold字节，each一次公共v2 all/text；旧指标不重评。keep仍为关系micro+macro严格升且macro不降，未自动迁原dev。

`inference_contract`目前null。后继root授权后须明确：complete_status（COMPLETE_FIXED58_*_NO_QUALITY_EVALUATION）、process_budget_seconds、TERM_grace_seconds、cached_indices、inference_source_basename、inference_launcher_basename、wrapper_coverage_flag、result_target_count_field。具体预算／cache划分／正式源名及终态未预设。`cached_rows`、6个endpoint refs和full_inference_authorization也为null。

正式runner须保留F1已审通用收据结构：原58顺序，new/reused计数、full gzip和uncompressed SHA、每row canonical_item_sha256+原row文件ref、EOS/cap/wall与fresh seed/cache原件绑定、完整native参数/scale及wrapper rc0/实际释放。全局wall/工程FAILED/不完整58不能quality。单行4096 cap或正常invalid按事前弃答政策进入完整58；原始IDs/raw及异常证据必须先保存。

`bind_completed_evaluation.py --binding OWNER_FROZEN_BINDING.json` 的输入字段精确为template_sha256、endpoint_references、cached_rows、inference_contract、full_inference_authorization。它只在完整58共同门禁通过后写独占evaluation-plan，既不组装预测也不评测；当前无可执行binding。

Gold路径/SHA630c9e39/507921B仅从已冻结patient-F0 endpoint-plan b719834e字符串复用，没有打开gold。`source-check.json`记录纯合成字节保持、invalid整行弃答、缺CID不补、literal PID/concept、独占freeze拒覆盖及未知完整合同拒绝。静态检查修正了初稿折名替换影响metric拼写的问题；最终依旧使用公共menF1/docF1/micF1/macF1字段，未有实际评测失败。
