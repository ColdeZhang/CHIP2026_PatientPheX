# 方案5：神经表型识别与患者图传播

主版本使用 SapBERT 冻结编码器（约110M参数，CLS表示），训练折拟合 LogisticRegression span 判别器，指定 HPO 与训练别名建立 dense concept bank，再通过患者 mention 锚点在句图中的衰减传播确定归属。MiniLM ONNX 后端仅作为低资源可运行对照。

## 数据与模型

- 模型 `cambridgeltl/SapBERT-from-PubMedBERT-fulltext`，revision `090663c3ae57bf35ffe4d0d468a2a88d03051a4d`，Apache-2.0；UMLS2020AA 自对齐，PubMedBERT基础。
- 需将公开仓库的 `config.json`、`model.safetensors`、`special_tokens_map.json`、`tokenizer_config.json`、`vocab.txt` 放入模型目录，并保存来源及SHA-256 manifest。
- 可运行 `python prepare_model.py --output /persistent/sapbert-fp32 --fp16-output /persistent/sapbert-fp16` 下载固定版本并制作半精度副本；该步骤额外需要 `safetensors==0.7.0` 与 numpy。转换来源、操作与新旧权重哈希均写入 manifest，方便网络受限服务器中转。
- 仅接受独立的带标签训练文件和去掉 entities/association 的推理文件；训练、推理 pmc_id 重叠时直接报错。
- 输入 HPO 必须由共享校验器核验为指定 2026-06-23；本程序只使用 HP:0000118 非obsolete分支，处理 alt_id。

## 运行

在独立 Python 环境安装 `requirements-gpu.txt`，或只读复用具备兼容 torch/transformers 的既有环境并在项目独立venv补装缺失依赖。不要修改共享环境。

```bash
CUDA_VISIBLE_DEVICES=<已分配空闲GPU的UUID> OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python solutions/semantic_neural/run.py \
  --train /path/to/train64.jsonl --input /path/to/dev16.input.jsonl \
  --hpo /path/to/hp.obo --model-dir /path/to/sapbert-model \
  --output /persistent/run/dev.pred.jsonl --work-dir /persistent/run \
  --encoder sapbert --device cuda --threads 1
```

首版阈值在独立验证前固定：精确候选判别概率0.48；神经候选0.72且概念相似度0.87；多患者句图最高归属比例0.60。不会读取推理输入的标注，且若发现这些字段会拒绝运行。

程序写标准JSONL及简短 manifest，包括所有输入/输出校验值、参数、源码SHA、模型来源、训练样本数和时间。embedding cache、模型与完整输出不进入Git。使用共享新评测和格式校验器获得近似四项指标；本实现不接触网站。

主要局限：短语候选仍依赖本体/训练词素，神经识别并非自由生成；冻结encoder不能直接理解否定/患者关系，使用显式否定范围和句图补足；句图在跨患者远距叙述中可能混淆。具体表现以独立验证为准。
