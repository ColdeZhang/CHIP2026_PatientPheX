from pathlib import Path
import hashlib,json
from datetime import datetime,timezone
u=Path(__file__).parent
r=u.parents[1]/'public-phenotagger-reuse'
def ref(p):
 return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)
a=json.loads((u/'A100_3_12-inventory.json').read_text());c=json.loads((u/'cpu-import-result.json').read_text())
libs={x['realpath']:x['bytes'] for x in a['CUDA_libraries'] if '/cuda-11.8/' in x['realpath']}
base=sum(x.get('bytes',0) for x in a['source_directory_sizes'])
aux=sum(c[k]['bytes'] for k in ['nltk_data','base_tokenizer_config','sources','label_file'])
weights=sum(x['bytes'] for x in a['top_level_weight_stat_only'])
summary=dict(schema=1,status='CPU_RESOURCE_AUDIT_COMPLETE_NO_GPU_OR_TRANSFER',created_utc=datetime.now(timezone.utc).isoformat(),original_runtime='A100 .3.12 public-phenotagger-reuse',fresh_cpu_import=dict(seconds=c['seconds'],tensorflow=c['tensorflow_version'],cuda_build=c['tensorflow_build_info']['cuda_version'],cudnn_distribution=c['cuDNN_distribution_version'],cuda_visible_devices=c['CUDA_VISIBLE_DEVICES'],no_model_or_tensor=True),compatibility=dict(source_python='CPython3.10.21',target_python='CPython3.10.12',all_SOABI='cpython-310-x86_64-linux-gnu',source_glibc='2.17',target_glibc='2.35',wheel_tags='cp310-cp310-manylinux_2_17_x86_64',copied_venv_bin_not_relocatable='original .venv/bin/python resolves /dataset/miniforge3/envs/vlm-det/bin/python3.10; target should use its verified /usr/bin/python3.10 with private relocated packages',driver_note='A30 driver580.178.04 is newer than source535.247.01; CUDA12/13 toolkit presence is not an incompatibility finding. Required CUDA11/cuDNN8 private libraries are not yet supplied on targets.',target_tf_import_verified=False),byte_budget=dict(venv_excluding_symlinks=a['source_directory_sizes'][0]['bytes'],gpu_overlay=a['source_directory_sizes'][1]['bytes'],cudnn_libraries_subset_of_overlay=sum(x['bytes'] for x in c['cuDNN_libraries']),cuda118_distinct_enumerated_libraries=len(libs),cuda118_conservative_library_bytes=sum(libs.values()),pubmed_weight=491312288,bioformer_weight=205710168,nltk_tokenizer_sources_label=aux,conservative_both_models_payload=base+sum(libs.values())+weights+aux,conservative_pubmed_only_payload=base+sum(libs.values())+491312288+aux,budget_limits='Stat-based uncompressed approximate payload, not a minimal dependency closure or measured transfer cost. Source site-packages includes CPU TensorFlow as well as separate GPU overlay. No archive, copy, installation or bandwidth benchmark performed.'),recommendation=dict(first='Wait for owner release of A100 GPU2, then fresh-check and reuse already validated original runtime; no migration needed.',alternative='If waiting is more costly, .100 offers two idle A30 GPUs and 446.8GB free at inventory. Transfer to an isolated private target only under new authorization, reuse target CPython3.10, validate CPU imports then separately authorized GPU smoke.',fallback='.3 GPU0 was idle and 191.8GB free; same conditional transfer/runtime validation applies.',fresh_resource_check_required=True),scan_scope='Project-owned runtime paths to depth5, known user conda sites and /usr/local library sites. No TF in scanned A30 paths does not assert whole-host absence.',forbidden_actions_performed=False,weights_rehashed=False,old_verified_evidence=[ref(r/p) for p in ['weight-source-manifest.json','installed-package-manifest.json','native-pipeline/gpu-overlay-state.json','native-pipeline/gpu-readiness.json','native-pipeline/full-inference-plan.json']],new_evidence=[ref(u/p) for p in ['A100_3_12-inventory.json','A30_3-inventory.json','A30_100-inventory.json','cpu-import-result.json','relocation-result.json','A100_3_12-abi.json','A30_3-abi.json','A30_100-abi.json']])
(u/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
report=f'''# PhenoTagger unigram 准备：运行环境只读核查

当前可直接复用的是 A100 `.3.12` 的原私有环境，但 GPU2 正在序列训练（23:21 UTC 实查 31,025 MiB），本任务没有占卡。优先等原 owner 释放后复用，无需搬运依赖。若需要改用 A30，`.100` 是条件性首选；它尚未具备经实际验证的 PhenoTagger TF 环境。

| 机器 | 23:21 UTC 空闲资源 | 解释器 / ABI / glibc | PhenoTagger 状态 |
|---|---|---|---|
| `.3.12` A100 | 全部卡有 compute PID；GPU2 为现有训练 | 原 `.venv/bin/python` → `/dataset/miniforge3/envs/vlm-det/bin/python3.10`，3.10.21；CPython310 / glibc2.17 | 原 TF2.12 / CUDA11.8 / 私有 cuDNN8.6.0.163；本次 CUDA 隐藏 CPU import 2.611s 通过 |
| `.3` A30 | GPU0 UUID `GPU-07726469-bf7a-b832-9dee-019dc276dc1d`，无 compute PID；磁盘191.8GB空闲 | `/usr/bin/python3.10` 3.10.12；CPython310 / glibc2.35 | 所查路径无原 TF；可搬私有依赖，尚未验证 |
| `.100` A30 | 两卡均无 compute PID；GPU1 UUID `GPU-fc5d0525-3ec1-074a-9fa3-35d92a9da539`；磁盘446.8GB空闲 | `/usr/bin/python3.10` 3.10.12；CPython310 / glibc2.35 | 所查路径无原 TF；可搬私有依赖，尚未验证 |

三机实际 `SOABI` 均为 `cpython-310-x86_64-linux-gnu`，x86_64；原 TF/numpy/h5py wheel 为 cp310/manylinux2.17，支持在目标较新 glibc 上复用的判断。原 venv 的绝对解释器符号链接不能原样照搬成目标可执行环境；可用目标现有 Python3.10 配合独立复制的 packages/overlay。唯一 `.pth` 是 setuptools 常规 shim，无项目绝对路径。原 TF 主 `.so` 的 `ldd` 未缺项，库内 TensorFlow 依赖走相对位置；这不证明目标 CUDA 动态依赖已经齐全。

A30 驱动580.178.04比原A100的535.247.01新。现有 CUDA12/13 toolkit 不代表 CUDA11运行库不兼容；缺口是尚未部署原私有 TF、cuDNN8及CUDA11库。目标现有 PyTorch cuDNN9/NumPy2 栈不可作为已验证的 TF2.12/NumPy1.23.5 环境。

原根 `P=/home/zhangyuheng/projects/CHIP2026_PatientPheX/public-phenotagger-reuse`：

- `P/.venv` 实际普通文件 {base-a['source_directory_sizes'][1]['bytes']:,} B（不含3个符号链接）；`P/tf-gpu-overlay` {a['source_directory_sizes'][1]['bytes']:,} B。两者共 {base:,} B；本次仅stat，不是重新包校验。
- overlay 中 `nvidia/cudnn/lib` 7个cuDNN8库共 {summary['byte_budget']['cudnn_libraries_subset_of_overlay']:,} B，已包含在overlay总量内。
- 原 `LD_LIBRARY_PATH=P/tf-gpu-overlay/nvidia/cudnn/lib:/usr/local/cuda-11.8/lib64`；CUDA11.8本次枚举11个独立相关共享库共 {sum(libs.values()):,} B，可作保守复制预算，尚未最小化依赖闭包。
- 原PubMed H5 491,312,288 B；Bioformer H5 205,710,168 B；原NLTK路径是 `P/nltk_data`（下划线）43,332,618 B。词表/配置/小源另约476KB。
- 完整保守 payload：只PubMed约 {summary['byte_budget']['conservative_pubmed_only_payload']/1e9:.3f} GB；两模型约 {summary['byte_budget']['conservative_both_models_payload']/1e9:.3f} GB，未压缩、未计传输缓存。未实测网络，不能据此承诺迁移分钟数。可去除重复CPU TF以缩包，但本次未执行、未验证精简版。

本次 CPU import 仅 TF/numpy/h5py；不构造模型、不创建张量、不枚举CUDA设备、不加载权重。模型SHA沿既审 `weight-source-manifest.json` 复用，未再hash权重。新证据绑定见 `summary.json`，原始MCP响应、可恢复脚本均在本目录。库存扫描仅项目自有runtime/已知conda路径，不能断言全主机无其他TF。GPU空闲仅此时快照，未来运行仍需 owner 调度及fresh核查。
'''
(u/'report.md').write_text(report)
files=[p for p in sorted(u.iterdir()) if p.is_file() and p.name!='index.json']
index=dict(status=summary['status'],created_utc=summary['created_utc'],files=[ref(p) for p in files])
(u/'index.json').write_text(json.dumps(index,indent=2)+'\n')
for p in [u/'report.md',u/'summary.json',u/'index.json']:print(json.dumps(ref(p)))
