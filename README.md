# SnapKV Experiment on DeepSeek-R1-Distill-Llama-8B

## 项目简介
在 DeepSeek-R1-Distill-Llama-8B 上手动实现 SnapKV（KV Cache Compression），
针对 transformers 5.x 的 DynamicCache 接口进行适配，支持 GQA（Grouped Query Attention）。

## 核心实现
- **GQA 适配**：DeepSeek 使用 32 Q heads / 8 KV heads，压缩前对 Q 按组取平均
- **压缩时机**：Prefill 结束后直接修改 `cache.layers[i].keys/values`，避免 position 错位
- **1D Pooling**：对 attention score 做 avg pooling，保留上下文连续性

## 实验结果（长文档问答）

| Config | 压缩比 | 正确率 | 速度提升 |
|---|---|---|---|
| Baseline | 无压缩 | 2/2 ✅ | 1.0x |
| SnapKV-512 | 1917→512 (73%压缩) | 2/2 ✅ | 1.26x |
| SnapKV-1024 | 1917→1024 (47%压缩) | 2/2 ✅ | 1.79x |
| SnapKV-2048 | 不触发 | 2/2 ✅ | 基本持平 |

## 文件说明
- `experiment1_longdoc.py` - Baseline 实验
- `experiment1_snapkv_v[1-5].py` - SnapKV 实现迭代过程
- `experiment1_snapkv_v5.py` - 最终版本
- `snapkv_utils.py` - SnapKV 核心算法（来自官方repo）
- `results_*.json` - 实验结果数据

## 环境
- GPU: NVIDIA L40S 45GB
- Model: DeepSeek-R1-Distill-Llama-8B
- transformers: 5.6.2
- vLLM: 0.20.0

## 参考
- [SnapKV 论文](https://arxiv.org/abs/2404.14469)
- [H2O 论文](https://arxiv.org/abs/2306.14048)
- [官方 SnapKV repo](https://github.com/FasterDecoding/SnapKV)
