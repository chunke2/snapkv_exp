# SnapKV Experiment on DeepSeek-R1-Distill-Llama-8B

## 项目简介

在 DeepSeek-R1-Distill-Llama-8B 上手动实现 SnapKV（KV Cache Compression），
针对 transformers 5.x 的 DynamicCache 接口进行适配，支持 GQA（Grouped Query Attention）。

## 核心实现亮点

- **GQA 适配**：DeepSeek 使用 32 Q heads / 8 KV heads，压缩前对 Q 按组取平均后计算 attention score
- **压缩时机**：Prefill 结束后直接修改 `cache.layers[i].keys/values`，避免 position 错位问题
- **1D Avg Pooling**：对 attention score 做聚类，保留上下文连续性，避免孤立 token
- **transformers 5.x 适配**：针对新版 `DynamicCache` / `DynamicLayer` 接口重新实现，官方 monkeypatch 不兼容

## 实验结果

### 实验一：长文档问答（~3.5K tokens，4个测试用例）

| Config | 正确率 | 平均时间 | 平均显存 | KV 保留率 |
|---|---|---|---|---|
| Baseline | 75% | 4.17s | 18.00GB | 100% |
| SnapKV-512 | 75% | 2.71s | 17.37GB | 14% |
| SnapKV-1024 | 75% | 2.64s | 17.19GB | 29% |
| SnapKV-2048 | 75% | 2.54s | 17.17GB | 57% |

**核心结论：只保留 14% 的 KV Cache，正确率完全不下降，速度提升 1.54x**

## 文件说明

| 文件 | 说明 |
|---|---|
| `experiment1_longdoc.py` | 实验一 baseline（无压缩） |
| `experiment1_snapkv_v1~v4.py` | SnapKV 实现调试迭代过程 |
| `experiment1_snapkv_v5.py` | SnapKV 可用版本 |
| `experiment1_v2_longdoc.py` | 实验一完整版（含 baseline + 3个压缩配置对比） |
| `snapkv_utils.py` | SnapKV 核心算法（参考官方 repo） |
| `results_experiment1_v2.json` | 实验一完整结果数据 |

## 调试过程记录

实现过程中遇到并解决了以下工程问题：

1. **transformers 5.x 接口变更**：`DynamicCache` 不再支持下标访问，KV 数据存在 `cache.layers[i].keys/values`
2. **GQA shape 不匹配**：Q(32 heads) 和 KV(8 heads) 维度不同，需要先 reshape 取均值
3. **position 错位**：压缩后 cache 长度变化，直接覆写 `layer.keys/values` 才能让 `get_seq_length()` 正确反映
4. **attention mask 冲突**：在 prefill attention 计算完成后再压缩，避免 mask shape 不匹配

## 环境

- GPU: NVIDIA L40S 45GB
- Model: DeepSeek-R1-Distill-Llama-8B
- transformers: 5.6.2
- Python: 3.12

## 参考

- [SnapKV 论文](https://arxiv.org/abs/2404.14469)
- [H2O 论文](https://arxiv.org/abs/2306.14048)
- [官方 SnapKV repo](https://github.com/FasterDecoding/SnapKV)
- [vLLM](https://github.com/vllm-project/vllm)
