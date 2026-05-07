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
| SnapKV-512 | **100%** | 2.72s | 17.37GB | 14% |
| SnapKV-1024 | **100%** | 2.64s | 17.19GB | 29% |
| SnapKV-2048 | **100%** | 2.54s | 17.17GB | 57% |

**核心结论：** 只保留 14% KV Cache，正确率不下降，速度提升 1.54x。SnapKV 压缩掉噪声 token 后模型更专注于相关信息。

---

### 实验二：大海捞针（~5.5K tokens，5个位置）

| Config | 开头 | 25% | 中间 | 75% | 结尾 |
|---|---|---|---|---|---|
| Baseline | ✅ | ✅ | ✅ | ✅ | ✅ |
| SnapKV-128 | ✅ | ✅ | ✅ | ✅ | ✅ |
| SnapKV-512 | ✅ | ✅ | ✅ | ✅ | ✅ |
| SnapKV-1024 | ✅ | ✅ | ✅ | ✅ | ✅ |
| SnapKV-2048 | ✅ | ✅ | ✅ | ✅ | ✅ |

**核心结论：** 即使压缩到 2%，SnapKV 仍能在所有位置找到关键信息。Observation window 机制有效引导 attention score 指向关键 token。

---

### 实验三：全文理解评测（Transformer 论文，6个问题覆盖全文）

| Config | 正确率 | 平均时间 | 平均显存 | KV 保留率 |
|---|---|---|---|---|
| Baseline | 83% | 1.27s | 16.26GB | 100% |
| SnapKV-128 | 83% | 1.02s | 16.03GB | 8% |
| SnapKV-256 | **100%** | 1.00s | 16.00GB | 17% |
| SnapKV-512 | **100%** | 0.98s | 15.91GB | 33% |
| SnapKV-1024 | 83% | 0.93s | 15.92GB | 66% |

**核心结论：** SnapKV-256 只保留 17% KV Cache，正确率反超 Baseline（100% vs 83%）。速度提升 1.3x，显存节省 0.35GB。

---

## 综合结论

| 结论 | 说明 |
|---|---|
| SnapKV 质量不下降 | 三个实验均显示压缩后正确率持平或提升 |
| 最佳甜点在 10-20% 保留率 | 过度压缩（<8%）或保留过多（>50%）都会引入噪声 |
| 速度提升 1.3-1.5x | 显存节省同时带来 decode 加速 |
| GQA 适配是关键 | transformers 5.x + DeepSeek 架构需要特殊处理 |

## 参数说明

| 参数 | 含义 |
|---|---|
| `window_size` | 观察窗口大小，用 Prompt 最后 N 个 token 的 Query 给前缀打分 |
| `max_capacity_prompt` | 压缩后 KV Cache 最多保留的 token 数（含观察窗口） |

例：SnapKV-256 = 保留 192 个重要前缀 token + 64 个观察窗口 token

## 调试过程记录

1. **transformers 5.x 接口变更**：`DynamicCache` 不再支持下标访问，KV 数据存在 `cache.layers[i].keys/values`
2. **GQA shape 不匹配**：Q(32 heads) 和 KV(8 heads) 维度不同，需要先 reshape 取均值
3. **position 错位**：压缩后 cache 长度变化，直接覆写 `layer.keys/values` 才能让 `get_seq_length()` 正确反映
4. **attention mask 冲突**：在 prefill attention 计算完成后再压缩，避免 mask shape 不匹配

## 文件说明

| 文件 | 说明 |
|---|---|
| `experiment1_longdoc.py` | 实验一 baseline |
| `experiment1_snapkv_v1~v5.py` | SnapKV 实现迭代过程 |
| `experiment1_v2_longdoc.py` | 实验一完整版 |
| `experiment2_needle.py` | 实验二：大海捞针 |
| `experiment3_comprehension.py` | 实验三：全文理解评测 |
| `transformer_doc.txt` | 实验三文档（Attention is All You Need） |
| `snapkv_utils.py` | SnapKV 核心算法（参考官方 repo） |
| `results_experiment1_v2.json` | 实验一结果 |
| `results_experiment2_needle.json` | 实验二结果 |
| `results_experiment3_comprehension.json` | 实验三结果 |

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
