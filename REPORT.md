# SnapKV 实验技术报告

> LLM 推理优化：KV Cache 压缩 | Python, PyTorch, transformers  
> 2026.03 – 2026.05

---

## 1. 问题背景

### 1.1 KV Cache 瓶颈

LLM 自回归推理时，每个 token 的 key/value 投影会被缓存（KV cache），避免重复计算。随着序列长度增长，KV cache 成为显存和带宽瓶颈：

- 显存：KV cache = `2 × num_layers × num_kv_heads × seq_len × head_dim × 2 bytes`
- 对于 DeepSeek-R1-Distill-Llama-8B（32 层, 8 KV heads, 128 dim, fp16）：
  - 1K tokens → 2 × 32 × 8 × 1024 × 128 × 2 = 16 MB
  - 16K tokens → 256 MB
- 带宽：decode 每步需要从 HBM 读取全部 KV cache 做 attention

### 1.2 SnapKV 核心思想

SnapKV 在 prefill 结束后、decode 开始前，根据注意力分数选择性地丢弃大部分 KV 对，只保留最重要的 K 个。关键假设：

1. **观察窗口**：最后 `window_size` 个 token 的 query 能代表整个序列的注意力需求
2. **重要性评分**：用这组 query 对所有 key 打分，分数高的 prefix token 被保留
3. **完整保留窗口**：末尾 `window_size` 个 token 的 KV 对始终保留

```
Score(k_i) = sum_q softmax(Q_obs · K^T / sqrt(d))[:, :, i]
Selected = TopK(avg_pool(Score), K=budget - window_size)
Compressed = {selected prefix tokens} ∪ {last window_size tokens}
```

---

## 2. 核心算法实现

### 2.1 GQA 适配

DeepSeek-R1-Distill-Llama-8B 使用 Grouped Query Attention（32 Q heads, 8 KV heads），每个 KV head 对应 4 个 Q head。SnapKV 打分需要将 Q 按 KV head 分组取均值，使 Q 和 K 的 head 数对齐：

```python
num_groups = num_q_heads // num_kv_heads  # 32 // 8 = 4
q_grouped = query_states.view(bsz, num_kv_heads, num_groups, seq_len, head_dim)
q_mean = q_grouped.mean(dim=2)  # [bsz, 8, seq_len, 128]
```

`head_dim` 为 128（DeepSeek-R1-Distill-Llama-8B 的 hidden_size=2048, num_q_heads=32 → 2048/32=64？不对，DeepSeek 实际是 hidden_size 4096, 32 heads → head_dim=128）。

### 2.2 打分与选择流程

```
Q_obs = q_mean[:, :, -window_size:, :]            # [1, 8, 64, 128]
attn = softmax(Q_obs @ K^T / sqrt(128))            # [1, 8, 64, full_len]
scores = attn[:, :, :, :-window_size].sum(dim=2)   # [1, 8, prefix_len]
smoothed = avg_pool1d(scores, kernel_size=5)        # 局部平滑
indices = topk(smoothed, k=budget - window_size)    # [1, 8, k]
K_new = gather(K_prefix, indices) + K_window
V_new = gather(V_prefix, indices) + V_window
```

---

## 3. 工程问题与解决

### 3.1 Cache 接口适配（transformers 5.x DynamicCache）

**问题**：官方 SnapKV 实现针对旧版 transformers，新版本（5.6.2）的 KV cache 改为 `DynamicCache` 对象，`past_key_values` 不再是简单的 tuple of tuples。

**解决**：通过 monkey-patch `LlamaAttention.forward` 捕获每层的 query，存储到 `self._last_query`。prefill 后通过 `past_kv.layers[i].keys` 访问并原地修改 cache：

```python
# 捕获 query（在 patched forward 中）
self._last_query = query_states.detach()

# prefill 后压缩（layer.keys 是 tensor，支持原地赋值）
layer.keys = k_new
layer.values = v_new
```

`DynamicCache.get_seq_length()` 会自动反映修改后的长度。

### 3.2 GQA Shape 匹配

**问题**：SnapKV 打分需要 Q 和 K 的 head 维度一致。但 GQA 下 Q 有 32 heads，K 只有 8 heads。

**解决**：Q 按 KV head 分组取 mean，降为 8 heads。同时从 model config 自动检测 head 数：

```python
num_q_heads = model.config.num_attention_heads     # 32
num_kv_heads = getattr(model.config, "num_key_value_heads", num_q_heads)  # 8
```

避免硬编码，支持换模型。

### 3.3 Attention Mask 冲突

**问题**：压缩后 `past_key_values` 的序列长度从 N 变为 K（K < N），但传给 `model.generate()` 的 `attention_mask` 仍是原始长度 N。导致 mask 长度和 KV cache 长度不一致，产生静默错误结果。

**定位**：表现为压缩后的模型输出与 baseline 差异巨大（准确率从 83% 暴跌至 0-17%）。

**解决**：利用 HuggingFace `generate()` 的内部机制——当 `past_key_values` 不为 None 时，`generate()` 会自动从 cache 长度推导 attention_mask。不手动传递 attention_mask，而是通过 `**inputs` 整体传入：

```python
# 错误做法（D3 bug）
model.generate(input_ids=input_ids[:, -1:],
               attention_mask=original_mask,  # 原始长度 N
               past_key_values=compressed_kv)  # 压缩后长度 K

# 正确做法
model.generate(**inputs,                      # 完整 inputs dict
               past_key_values=compressed_kv)  # generate 自动匹配
```

---

## 4. 代码架构演进

### 4.1 初始状态：5 份重复代码

每个实验脚本独立定义 `snapkv_compress_gqa`、`patch_llama_attention`、`compress_cache`，存在细微差异（返回值 2 元组 vs 3 元组），bug 修复需改 5 处。

### 4.2 统一库：snapkv_lib.py

将公共逻辑提取到 `snapkv_lib.py`，所有实验 `import` 同一份实现。模块结构：

```
snapkv_lib.py
├── _get_head_counts()           # auto-detect GQA from model config
├── _snapkv_compute_indices()    # attention scoring → topk indices
├── _apply_indices()             # apply indices to compress KV
├── snapkv_compress_gqa()        # SnapKV compression (delegates to above)
├── random_compress()            # Control baseline: random selection
├── keepfirst_compress()         # Control baseline: keep first K tokens
├── patch_llama_attention()      # Monkey-patch forward to capture query
├── restore_attention()          # Restore original forward
├── compress_cache()             # Per-layer cache compression with stride
├── run_inference()              # Unified prefill→compress→decode pipeline
├── check_correct_keyword()      # Answer evaluation with negation detection
└── collect_metadata()           # Environment/git version info
```

### 4.3 实验脚本

| 脚本 | 用途 | 使用共享库 |
|------|------|:--:|
| `experiment3_comprehension.py` | 6 题阅读理解，Baseline + 4 SnapKV + 2 Random + 2 KeepFirst | ✅ |
| `experiment5_position.py` | 8157 token 位置注入，5 位置 × 9 配置 | ✅ |
| `experiment4_longbench.py` | LongBench qasper 50 样本 | ✅ |

---

## 5. 性能优化

### 5.1 延迟分解

将端到端时间拆分为三段，定位瓶颈：

| 阶段 | 含义 | 决定因素 |
|------|------|---------|
| prefill_time | 模型前向计算，产出完整 KV cache | 输入长度 |
| compress_time | SnapKV 打分 + 选择 + 裁剪 | 输入长度 × 打分窗口 |
| decode_time | 逐 token 生成（attention 使用压缩后 KV） | 输出长度 × 压缩后 KV 长度 |

**发现**：在 8157 token 输入上，compress_time = 0.46s，decode_time 从 5.67s 降至 2.50s。压缩开销 0.46s 换来 decode 节省 3.17s，净收益 2.71s（+47%）。

### 5.2 层采样（Layer Stride）

**问题**：每层都做完整 SnapKV 打分（QK matmul + softmax + pool + topk），32 层合计 0.46s。

**方案**：只对每第 N 层做完整打分，其余层复用最近采样层的 topk 索引。

**原理**：相邻层的注意力模式高度相似，Layer 0 认为重要的 token，Layer 1-3 大概率也认为重要。

**实现**：
```python
sampled_layers = list(range(0, n_layers, stride))  # stride=4 → [0,4,8,12,16,20,24,28]
sampled_layers.append(n_layers - 1)                 # 总是包含最后一层

# Phase 1: 只在采样层计算索引
for i in sampled_layers:
    sampled_indices[i] = compute_attention_scores(layer_i_query, layer_i_keys)

# Phase 2: 所有层用最近采样层的索引
for i in range(n_layers):
    nearest = min(sampled_indices.keys(), key=lambda s: abs(s - i))
    compress_layer_i_with_indices(sampled_indices[nearest])
```

**效果**：

| 配置 | 准确率 | 压缩时间 | 加速比 |
|------|--------|---------|--------|
| stride=1（全量） | 100% | 0.460s | 1× |
| stride=4（9/32 层） | **100%** | **0.136s** | **3.4×** |
| stride=8（5/32 层） | 80% | 0.121s | 3.8× |

stride=4 是甜点：准确率无损，压缩开销从 KeepFirst 的 10× 降至 2.9×。

### 5.3 打分窗口解耦

将"打分用的 query 数"和"观察窗口大小"分离为独立参数：

- `window_size`（默认 64）：末尾始终保留的 token 数
- `scoring_window`（默认 = window_size）：用于打分的 query 数

减小 `scoring_window` 可降低 QK matmul 开销，但过度减小会丢失打分多样性，导致中间位置的 token 无法被正确评分。实验中 scoring=8 或 4 时 P75（文档 73% 位置）均丢失。该参数保留为灵活性工具，不推荐 < 16。

---

## 6. 实验体系

### 6.1 实验矩阵

| 实验 | 脚本 | 测试内容 | 序列长度 | 样本数 |
|------|------|---------|---------|--------|
| Exp1 | experiment1_longdoc.py | 长文档问答（2 主题） | ~5K tokens | 2 |
| Exp2 | experiment2_needle.py | 大海捞针（5 位置） | ~5.5K tokens | 25 |
| Exp3 | experiment3_comprehension.py | 论文全文理解（6 题） | ~1.5K tokens | 6 |
| Exp5 | experiment5_position.py | 位置注入检索（5 位置） | ~8.2K tokens | 5 |

### 6.2 对照基线

在 Exp3 和 Exp5 中引入两个零成本基线，验证 SnapKV 打分是否有信息量：

| 基线 | 策略 | 含义 |
|------|------|------|
| **Random-K** | 随机选 K 个 token + 观察窗口 | 如果和 SnapKV 同分，说明打分无意义 |
| **KeepFirst-K** | 保留前 K 个 token + 观察窗口 | 模拟文档开头有重要信息的场景 |

**关键发现**：

在短文档（1.5K tokens，transformer 论文）上，SnapKV-256 和 KeepFirst-256 同为 83%，无法区分。但换到长文档位置注入测试（8.2K tokens），SnapKV-256 保持 100% 而 KeepFirst-256 跌至 80%——差异在文档 73% 位置的 needle（P75），KeepFirst 无法到达。

### 6.3 位置注入实验设计

将 4 篇技术文章（Transformer / vLLM / CUDA / RLHF）各重复 6 次拼接成 8157 token 长文档，在 5 个位置埋入独特的数字事实：

| 位置 | Token | 事实 | 问题 |
|------|-------|------|------|
| P0 (0%) | 0 | 端口号 7742 | 端口号是多少？ |
| P25 (26%) | 2119 | 加速比 94 | 加速比是多少？ |
| P50 (50%) | 4040 | 批次大小 47 | 最优批次大小？ |
| P75 (73%) | 5956 | checkpoint 版本 305 | checkpoint 版本？ |
| P100 (99%) | 8144 | 超时值 73 秒 | 超时值是多少？ |

**设计要点**：
- 纯数字答案，避免 BPE tokenizer 切碎连字符/符号
- 不冲突模型先验（如不用 "temperature=73"，模型认为温度应为 0-1）
- Baseline 100% 验证所有 needle 可检索

---

## 7. 关键数据

### 7.1 位置注入测试（8157 tokens）

| Config | Acc | Compress | Decode | P75 |
|--------|-----|----------|--------|:--:|
| Baseline | 100% | — | 5.67s | OK |
| SnapKV-256 | 100% | 0.460s | 2.50s | OK |
| SnapKV-256-stride4 | 100% | 0.136s | 2.50s | OK |
| SnapKV-256-fast4 | 80% | 0.279s | 2.51s | XX |
| KeepFirst-256 | 80% | 0.047s | 2.15s | XX |
| Random-256 | 20% | 0.566s | 2.50s | XX |

### 7.2 阅读理解测试（6 题，transformer 论文）

| Config | Acc | Time | KV Retention |
|--------|-----|------|:--:|
| Baseline | 83% | 1.52s | 100% |
| SnapKV-256 | 83% | 1.06s | 17% |
| SnapKV-512 | 83% | 1.05s | 33% |
| KeepFirst-256 | 83% | 1.06s | 17% |
| Random-256 | 0% | 1.06s | 17% |

### 7.3 速度收益汇总

| 输入长度 | Decode 加速 | 压缩开销 | 净收益 |
|---------|:--:|:--:|:--:|
| ~1.5K tokens | 1.3× | 0.04s | 31% |
| ~8.2K tokens | 2.3× | 0.14s | 47% |

---

## 8. 经验教训

### 8.1 为什么 KeepFirst 能和 SnapKV 同分？

在短文档（1.5K tokens）且重要信息在开头（论文摘要）的场景下，KeepFirst-256 恰好保留了 256/1540 = 16.6% 的关键信息。这不是 SnapKV 的失败，而是测试设计没有区分度。

**教训**：评估压缩策略时，必须用信息分布在文档不同位置的测试。位置注入实验正是为此设计。

### 8.2 层采样为什么有效？

相邻层的 attention 模式高度相关（层 0-3 关注相似的 token 集合）。SnapKV 的打分基于末尾 query 的注意力分布，而这些 query 在不同层间也呈现相似的模式。因此每 4 层采样一次足以捕获跨层的共同信息。

### 8.3 观测窗口的双重角色

SnapKV 的 window_size 同时控制：1) 打分 query 数（计算成本）；2) 末尾保留 token 数（安全边际）。将两者解耦（scoring_window vs window_size）增加了灵活性，但缩小 scoring_window 会导致打分多样性不足——中间位置的 token 无法被足够多的 query 注意到。实践中 scoring_window 不应低于 window_size 的 25%。

### 8.4 测试设计优先于优化

早期实验（Exp1/Exp3）缺少对照基线，导致无法区分"压缩本身有效"和"SnapKV 的打分方法有效"。加入 Random-K 和 KeepFirst-K 后，实验的因果推断能力大幅提升。**先确保实验能回答核心问题，再优化代码。**

---

## 9. 残余问题

| 级别 | 问题 | 描述 |
|------|------|------|
| Blocker | D1 | experiment1_snapkv.py 的 patch 从未调用 compress() |
| Medium | D8-D12 | 代码细节问题（cache 清理、prompt 截断统一等） |
| Low | D13-D15 | 方法论笔记（pooling 平滑、位置先验、观察窗口假设） |
| 数据 | — | LongBench qasper 数据需下载到服务器以验证 Exp4 |

---

## 附录：Git 提交历史

| Commit | 轮次 | 内容 |
|--------|:--:|------|
| `5115248` | 启动 | BASELINE.md + DEFECTS.md |
| `9ea3163` | R1 | snapkv_lib.py 统一库、对照基线、计时分解 |
| `1c89774` | R2 | 位置注入实验、D16 解决 |
| `355a92e` | R3-4 | 层采样优化、D17 解决 |
| `85fa22b` | R5 | D2/D3 解决、Exp4 迁移 |
