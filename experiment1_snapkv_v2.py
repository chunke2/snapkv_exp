import torch
import time
import math
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.llama.modeling_llama import LlamaAttention
from transformers.cache_utils import Cache
import json
import types

MODEL_PATH = "/workspace/models/models--deepseek-ai--DeepSeek-R1-Distill-Llama-8B/snapshots/6a6f4aa4197940add57724a7707d069478df56b1"

TEST_CASES = [
    {
        "context": """The transformer architecture was introduced in the paper "Attention is All You Need" 
        by Vaswani et al. in 2017. It relies entirely on attention mechanisms, dispensing with recurrence 
        and convolutions entirely. The model consists of an encoder and decoder, each composed of 
        multiple layers. Each layer has two sub-layers: a multi-head self-attention mechanism and a 
        position-wise fully connected feed-forward network.
        """ * 20,
        "question": "What year was the transformer architecture introduced?",
        "answer": "2017"
    },
    {
        "context": """PagedAttention was introduced by the vLLM team to solve memory management 
        challenges in large language model inference. The key insight is that the KV cache of a 
        request can be stored in non-contiguous memory spaces, similar to how operating systems 
        manage virtual memory with paging. The PagedAttention algorithm divides the KV cache into 
        fixed-size blocks, each containing the key-value pairs for a fixed number of tokens. 
        The implementation uses a block size of 16 tokens per block by default.
        """ * 20,
        "question": "What is the default block size used in PagedAttention?",
        "answer": "16"
    }
]

def snapkv_compress(query_states, key_states, value_states, window_size, max_capacity_prompt, kernel_size=5):
    """
    SnapKV 压缩核心逻辑:
    1. 用观察窗口的 query 对前缀 key 打分
    2. 1D avg pooling 聚类
    3. 选 top-k 保留
    4. 拼接压缩后前缀 + 完整观察窗口
    """
    bsz, num_heads, q_len, head_dim = query_states.shape

    # 只在 prefill 且超过预算时压缩
    if q_len <= max_capacity_prompt:
        return key_states, value_states

    # 1. 观察窗口的 query 对所有 key 计算 attention score
    attn_weights = torch.matmul(
        query_states[:, :, -window_size:, :],
        key_states.transpose(2, 3)
    ) / math.sqrt(head_dim)
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

    # 2. 只看前缀部分（去掉观察窗口自身）并求和
    prefix_weights = attn_weights[:, :, :, :-window_size].sum(dim=2)

    # 3. 1D avg pooling 聚类相邻 token
    pooled = F.avg_pool1d(
        prefix_weights,
        kernel_size=kernel_size,
        padding=kernel_size // 2,
        stride=1
    )

    # 4. 选 top-k
    k = max_capacity_prompt - window_size
    indices = pooled.topk(k, dim=-1).indices
    indices_expanded = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)

    # 5. 拼接
    k_compressed = key_states[:, :, :-window_size, :].gather(dim=2, index=indices_expanded)
    v_compressed = value_states[:, :, :-window_size, :].gather(dim=2, index=indices_expanded)
    key_out = torch.cat([k_compressed, key_states[:, :, -window_size:, :]], dim=2)
    value_out = torch.cat([v_compressed, value_states[:, :, -window_size:, :]], dim=2)

    return key_out, value_out

def patch_llama_attention(model, window_size=64, max_capacity_prompt=512, kernel_size=5):
    """把 SnapKV 注入到每一层的 LlamaAttention.forward"""
    from transformers.models.llama import modeling_llama as llama_module

    # 拿到原始 forward
    original_forward = LlamaAttention.forward

    def new_forward(
        self,
        hidden_states,
        position_embeddings=None,
        attention_mask=None,
        past_key_values=None,
        **kwargs
    ):
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, ALL_ATTENTION_FUNCTIONS, eager_attention_forward
        from typing import Callable

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # SnapKV 压缩：在存入 cache 之前压缩
        seq_len = query_states.shape[2]
        if seq_len > 1:  # prefill 阶段
            key_states, value_states = snapkv_compress(
                query_states, key_states, value_states,
                window_size, max_capacity_prompt, kernel_size
            )

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        attn_output, attn_weights = attention_interface(
            self, query_states, key_states, value_states,
            attention_mask, dropout=0.0, scaling=self.scaling, **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    # Monkey-patch
    LlamaAttention.forward = new_forward
    print(f"Patched LlamaAttention.forward with SnapKV (window={window_size}, budget={max_capacity_prompt})")

def restore_llama_attention():
    from transformers.models.llama.modeling_llama import LlamaAttention
    # 重新 import 恢复原始
    import importlib
    import transformers.models.llama.modeling_llama as mod
    importlib.reload(mod)
    print("Restored original LlamaAttention.forward")

def get_peak_memory_gb():
    return torch.cuda.max_memory_allocated() / 1024**3

def run_inference(model, tokenizer, context, question):
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192).to("cuda")
    input_len = inputs.input_ids.shape[-1]

    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False, temperature=1.0)
    elapsed = time.time() - start

    generated = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    return {
        "answer": generated.strip(),
        "time": elapsed,
        "peak_mem_gb": get_peak_memory_gb(),
        "input_len": input_len,
    }

def main():
    with open("results_baseline.json") as f:
        baseline_results = json.load(f)

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.float16, device_map="cuda", use_cache=True,
    )
    model.eval()
    print("Model loaded.\n")

    configs = [
        {"window_size": 64, "max_capacity_prompt": 512,  "label": "SnapKV-512"},
        {"window_size": 64, "max_capacity_prompt": 1024, "label": "SnapKV-1024"},
        {"window_size": 64, "max_capacity_prompt": 2048, "label": "SnapKV-2048"},
    ]

    all_results = []

    for cfg in configs:
        print(f"\n{'='*50}")
        print(f"Config: {cfg['label']}  (window={cfg['window_size']}, budget={cfg['max_capacity_prompt']})")
        print(f"{'='*50}")

        # 注入 SnapKV
        patch_llama_attention(model, cfg["window_size"], cfg["max_capacity_prompt"])

        cfg_results = {"config": cfg["label"], "cases": []}
        for i, test in enumerate(TEST_CASES):
            print(f"\n[Test {i+1}] {test['question']}")
            result = run_inference(model, tokenizer, test["context"], test["question"])
            baseline = baseline_results[i]["baseline"]

            correct = test["answer"] in result["answer"]
            print(f"  Answer:   {result['answer'][:80]}")
            print(f"  Correct:  {'✅' if correct else '❌'} (expected: {test['answer']})")
            print(f"  Time:     {result['time']:.2f}s  (baseline: {baseline['time']:.2f}s)")
            print(f"  Peak mem: {result['peak_mem_gb']:.2f} GB")

            cfg_results["cases"].append({
                "question": test["question"],
                "correct": correct,
                "time": result["time"],
                "baseline_time": baseline["time"],
                "peak_mem_gb": result["peak_mem_gb"],
            })
        all_results.append(cfg_results)

    with open("results_snapkv_v2.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nAll results saved to results_snapkv_v2.json")

if __name__ == "__main__":
    main()
