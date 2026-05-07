import torch
import time
import math
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.llama.modeling_llama import LlamaAttention
import json

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

def snapkv_compress_gqa(query_states, key_states, value_states,
                         window_size, max_capacity_prompt, kernel_size=5,
                         num_q_heads=32, num_kv_heads=8):
    """
    GQA 版本的 SnapKV 压缩
    Q: [bsz, 32, seq, head_dim]
    K/V: [bsz, 8, seq, head_dim]
    每 4 个 Q head 共享 1 个 KV head
    """
    bsz, num_kv, q_len, head_dim = key_states.shape
    if q_len <= max_capacity_prompt:
        return key_states, value_states

    num_groups = num_q_heads // num_kv_heads  # 4

    # 把 Q reshape 成 [bsz, kv_heads, groups, seq, head_dim]
    # 然后在 group 维度取平均，得到 [bsz, kv_heads, seq, head_dim]
    q_grouped = query_states.view(bsz, num_kv_heads, num_groups, q_len, head_dim)
    q_mean = q_grouped.mean(dim=2)  # [bsz, 8, seq, head_dim]

    # 用观察窗口的平均 Q 对 K 打分
    attn_weights = torch.matmul(
        q_mean[:, :, -window_size:, :],   # [bsz, 8, window, head_dim]
        key_states.transpose(2, 3)         # [bsz, 8, head_dim, seq]
    ) / math.sqrt(head_dim)
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

    # 前缀部分打分并求和
    prefix_weights = attn_weights[:, :, :, :-window_size].sum(dim=2)  # [bsz, 8, prefix_len]

    # 1D avg pooling 聚类
    pooled = F.avg_pool1d(
        prefix_weights,
        kernel_size=kernel_size,
        padding=kernel_size // 2,
        stride=1
    )

    # 选 top-k
    k = max_capacity_prompt - window_size
    indices = pooled.topk(k, dim=-1).indices          # [bsz, 8, k]
    indices_expanded = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)

    # 压缩 KV
    k_compressed = key_states[:, :, :-window_size, :].gather(dim=2, index=indices_expanded)
    v_compressed = value_states[:, :, :-window_size, :].gather(dim=2, index=indices_expanded)
    key_out   = torch.cat([k_compressed, key_states[:, :, -window_size:, :]], dim=2)
    value_out = torch.cat([v_compressed, value_states[:, :, -window_size:, :]], dim=2)

    compression_ratio = q_len / key_out.shape[2]
    return key_out, value_out

def patch_llama_attention(window_size=64, max_capacity_prompt=512, kernel_size=5,
                          num_q_heads=32, num_kv_heads=8):
    original_forward = LlamaAttention.forward

    def new_forward(self, hidden_states, position_embeddings=None,
                    attention_mask=None, past_key_values=None, **kwargs):
        from transformers.models.llama.modeling_llama import (
            apply_rotary_pos_emb, ALL_ATTENTION_FUNCTIONS, eager_attention_forward
        )
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states   = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # SnapKV: prefill 阶段压缩
        if query_states.shape[2] > 1:
            key_states, value_states = snapkv_compress_gqa(
                query_states, key_states, value_states,
                window_size, max_capacity_prompt, kernel_size,
                num_q_heads, num_kv_heads
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

    LlamaAttention.forward = new_forward
    print(f"SnapKV patched: window={window_size}, budget={max_capacity_prompt}")
    return original_forward

def restore_attention(original_forward):
    LlamaAttention.forward = original_forward
    print("Restored original attention")

def run_inference(model, tokenizer, context, question):
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192).to("cuda")
    input_len = inputs.input_ids.shape[-1]
    print(f"  Input length: {input_len} tokens")

    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False, temperature=1.0)
    elapsed = time.time() - start

    generated = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    return {
        "answer": generated.strip(),
        "time": elapsed,
        "peak_mem_gb": torch.cuda.max_memory_allocated() / 1024**3,
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

    original_forward = LlamaAttention.forward

    configs = [
        {"window_size": 64, "max_capacity_prompt": 512,  "label": "SnapKV-512"},
        {"window_size": 64, "max_capacity_prompt": 1024, "label": "SnapKV-1024"},
        {"window_size": 64, "max_capacity_prompt": 2048, "label": "SnapKV-2048"},
    ]

    all_results = []

    for cfg in configs:
        print(f"\n{'='*50}")
        print(f"Config: {cfg['label']}")
        print(f"{'='*50}")

        orig = patch_llama_attention(
            cfg["window_size"], cfg["max_capacity_prompt"],
            num_q_heads=32, num_kv_heads=8
        )

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

        # 恢复原始 forward 再换下一个 config
        restore_attention(orig)

    with open("results_snapkv_v3.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    for r in all_results:
        print(f"\n{r['config']}:")
        for c in r["cases"]:
            print(f"  Q: {c['question'][:50]}...")
            print(f"     {'✅' if c['correct'] else '❌'}  time={c['time']:.2f}s  mem={c['peak_mem_gb']:.2f}GB")

if __name__ == "__main__":
    main()
