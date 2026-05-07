import torch
import time
import math
import torch.nn.functional as F
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import json

MODEL_PATH = "/workspace/models/models--deepseek-ai--DeepSeek-R1-Distill-Llama-8B/snapshots/6a6f4aa4197940add57724a7707d069478df56b1"

TEST_CASES = [
    {
        "context": """The transformer architecture was introduced in the paper "Attention is All You Need" 
        by Vaswani et al. in 2017. It relies entirely on attention mechanisms, dispensing with recurrence 
        and convolutions entirely. The model consists of an encoder and decoder, each composed of 
        multiple layers. Each layer has two sub-layers: a multi-head self-attention mechanism and a 
        position-wise fully connected feed-forward network. The encoder maps an input sequence of symbol 
        representations to a sequence of continuous representations. Given this sequence, the decoder 
        then generates an output sequence of symbols one element at a time.
        
        The attention function maps a query and a set of key-value pairs to an output. The output is 
        computed as a weighted sum of the values, where the weight assigned to each value is computed 
        by a compatibility function of the query with the corresponding key. The model uses multi-head 
        attention, which allows the model to jointly attend to information from different representation 
        subspaces at different positions.
        
        Positional encoding is added to the input embeddings to give the model information about the 
        relative or absolute position of the tokens in the sequence. The positional encodings have the 
        same dimension as the embeddings so they can be summed. The authors used sine and cosine 
        functions of different frequencies for positional encoding.
        """ * 20,
        "question": "What year was the transformer architecture introduced?",
        "answer": "2017"
    },
    {
        "context": """PagedAttention was introduced by the vLLM team to solve memory management 
        challenges in large language model inference. The key insight is that the KV cache of a 
        request can be stored in non-contiguous memory spaces, similar to how operating systems 
        manage virtual memory with paging. In traditional LLM serving systems, the KV cache for 
        each request is stored in contiguous GPU memory, which leads to significant memory 
        fragmentation and limits the number of requests that can be served concurrently.
        
        The PagedAttention algorithm divides the KV cache into fixed-size blocks, each containing 
        the key-value pairs for a fixed number of tokens. These blocks can be stored in 
        non-contiguous physical memory, and a block table maintains the mapping between logical 
        and physical blocks for each request. This approach virtually eliminates memory 
        fragmentation and enables more efficient memory sharing between requests.
        
        The implementation uses a block size of 16 tokens per block by default. When a request 
        needs more KV cache space, the system allocates new blocks rather than requiring 
        contiguous memory. This allows the system to pack more requests into the same amount 
        of GPU memory, increasing throughput significantly.
        """ * 20,
        "question": "What is the default block size used in PagedAttention?",
        "answer": "16 tokens"
    }
]

# SnapKV hook 注入
def make_snapkv_hook(window_size=64, max_capacity_prompt=512, kernel_size=5):
    def hook(module, args, kwargs):
        # 从 kwargs 里拿 hidden_states
        hidden_states = kwargs.get("hidden_states", args[0] if args else None)
        if hidden_states is None:
            return
        seq_len = hidden_states.shape[1]
        # 只在 prefill 阶段（seq_len > max_capacity_prompt）触发
        if seq_len <= max_capacity_prompt:
            return
        module._snapkv_seq_len = seq_len
    return hook

class SnapKVCluster:
    def __init__(self, window_size=64, max_capacity_prompt=512, kernel_size=5):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        self.kernel_size = kernel_size

    def compress(self, key_states, query_states, value_states):
        bsz, num_heads, q_len, head_dim = query_states.shape
        if q_len <= self.max_capacity_prompt:
            return key_states, value_states

        # 用观察窗口的 query 对所有 key 计算 attention
        attn_weights = torch.matmul(
            query_states[:, :, -self.window_size:, :],
            key_states.transpose(2, 3)
        ) / math.sqrt(head_dim)
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        # 对前缀部分（非观察窗口）求和打分
        attn_weights_sum = attn_weights[:, :, :, :-self.window_size].sum(dim=2)

        # 1D avg pooling 聚类
        attn_cache = F.avg_pool1d(
            attn_weights_sum,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
            stride=1
        )

        # 选 top-k 重要 token
        k = self.max_capacity_prompt - self.window_size
        indices = attn_cache.topk(k, dim=-1).indices
        indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)

        # 压缩前缀 KV
        k_compress = key_states[:, :, :-self.window_size, :].gather(dim=2, index=indices)
        v_compress = value_states[:, :, :-self.window_size, :].gather(dim=2, index=indices)

        # 拼接：压缩后的前缀 + 完整观察窗口
        k_out = torch.cat([k_compress, key_states[:, :, -self.window_size:, :]], dim=2)
        v_out = torch.cat([v_compress, value_states[:, :, -self.window_size:, :]], dim=2)

        return k_out, v_out

def patch_model_with_snapkv(model, window_size=64, max_capacity_prompt=512, kernel_size=5):
    cluster = SnapKVCluster(window_size, max_capacity_prompt, kernel_size)
    patched_layers = 0

    for layer in model.model.layers:
        attn = layer.self_attn
        original_forward = attn.forward

        def make_new_forward(orig_forward, cluster):
            def new_forward(*args, **kwargs):
                # 拿到 hidden_states
                hidden_states = kwargs.get("hidden_states", args[0] if args else None)
                if hidden_states is None:
                    return orig_forward(*args, **kwargs)

                seq_len = hidden_states.shape[1]

                # decode 阶段不压缩
                if seq_len <= 1:
                    return orig_forward(*args, **kwargs)

                # prefill 阶段且超过预算才压缩
                if seq_len <= cluster.max_capacity_prompt:
                    return orig_forward(*args, **kwargs)

                # 先跑原始 forward 拿到 Q/K/V（通过临时 hook）
                result = orig_forward(*args, **kwargs)
                return result
            return new_forward

        attn.forward = make_new_forward(original_forward, cluster)
        patched_layers += 1

    print(f"Patched {patched_layers} attention layers")
    return cluster

def get_gpu_memory():
    return torch.cuda.memory_allocated() / 1024**3

def run_inference(model, tokenizer, context, question):
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192).to("cuda")
    input_len = inputs.input_ids.shape[-1]
    print(f"  Input length: {input_len} tokens")

    mem_before = get_gpu_memory()
    start_time = time.time()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            temperature=1.0,
        )

    elapsed = time.time() - start_time
    mem_after = get_gpu_memory()
    generated = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)

    return {
        "answer": generated.strip(),
        "time": elapsed,
        "mem_peak_gb": torch.cuda.max_memory_allocated() / 1024**3,
        "input_len": input_len
    }

def main():
    # 读 baseline 结果
    with open("results_baseline.json") as f:
        baseline_results = json.load(f)

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.float16,
        device_map="cuda",
        use_cache=True,
    )
    model.eval()
    print("Model loaded.\n")

    # 测试不同压缩率
    configs = [
        {"window_size": 64, "max_capacity_prompt": 512,  "label": "SnapKV-512  (~9% budget)"},
        {"window_size": 64, "max_capacity_prompt": 1024, "label": "SnapKV-1024 (~18% budget)"},
        {"window_size": 64, "max_capacity_prompt": 2048, "label": "SnapKV-2048 (~36% budget)"},
    ]

    all_results = {"baseline": baseline_results, "snapkv": []}

    for cfg in configs:
        print(f"\n{'='*50}")
        print(f"Config: {cfg['label']}")
        print(f"{'='*50}")

        cluster = SnapKVCluster(cfg["window_size"], cfg["max_capacity_prompt"])
        cfg_results = {"config": cfg["label"], "cases": []}

        for i, test in enumerate(TEST_CASES):
            print(f"\n[Test Case {i+1}] {test['question']}")
            torch.cuda.reset_peak_memory_stats()

            result = run_inference(model, tokenizer, test["context"], test["question"])
            baseline = baseline_results[i]

            correct = test["answer"].lower() in result["answer"].lower()
            print(f"  Answer:   {result['answer'][:80]}")
            print(f"  Expected: {test['answer']}  → {'✅' if correct else '❌'}")
            print(f"  Time:     {result['time']:.2f}s (baseline: {baseline['baseline']['time']:.2f}s)")
            print(f"  Peak mem: {result['mem_peak_gb']:.2f} GB")

            cfg_results["cases"].append({
                "question": test["question"],
                "expected": test["answer"],
                "got": result["answer"][:100],
                "correct": correct,
                "time": result["time"],
                "baseline_time": baseline["baseline"]["time"],
            })

        all_results["snapkv"].append(cfg_results)

    with open("results_snapkv.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nResults saved to results_snapkv.json")

if __name__ == "__main__":
    main()
