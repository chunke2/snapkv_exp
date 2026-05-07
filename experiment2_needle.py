import torch
import time
import math
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.llama.modeling_llama import LlamaAttention
import json
import random

MODEL_PATH = "/workspace/models/models--deepseek-ai--DeepSeek-R1-Distill-Llama-8B/snapshots/6a6f4aa4197940add57724a7707d069478df56b1"

# 背景文（足够长的技术文章）
BACKGROUND = """
The transformer architecture was introduced in the paper "Attention is All You Need" 
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
subspaces at different positions. With h heads, MultiHead(Q,K,V) = Concat(head_1,...,head_h)W^O.

Positional encoding is added to the input embeddings to give the model information about the 
relative or absolute position of the tokens in the sequence. The positional encodings have the 
same dimension as the embeddings so they can be summed. The authors used sine and cosine 
functions of different frequencies for positional encoding. PE(pos,2i)=sin(pos/10000^(2i/d)).

PagedAttention was introduced by the vLLM team to solve memory management challenges in large 
language model inference. The key insight is that the KV cache of a request can be stored in 
non-contiguous memory spaces, similar to how operating systems manage virtual memory with paging. 
In traditional LLM serving systems, the KV cache for each request is stored in contiguous GPU 
memory, which leads to significant memory fragmentation and limits concurrent requests.

The PagedAttention algorithm divides the KV cache into fixed-size blocks, each containing 
the key-value pairs for a fixed number of tokens. These blocks can be stored in non-contiguous 
physical memory, and a block table maintains the mapping between logical and physical blocks 
for each request. This approach virtually eliminates memory fragmentation and enables more 
efficient memory sharing between requests with common prefixes.

CUDA (Compute Unified Device Architecture) is NVIDIA parallel computing platform and programming 
model. It enables dramatic increases in computing performance by harnessing the power of the GPU. 
The GPU architecture consists of thousands of smaller, more efficient cores designed for handling 
multiple tasks simultaneously. Modern NVIDIA GPUs contain Streaming Multiprocessors (SMs), each 
containing multiple CUDA cores, Tensor Cores for matrix operations, and shared memory.

Warps are the fundamental unit of execution on NVIDIA GPUs. A warp consists of 32 threads that 
execute in lockstep (SIMT - Single Instruction Multiple Thread). If threads in a warp take 
different execution paths, the GPU serializes the divergent paths, reducing efficiency. Occupancy 
refers to the ratio of active warps to the maximum number of warps supported by an SM.

Reinforcement Learning from Human Feedback (RLHF) is a technique for aligning language models 
with human preferences. It was popularized by InstructGPT in 2022 and has become a standard 
component of training state-of-the-art language models. The RLHF pipeline consists of three 
main stages: supervised fine-tuning, reward model training, and reinforcement learning with PPO.

Direct Preference Optimization (DPO) is an alternative to PPO that directly optimizes the 
policy using preference data without explicit reward model training. DPO derives a closed-form 
expression for the optimal policy given the preference data and optimizes it via supervised 
learning. This avoids the instability of RL training while achieving comparable alignment results.

Flash Attention is a memory-efficient attention algorithm that computes exact attention without 
storing the full attention matrix. It tiles the computation to fit in SRAM, reducing HBM memory 
accesses from O(N^2) to O(N). Flash Attention 2 further improved parallelism and work 
partitioning, achieving near-peak hardware utilization on modern GPUs with large sequence lengths.

Speculative decoding accelerates autoregressive generation by using a smaller draft model to 
propose multiple tokens at once, which are then verified in parallel by the larger target model. 
This achieves significant speedups when the draft model proposals are frequently accepted by 
the target model, reducing the number of sequential forward passes required for generation.

Quantization reduces model size and inference latency by representing weights and activations 
with lower precision. Common methods include post-training quantization (PTQ) and quantization-aware 
training (QAT). INT8 quantization typically preserves most model quality while halving memory 
usage. INT4 quantization offers further compression but requires careful calibration to avoid 
significant quality degradation on downstream tasks.
""" * 5  # 重复让背景文足够长

def build_document(needle, position_ratio, background):
    """
    把针插入背景文的指定位置
    position_ratio: 0.0=开头, 0.5=中间, 1.0=结尾
    """
    sentences = background.strip().split('\n\n')
    sentences = [s.strip() for s in sentences if s.strip()]
    
    insert_idx = int(len(sentences) * position_ratio)
    insert_idx = max(0, min(insert_idx, len(sentences)))
    
    needle_sentence = f"\n\nIMPORTANT: {needle}\n\n"
    sentences.insert(insert_idx, needle_sentence)
    
    return '\n\n'.join(sentences)

def snapkv_compress_gqa(query_states, key_states, value_states,
                         window_size, max_capacity_prompt, kernel_size=5,
                         num_q_heads=32, num_kv_heads=8):
    bsz, num_kv, q_len, head_dim = key_states.shape
    if q_len <= max_capacity_prompt:
        return key_states, value_states, False

    num_groups = num_q_heads // num_kv_heads
    q_grouped = query_states.view(bsz, num_kv_heads, num_groups, q_len, head_dim).mean(dim=2)

    attn_weights = torch.matmul(
        q_grouped[:, :, -window_size:, :],
        key_states.transpose(2, 3)
    ) / math.sqrt(head_dim)
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

    prefix_weights = attn_weights[:, :, :, :-window_size].sum(dim=2)
    pooled = F.avg_pool1d(prefix_weights, kernel_size=kernel_size,
                          padding=kernel_size // 2, stride=1)

    k = max_capacity_prompt - window_size
    indices = pooled.topk(k, dim=-1).indices
    indices_exp = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)

    k_comp = key_states[:, :, :-window_size, :].gather(dim=2, index=indices_exp)
    v_comp = value_states[:, :, :-window_size, :].gather(dim=2, index=indices_exp)
    key_out   = torch.cat([k_comp, key_states[:, :, -window_size:, :]], dim=2)
    value_out = torch.cat([v_comp, value_states[:, :, -window_size:, :]], dim=2)

    return key_out, value_out, True

def patch_llama_attention():
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
        self._last_query = query_states.detach()

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
    return original_forward

def restore_attention(orig):
    LlamaAttention.forward = orig

def compress_cache(model, past_kv, window_size, max_capacity_prompt):
    for i, layer in enumerate(past_kv.layers):
        q = model.model.layers[i].self_attn._last_query
        k_new, v_new, did = snapkv_compress_gqa(
            q, layer.keys, layer.values, window_size, max_capacity_prompt
        )
        if did:
            layer.keys = k_new
            layer.values = v_new

def run_inference(model, tokenizer, document, question,
                  use_snapkv=False, window_size=64, max_capacity_prompt=512):
    prompt = f"Document:\n{document}\n\nQuestion: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192).to("cuda")
    input_len = inputs.input_ids.shape[-1]

    torch.cuda.reset_peak_memory_stats()
    start = time.time()

    with torch.no_grad():
        out = model(**inputs, use_cache=True)
        past_kv = out.past_key_values

        compressed_len = input_len
        if use_snapkv:
            compress_cache(model, past_kv, window_size, max_capacity_prompt)
            compressed_len = past_kv.layers[0].keys.shape[2]

        outputs = model.generate(
            **inputs,
            past_key_values=past_kv,
            max_new_tokens=30,
            do_sample=False,
            temperature=1.0,
        )

    elapsed = time.time() - start
    generated = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)

    return {
        "answer": generated.strip(),
        "time": elapsed,
        "peak_mem_gb": torch.cuda.max_memory_allocated() / 1024**3,
        "input_len": input_len,
        "compressed_len": compressed_len,
    }

def main():
    random.seed(42)

    # 生成随机针
    secret_code = str(random.randint(10000, 99999))
    needle = f"In a related experiment, researchers recorded the value {secret_code} as the baseline measurement."
    question = "What value did researchers record as the baseline measurement in the related experiment?"

    print(f"needle: {needle}")
    print(f"secret_code: {secret_code}\n")

    # 针的位置
    positions = [
        (0.0,  "开头 (0%)"),
        (0.25, "25%处"),
        (0.5,  "中间 (50%)"),
        (0.75, "75%处"),
        (1.0,  "结尾 (100%)"),
    ]

    # 压缩配置
    configs = [
        {"use_snapkv": False,  "label": "Baseline"},
        {"use_snapkv": True, "window_size": 64, "max_capacity_prompt": 128,  "label": "SnapKV-128"},
        {"use_snapkv": True, "window_size": 64, "max_capacity_prompt": 512,  "label": "SnapKV-512"},
        {"use_snapkv": True, "window_size": 64, "max_capacity_prompt": 1024, "label": "SnapKV-1024"},
        {"use_snapkv": True, "window_size": 64, "max_capacity_prompt": 2048, "label": "SnapKV-2048"},
    ]

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.float16, device_map="cuda", use_cache=True,
    )
    model.eval()
    orig = patch_llama_attention()
    print("Model loaded.\n")

    # 结果矩阵: results[position_label][config_label] = {correct, answer, ...}
    results = {pos_label: {} for _, pos_label in positions}

    for pos_ratio, pos_label in positions:
        document = build_document(needle, pos_ratio, BACKGROUND)
        print(f"\n{'='*60}")
        print(f"针的位置: {pos_label}")
        print(f"{'='*60}")

        for cfg in configs:
            result = run_inference(
                model, tokenizer, document, question,
                use_snapkv=cfg.get("use_snapkv", False),
                window_size=cfg.get("window_size", 64),
                max_capacity_prompt=cfg.get("max_capacity_prompt", 512),
            )
            correct = secret_code in result["answer"].replace(" ", "").replace("-", "")
            ratio = result["compressed_len"] / result["input_len"] * 100

            print(f"  [{cfg['label']}] tokens: {result['input_len']}→{result['compressed_len']} ({ratio:.0f}%)  "
                  f"{'✅' if correct else '❌'}  {result['answer'][:60]}")

            results[pos_label][cfg["label"]] = {
                "correct": correct,
                "answer": result["answer"][:100],
                "input_len": result["input_len"],
                "compressed_len": result["compressed_len"],
                "retention_pct": round(ratio, 1),
                "time": result["time"],
            }

    restore_attention(orig)

    # 打印热力图
    print(f"\n{'='*60}")
    print("热力图（✅=找到针，❌=丢失）")
    print(f"{'='*60}")
    header = f"{'位置':<12}" + "".join(f"{cfg['label']:<15}" for cfg in configs)
    print(header)
    print("-" * 60)
    for _, pos_label in positions:
        row = f"{pos_label:<12}"
        for cfg in configs:
            r = results[pos_label][cfg["label"]]
            row += f"{'✅' if r['correct'] else '❌':<15}"
        print(row)

    with open("results_experiment2_needle.json", "w") as f:
        json.dump({
            "needle": needle,
            "secret_code": secret_code,
            "results": results
        }, f, indent=2, ensure_ascii=False)
    print("\nResults saved to results_experiment2_needle.json")

if __name__ == "__main__":
    main()
