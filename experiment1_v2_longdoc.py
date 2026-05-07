import torch
import time
import math
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.llama.modeling_llama import LlamaAttention
import json

MODEL_PATH = "/workspace/models/models--deepseek-ai--DeepSeek-R1-Distill-Llama-8B/snapshots/6a6f4aa4197940add57724a7707d069478df56b1"

# 用不同主题的长文本，避免简单重复
LONG_TEXTS = {
    "transformer": """
The transformer architecture was introduced in the seminal paper "Attention is All You Need" 
by Vaswani et al. in 2017. Unlike previous sequence-to-sequence models that relied on 
recurrent neural networks or convolutional neural networks, the transformer relies entirely 
on attention mechanisms to draw global dependencies between input and output.

The core of the transformer is the multi-head self-attention mechanism. Given a sequence of 
tokens, each token attends to every other token in the sequence, computing a weighted sum 
of value vectors based on the compatibility between query and key vectors. This allows the 
model to capture long-range dependencies that RNNs struggle with due to vanishing gradients.

The encoder consists of N identical layers, each with two sub-layers: multi-head self-attention 
and a position-wise feed-forward network. Residual connections and layer normalization are 
applied around each sub-layer. The decoder is similar but adds a third sub-layer performing 
multi-head attention over the encoder output.

Positional encoding is added to the input embeddings since the transformer contains no 
recurrence or convolution. The authors used sine and cosine functions of different frequencies:
PE(pos, 2i) = sin(pos/10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos/10000^(2i/d_model))

The attention function maps a query and a set of key-value pairs to an output:
Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V

Multi-head attention allows the model to jointly attend to information from different 
representation subspaces. With h heads:
MultiHead(Q,K,V) = Concat(head_1,...,head_h)W^O
where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)

The transformer was first applied to machine translation, achieving state-of-the-art results 
on English-to-German and English-to-French translation tasks. The model was trained on the 
WMT 2014 dataset and outperformed all previously published models including ensembles.

Training used the Adam optimizer with beta_1=0.9, beta_2=0.98, epsilon=10^-9, and a custom 
learning rate schedule that increases linearly for the first warmup_steps training steps, 
then decreases proportionally to the inverse square root of the step number.

The base model had 6 encoder and 6 decoder layers, with d_model=512, h=8, d_ff=2048, 
resulting in 65M parameters. The large model used d_model=1024, h=16, d_ff=4096, resulting 
in 213M parameters. Both were trained for 100,000 steps on 8 NVIDIA P100 GPUs.

Regularization techniques included residual dropout applied to the output of each sub-layer 
before it is added to the sub-layer input and normalized, as well as label smoothing of 
epsilon_ls=0.1. Label smoothing hurts perplexity but improves accuracy and BLEU score.

The transformer has since become the dominant architecture in NLP, forming the basis of 
models like BERT, GPT, T5, and many others. Its ability to parallelize computation makes 
it much faster to train than RNNs, and its attention mechanism provides better interpretability.

BERT (Bidirectional Encoder Representations from Transformers) was introduced by Google in 2018.
It uses only the encoder portion of the transformer and is pre-trained on two tasks: masked 
language modeling (MLM) and next sentence prediction (NSP). BERT achieved state-of-the-art 
results on 11 NLP tasks when it was released.

GPT (Generative Pre-trained Transformer) uses only the decoder portion and is trained with 
causal language modeling - predicting the next token given all previous tokens. GPT-2 was 
released in 2019 with 1.5B parameters, and GPT-3 in 2020 with 175B parameters.

The scaling laws for neural language models were studied by Kaplan et al. in 2020, showing 
that model performance improves predictably as a power law with model size, dataset size, 
and compute budget. This motivated the development of increasingly large language models.

Vision Transformer (ViT) demonstrated in 2020 that pure transformer architectures can 
achieve excellent results in computer vision when applied to sequences of image patches, 
challenging the dominance of convolutional neural networks in vision tasks.
""" * 4,

    "vllm": """
vLLM is a fast and memory-efficient inference and serving engine for large language models.
It was developed by researchers at UC Berkeley and introduced in 2023. The key innovation 
is PagedAttention, a novel attention algorithm that manages the KV cache with non-contiguous 
memory similar to virtual memory paging in operating systems.

Traditional LLM serving systems pre-allocate a contiguous chunk of GPU memory for each 
request's KV cache based on the maximum possible sequence length. This leads to three types 
of memory waste: internal fragmentation (reserved but unused slots within a sequence), 
external fragmentation (free memory that cannot be used because it is not contiguous), 
and reservation (memory reserved for future tokens that may never be generated).

PagedAttention solves this by dividing the KV cache into fixed-size blocks (pages), each 
storing the key-value pairs for a fixed number of tokens (typically 16). These blocks can 
be stored in non-contiguous physical memory locations, with a block table maintaining the 
mapping between logical and physical block positions for each sequence.

When a new request arrives, vLLM allocates blocks on-demand as tokens are generated, 
rather than pre-allocating the full potential sequence length. This allows multiple requests 
to efficiently share the GPU memory, dramatically increasing the number of requests that 
can be served simultaneously.

The continuous batching technique allows new requests to join a running batch dynamically,
rather than waiting for the entire batch to finish. This is crucial for keeping GPU utilization 
high when requests have varying lengths. Combined with PagedAttention, continuous batching 
enables throughput improvements of 2-4x compared to static batching approaches.

Memory sharing is another key feature of PagedAttention. When multiple sequences share a 
common prefix (e.g., system prompts in chat applications), they can share the physical 
KV cache blocks for that prefix, with copy-on-write semantics when a sequence needs to 
modify shared blocks. This is particularly beneficial for applications like parallel sampling 
and beam search.

vLLM supports various optimization techniques including tensor parallelism for distributing 
model weights across multiple GPUs, quantization methods like AWQ and GPTQ for reducing 
model size, speculative decoding for accelerating inference on smaller models, and prefix 
caching for reusing KV cache across requests with shared prefixes.

The vLLM serving framework exposes an OpenAI-compatible API, making it easy to use as a 
drop-in replacement for the OpenAI API. It supports both online serving (continuous request 
handling) and offline inference (batch processing of datasets).

Performance benchmarks show vLLM achieving 2-24x higher throughput than HuggingFace 
Transformers with naive implementations, depending on the model size and hardware. On 
an A100 GPU with LLaMA-13B, vLLM can serve over 2000 tokens per second with concurrent 
users, compared to around 200 for naive implementations.

The block size parameter controls how many tokens are stored in each KV cache block.
The default block size is 16 tokens per block. Larger block sizes reduce overhead from 
block table lookups but increase internal fragmentation. Smaller block sizes reduce 
fragmentation but increase management overhead.

GPU memory utilization in vLLM is controlled by the gpu_memory_utilization parameter,
which determines what fraction of the total GPU memory can be used for the KV cache.
The default value is 0.9, meaning 90% of GPU memory can be allocated to KV cache blocks.

vLLM also implements chunked prefill, which breaks long prompts into smaller chunks 
processed across multiple iterations. This prevents a single long prompt from blocking 
the batch for too long, improving overall system responsiveness and fairness.

The scheduler in vLLM uses a priority-based system to determine which requests to process 
in each iteration. Running requests have higher priority than waiting requests. When memory 
is insufficient, the scheduler may preempt lower-priority requests by swapping their KV 
cache to CPU memory or recomputing it when they are resumed.
""" * 4,

    "cuda": """
CUDA (Compute Unified Device Architecture) is NVIDIA's parallel computing platform and 
programming model. It enables dramatic increases in computing performance by harnessing 
the power of the GPU. CUDA was first introduced in 2006 with the GeForce 8800 GPU.

The GPU architecture consists of thousands of smaller, more efficient cores designed for 
handling multiple tasks simultaneously. Modern NVIDIA GPUs contain Streaming Multiprocessors 
(SMs), each containing multiple CUDA cores, Tensor Cores for matrix operations, and 
shared memory for fast intra-SM communication.

CUDA programs are organized into grids of thread blocks. Each thread block contains up to 
1024 threads that execute concurrently on a single SM. Threads within a block can communicate 
via shared memory and synchronize using __syncthreads(). The grid dimensions determine how 
many blocks are launched in total.

Warps are the fundamental unit of execution on NVIDIA GPUs. A warp consists of 32 threads 
that execute in lockstep (SIMT - Single Instruction Multiple Thread). If threads in a warp 
take different execution paths (branch divergence), the GPU serializes the divergent paths, 
reducing efficiency.

Memory hierarchy in CUDA includes registers (fastest, private to each thread), shared memory 
(fast, shared within a thread block), L1/L2 cache (automatic), and global memory (slowest, 
accessible by all threads). Efficient CUDA programming requires maximizing use of fast 
memory and minimizing global memory accesses.

Coalesced memory access is critical for performance. When threads in a warp access consecutive 
memory locations, the GPU can combine these into a single memory transaction. Non-coalesced 
accesses require multiple transactions and significantly reduce bandwidth utilization.

Occupancy refers to the ratio of active warps to the maximum number of warps supported by 
an SM. Higher occupancy generally leads to better latency hiding, as the GPU can switch to 
other warps while waiting for memory operations to complete. Factors affecting occupancy 
include register usage, shared memory usage, and block size.

cuBLAS is NVIDIA's GPU-accelerated library for basic linear algebra subroutines. It provides 
highly optimized implementations of GEMM (General Matrix Multiplication), GEMV (General 
Matrix-Vector Multiplication), and other operations. cuBLASLt extends cuBLAS with support 
for matrix multiplication with fused epilogue operations.

Tensor Cores are specialized hardware units introduced in the Volta architecture for 
accelerating matrix operations. They can perform 4x4 matrix multiply-accumulate operations 
in a single clock cycle, providing massive speedups for deep learning workloads. Effective 
use of Tensor Cores requires specific data layouts and dimensions aligned to multiples of 8.

NVIDIA Nsight Systems is a system-wide performance analysis tool for optimizing software 
performance. It provides timeline views of CPU and GPU activity, API calls, and memory 
transfers. Nsight Compute provides detailed kernel-level profiling with hardware performance 
counters, helping identify bottlenecks like memory bandwidth saturation or low occupancy.

The CUDA programming model requires careful attention to synchronization. Atomic operations 
like atomicAdd and atomicCAS provide thread-safe updates to shared variables. CUDA streams 
enable concurrent execution of independent operations, while events allow fine-grained 
synchronization between streams.

Flash Attention is a memory-efficient attention algorithm that computes exact attention 
without storing the full attention matrix. It tiles the computation to fit in SRAM, 
reducing HBM memory accesses from O(N^2) to O(N). Flash Attention 2 further improved 
parallelism and work partitioning, achieving near-peak hardware utilization on modern GPUs.

Mixed precision training uses FP16 or BF16 for forward and backward passes while maintaining 
FP32 master weights for numerical stability. NVIDIA's automatic mixed precision (AMP) library 
handles loss scaling to prevent gradient underflow with FP16. BF16 has the same dynamic range 
as FP32 and generally does not require loss scaling.
""" * 4,

    "rlhf": """
Reinforcement Learning from Human Feedback (RLHF) is a technique for aligning language models 
with human preferences. It was popularized by InstructGPT in 2022 and has become a standard 
component of training state-of-the-art language models like GPT-4, Claude, and Gemini.

The RLHF pipeline consists of three main stages. First, supervised fine-tuning (SFT) trains 
the base language model on high-quality demonstration data to produce a initial policy. Second, 
reward model training uses human preference data to train a model that scores responses. Third, 
reinforcement learning uses the reward model to further fine-tune the policy via PPO.

Collecting human preference data involves showing annotators pairs of model responses and asking 
them to indicate which they prefer. These comparisons are used to train the reward model using 
the Bradley-Terry model: P(A preferred over B) = sigmoid(r(A) - r(B)), where r is the reward.

Proximal Policy Optimization (PPO) is the RL algorithm most commonly used in RLHF. It updates 
the policy to maximize expected reward while staying close to the reference policy via a KL 
divergence penalty: reward = r_theta(x,y) - beta * KL(pi_theta || pi_ref). This prevents the 
model from exploiting the reward model with out-of-distribution outputs.

Constitutional AI (CAI), introduced by Anthropic, extends RLHF with a set of principles (the 
constitution) that guide the model's behavior. In the RL phase, an AI assistant evaluates and 
revises its own responses according to constitutional principles, generating preference data 
without human labeling. This is called Reinforcement Learning from AI Feedback (RLAIF).

Direct Preference Optimization (DPO) is an alternative to PPO that directly optimizes the 
policy using preference data without explicit reward model training. DPO derives a closed-form 
expression for the optimal policy given the preference data and optimizes it directly via 
supervised learning: L_DPO = -E[log sigma(beta * log(pi_theta(y_w|x)/pi_ref(y_w|x)) - 
beta * log(pi_theta(y_l|x)/pi_ref(y_l|x)))]

Reward hacking occurs when the policy finds ways to achieve high reward that do not correspond 
to actually helpful or harmless behavior. This is a fundamental challenge in RLHF because the 
reward model is an imperfect proxy for true human preferences. Techniques to mitigate reward 
hacking include KL constraints, ensemble reward models, and careful data collection.

Scalable oversight is a research direction aimed at developing techniques for supervising AI 
systems that may exceed human capabilities in certain domains. This includes approaches like 
debate (where two AI agents argue opposing positions for a human judge), amplification (using 
AI assistance to help humans evaluate complex outputs), and process-based supervision 
(rewarding correct reasoning steps rather than just final answers).

Red-teaming involves systematically testing AI systems for harmful or unintended behaviors. 
Human red-teamers attempt to elicit problematic outputs through adversarial prompting, while 
automated red-teaming uses other AI models to generate diverse adversarial inputs at scale.

Instruction following capability is a key outcome of RLHF. Models trained with RLHF are much 
better at following complex, multi-step instructions compared to base language models. This 
is because the SFT and RL stages expose the model to diverse instruction types and reinforce 
helpful, accurate responses.

The reward model architecture is typically initialized from the same base model as the policy.
A linear head is added on top to output a scalar reward value. The reward model is trained 
on pairs of (prompt, chosen response, rejected response) using the preference loss function.
""" * 4,
}

TEST_CASES = [
    {
        "text_key": "transformer",
        "question": "What year was the transformer architecture introduced and in which paper?",
        "answer": "2017",
        "key_fact": "2017"
    },
    {
        "text_key": "vllm",
        "question": "What is the default block size used in PagedAttention?",
        "answer": "16 tokens",
        "key_fact": "16"
    },
    {
        "text_key": "cuda",
        "question": "How many threads does a warp consist of in CUDA?",
        "answer": "32 threads",
        "key_fact": "32"
    },
    {
        "text_key": "rlhf",
        "question": "What are the three main stages of the RLHF pipeline?",
        "answer": "SFT, reward model training, RL",
        "key_fact": "supervised fine-tuning"
    },
]

# SnapKV 核心
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
    compressed = 0
    for i, layer in enumerate(past_kv.layers):
        q = model.model.layers[i].self_attn._last_query
        k_new, v_new, did_compress = snapkv_compress_gqa(
            q, layer.keys, layer.values,
            window_size, max_capacity_prompt
        )
        if did_compress:
            layer.keys = k_new
            layer.values = v_new
            compressed += 1
    return compressed

def run_inference(model, tokenizer, context, question,
                  use_snapkv=False, window_size=64, max_capacity_prompt=512):
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192).to("cuda")
    input_len = inputs.input_ids.shape[-1]

    torch.cuda.reset_peak_memory_stats()
    start = time.time()

    with torch.no_grad():
        out = model(**inputs, use_cache=True)
        past_kv = out.past_key_values

        compressed_len = input_len
        if use_snapkv:
            n = compress_cache(model, past_kv, window_size, max_capacity_prompt)
            if n > 0:
                compressed_len = past_kv.layers[0].keys.shape[2]

        outputs = model.generate(
            **inputs,
            past_key_values=past_kv,
            max_new_tokens=80,
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
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.float16, device_map="cuda", use_cache=True,
    )
    model.eval()
    orig = patch_llama_attention()
    print("Model loaded.\n")

    configs = [
        {"use_snapkv": False, "label": "Baseline"},
        {"use_snapkv": True, "window_size": 64, "max_capacity_prompt": 512,  "label": "SnapKV-512"},
        {"use_snapkv": True, "window_size": 64, "max_capacity_prompt": 1024, "label": "SnapKV-1024"},
        {"use_snapkv": True, "window_size": 64, "max_capacity_prompt": 2048, "label": "SnapKV-2048"},
    ]

    all_results = {cfg["label"]: [] for cfg in configs}

    for test in TEST_CASES:
        context = LONG_TEXTS[test["text_key"]]
        print(f"\n{'='*60}")
        print(f"Question: {test['question']}")
        print(f"Expected: {test['answer']}")
        print(f"{'='*60}")

        for cfg in configs:
            result = run_inference(
                model, tokenizer, context, test["question"],
                use_snapkv=cfg.get("use_snapkv", False),
                window_size=cfg.get("window_size", 64),
                max_capacity_prompt=cfg.get("max_capacity_prompt", 512),
            )
            correct = test["key_fact"].lower() in result["answer"].lower()
            ratio = result["compressed_len"] / result["input_len"] * 100

            print(f"\n[{cfg['label']}]")
            print(f"  tokens: {result['input_len']} → {result['compressed_len']} ({ratio:.0f}% retained)")
            print(f"  answer: {result['answer'][:80]}")
            print(f"  correct: {'✅' if correct else '❌'}  time: {result['time']:.2f}s  mem: {result['peak_mem_gb']:.2f}GB")

            all_results[cfg["label"]].append({
                "question": test["question"],
                "key_fact": test["key_fact"],
                "correct": correct,
                "input_len": result["input_len"],
                "compressed_len": result["compressed_len"],
                "retention_pct": round(ratio, 1),
                "time": result["time"],
                "peak_mem_gb": result["peak_mem_gb"],
                "answer": result["answer"][:150],
            })

    restore_attention(orig)

    with open("results_experiment1_v2.json", "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # 打印汇总表
    print(f"\n{'='*60}")
    print("SUMMARY TABLE")
    print(f"{'='*60}")
    print(f"{'Config':<15} {'正确率':<8} {'平均时间':<10} {'平均显存':<10} {'平均保留率'}")
    print("-"*60)
    for cfg in configs:
        label = cfg["label"]
        cases = all_results[label]
        acc = sum(c["correct"] for c in cases) / len(cases) * 100
        avg_time = sum(c["time"] for c in cases) / len(cases)
        avg_mem = sum(c["peak_mem_gb"] for c in cases) / len(cases)
        avg_ret = sum(c["retention_pct"] for c in cases) / len(cases)
        print(f"{label:<15} {acc:.0f}%{'':<5} {avg_time:.2f}s{'':<5} {avg_mem:.2f}GB{'':<4} {avg_ret:.0f}%")

if __name__ == "__main__":
    main()
