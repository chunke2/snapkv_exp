"""
Experiment 5 -- Position-encoded needle test (v2).
Long document (~5000 tokens) with simple numeric needles at 5 positions.
Measures retrieval range of each strategy.
"""
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from snapkv_lib import (
    patch_llama_attention, restore_attention,
    run_inference, check_correct_keyword, collect_metadata,
)

MODEL_PATH = "/workspace/models/models--deepseek-ai--DeepSeek-R1-Distill-Llama-8B/snapshots/6a6f4aa4197940add57724a7707d069478df56b1"

# ── 4 tech articles, each repeated to build length ──
ARTICLE_BASE = [
    # Article 1: Transformer
    """The transformer architecture was introduced in the paper Attention is All You Need
by Vaswani et al. in 2017. It relies entirely on attention mechanisms, dispensing with
recurrence and convolutions entirely. The model consists of an encoder and decoder, each
composed of multiple layers. Each layer has two sub-layers: a multi-head self-attention
mechanism and a position-wise fully connected feed-forward network. The encoder maps an
input sequence of symbol representations to a sequence of continuous representations.
The attention function maps a query and a set of key-value pairs to an output, computed
as a weighted sum of the values. Multi-head attention allows the model to jointly attend
to information from different representation subspaces at different positions. Positional
encoding is added to the input embeddings to give the model information about the relative
or absolute position of the tokens in the sequence, using sine and cosine functions of
different frequencies. The transformer was first applied to machine translation, achieving
state-of-the-art results on English-to-German and English-to-French translation using the
WMT 2014 dataset. The base model had 6 encoder and 6 decoder layers with 65M parameters.
The large model used 16 heads and 213M parameters trained for 100000 steps on 8 P100 GPUs.
Regularization included residual dropout and label smoothing of epsilon 0.1. BERT later
used only the encoder for masked language modeling, while GPT used only the decoder for
causal language modeling. Vision Transformer showed transformers work for computer vision
by applying self-attention to sequences of image patches instead of convolution.""",

    # Article 2: vLLM / PagedAttention
    """vLLM is a fast inference engine for large language models developed at UC Berkeley.
The key innovation is PagedAttention, which manages the KV cache with non-contiguous memory
similar to virtual memory paging in operating systems. Traditional LLM serving systems
pre-allocate contiguous GPU memory for each request's KV cache, leading to internal and
external fragmentation. PagedAttention divides the KV cache into fixed-size blocks of 16
tokens each, stored in non-contiguous physical memory with a block table maintaining the
mapping between logical and physical positions. When a new request arrives, vLLM allocates
blocks on-demand as tokens are generated, rather than pre-allocating the full sequence length.
Continuous batching allows new requests to join a running batch dynamically, enabling 2-4x
throughput improvements over static batching. Memory sharing allows multiple sequences with
common prefixes to share physical KV cache blocks with copy-on-write semantics. vLLM supports
tensor parallelism for distributing model weights across GPUs, quantization methods like AWQ
and GPTQ, speculative decoding, and prefix caching. On an A100 GPU with LLaMA-13B, vLLM can
serve over 2000 tokens per second with concurrent users compared to around 200 for naive
implementations. GPU memory utilization defaults to 90 percent. Chunked prefill breaks long
prompts into smaller chunks to prevent a single long request from blocking the batch.""",

    # Article 3: CUDA / GPU architecture
    """CUDA is NVIDIA parallel computing platform and programming model, first introduced in
2006 with the GeForce 8800 GPU. The GPU architecture contains thousands of efficient cores
designed for handling multiple tasks simultaneously. Modern NVIDIA GPUs contain Streaming
Multiprocessors, each with multiple CUDA cores, Tensor Cores for matrix operations, and
shared memory for fast intra-SM communication. CUDA programs organize threads into grids of
thread blocks, each containing up to 1024 threads executing concurrently on a single SM.
Threads within a block communicate via shared memory and synchronize using syncthreads.
Warps are the fundamental unit of execution, consisting of 32 threads executing in lockstep
via SIMT. Branch divergence causes the GPU to serialize divergent paths, reducing efficiency.
The memory hierarchy includes registers, shared memory, L1 and L2 cache, and global memory.
Coalesced memory access is critical: when threads in a warp access consecutive addresses, the
GPU combines them into a single transaction. Occupancy refers to the ratio of active warps to
maximum warps supported by an SM. Higher occupancy improves latency hiding as the GPU switches
between warps during memory operations. Tensor Cores accelerate matrix multiply-accumulate
operations introduced with the Volta architecture. Flash Attention computes exact attention
without storing the full matrix, tiling computation to fit in SRAM and reducing HBM accesses.""",

    # Article 4: RLHF / alignment
    """Reinforcement Learning from Human Feedback is a technique for aligning language models
with human preferences, popularized by InstructGPT in 2022. The RLHF pipeline has three
stages: supervised fine-tuning on demonstration data, reward model training on human
preference comparisons, and reinforcement learning via PPO to optimize the policy against
the reward model while staying close to the reference policy through a KL divergence penalty.
Collecting human preference data involves showing annotators pairs of model responses and
asking which they prefer, training the reward model using the Bradley-Terry preference model.
Constitutional AI extends RLHF by having an AI assistant evaluate and revise its own responses
according to constitutional principles, generating preference data without human labeling.
Direct Preference Optimization is an alternative that directly optimizes the policy using
preference data without explicit reward model training, deriving a closed-form expression
for the optimal policy. Reward hacking occurs when the policy exploits the imperfect reward
model to achieve high scores without actually being helpful or harmless. Mitigation techniques
include KL constraints, ensemble reward models, and careful data collection. Scalable oversight
research develops techniques for supervising AI systems that may exceed human capabilities,
including debate between AI agents and amplification using AI assistance. Red-teaming involves
systematically testing AI systems for harmful behaviors through adversarial prompting.
Instruction following capability is a key outcome of RLHF, with models becoming much better at
following complex multi-step instructions compared to base language models.""",
]

# ── Simple numeric needles ──
NEEDLES = [
    {
        "id": "P0",
        "insert_at_article": 0,
        "needle": "SPECIAL NOTE: The server port number is 7742.",
        "question": "What is the server port number?",
        "answer": "7742|77",
    },
    {
        "id": "P25",
        "insert_after_article": 0,  # after article 0, before article 1
        "needle": "SPECIAL NOTE: The prototype achieved a speedup factor of 94.",
        "question": "What speedup factor did the prototype achieve?",
        "answer": "94",
    },
    {
        "id": "P50",
        "insert_after_article": 1,  # after article 1, before article 2
        "needle": "SPECIAL NOTE: The optimal batch size was found to be 47.",
        "question": "What is the optimal batch size?",
        "answer": "47",
    },
    {
        "id": "P75",
        "insert_after_article": 2,  # after article 2, before article 3
        "needle": "SPECIAL NOTE: The model checkpoint version is 305.",
        "question": "What is the model checkpoint version?",
        "answer": "305",
    },
    {
        "id": "P100",
        "insert_after_article": 99,
        "needle": "SPECIAL NOTE: The server timeout value is 73 seconds.",
        "question": "What is the server timeout value in seconds?",
        "answer": "73",
    },
]


def build_document():
    """Build long document by repeating articles, with needles at known positions."""
    REPEAT = 6
    articles = [a * REPEAT for a in ARTICLE_BASE]

    parts = []
    # P0: at very start
    parts.append(NEEDLES[0]["needle"] + "\n\n")

    for i, article in enumerate(articles):
        parts.append(article)

        # Insert needles after specific articles
        for n in NEEDLES[1:]:
            if n.get("insert_after_article") == i:
                parts.append("\n\n" + n["needle"] + "\n")

    # P100: at very end (last needle)
    parts.append("\n\n" + NEEDLES[-1]["needle"] + "\n")

    return "\n\n".join(parts)


DOCUMENT = build_document()

STRATEGY_CONFIGS = [
    # (type, label, comp_strat, ws, cap, scoring_window, layer_stride)
    (None,         "Baseline",             None,         64, None, None, 1),
    ("snapkv",     "SnapKV-256",           "snapkv",     64, 256,  None, 1),
    ("snapkv",     "SnapKV-256-stride4",   "snapkv",     64, 256,  None, 4),
    ("snapkv",     "SnapKV-256-stride8",   "snapkv",     64, 256,  None, 8),
    ("snapkv",     "SnapKV-256-fast4",     "snapkv",     64, 256,  4,    1),
    ("snapkv",     "SnapKV-512",           "snapkv",     64, 512,  None, 1),
    ("keepfirst",  "KeepFirst-256",        "keepfirst",  64, 256,  None, 1),
    ("keepfirst",  "KeepFirst-512",        "keepfirst",  64, 512,  None, 1),
    ("random",     "Random-256",           "random",     64, 256,  None, 1),
]


def main():
    meta = collect_metadata(MODEL_PATH)
    print(f"git: {meta['git_commit'][:7]}")

    temp_tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    doc_tokens = len(temp_tok.encode(DOCUMENT))
    print(f"Document: {doc_tokens} tokens")
    for n in NEEDLES:
        idx = DOCUMENT.find(n["needle"])
        if idx >= 0:
            prefix_tokens = len(temp_tok.encode(DOCUMENT[:idx]))
            pct = prefix_tokens / doc_tokens * 100
            # Verify: show surrounding context
            ctx = DOCUMENT[max(0,idx-30):idx+len(n["needle"])+30]
            print(f"  Needle {n['id']} at token ~{prefix_tokens} (~{pct:.0f}%)  "
                  f"answer={n['answer'].split('|')[0]}  ctx='...{ctx[:60]}...'")
        else:
            print(f"  Needle {n['id']}: NOT FOUND in document!")
    del temp_tok
    torch.cuda.empty_cache()

    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, device_map="cuda", use_cache=True,
    )
    model.eval()
    orig_forward = patch_llama_attention()
    print("Model loaded.\n")

    all_results = {}

    for str_type, label, comp_strat, ws, cap, sw, ls in STRATEGY_CONFIGS:
        extra = ""
        if sw:
            extra += f"  scoring={sw}"
        if ls > 1:
            extra += f"  stride={ls}"
        print(f"{'='*60}")
        print(f"{label}  budget={cap or 'full'}{extra}")
        print(f"{'='*60}")

        cases = []
        for test in NEEDLES:
            prompt = f"Document:\n{DOCUMENT}\n\nQuestion: {test['question']}\nAnswer:"
            result = run_inference(
                model, tokenizer, prompt,
                compress_strategy=comp_strat,
                window_size=ws,
                max_capacity_prompt=cap or 99999,
                max_new_tokens=50,
                scoring_window=sw,
                layer_stride=ls,
            )
            correct = check_correct_keyword(result["answer"], test["answer"])

            print(f"  [{test['id']:<4}] {'OK' if correct else 'XX'}  "
                  f"exp={test['answer']}  "
                  f"got='{result['answer'][:70]}'  "
                  f"ret={result['retention_pct']:.0f}%")

            cases.append({
                "position": test["id"],
                "expected": test["answer"],
                "got": result["answer"][:120],
                "correct": correct,
                **{k: result[k] for k in [
                    "input_len", "compressed_len", "retention_pct",
                    "prefill_time", "compress_time", "decode_time", "total_time",
                    "peak_mem_gb",
                ]},
            })
        all_results[label] = cases

    restore_attention(orig_forward)

    n = len(NEEDLES)
    pos_ids = [n2["id"] for n2 in NEEDLES]

    # ── Position heatmap ──
    print(f"\n{'='*65}")
    print("POSITION HEATMAP")
    print(f"{'='*65}")
    hdr = f"{'Config':<16}"
    for p in pos_ids:
        hdr += f" {p:<6}"
    print(hdr)
    print("-" * 65)
    for _, label, _, _, _, _, _ in STRATEGY_CONFIGS:
        cases = all_results[label]
        row = f"{label:<16}"
        for c in cases:
            row += f" {'OK' if c['correct'] else 'XX':<6}"
        print(row)

    # ── Summary ──
    print(f"\n{'='*65}")
    print("SUMMARY")
    print(f"{'='*65}")
    print(f"{'Config':<16} {'Acc':<6} {'Dec(s)':<8} {'Total(s)':<8} {'Ret%':<6}")
    print("-" * 55)
    for _, label, _, _, _, _, _ in STRATEGY_CONFIGS:
        cases = all_results[label]
        acc = sum(c["correct"] for c in cases) / n * 100
        dec = sum(c["decode_time"] for c in cases) / n
        tot = sum(c["total_time"] for c in cases) / n
        ret = sum(c["retention_pct"] for c in cases) / n
        print(f"{label:<16} {acc:.0f}%{'':<2} {dec:.2f}{'':<4} {tot:.2f}{'':<4} {ret:.0f}%")

    with open("results_experiment5_position.json", "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print("\nSaved.")


if __name__ == "__main__":
    main()
