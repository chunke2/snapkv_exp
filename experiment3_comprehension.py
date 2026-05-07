import torch
import time
import math
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.llama.modeling_llama import LlamaAttention
import json

MODEL_PATH = "/workspace/models/models--deepseek-ai--DeepSeek-R1-Distill-Llama-8B/snapshots/6a6f4aa4197940add57724a7707d069478df56b1"

DOCUMENT = open("transformer_doc.txt").read()

TEST_CASES = [
    {
        "question": "What year is the WMT English-to-French translation task that the model was evaluated on?",
        "answer": "2014",
        "section": "Abstract"
    },
    {
        "question": "The Transformer avoids recurrence by using what mechanism to capture global dependencies?",
        "answer": "attention mechanisms|Attention mechanisms|attention",
        "section": "Introduction"
    },
    {
        "question": "In the Scaled Dot-Product Attention formula, what do we divide the dot products by?",
        "answer": "sqrt(dk)|sqrt(d",
        "section": "Attention"
    },
    {
        "question": "Positional encodings are added to the input embeddings at the what of the encoder and decoder stacks?",
        "answer": "bottoms|bottom|encoder and decoder",
        "section": "Positional Encoding"
    },
    {
        "question": "What is the label smoothing value eps_ls used during training?",
        "answer": "0.1",
        "section": "Training"
    },
    {
        "question": "How long did it take to train the big transformer model on 8 P100 GPUs?",
        "answer": "3.5 days|3.5days|3.5",
        "section": "Results"
    },
]

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
            torch.cuda.empty_cache()

        outputs = model.generate(
            **inputs,
            past_key_values=past_kv,
            max_new_tokens=30,
            do_sample=False,
            temperature=1.0,
        )

    elapsed = time.time() - start
    generated = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    # 清理 BPE 特殊字符
    generated = generated.replace("\u0120", " ").replace("\u010a", "\n")

    return {
        "answer": generated.strip(),
        "time": elapsed,
        "peak_mem_gb": torch.cuda.max_memory_allocated() / 1024**3,
        "input_len": input_len,
        "compressed_len": compressed_len,
    }

def check_correct(answer, expected):
    clean = answer.lower().replace(" ", "").replace("-", "")
    return any(e.strip().lower().replace(" ", "").replace("-", "") in clean 
               for e in expected.split("|"))

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
        {"use_snapkv": False,  "label": "Baseline"},
        {"use_snapkv": True, "window_size": 64, "max_capacity_prompt": 128,  "label": "SnapKV-128"},
        {"use_snapkv": True, "window_size": 64, "max_capacity_prompt": 256,  "label": "SnapKV-256"},
        {"use_snapkv": True, "window_size": 64, "max_capacity_prompt": 512,  "label": "SnapKV-512"},
        {"use_snapkv": True, "window_size": 64, "max_capacity_prompt": 1024, "label": "SnapKV-1024"},
    ]

    all_results = {cfg["label"]: [] for cfg in configs}

    for cfg in configs:
        print(f"\n{'='*60}")
        print(f"Config: {cfg['label']}")
        print(f"{'='*60}")
        correct_count = 0

        for test in TEST_CASES:
            result = run_inference(
                model, tokenizer, DOCUMENT, test["question"],
                use_snapkv=cfg.get("use_snapkv", False),
                window_size=cfg.get("window_size", 64),
                max_capacity_prompt=cfg.get("max_capacity_prompt", 512),
            )
            correct = check_correct(result["answer"], test["answer"])
            correct_count += correct
            ratio = result["compressed_len"] / result["input_len"] * 100

            print(f"  [{test['section']}] {'✅' if correct else '❌'} "
                  f"expected={test['answer']}  got={result['answer'][:50]}")

            all_results[cfg["label"]].append({
                "section": test["section"],
                "question": test["question"],
                "expected": test["answer"],
                "got": result["answer"][:100],
                "correct": correct,
                "input_len": result["input_len"],
                "compressed_len": result["compressed_len"],
                "retention_pct": round(ratio, 1),
                "time": result["time"],
                "peak_mem_gb": result["peak_mem_gb"],
            })

        acc = correct_count / len(TEST_CASES) * 100
        avg_time = sum(r["time"] for r in all_results[cfg["label"]]) / len(TEST_CASES)
        print(f"  → 正确率: {acc:.0f}%  平均时间: {avg_time:.2f}s")

    restore_attention(orig)

    with open("results_experiment3_comprehension.json", "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # 汇总表
    print(f"\n{'='*60}")
    print("SUMMARY TABLE")
    print(f"{'='*60}")
    print(f"{'Config':<15} {'正确率':<8} {'平均时间':<10} {'平均显存':<10} {'KV保留率'}")
    print("-"*60)
    for cfg in configs:
        cases = all_results[cfg["label"]]
        acc = sum(c["correct"] for c in cases) / len(cases) * 100
        avg_time = sum(c["time"] for c in cases) / len(cases)
        avg_mem = sum(c["peak_mem_gb"] for c in cases) / len(cases)
        avg_ret = sum(c["retention_pct"] for c in cases) / len(cases)
        print(f"{cfg['label']:<15} {acc:.0f}%{'':<5} {avg_time:.2f}s{'':<5} {avg_mem:.2f}GB{'':<4} {avg_ret:.0f}%")

if __name__ == "__main__":
    main()
