import torch
import time
import math
import json
import string
import re
from collections import Counter
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.llama.modeling_llama import LlamaAttention

MODEL_PATH = "/workspace/models/models--deepseek-ai--DeepSeek-R1-Distill-Llama-8B/snapshots/6a6f4aa4197940add57724a7707d069478df56b1"
DATA_PATH = "/workspace/longbench_data/data/qasper.jsonl"

# F1 评分
def normalize_answer(s):
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ''.join(ch for ch in s if ch not in string.punctuation)
    s = ' '.join(s.split())
    return s

def normalize_nospace(s):
    """去掉空格版本，用于处理 tokenizer 不保留空格的情况"""
    s = s.lower()
    s = ''.join(ch for ch in s if ch not in string.punctuation and ch != ' ')
    return s

def f1_score(prediction, ground_truth):
    # 方式1：正常带空格
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens   = normalize_answer(ground_truth).split()
    if pred_tokens and gt_tokens:
        common = Counter(pred_tokens) & Counter(gt_tokens)
        num_same = sum(common.values())
        if num_same > 0:
            precision = num_same / len(pred_tokens)
            recall    = num_same / len(gt_tokens)
            f1_normal = 2 * precision * recall / (precision + recall)
        else:
            f1_normal = 0.0
    else:
        f1_normal = 0.0

    # 方式2：去掉空格，字符级匹配
    pred_ns = normalize_nospace(prediction)
    gt_ns   = normalize_nospace(ground_truth)
    if pred_ns and gt_ns:
        # 检查 ground truth 是否包含在 prediction 里（或反之）
        if gt_ns in pred_ns or pred_ns in gt_ns:
            overlap = min(len(pred_ns), len(gt_ns))
            precision = overlap / len(pred_ns)
            recall    = overlap / len(gt_ns)
            f1_nospace = 2 * precision * recall / (precision + recall)
        else:
            f1_nospace = 0.0
    else:
        f1_nospace = 0.0

    return max(f1_normal, f1_nospace)

def best_f1(prediction, ground_truths):
    return max(f1_score(prediction, gt) for gt in ground_truths)

# SnapKV 压缩
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

def make_snapkv_forward(window_size, max_capacity_prompt):
    """
    方案三：在 attention hook 里检测 prefill 结束，自动压缩
    prefill: seq_len > 1，第一次调用后压缩
    decode:  seq_len == 1，直接跑
    """
    original_forward = LlamaAttention.forward

    # 用一个共享状态记录是否已经压缩过
    state = {"prefill_done": False, "compressed_len": None}

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

        # 存入 cache
        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx
            )

        seq_len = query_states.shape[2]

        # prefill 结束后（最后一层）压缩
        is_last_layer = (self.layer_idx == 31)
        if seq_len > 1 and not state["prefill_done"] and is_last_layer:
            # 压缩所有层的 cache
            for layer_cache in past_key_values.layers:
                k, v = layer_cache.keys, layer_cache.values
                # 用当前层的 query 近似（实际每层都不同，但这里简化）
                k_new, v_new, did = snapkv_compress_gqa(
                    query_states, k, v, window_size, max_capacity_prompt
                )
                if did:
                    layer_cache.keys = k_new
                    layer_cache.values = v_new

            state["prefill_done"] = True
            state["compressed_len"] = past_key_values.layers[0].keys.shape[2]

            # 更新当前层的 key_states/value_states 为压缩后的
            key_states   = past_key_values.layers[self.layer_idx].keys
            value_states = past_key_values.layers[self.layer_idx].values

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
    return original_forward, state

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

def run_inference(model, tokenizer, context, question,
                  use_snapkv=False, window_size=64, max_capacity_prompt=512):
    # LongBench 官方 qasper prompt 模板
    prompt = (f"Answer the question based on the given passages. "
              f"Only give me the answer and do not output any other words.\n\n"
              f"Passages: {context}\nQuestion: {question}\nAnswer:")
    
    # 官方截断方式：从中间截断，保留开头和结尾
    tokenized = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
    max_length = 8000
    if len(tokenized) > max_length:
        half = max_length // 2
        input_ids = torch.cat([tokenized[:half], tokenized[-half:]]).unsqueeze(0).to("cuda")
        attention_mask = torch.ones_like(input_ids)
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
    else:
        inputs = {"input_ids": tokenized.unsqueeze(0).to("cuda"), "attention_mask": torch.ones(1, len(tokenized)).to("cuda")}
    input_len = inputs["input_ids"].shape[-1] if isinstance(inputs, dict) else inputs.input_ids.shape[-1]

    torch.cuda.reset_peak_memory_stats()
    start = time.time()

    with torch.no_grad():
        if use_snapkv:
            # prefill
            out = model(**inputs, use_cache=True)
            past_kv = out.past_key_values
            # 压缩
            compress_cache(model, past_kv, window_size, max_capacity_prompt)
            compressed_len = past_kv.layers[0].keys.shape[2]
            torch.cuda.empty_cache()
            # decode：只传最后一个 token + 完整 attention_mask
            decode_out = model.generate(
                input_ids=inputs["input_ids"][:, -1:],
                attention_mask=inputs["attention_mask"],
                past_key_values=past_kv,
                max_new_tokens=64,
                do_sample=False,
                temperature=1.0,
            )
            # 拼回完整输出
            outputs = torch.cat([inputs["input_ids"], decode_out], dim=1)
        else:
            outputs = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                temperature=1.0,
            )
            compressed_len = input_len

    elapsed = time.time() - start
    raw = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=False)
    # 把 BPE 空格符还原成真实空格
    generated = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    # 清理 BPE 特殊字符
    import unicodedata
    generated = "".join(
        " " if unicodedata.name(c, "").startswith("LATIN SMALL LETTER DOTLESS") or ord(c) == 0x0120
        else "\n" if ord(c) == 0x010a
        else c
        for c in generated
    ).strip()
    generated = generated.split("\n")[0].strip()

    return {
        "answer": generated,
        "time": elapsed,
        "input_len": input_len,
        "compressed_len": compressed_len,
    }

def main():
    with open(DATA_PATH) as f:
        data = [json.loads(l) for l in f][:50]
    print(f"Loaded {len(data)} samples from qasper")

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
        {"use_snapkv": True, "window_size": 64, "max_capacity_prompt": 256,  "label": "SnapKV-256"},
        {"use_snapkv": True, "window_size": 64, "max_capacity_prompt": 512,  "label": "SnapKV-512"},
        {"use_snapkv": True, "window_size": 64, "max_capacity_prompt": 1024, "label": "SnapKV-1024"},
    ]

    all_results = {
        cfg["label"]: {"f1_scores": [], "times": [], "retention_pcts": []}
        for cfg in configs
    }

    for idx, sample in enumerate(data):
        context  = sample["context"]
        question = sample["input"]
        answers  = sample["answers"]
        print(f"\n[{idx+1}/50] {question[:60]}...")

        for cfg in configs:
            result = run_inference(
                model, tokenizer, context, question,
                use_snapkv=cfg.get("use_snapkv", False),
                window_size=cfg.get("window_size", 64),
                max_capacity_prompt=cfg.get("max_capacity_prompt", 512),
            )
            f1 = best_f1(result["answer"], answers)
            retention = result["compressed_len"] / result["input_len"] * 100

            all_results[cfg["label"]]["f1_scores"].append(f1)
            all_results[cfg["label"]]["times"].append(result["time"])
            all_results[cfg["label"]]["retention_pcts"].append(retention)

            print(f"  [{cfg['label']}] F1={f1:.2f}  "
                  f"retention={retention:.0f}%  time={result['time']:.2f}s  "
                  f"got={result['answer'][:50]}")

    print(f"\n{'='*60}")
    print("FINAL RESULTS (50 samples, LongBench qasper)")
    print(f"{'='*60}")
    print(f"{'Config':<15} {'Avg F1':<10} {'Avg Time':<10} {'Avg Retention'}")
    print("-"*55)
    for cfg in configs:
        r = all_results[cfg["label"]]
        avg_f1   = sum(r["f1_scores"]) / len(r["f1_scores"]) * 100
        avg_time = sum(r["times"]) / len(r["times"])
        avg_ret  = sum(r["retention_pcts"]) / len(r["retention_pcts"])
        print(f"{cfg['label']:<15} {avg_f1:.1f}%{'':<5} {avg_time:.2f}s{'':<5} {avg_ret:.0f}%")

    restore_attention(orig)

    with open("results_longbench_qasper.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved to results_longbench_qasper.json")

if __name__ == "__main__":
    main()
