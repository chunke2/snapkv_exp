"""
SnapKV shared library — single source of truth for all experiments.

Provides:
- snapkv_compress_gqa (auto-detect GQA heads from model config)
- random_compress / keepfirst_compress (control baselines)
- patch_llama_attention / restore_attention
- compress_cache (per-layer, using _last_query)
- run_inference (unified, with timing breakdown)
- check_correct_keyword
"""
import torch
import time
import math
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import LlamaAttention


# ---------------------------------------------------------------------------
# Compression strategies
# ---------------------------------------------------------------------------

def _get_head_counts(model):
    """Auto-detect num_q_heads and num_kv_heads from model config."""
    config = model.config
    num_q_heads = config.num_attention_heads
    num_kv_heads = getattr(config, "num_key_value_heads", num_q_heads)
    return num_q_heads, num_kv_heads


def snapkv_compress_gqa(query_states, key_states, value_states,
                         window_size, max_capacity_prompt, kernel_size=5,
                         num_q_heads=None, num_kv_heads=None):
    """
    GQA-aware SnapKV compression.
    Returns (compressed_key, compressed_value, did_compress).
    """
    bsz, num_kv, q_len, head_dim = key_states.shape
    if q_len <= max_capacity_prompt:
        return key_states, value_states, False

    if num_q_heads is None or num_kv_heads is None:
        raise ValueError("num_q_heads and num_kv_heads must be provided")

    num_groups = num_q_heads // num_kv_heads
    q_grouped = query_states.view(bsz, num_kv_heads, num_groups, q_len, head_dim)
    q_mean = q_grouped.mean(dim=2)

    attn_weights = torch.matmul(
        q_mean[:, :, -window_size:, :],
        key_states.transpose(2, 3)
    ) / math.sqrt(head_dim)
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

    prefix_weights = attn_weights[:, :, :, :-window_size].sum(dim=2)
    pooled = F.avg_pool1d(prefix_weights, kernel_size=kernel_size,
                          padding=kernel_size // 2, stride=1)

    k = max_capacity_prompt - window_size
    indices = pooled.topk(k, dim=-1).indices
    indices_expanded = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)

    k_compressed = key_states[:, :, :-window_size, :].gather(dim=2, index=indices_expanded)
    v_compressed = value_states[:, :, :-window_size, :].gather(dim=2, index=indices_expanded)
    key_out = torch.cat([k_compressed, key_states[:, :, -window_size:, :]], dim=2)
    value_out = torch.cat([v_compressed, value_states[:, :, -window_size:, :]], dim=2)

    return key_out, value_out, True


def random_compress(key_states, value_states, window_size, max_capacity_prompt):
    """Control baseline: randomly select tokens from prefix."""
    bsz, num_kv, q_len, head_dim = key_states.shape
    if q_len <= max_capacity_prompt:
        return key_states, value_states, False

    k = max_capacity_prompt - window_size
    prefix_len = q_len - window_size
    indices = torch.randperm(prefix_len, device=key_states.device)[:k]
    indices = indices.sort().values  # preserve order
    indices = indices.unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand(bsz, num_kv, -1, head_dim)

    k_compressed = key_states[:, :, :-window_size, :].gather(dim=2, index=indices)
    v_compressed = value_states[:, :, :-window_size, :].gather(dim=2, index=indices)
    key_out = torch.cat([k_compressed, key_states[:, :, -window_size:, :]], dim=2)
    value_out = torch.cat([v_compressed, value_states[:, :, -window_size:, :]], dim=2)
    return key_out, value_out, True


def keepfirst_compress(key_states, value_states, window_size, max_capacity_prompt):
    """Control baseline: keep the first K tokens from prefix."""
    bsz, num_kv, q_len, head_dim = key_states.shape
    if q_len <= max_capacity_prompt:
        return key_states, value_states, False

    k = max_capacity_prompt - window_size
    key_out = torch.cat([key_states[:, :, :k, :], key_states[:, :, -window_size:, :]], dim=2)
    value_out = torch.cat([value_states[:, :, :k, :], value_states[:, :, -window_size:, :]], dim=2)
    return key_out, value_out, True


COMPRESS_STRATEGIES = {
    "snapkv": snapkv_compress_gqa,
    "random": random_compress,
    "keepfirst": keepfirst_compress,
}


# ---------------------------------------------------------------------------
# LlamaAttention monkey-patch
# ---------------------------------------------------------------------------

def patch_llama_attention():
    """
    Monkey-patch LlamaAttention.forward to capture _last_query on each layer.
    Returns the original forward for later restoration.
    """
    original_forward = LlamaAttention.forward

    def new_forward(self, hidden_states, position_embeddings=None,
                    attention_mask=None, past_key_values=None, **kwargs):
        from transformers.models.llama.modeling_llama import (
            apply_rotary_pos_emb, ALL_ATTENTION_FUNCTIONS, eager_attention_forward,
        )
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        self._last_query = query_states.detach()

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx
            )

        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward,
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


def restore_attention(original_forward):
    LlamaAttention.forward = original_forward


# ---------------------------------------------------------------------------
# Cache compression (per-layer)
# ---------------------------------------------------------------------------

def compress_cache(model, past_key_values, window_size, max_capacity_prompt,
                   num_q_heads=None, num_kv_heads=None, strategy="snapkv"):
    """
    Compress KV cache in-place using the given strategy.
    Returns compression info dict.
    """
    compress_fn = COMPRESS_STRATEGIES[strategy]
    info = {"compressed_count": 0, "before": None, "after": None}

    for i, layer in enumerate(past_key_values.layers):
        before = layer.keys.shape[2]
        if i == 0:
            info["before"] = before

        if strategy == "snapkv":
            q = model.model.layers[i].self_attn._last_query
            k_new, v_new, did = compress_fn(
                q, layer.keys, layer.values,
                window_size, max_capacity_prompt,
                num_q_heads=num_q_heads, num_kv_heads=num_kv_heads,
            )
        else:
            k_new, v_new, did = compress_fn(
                layer.keys, layer.values,
                window_size, max_capacity_prompt,
            )

        if did:
            layer.keys = k_new
            layer.values = v_new
            info["compressed_count"] += 1
        if i == 0:
            info["after"] = layer.keys.shape[2]

    return info


# ---------------------------------------------------------------------------
# Unified inference
# ---------------------------------------------------------------------------

def run_inference(model, tokenizer, prompt,
                  compress_strategy=None,
                  window_size=64, max_capacity_prompt=512,
                  max_new_tokens=50):
    """
    Unified inference: prefill → optional compression → decode.

    compress_strategy: None (baseline), "snapkv", "random", "keepfirst"

    Returns dict with:
      answer, input_len, compressed_len, retention_pct,
      prefill_time, compress_time, decode_time, total_time,
      peak_mem_gb
    """
    num_q_heads, num_kv_heads = _get_head_counts(model)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192).to("cuda")
    input_len = inputs.input_ids.shape[-1]

    torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        # ── Prefill ──
        t0 = time.time()
        out = model(**inputs, use_cache=True)
        past_kv = out.past_key_values
        prefill_time = time.time() - t0

        # ── Compress ──
        compress_time = 0.0
        compressed_len = input_len
        compress_info = None

        do_compress = (
            compress_strategy is not None
            and compress_strategy != "baseline"
            and input_len > max_capacity_prompt
        )
        if do_compress:
            t1 = time.time()
            compress_info = compress_cache(
                model, past_kv, window_size, max_capacity_prompt,
                num_q_heads=num_q_heads, num_kv_heads=num_kv_heads,
                strategy=compress_strategy,
            )
            compress_time = time.time() - t1
            compressed_len = past_kv.layers[0].keys.shape[2]
            torch.cuda.empty_cache()

        # ── Decode ──
        t2 = time.time()
        # Match old working pattern: pass full inputs dict to generate.
        # HuggingFace generate() trims input_ids to [:, -1:] when past_key_values is set.
        decode_out = model.generate(
            **inputs,
            past_key_values=past_kv,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
        decode_time = time.time() - t2

    total_time = time.time() - t0
    generated = tokenizer.decode(decode_out[0][input_len:], skip_special_tokens=True)

    return {
        "answer": generated.strip(),
        "input_len": input_len,
        "compressed_len": compressed_len,
        "retention_pct": round(compressed_len / input_len * 100, 1) if input_len > 0 else 100.0,
        "prefill_time": prefill_time,
        "compress_time": compress_time,
        "decode_time": decode_time,
        "total_time": total_time,
        "peak_mem_gb": torch.cuda.max_memory_allocated() / 1024**3,
        "compress_info": compress_info,
    }


# ---------------------------------------------------------------------------
# Correctness checking
# ---------------------------------------------------------------------------

def check_correct_keyword(answer, keywords, check_negation=True):
    """
    Check correctness with basic negation detection.
    keywords: '|' separated alternatives.
    """
    answer_lower = answer.lower()
    negation_patterns = ["not ", "not\n", "wasn't", "isn't", "was not", "is not",
                         "weren't", "were not", "aren't", "are not"]

    for kw in keywords.split("|"):
        kw_lower = kw.strip().lower()

        if check_negation and len(kw_lower) > 2:
            negated = False
            for neg in negation_patterns:
                idx = answer_lower.find(kw_lower)
                if idx < 0:
                    continue
                context_start = max(0, idx - 20)
                context = answer_lower[context_start:idx]
                if neg in context:
                    negated = True
                    break
            if negated:
                continue

        if kw_lower in answer_lower:
            return True
    return False


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def collect_metadata(model_path):
    import sys
    import subprocess
    from datetime import datetime, timezone

    metadata = {
        "python_version": sys.version,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_path": model_path,
    }

    try:
        import transformers
        metadata["transformers_version"] = transformers.__version__
    except Exception:
        pass

    if torch.cuda.is_available():
        metadata["gpu_name"] = torch.cuda.get_device_name(0)
        metadata["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1024**3
        metadata["cuda_version"] = torch.version.cuda

    try:
        rev = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        metadata["git_commit"] = rev
    except Exception:
        metadata["git_commit"] = "unknown"

    return metadata
