"""
Experiment 4 -- LongBench qasper evaluation.
Uses snapkv_lib.py shared inference (D3 fixed, D2 removed).
Requires LongBench qasper data at DATA_PATH.
"""
import torch
import json
import time
import string
import re
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM
from snapkv_lib import (
    patch_llama_attention, restore_attention,
    run_inference, collect_metadata,
)

MODEL_PATH = "/workspace/models/models--deepseek-ai--DeepSeek-R1-Distill-Llama-8B/snapshots/6a6f4aa4197940add57724a7707d069478df56b1"
DATA_PATH = "/workspace/longbench_data/data/qasper.jsonl"


# ── F1 scoring (unchanged from LongBench official) ──

def normalize_answer(s):
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ''.join(ch for ch in s if ch not in string.punctuation)
    s = ' '.join(s.split())
    return s


def normalize_nospace(s):
    s = s.lower()
    s = ''.join(ch for ch in s if ch not in string.punctuation and ch != ' ')
    return s


def f1_score(prediction, ground_truth):
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    if pred_tokens and gt_tokens:
        common = Counter(pred_tokens) & Counter(gt_tokens)
        num_same = sum(common.values())
        if num_same > 0:
            precision = num_same / len(pred_tokens)
            recall = num_same / len(gt_tokens)
            f1_normal = 2 * precision * recall / (precision + recall)
        else:
            f1_normal = 0.0
    else:
        f1_normal = 0.0

    pred_ns = normalize_nospace(prediction)
    gt_ns = normalize_nospace(ground_truth)
    if pred_ns and gt_ns:
        if gt_ns in pred_ns or pred_ns in gt_ns:
            overlap = min(len(pred_ns), len(gt_ns))
            precision = overlap / len(pred_ns)
            recall = overlap / len(gt_ns)
            f1_nospace = 2 * precision * recall / (precision + recall)
        else:
            f1_nospace = 0.0
    else:
        f1_nospace = 0.0

    return max(f1_normal, f1_nospace)


def best_f1(prediction, ground_truths):
    return max(f1_score(prediction, gt) for gt in ground_truths)


# ── Main ──

def main():
    meta = collect_metadata(MODEL_PATH)
    print(f"git: {meta['git_commit'][:7]}")

    if not __import__('os').path.exists(DATA_PATH):
        print(f"ERROR: Data not found at {DATA_PATH}")
        print("Download LongBench qasper data or set DATA_PATH.")
        return

    with open(DATA_PATH) as f:
        data = [json.loads(l) for l in f][:50]
    print(f"Loaded {len(data)} samples from qasper")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, device_map="cuda", use_cache=True,
    )
    model.eval()
    orig_forward = patch_llama_attention()
    print("Model loaded.\n")

    configs = [
        ("snapkv",   "Baseline",      None,   64, 99999, None, 1),
        ("snapkv",   "SnapKV-256",    "snapkv", 64, 256,  None, 4),
        ("snapkv",   "SnapKV-512",    "snapkv", 64, 512,  None, 4),
        ("snapkv",   "SnapKV-1024",   "snapkv", 64, 1024, None, 4),
        ("keepfirst","KeepFirst-256", "keepfirst", 64, 256, None, 1),
        ("keepfirst","KeepFirst-512", "keepfirst", 64, 512, None, 1),
    ]

    all_results = {
        cfg[1]: {"f1_scores": [], "times": [], "retention_pcts": []}
        for cfg in configs
    }

    for idx, sample in enumerate(data):
        context = sample["context"]
        question = sample["input"]
        answers = sample["answers"]
        print(f"\n[{idx+1}/{len(data)}] {question[:60]}...")

        for _, label, comp_strat, ws, cap, sw, ls in configs:
            prompt = (
                f"Answer the question based on the given passages. "
                f"Only give me the answer and do not output any other words.\n\n"
                f"Passages: {context}\nQuestion: {question}\nAnswer:"
            )

            # Middle-truncation (LongBench official)
            tokenized = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
            max_length = 8000
            if len(tokenized) > max_length:
                half = max_length // 2
                input_ids = torch.cat([tokenized[:half], tokenized[-half:]]).unsqueeze(0).to("cuda")
                attention_mask = torch.ones_like(input_ids)
                inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
            else:
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192)
                inputs = {k: v.to("cuda") for k, v in inputs.items()}

            input_len = inputs["input_ids"].shape[-1]

            torch.cuda.reset_peak_memory_stats()
            start = time.time()

            with torch.no_grad():
                # Prefill
                out = model(**inputs, use_cache=True)
                past_kv = out.past_key_values

                compressed_len = input_len
                if comp_strat and comp_strat != "baseline" and input_len > cap:
                    from snapkv_lib import compress_cache, _get_head_counts
                    nq, nk = _get_head_counts(model)
                    compress_cache(
                        model, past_kv, ws, cap,
                        num_q_heads=nq, num_kv_heads=nk,
                        strategy=comp_strat,
                        scoring_window=sw,
                        layer_stride=ls,
                    )
                    compressed_len = past_kv.layers[0].keys.shape[2]
                    torch.cuda.empty_cache()

                # Decode
                decode_out = model.generate(
                    **inputs,
                    past_key_values=past_kv,
                    max_new_tokens=64,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=tokenizer.eos_token_id,
                )

            elapsed = time.time() - start

            # Extract generated text
            raw_output = decode_out[0]
            if raw_output.shape[0] > input_len:
                generated = tokenizer.decode(raw_output[input_len:], skip_special_tokens=True)
            else:
                generated = tokenizer.decode(raw_output, skip_special_tokens=True)

            import unicodedata
            generated = "".join(
                " " if unicodedata.name(c, "").startswith("LATIN SMALL LETTER DOTLESS") or ord(c) == 0x0120
                else "\n" if ord(c) == 0x010a
                else c
                for c in generated
            ).strip()
            generated = generated.split("\n")[0].strip()

            f1 = best_f1(generated, answers)
            retention = compressed_len / input_len * 100

            all_results[label]["f1_scores"].append(f1)
            all_results[label]["times"].append(elapsed)
            all_results[label]["retention_pcts"].append(retention)

            print(f"  [{label}] F1={f1:.2f}  ret={retention:.0f}%  "
                  f"t={elapsed:.2f}s  got='{generated[:50]}'")

    restore_attention(orig_forward)

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS ({len(data)} samples, LongBench qasper)")
    print(f"{'='*60}")
    print(f"{'Config':<16} {'Avg F1':<10} {'Avg Time':<10} {'Avg Retention'}")
    print("-" * 55)
    for _, label, _, _, _, _, _ in configs:
        r = all_results[label]
        avg_f1 = sum(r["f1_scores"]) / len(r["f1_scores"]) * 100
        avg_time = sum(r["times"]) / len(r["times"])
        avg_ret = sum(r["retention_pcts"]) / len(r["retention_pcts"])
        print(f"{label:<16} {avg_f1:.1f}%{'':<5} {avg_time:.2f}s{'':<5} {avg_ret:.0f}%")

    with open("results_longbench_qasper.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved to results_longbench_qasper.json")


if __name__ == "__main__":
    main()
