"""
Experiment 3 — Comprehension eval with control baselines.
Tests: Baseline vs SnapKV vs Random-K vs Keep-First-K
Measures: accuracy, prefill/compress/decode timing, peak memory, KV retention.
"""
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from snapkv_lib import (
    patch_llama_attention, restore_attention,
    run_inference, check_correct_keyword, collect_metadata,
)

MODEL_PATH = "/workspace/models/models--deepseek-ai--DeepSeek-R1-Distill-Llama-8B/snapshots/6a6f4aa4197940add57724a7707d069478df56b1"

DOCUMENT = open("transformer_doc.txt").read()

TEST_CASES = [
    {
        "question": "What year is the WMT English-to-French translation task that the model was evaluated on?",
        "answer": "2014",
        "section": "Abstract",
    },
    {
        "question": "The Transformer avoids recurrence by using what mechanism to capture global dependencies?",
        "answer": "attention mechanisms|Attention mechanisms|attention",
        "section": "Introduction",
    },
    # Q3 flagged: baseline often fails due to BPE tokenizer not preserving math notation
    {
        "question": "In the Scaled Dot-Product Attention formula, what do we divide the dot products by?",
        "answer": "sqrt(dk)|sqrt(d",
        "section": "Attention",
        "flag": "tokenizer-sensitive",
    },
    {
        "question": "Positional encodings are added to the input embeddings at the what of the encoder and decoder stacks?",
        "answer": "bottoms|bottom|encoder and decoder",
        "section": "Positional Encoding",
    },
    {
        "question": "What is the label smoothing value eps_ls used during training?",
        "answer": "0.1",
        "section": "Training",
    },
    {
        "question": "How long did it take to train the big transformer model on 8 P100 GPUs?",
        "answer": "3.5 days|3.5days|3.5",
        "section": "Results",
    },
]

STRATEGY_CONFIGS = [
    (None,                             "Baseline",        None,         64, None),
    ("snapkv",                         "SnapKV-128",      "snapkv",     64, 128),
    ("snapkv",                         "SnapKV-256",      "snapkv",     64, 256),
    ("snapkv",                         "SnapKV-512",      "snapkv",     64, 512),
    ("snapkv",                         "SnapKV-1024",     "snapkv",     64, 1024),
    ("random",                         "Random-256",       "random",     64, 256),
    ("random",                         "Random-512",       "random",     64, 512),
    ("keepfirst",                      "KeepFirst-256",    "keepfirst",  64, 256),
    ("keepfirst",                      "KeepFirst-512",    "keepfirst",  64, 512),
]


def main():
    meta = collect_metadata(MODEL_PATH)
    print(f"Python: {meta['python_version'].split()[0]}  "
          f"transformers: {meta.get('transformers_version','?')}  "
          f"git: {meta['git_commit'][:7]}")

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, device_map="cuda", use_cache=True,
    )
    model.eval()
    num_q_heads = model.config.num_attention_heads
    num_kv_heads = getattr(model.config, "num_key_value_heads", num_q_heads)
    print(f"Model loaded.  Q heads: {num_q_heads}  KV heads: {num_kv_heads}")

    orig_forward = patch_llama_attention()

    all_results = {}

    for str_type, label, comp_strat, ws, cap in STRATEGY_CONFIGS:
        budget_info = f"  budget={cap}" if cap else ""
        print(f"\n{'='*60}")
        print(f"{label}{budget_info}  strategy={comp_strat or 'baseline'}")
        print(f"{'='*60}")

        cases = []
        for test in TEST_CASES:
            prompt = f"Document:\n{DOCUMENT}\n\nQuestion: {test['question']}\nAnswer:"
            result = run_inference(
                model, tokenizer, prompt,
                compress_strategy=comp_strat,
                window_size=ws,
                max_capacity_prompt=cap or 99999,
                max_new_tokens=30,
            )
            correct = check_correct_keyword(result["answer"], test["answer"])

            flag = f" [{test.get('flag','')}]" if test.get("flag") else ""
            print(f"  [{test['section']}{flag}] {'OK' if correct else 'XX'}  "
                  f"exp='{test['answer'].split('|')[0]}'  "
                  f"got='{result['answer'][:60]}'  "
                  f"pf={result['prefill_time']:.2f}s  "
                  f"cmp={result['compress_time']:.3f}s  "
                  f"dec={result['decode_time']:.2f}s  "
                  f"ret={result['retention_pct']:.0f}%")

            cases.append({
                "section": test["section"],
                "question": test["question"],
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

    # ── Summary table ──
    n = len(TEST_CASES)
    print(f"\n{'='*80}")
    print("SUMMARY TABLE")
    print(f"{'='*80}")
    hdr = (f"{'Config':<16} {'Acc':<6} {'Pf(s)':<7} {'Cmp(s)':<7} "
           f"{'Dec(s)':<7} {'Total(s)':<8} {'Mem(GB)':<8} {'Ret%':<6}")
    print(hdr)
    print("-" * 80)

    for str_type, label, comp_strat, ws, cap in STRATEGY_CONFIGS:
        cases = all_results[label]
        acc = sum(c["correct"] for c in cases) / n * 100
        pf  = sum(c["prefill_time"] for c in cases) / n
        cmp = sum(c["compress_time"] for c in cases) / n
        dec = sum(c["decode_time"] for c in cases) / n
        tot = sum(c["total_time"] for c in cases) / n
        mem = sum(c["peak_mem_gb"] for c in cases) / n
        ret = sum(c["retention_pct"] for c in cases) / n
        print(f"{label:<16} {acc:.0f}%{'':<2} {pf:.2f}{'':<3} {cmp:.3f}{'':<2} "
              f"{dec:.2f}{'':<3} {tot:.2f}{'':<3} {mem:.2f}{'':<3} {ret:.0f}%")

    # ── Strategy showdown ──
    print(f"\n{'='*80}")
    print("STRATEGY COMPARISON: SnapKV vs Random vs KeepFirst at same budgets")
    print(f"{'='*80}")
    for budget in [256, 512]:
        print(f"\n  Budget = {budget}:")
        for lbl in [f"SnapKV-{budget}", f"Random-{budget}", f"KeepFirst-{budget}"]:
            if lbl in all_results:
                c = all_results[lbl]
                acc = sum(x["correct"] for x in c) / n * 100
                print(f"    {lbl:<18} accuracy={acc:.0f}%")

    with open("results_experiment3_comprehension.json", "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print("\nSaved to results_experiment3_comprehension.json")


if __name__ == "__main__":
    main()
