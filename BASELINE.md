# BASELINE — SnapKV Experiment

**Date**: 2026-05-14
**Script**: experiment3_comprehension.py
**Model**: DeepSeek-R1-Distill-Llama-8B (fp16, L40S 48GB)
**Test data**: transformer_doc.txt (6575 bytes, the original Transformer paper)
**Test cases**: 6 comprehension questions covering Abstract, Introduction, Attention, Positional Encoding, Training, Results
**Input length**: ~1540 tokens per question (document + prompt)
**Generation**: greedy, max 30 new tokens

---

## Key Metrics

| Config | Accuracy | Time (avg) | Peak Mem (avg) | KV Retention |
|--------|----------|------------|----------------|--------------|
| Baseline | 83% (5/6) | 1.32s | 16.26 GB | 100% |
| SnapKV-128 | 83% (5/6) | 1.08s | 16.03 GB | 8.3% |
| **SnapKV-256** | **100% (6/6)** | **1.05s** | **16.00 GB** | **16.6%** |
| **SnapKV-512** | **100% (6/6)** | **1.04s** | **15.91 GB** | **33%** |
| SnapKV-1024 | 83% (5/6) | 1.00s | 15.92 GB | 66% |

## Per-question Baseline Detail

| # | Section | Expected | Got (truncated) | OK |
|---|---------|----------|-----------------|----|
| 1 | Abstract | 2014 | 2014 | Y |
| 2 | Introduction | attention mechanisms | Attention mechanisms | Y |
| 3 | Attention | sqrt(dk) | garbled BPE | N |
| 4 | Positional Encoding | bottoms | bottom | Y |
| 5 | Training | 0.1 | 0.1 | Y |
| 6 | Results | 3.5 days | 3.5 | Y |

## Key Observations

1. SnapKV-256 and SnapKV-512 beat baseline (83% to 100%). Compression acts as noise filter.
2. SnapKV-1024 drops back to 83% — more budget keeps distracting tokens.
3. Peak memory differences are small (~0.3 GB). KV cache for ~1500 tokens is ~19 MB vs 16 GB weights.
4. Time savings: 18-24% from shorter decode attention.
5. Question 3 (sqrt(dk)) only answered correctly by SnapKV-256 and SnapKV-512.
