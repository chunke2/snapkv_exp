# CHANGELOG -- SnapKV Experiment

---

## Round 1 -- 2026-05-14: Code unification + control baselines + timing breakdown

**Decision**: D (partial keep WIP)

### What was done
- Extracted snapkv_compress_gqa, patch_llama_attention, restore_attention, compress_cache, run_inference into shared snapkv_lib.py (fixes D5, F7)
- Auto-detect GQA head counts from model config (fixes D6)
- Added two control baselines: Random-K and KeepFirst-K (F2)
- Added prefill / compress / decode timing breakdown (F4)
- Fixed D3: attention_mask handling in decode matched to old working pattern
- Rewrote experiment3_comprehension.py to use shared lib with 9 configs

### Key results (experiment3, 6 questions on transformer paper, ~1540 tokens)
| Config | Acc | Dec(s) | Ret% |
|--------|-----|--------|------|
| Baseline | 83% | 1.42 | 100% |
| SnapKV-256 | 83% | 0.92 | 17% |
| SnapKV-512 | 83% | 0.90 | 33% |
| Random-256 | 0% | 0.92 | 17% |
| KeepFirst-256 | 83% | 0.92 | 17% |
| KeepFirst-512 | 83% | 0.90 | 33% |

### Critical finding
**SnapKV and KeepFirst achieve identical accuracy (83%) at the same budgets.**
- Keeping the first 256/512 tokens (document abstract + intro) is as effective as SnapKV attention-based selection on this test
- Random is terrible (0-33%) -- proves selection strategy matters, but attention scoring does not beat simple position heuristic
- Decode speedup: 37% (1.42s to 0.90s)

### New defects identified
- D16: SnapKV does not outperform KeepFirst on short-document factual QA -- need tests where important info is NOT at the beginning
- D17: SnapKV compress overhead 0.041s vs KeepFirst 0.006s (7x) with no accuracy gain on this test

### Next round candidates
- Download LongBench data for standard benchmark
- Construct needle-in-middle/late tests where KeepFirst will fail
- Add per-question answer-token retention analysis
