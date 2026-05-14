# CLAUDE.md — SnapKV Experiment

## Project
KV cache compression research (SnapKV). Iterative R&D: Audit → Hypothesize → Patch → Verify → Decide.
Model: DeepSeek-R1-Distill-Llama-8B (32 Q heads / 8 KV heads, fp16).

## Server
```
ssh -p 40121 root@18.232.140.143 -L 8080:localhost:8080
```
- GPU: L40S 48GB, vast.ai instance 36721650
- Repo: `/workspace/snapkv_exp/`
- Model: `/workspace/models/models--deepseek-ai--DeepSeek-R1-Distill-Llama-8B/snapshots/6a6f4*/`
- Python: 3.12.13, transformers 5.6.2, torch
- Use `python3` not `python`

## GitHub
- Repo: `https://github.com/chunke2/snapkv_exp`
- Push: set GH_TOKEN env var or use `git remote set-url origin https://TOKEN@github.com/...` then restore URL after push

## Key Files

### Shared Library
- `snapkv_lib.py` — unified SnapKV implementation. All experiments import from here.
  - `snapkv_compress_gqa()` — GQA-aware compression (auto-detect heads from config)
  - `random_compress()` / `keepfirst_compress()` — control baselines
  - `patch_llama_attention()` / `restore_attention()` — monkey-patch forward
  - `compress_cache()` — per-layer compression with `layer_stride` support
  - `run_inference()` — unified prefill→compress→decode pipeline
  - Parameters: `window_size=64`, `max_capacity_prompt`, `scoring_window`, `layer_stride`
  - Best config: `stride=4` (3.4x compress speedup with 100% accuracy)

### Active Experiments
- `experiment3_comprehension.py` — 6 questions on transformer paper (~1.5K tokens), 9 configs
- `experiment5_position.py` — 5 needles at 0/25/50/75/100% in 8.2K-token document, 9 configs
- `experiment4_longbench.py` — LongBench qasper (needs data at `/workspace/longbench_data/data/qasper.jsonl`)

### Legacy (kept as iteration history, do not modify)
- `experiment1_*.py`, `experiment2_needle.py`, `snapkv_utils.py`, `fix_compress.py`

### Documentation
- `BASELINE.md` — initial baseline metrics
- `DEFECTS.md` — living defect inventory (resolved items struck through)
- `CHANGELOG.md` — per-round what was done and decided
- `REPORT.md` — comprehensive technical report (algorithms, engineering, experiments)

## State: What's Done & What's Left

### Done (5 rounds)
- R1: Code unification (snapkv_lib.py), control baselines, timing breakdown (D)
- R2: Position-injection test, SnapKV 100% vs KeepFirst 80% (A)
- R3: scoring_window parameter, not the right fix for D17 (B)
- R4: Layer sampling stride=4, 3.4x compress speedup (A)
- R5: Fix D2/D3, migrate experiment4 to shared lib (A)

### Remaining Defects
- **D1 (Blocker)**: experiment1_snapkv.py patch never calls compress() — `results_snapkv.json` is invalid
- D8-D12 (Medium): misc code quality issues
- D13-D15 (Low): methodology notes (pooling, position prior, observation window)
- LongBench qasper data needed for experiment4 verification

### Key Numbers
| Test | SnapKV-256-stride4 | KeepFirst-256 | Baseline |
|------|:--:|:--:|:--:|
| Position injection (8.2K tok) | 100% acc, 0.14s compress | 80%, 0.05s | 100%, 5.67s |
| Comprehension (1.5K tok) | 83%, 0.04s | 83%, 0.006s | 83%, 1.42s |

## Workflow Rules
1. Three docs strictly maintained: BASELINE (append only), DEFECTS (strike resolved, add new), CHANGELOG (per-round entry)
2. No silent refactoring — if you see unrelated issues, add to DEFECTS, don't fix in same round
3. Honesty > progress — failed attempts go in DEFECTS with reason
4. Each round: git commit with descriptive message
5. Keep old scripts as iteration history, don't delete
6. transformers 5.x compatibility issues: report to user first, don't rewrite experiment code without discussion
7. Minimum change principle — don't add abstractions beyond what the fix needs
8. After each round: push to GitHub with token
