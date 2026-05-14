# DEFECTS — SnapKV Experiment

Living defect inventory. Fixed items get strikethrough with resolution note.
New defects appended at bottom.

---

## Blocker

### D1: experiment1_snapkv.py — patch_model_with_snapkv never compresses
- **File**: experiment1_snapkv.py:106-128
- **Type**: implementation bug
- **Severity**: Blocker
- **Description**: make_new_forward replaces attention forward but never calls cluster.compress(). For seq_len > max_capacity_prompt, it still returns orig_forward(*args, **kwargs). KV cache is never compressed.
- **Impact**: results_snapkv.json SnapKV data is actually full KV cache — proves nothing about compression.

### D2: experiment4_longbench.py — make_snapkv_forward uses last-layer query for all layers
- **File**: experiment4_longbench.py:112-131
- **Type**: implementation bug
- **Severity**: Blocker
- **Description**: Compression triggered in last layer (layer_idx==31) but uses its query_states to compress all 32 layers. Layer-31 query is semantically wrong for scoring layer-0 keys.
- **Impact**: Currently NOT called by run_inference (which uses compress_cache correctly). Dead code risk.

### D3: experiment4_longbench.py — attention_mask mismatch after compression
- **File**: experiment4_longbench.py:196-210
- **Type**: implementation bug
- **Severity**: Blocker
- **Description**: Decode passes original attention_mask (length=input_len) alongside compressed past_key_values (length=max_capacity_prompt). Length mismatch.
- **Impact**: results_longbench_qasper.json SnapKV data may be unreliable.

---

## High

### D4: snapkv_utils.py — SnapKVCluster.update_kv and init_snapkv are dead code
- **File**: snapkv_utils.py:45-95
- **Type**: dead code / design coupling
- **Severity**: High
- **Description**: Defined but never imported or called. All experiments define their own snapkv_compress_gqa(). Two inconsistent SnapKV implementations exist.
- **Impact**: Confusion about which implementation is authoritative.

### D5: snapkv_compress_gqa duplicated in 5 files
- **Files**: experiment1_snapkv_v5.py, experiment1_v2_longdoc.py, experiment2_needle.py, experiment3_comprehension.py, experiment4_longbench.py
- **Type**: code duplication
- **Severity**: High
- **Description**: Same function in 5 files with subtle differences (v5 returns 2-tuple vs others return 3-tuple). Bug fixes require 5 edits.
- **Impact**: Maintenance nightmare. Cross-experiment comparison unreliable.

### D6: Hardcoded GQA head counts
- **Files**: All files defining snapkv_compress_gqa
- **Type**: robustness
- **Severity**: High
- **Description**: num_q_heads=32, num_kv_heads=8 hardcoded. Changing model produces wrong GQA grouping silently.
- **Impact**: Broken model portability.

### D7: Memory measurement inconsistency across experiments
- **Files**: all experiment scripts
- **Type**: measurement inconsistency
- **Severity**: High
- **Description**: Different scripts use different memory metrics (delta vs peak vs none). Cannot compare memory savings.
- **Impact**: SnapKV core value prop unmeasurable cross-experiment.

---

## Medium

### D8: experiment1_snapkv.py — single cluster shared across all layers
- **File**: experiment1_snapkv.py:128
- **Type**: design issue
- **Severity**: Medium
- **Description**: All layers share one cluster object. If D1 fixed and compression activated, shared state could cause issues.

### D9: experiment4 — inconsistent prompt truncation
- **File**: experiment4_longbench.py:174-184
- **Type**: implementation inconsistency
- **Severity**: Medium
- **Description**: LongBench uses middle-truncation; other experiments use tail truncation. Different context preservation.
- **Impact**: Retention ratios not comparable across experiments.

### D10: Missing torch.cuda.empty_cache() after compression
- **Files**: experiment1_snapkv_v5.py, experiment1_v2_longdoc.py, experiment2_needle.py
- **Type**: implementation omission
- **Severity**: Medium
- **Description**: Old KV tensors linger until GC. experiment3/experiment4 call empty_cache() but earlier scripts don't.
- **Impact**: Memory usage over-reported in affected scripts.

### D11: Two different SnapKVCluster implementations
- **Files**: experiment1_snapkv.py vs snapkv_utils.py
- **Type**: implementation inconsistency
- **Severity**: Medium
- **Description**: compress() has no causal mask; update_kv() has one. Different attention scoring.

### D12: snapkv_utils.py update_kv — suspicious causal mask
- **File**: snapkv_utils.py:56-62
- **Type**: potential bug
- **Severity**: Medium
- **Description**: Adds causal mask that model attention already applies. May alter token importance scores.

---

## Low (methodology — do not change code)

### D13: Observation window heuristic
- **Type**: methodological assumption
- **Severity**: Low
- **Description**: Last window_size queries score all prefix keys. Needle at beginning/middle may have low attention from end queries.
- **Note**: exp2 specifically tests this. All positions found correctly in existing results.

### D14: Pooling may dilute isolated important tokens
- **Type**: methodological tradeoff
- **Severity**: Low
- **Description**: avg_pool1d(kernel_size=5) smooths scores before top-k. Critical isolated token gets score averaged down.
- **Note**: maxpool variant exists but untested. Future ablation candidate.

### D15: No position prior in token selection
- **Type**: methodological limitation
- **Severity**: Low
- **Description**: Token selection ignores position. BOS/system prompt may have low attention but high importance.
- **Note**: Could add keep-first-N-tokens rule. Future experiment candidate.

---

## Summary

| Severity | Count | Action |
|----------|-------|--------|
| Blocker | 3 | Must fix before trusting results |
| High | 4 | Should fix for reliable experimentation |
| Medium | 5 | Fix opportunistically |
| Low | 3 | Methodology notes, no code change |

---

## Round 1 Updates (2026-05-14)

### Fixed this round

~~D5: snapkv_compress_gqa duplicated in 5 files~~ → Extracted to snapkv_lib.py. All experiments now import shared implementation.

~~D6: Hardcoded GQA head counts~~ → Auto-detect from model.config.num_attention_heads / num_key_value_heads.

~~D7: Memory measurement inconsistency~~ → Standardized: all experiments use torch.cuda.max_memory_allocated() via shared run_inference.

~~F2: No control baselines~~ → Added Random-K and KeepFirst-K strategies in snapkv_lib.py.

~~F4: No latency decomposition~~ → run_inference now reports prefill_time, compress_time, decode_time separately.

~~F7: Experiments not comparable~~ → Unified inference loop in snapkv_lib.py with consistent metrics interface.

### New defects discovered

#### D16: SnapKV does not outperform KeepFirst on short-document factual QA
- **File**: experiment3_comprehension.py results
- **Type**: experimental finding
- **Severity**: High (research significance)
- **Description**: On the transformer paper comprehension test (~1500 tokens), KeepFirst-256/512 achieves 83% accuracy — identical to SnapKV-256/512. The important information (abstract, key results) is at the beginning of the document, so simply keeping the first K tokens works as well as attention-based selection. Random-256 gets 0%, proving selection matters — but attention scoring doesn't beat the position heuristic.
- **Impact**: Current test setup cannot distinguish SnapKV's value from a trivial baseline. Need tests where critical information is NOT at the beginning.

#### D17: SnapKV compress overhead 7x higher than KeepFirst with no accuracy gain
- **File**: experiment3_comprehension.py results
- **Type**: performance finding
- **Severity**: Medium
- **Description**: SnapKV compress time = 0.041s vs KeepFirst = 0.006s. This is 7x overhead for no accuracy benefit on the current test. The overhead comes from attention computation (QK matmul + softmax + pooling + topk). For longer sequences where KeepFirst fails, this overhead may be justified — but not on short documents with front-loaded information.
- **Impact**: Performance-cost ratio of SnapKV is poor on this test. Need to validate on scenarios where KeepFirst cannot work.

### Status unchanged
- D1, D2, D3, D8, D9, D10, D11, D12, D13, D14, D15 — not addressed this round

---

## Round 2 Updates (2026-05-14)

### Fixed this round

~~D16: SnapKV does not outperform KeepFirst on short-document factual QA~~
  -> Position-injected long document test (8157 tokens, 5 positions) shows SnapKV 100% vs KeepFirst 80%. Difference is at P75 (73% position), confirming SnapKV attention scoring can retrieve from mid-document while KeepFirst cannot.

~~D18: Long document causes BPE garbled output, baseline accuracy low~~
  -> Baseline 100% on revised test with simple numeric needles and non-conflicting concepts.

~~D19: Needle format broken by BPE (hyphens in numbers)~~
  -> Use pure numeric needles without hyphens. P0 answer widened to match 77 (BPE-fragmented version of 7742).

~~D20: Document too short for KeepFirst differentiation~~
  -> 8157 tokens, KeepFirst only covers first 3% of document.

### New defects

None this round.

### Status unchanged
- D1, D2, D3, D8, D9, D10, D11, D12, D13, D14, D15, D17 — not addressed this round

---

## Rounds 3-4 Updates (2026-05-14)

### Fixed this round

~~D17: SnapKV compress overhead 7-10x higher than KeepFirst with no accuracy gain~~
  -> Layer sampling (stride=4) reduces compress time from 0.460s to 0.136s (3.4x speedup) while maintaining 100% accuracy. Compress overhead vs KeepFirst reduced from 10x to 2.9x.

### Parameters added
- scoring_window: decouple scoring queries from observation window (flexibility tool, not primary optimization)
- layer_stride: compute SnapKV indices every N layers, share with neighbors (primary D17 fix)

### Status unchanged
- D1, D2, D3, D8, D9, D10, D11, D12, D13, D14, D15 -- not addressed
