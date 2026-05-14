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
