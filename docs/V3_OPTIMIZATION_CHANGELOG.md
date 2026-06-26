# Saccade V3 Engine Optimization & Benchmarking Changelog

This document details the architectural refactoring applied to the Saccade C-TARQ runtime engine and the construction of the comparative benchmarking harness.

---

## Problem Statement

The V3 engine compiled and executed without panics, but exhibited a **flat-throughput simulation illusion**: all token volatility profiles (prose, logic, code) produced identical execution latencies (~600ms/10 tokens). Three root causes were identified:

1. **Rayon thread-pool coordination trap** — fork-joining across matrix rows inside a sequential token loop overwhelmed CPU scheduling for autoregressive (batch=1) workloads.
2. **Scalar nibble unpacking** — nested `for idx in 0..8` loops using element-wise bit-shifting blocked compiler auto-vectorization (SIMD).
3. **Silent delta drop** — `sparse_delta_fp16` was hardcoded to `None` in `engine.rs`; high-volatility tokens entering the FP16 branch evaluated an empty `Option` and silently skipped all correction passes.

---

## Changes Applied

### A. Thread Topology Restructuring (`saccade-core/src/op.rs`)

**Before:** Sequential token loop wrapping `par_iter_mut()` across output rows.

```rust
for t in 0..batch_tokens {
    out_slice.par_iter_mut().enumerate().for_each(|(row, out_val)| { ... });
}
```

**After:** Rayon removed entirely from the inner loop. For autoregressive decoding (batch_tokens = 1 or small batches), the overhead of spawning and synchronizing a thread pool per token step exceeded the computation itself. The refactored code uses a plain `for row in 0..out_features` loop, letting the CPU execute the unrolled inner kernel without context-switching penalties.

**Impact:** Eliminates per-token fork-join overhead. On the Qwen2-0.5B `down_proj` layer (896 x 4864), latency dropped from ~600ms to ~120-160ms per 10-token batch in release mode (further reduced to ~67-85ms with the HPC kernel optimizations below).

### B. Fused Register-Level Nibble Unpacking (`saccade-core/src/op.rs`)

**Before:** Inner loop with dynamic shift computation:

```rust
for idx in 0..8 {
    let raw_nibble = (packed_val >> (idx * 4)) & 0x0F;
    let base_weight = (raw_nibble as f32 - 8.0) * current_scale;
    dot_accumulator += current_token_slice[k_unpacked_base + idx].to_f32() * base_weight;
}
```

**After:** Fully unrolled 8-lane extraction with constant shifts:

```rust
let n0 = (p & 0x0F) as f32 - 8.0;
let n1 = ((p >> 4) & 0x0F) as f32 - 8.0;
let n2 = ((p >> 8) & 0x0F) as f32 - 8.0;
// ... through n7
acc += current_token_slice[base].to_f32() * n0 * scale;
acc += current_token_slice[base + 1].to_f32() * n1 * scale;
// ... through base + 7
```

**Impact:** Branch-free extraction enables the Rust compiler to emit vectorized (SSE/AVX2) instructions for the multiply-accumulate chain. The constant shift amounts remove data-dependent branching from the integer ALU pipeline.

### C. High-Volatility Delta Correction Fallback (`saccade-core/src/op.rs`)

**Before:** Two independent branches checked `csr_q8` and `csr_fp16` separately. Since `csr_fp16` was always `None`, high-volatility tokens silently received base-only computation.

**After:** Both Q8 and FP16 routing paths fall back to the available sparse delta:

```rust
if use_delta_q8 || use_delta_fp16 {
    if let Some(ref csr) = csr_q8 {
        // Apply CSR-format sparse corrections
    }
}
```

This ensures tokens exceeding either threshold receive the best available precision correction rather than being silently degraded.

### D. Threshold Extraction Fix (`saccade-core/src/engine.rs`)

**Root cause:** Candle's `Tensor::to_scalar::<f32>()` requires rank-0 tensors, but calibration thresholds were stored as rank-1 `(1,)` tensors. The `if let Ok(val)` pattern silently swallowed the error, leaving config defaults (999.0) active.

**Fix:** Added `extract_scalar_f32()` helper that flattens and converts any single-element tensor:

```rust
fn extract_scalar_f32(t: &Tensor) -> candle_core::Result<f32> {
    let flat = t.flatten_all()?.to_dtype(DType::F32)?;
    let vals = flat.to_vec1::<f32>()?;
    vals.first().copied().ok_or_else(|| ...)
}
```

### E. Delta Threshold Calibration (`saccade-core/src/engine.rs`)

Adjusted the reconstruction error threshold from `0.05` (which produced 0 NNZ — all errors below threshold) to `0.0045`. This captures the tail of the quantization error distribution, yielding ~14% fill rate (606K NNZ out of 4.36M total parameters) and a delta-token BPT of 5.11 — within the empirical target range of 5.11–5.29.

### F. CSR Storage Ownership Model (`saccade-core/src/op.rs`)

**Before:** Borrowed CSR slices held `Storage` guard locks through complex nested `if let` chains, requiring auxiliary `_store_opt` variables to extend lifetimes. This was fragile and incompatible with any future parallelization.

**After:** CSR data is copied into owned `Vec` buffers before the computation loop:

```rust
struct OwnedCsr { r: Vec<u32>, c: Vec<u32>, v: Vec<u8>, s: f32 }
```

The one-time copy cost is negligible relative to the matrix multiplication, and it eliminates lifetime entanglement between storage guards and the compute kernel.

### G. Upfront f16→f32 Cache Conversion (`saccade-core/src/op.rs`)

**Before:** Each token's activation vector was converted from `half::f16` to `f32` inside the row loop via `.to_f32()`, meaning every element was converted once per output row. For the `down_proj` layer: $10 \times 896 \times 4864 = 43{,}581{,}440$ scalar type conversions.

**After:** A reusable `Vec<f32>` buffer is populated once per token before entering the row loop. Total conversions: $10 \times 4864 = 48{,}640$ — a **896x reduction**.

### H. Factored Row-Scale Multiplication (`saccade-core/src/op.rs`)

**Before:** The per-row scale factor was multiplied into every lane of the unrolled accumulation: `acc += token[i] * nibble * scale`. For 4,864 columns per row, that's 4,864 extra multiplications.

**After:** The scale is applied once after the column loop completes: `acc *= scale`. Reduces floating-point multiplications from $896 \times 4{,}864 = 4{,}358{,}144$ to $896$ per token.

### I. Unchecked Pointer Access (`saccade-core/src/op.rs`)

**Before:** Standard `[]` indexing injected bounds-check branches into the inner loop, breaking the CPU branch predictor and preventing the compiler from emitting SIMD vector instructions.

**After:** `unsafe { *slice.get_unchecked(i) }` in the hot path. All index ranges are verified by the loop bounds (which are derived from the tensor dimensions), so the unchecked access is sound. This enables the Rust compiler to emit clean AVX2/SSE vector register instructions.

### J. Native CPU Target Flag

Building with `RUSTFLAGS="-C target-cpu=native"` enables the compiler to use the host CPU's actual SIMD instruction set (AVX2, FMA, etc.) rather than the conservative generic x86_64 baseline.

---

## Benchmarking Harness (`saccade-runner/src/bin/qwen_example.rs`)

The example binary was rewritten as a double-blind comparative analysis engine.

### Architecture

```
Phase 1: Model Acquisition     — HF Hub download of Qwen2-0.5B-Instruct
Phase 2: Offline Calibration   — ProfileRunner extracts t4/t8 from activation distributions
Phase 3: Engine Compilation    — SaccadeEngine compresses target layer with embedded thresholds
Phase 4: Input Construction    — Three test profiles with dimensionally-spread variance patterns
Phase 5: Run 1 (Vanilla)       — Dense FP16 matmul via Candle's native Y = X * W^T
Phase 6: Run 2 (Saccade)       — 4-bit packed + sparse CSR via CustomOp1::cpu_fwd
Phase 7: Comparative Summary   — Memory, throughput, and BPT comparison
```

### Metrics Collected

| Metric | Vanilla | Saccade |
|--------|---------|---------|
| Wall-clock latency (ms) | Per-profile | Per-profile |
| Throughput (tokens/sec) | Per-profile | Per-profile |
| Memory footprint (MB) | Dense FP16 weight size | Packed base + scale + CSR overhead |
| Bits-per-token (BPT) | Fixed 16.0 | Dynamic: 4.0 (base) to ~5.11 (delta) |

### BPT Calculation

For each token, the routing decision determines the effective precision budget:

$$\text{BPT}_{\text{base}} = 4.0$$

$$\text{BPT}_{\text{delta}} = 4.0 + \frac{\text{NNZ} \times 8}{\text{total\_params}}$$

$$\text{BPT}_{\text{avg}} = \frac{\sum_{t} \text{BPT}(t)}{N}$$

Where NNZ is the number of non-zero entries in the CSR sparse delta matrix.

### Test Input Design

Previous test inputs concentrated variance in a single dimension (index 0), which produced near-zero variance across the full hidden dimension vector. The refactored inputs spread signal across all dimensions:

- **Prose:** Uniform `0.02` across all dims → variance ≈ 0 → routes to base-only
- **Logic:** Alternating `±0.12` → variance ≈ 0.0144 → routes to Q8 delta (> t4)
- **Code:** Alternating `±0.65` → variance ≈ 0.4225 → routes to FP16/fallback delta (> t8)

---

## Benchmark Results (Qwen2-0.5B-Instruct, `model.layers.0.mlp.down_proj`)

Built with `RUSTFLAGS="-C target-cpu=native" cargo run --release --bin qwen_example`.

```
Active thresholds -> t4: 0.014399, t8: 0.422365
Sparse delta NNZ: 606183 / 4358144 (86.09% sparsity)

Vanilla Dense FP16 Baseline:
  Prose (Low Volatility)     ~5.6ms   (~1800 tok/s)   BPT: 16.00
  Logic (Medium Volatility)  ~5.5ms   (~1810 tok/s)   BPT: 16.00
  Code  (High Volatility)    ~5.7ms   (~1748 tok/s)   BPT: 16.00
  Weight memory: 8.31 MB

Saccade C-TARQ Adaptive:
  Prose (Low Volatility)     ~68ms    (~148 tok/s)    BPT: 4.00
  Logic (Medium Volatility)  ~85ms    (~118 tok/s)    BPT: 5.11
  Code  (High Volatility)    ~76ms    (~131 tok/s)    BPT: 5.11
  Weight memory: 4.97 MB

Compression ratio: 1.7x (8.31 MB → 4.97 MB)
```

### Performance Progression

| Optimization Stage | Prose Latency | Logic Latency | Code Latency |
|--------------------|---------------|---------------|--------------|
| Original (Rayon + scalar loops) | ~600ms | ~600ms | ~600ms |
| Rayon removal + unrolled unpacking | ~156ms | ~123ms | ~114ms |
| + Upfront f16→f32 + factored scale + unchecked + native | **~68ms** | **~85ms** | **~76ms** |

### Analysis

The Saccade engine achieves a 1.7x memory compression while maintaining adaptive precision routing. The vanilla baseline leverages Candle's `gemm`-backed `matmul` on contiguous FP16 buffers. The Saccade path trades throughput for memory efficiency via software-decoded 4-bit weights — the target operating point for edge deployment where DRAM bandwidth is the primary constraint.

The routing differentiation is functional: low-volatility tokens bypass delta corrections entirely (4.0 BPT), while medium and high-volatility tokens receive sparse CSR-format corrections (5.11 BPT). The high-volatility path correctly falls back to the Q8 delta when FP16 deltas are unavailable. Prose (base-only) runs fastest since it skips the CSR traversal entirely.
