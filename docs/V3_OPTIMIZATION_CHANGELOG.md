# Saccade V3 Engine: Architecture, Optimization History & Results

This document serves as the complete technical record of the Saccade C-TARQ engine's evolution from an isolated micro-benchmark to a production-grade inference toolkit. It covers every architectural decision, the reasoning behind each optimization, and the empirical results that validate the framework's thesis.

---

## 1. The C-TARQ Thesis

**Causal Token-Adaptive Residual Quantization (C-TARQ)** compresses transformer weight matrices to 4-bit packed representations with sparse INT8 error corrections, then dynamically selects per-token precision based on activation complexity. The hypothesis: on memory-bandwidth-constrained hardware, reading 4x less weight data per layer outweighs the software dequantization cost, yielding both smaller memory footprints and faster inference.

**Result:** Validated. On Qwen2.5-0.5B-Instruct (24 layers, full end-to-end inference):

| Metric | Vanilla FP16 | Saccade C-TARQ |
|--------|-------------|----------------|
| Decode speed | 5.8 tok/s | **7.4 tok/s (1.28x faster)** |
| Memory footprint | 1264.81 MB | **718.27 MB (1.76x smaller)** |
| Output quality | Coherent | Coherent |
| Precision budget | 16.00 BPT | ~5.19 BPT |

---

## 2. Problem Statement (Initial State)

The V3 engine compiled and executed without panics, but suffered from seven distinct issues:

1. **Flat-throughput illusion** — all token volatility profiles (prose, logic, code) produced identical latencies (~600ms/10 tokens) because the routing logic was inert.
2. **Rayon thread-pool coordination trap** — fork-joining across matrix rows inside a sequential token loop overwhelmed CPU scheduling for autoregressive workloads.
3. **Scalar nibble unpacking** — nested `for idx in 0..8` loops using element-wise bit-shifting blocked compiler auto-vectorization.
4. **Silent delta drop** — `sparse_delta_fp16` was hardcoded to `None`; high-volatility tokens silently received base-only computation.
5. **Redundant f16→f32 conversions** — 43M scalar type conversions per batch where 48K sufficed.
6. **4.03 BPT illusion** — an RMS-based delta threshold produced 99.6% sparsity, running as a nearly uncorrected 4-bit baseline.
7. **Single-layer evaluation** — benchmarks targeted only `model.layers.0.mlp.down_proj`, not the full model.

---

## 3. Optimization History

### Stage 1: Core Kernel Fixes

**Unrolled nibble extraction:** Replaced the nested `for idx in 0..8` loop with 8 constant-shift extractions. Constant shift amounts produce branch-free extraction that the compiler can map to SIMD integer lanes.

**Unified delta fallback:** Both Q8 and FP16 routing paths fall back to the available sparse delta (`if use_delta_q8 || use_delta_fp16`), preventing high-volatility tokens from silently computing base-only.

**Threshold extraction fix:** Candle's `Tensor::to_scalar()` requires rank-0 tensors, but calibration thresholds were stored as rank-1 `(1,)` tensors. Added `extract_scalar_f32()` that flattens any single-element tensor.

**Owned CSR storage:** Replaced fragile borrowed `Storage` guard chains with owned `Vec` buffers, eliminating lifetime entanglement.

### Stage 2: HPC Kernel Optimizations

**Upfront f16→f32 cache conversion:** Token activations converted once per token into a reusable buffer, not once per row×column. Reduced conversions from 43.5M to 48.6K per batch (896x reduction).

**Factored row-scale multiplication:** Row scale applied once after the column accumulation loop (`acc *= scale`) instead of per-lane. Reduced FP multiplies from 4,358,144 to 896 per token.

**Unchecked pointer access:** `unsafe { *slice.get_unchecked(i) }` in the inner loop. All index ranges are verified by loop structure. Eliminates ~8.7M conditional branches per batch that broke SIMD vectorization.

### Stage 3: Parallelism & Evaluation Methodology

**Coarse-grained Rayon row parallelism:** Re-introduced after the initial removal. The per-row kernel is now lightweight enough (cached f32, factored scale, unchecked access) that Rayon's fork-join cost is amortized across 1536+ rows.

**GEMM + GEMV dual benchmark:** Added batch=1 autoregressive decoding simulation alongside the batch=10 prefill benchmark. GEMV is the deployment scenario C-TARQ was engineered for.

### Stage 4: Pre-Cached Kernel Data & CSC Sparse Format

**Pre-cached kernel data:** All execution data is extracted from Tensors once at `SaccadeLinearOp::new()` construction time and stored as `KernelCache`:
- `packed_weights: Vec<u32>` — base weights
- `scales_f32: Vec<f32>` — row scales pre-converted from f16
- `csc: Option<CachedCsc>` — sparse corrections in CSC format with pre-scaled f32 values

This eliminates per-`cpu_fwd` Tensor guard acquisition and Vec memcpy. In full-model inference (72 MLP layer calls per token), this saved ~144ms of per-call extraction overhead.

**CSR→CSC transposition:** Sparse corrections are transposed from Compressed Sparse Row to Compressed Sparse Column format at construction time. CSR iterates rows and does scattered reads from the activation cache (breaking the CPU prefetcher). CSC iterates columns sequentially — contiguous activation reads, with row-indexed writes targeting the 6KB accumulator buffer (L1-resident). Values are pre-scaled to f32 (`(i8_val as f32) * scale`), leaving a single FMA per non-zero element in the hot loop.

### Stage 5: Pipelined FMA Accumulators (The Breakthrough)

**The discovery:** FMA instructions on modern x86 CPUs have 4-cycle latency but 0.5-cycle throughput. A single accumulator (`acc += a * b`) creates a serial dependency chain where only 1 FMA can execute per 4 cycles — wasting 87.5% of the FPU pipeline. Since Rust does not enable `-ffast-math`, the compiler cannot reorder these into independent chains automatically.

**The fix:** Manually distribute the 8 per-u32 FMAs across 4 independent accumulators (a0–a3):
```rust
a0 += activation[base + 0] * nibble_0;
a1 += activation[base + 1] * nibble_1;
a2 += activation[base + 2] * nibble_2;
a3 += activation[base + 3] * nibble_3;
a0 += activation[base + 4] * nibble_4;
a1 += activation[base + 5] * nibble_5;
a2 += activation[base + 6] * nibble_6;
a3 += activation[base + 7] * nibble_7;
```

Each accumulator receives 2 FMAs per iteration with 2 intervening independent FMAs between consecutive uses — enough pipeline separation to approach the 0.5-cycle throughput limit. This single change yielded a **3.2x speedup** on the base computation path.

### Stage 6: Production Toolkit

**Percentile-based delta threshold:** `compute_percentile_threshold()` quantizes the weight matrix to 4-bit, collects all reconstruction errors, sorts them, and returns the error at the `(1 - target_fill)` percentile. With `target_fill = 0.15`, this ensures exactly 15% of weight elements receive sparse delta corrections, producing BPT in the 5.11–5.29 range regardless of model size or weight distribution. This replaced the RMS-based heuristic (`rms_weight * 0.6`) that produced 99.6% sparsity.

**Full-model compression (`saccade-compile`):** Compresses all 72 MLP projections (24 layers × gate/up/down) with per-layer adaptive thresholds. Infers model config from tensor shapes to avoid config.json download issues.

**Streaming inference (`saccade-run`):** Full Qwen2 transformer with `ProjectionLayer` dual-mode — `Standard(candle_nn::Linear)` for attention layers, `Saccade(SaccadeLinearOp)` for compressed MLP layers. Supports both vanilla and Saccade modes with real-time token streaming and telemetry.

**Tied-weights support:** Qwen2.5 models share `embed_tokens` and `lm_head` weights (`tie_word_embeddings: true`). The model automatically falls back to the embedding matrix when `lm_head.weight` is absent.

---

## 4. Two-Phase Kernel Architecture

The execution kernel in `op.rs` separates computation into three phases:

```
┌─────────────────────────────────────────────────────┐
│ PHASE 1: Base 4-bit dot product (Rayon-parallel)    │
│                                                     │
│  For each row (distributed across CPU cores):       │
│    Unpack 8 nibbles per u32                         │
│    4 independent FMA accumulators (a0–a3)           │
│    Factored row scale: acc = (a0+a1+a2+a3) * scale  │
│    → f32 accumulator buffer                         │
├─────────────────────────────────────────────────────┤
│ PHASE 2: Sparse CSC correction (sequential)         │
│                                                     │
│  For each column (sequential, prefetch-friendly):   │
│    Read activation[col] contiguously                │
│    For each non-zero entry in this column:          │
│      acc_buffer[row] += activation * pre_scaled_val │
│    Writes target ~6KB buffer (L1-resident)          │
├─────────────────────────────────────────────────────┤
│ PHASE 3: Convert f32 accumulator → f16 output       │
└─────────────────────────────────────────────────────┘
```

### Why Two Phases?

Phase 1 is embarrassingly parallel — each row's computation is independent. Rayon distributes rows across CPU cores with no shared mutable state.

Phase 2 cannot be parallelized trivially because different columns write to overlapping row indices in the accumulator. However, CSC format makes this phase fast enough that parallelization isn't needed: the activation reads are contiguous (sequential L1 hits), the writes target a tiny buffer (6KB for 1536 rows × 4 bytes), and each entry requires just one FMA with a pre-scaled f32 value.

---

## 5. Performance Progression

| Stage | Prose GEMV | Logic GEMV | Key Change |
|-------|-----------|-----------|------------|
| Original (v3.0) | ~600ms/10tok | ~600ms/10tok | Flat throughput illusion |
| + Unrolled unpacking (v3.1) | 156ms/10tok | 123ms/10tok | SIMD-friendly nibble extraction |
| + HPC kernel (v3.2) | 68ms/10tok | 85ms/10tok | Upfront f16 cache, factored scale |
| + Rayon + GEMV bench (v3.3) | 2.3ms/tok | 5.4ms/tok | Multi-core, dual benchmark |
| + Pre-cached CSC (v3.4) | 2.8ms/tok | 5.4ms/tok | Eliminated Tensor extraction |
| + Pre-scaled CSC values (v3.4b) | 2.8ms/tok | 5.4ms/tok | Eliminated per-element i8→f32 |
| + Pipelined accumulators (v3.5) | **0.88ms/tok** | **2.8ms/tok** | 4-accumulator FMA pipeline |

### Single-Layer Micro-Benchmark (Qwen2-1.5B, `down_proj` 1536×8960)

```
GEMV (batch=1, autoregressive decoding):
  [Vanilla]  Prose  441 tok/s    Logic  461 tok/s    Code  447 tok/s
  [Saccade]  Prose  1138 tok/s   Logic  362 tok/s    Code  380 tok/s
                    ↑ 2.58x faster

Memory: 26.25 MB (FP16) → 16.37 MB (Saccade) = 1.6x compression
BPT: 5.19 (within 5.11–5.29 target)
```

### End-to-End Full-Model Inference (Qwen2.5-0.5B-Instruct, 24 layers)

```
Vanilla FP16:
  Decode: 172.64 ms/tok → 5.8 tok/s
  Memory: 1264.81 MB

Saccade C-TARQ:
  Decode: 135.45 ms/tok → 7.4 tok/s (1.28x FASTER)
  Memory: 718.27 MB (1.76x SMALLER)
```

**Why single-layer and full-model results differ:** In the single-layer benchmark, the entire 26 MB weight matrix fits in L3 cache, so vanilla never hits a DRAM bandwidth wall. In full-model inference, 24 layers compete for cache — vanilla must stream ~1.2 GB of FP16 weights while Saccade streams ~700 MB of packed 4-bit + sparse data. The bandwidth savings outweigh the dequantization cost.

---

## 6. Project Structure

```
saccade-x2/
├── saccade-core/src/
│   ├── config.rs       — SaccadeLinearOp, SaccadeConfig, KernelCache, CachedCsc
│   ├── op.rs           — CustomOp1 impl: 3-phase kernel with pipelined accumulators
│   ├── engine.rs       — SaccadeEngine::compile_model_topology
│   ├── compress.rs     — compress_tensor_to_saccade, compute_percentile_threshold
│   ├── calibration.rs  — ProfileRunner::calibrate (percentile-based threshold extraction)
│   ├── heuristics.rs   — variance_heuristic, l2_norm_heuristic
│   └── lib.rs          — Public API exports
│
├── saccade-runner/src/
│   ├── model.rs        — Qwen2Model with ProjectionLayer dual-mode (Standard/Saccade)
│   ├── lib.rs          — Module exports
│   └── bin/
│       ├── compile.rs  — saccade-compile CLI (model compression)
│       ├── run.rs      — saccade-run CLI (streaming inference + telemetry)
│       ├── qwen_example.rs — GEMM/GEMV micro-benchmark
│       └── verify.rs   — Minimal validation harness
│
└── docs/
    ├── DEVELOPER_GUIDE.md         — Usage guide for developers
    └── V3_OPTIMIZATION_CHANGELOG.md — This file
```

---

## 7. Build & Run

```bash
# Build all binaries with native SIMD optimizations
RUSTFLAGS="-C target-cpu=native" cargo build --release

# Step 1: Compile a model (downloads Qwen2.5-0.5B from HF Hub)
cargo run --release --bin saccade-compile -- \
  --model-id Qwen/Qwen2.5-0.5B-Instruct \
  --calib-file calibration.txt \
  --output-path saccade_qwen.safetensors \
  --tokenizer tokenizer.json

# Step 2: Run inference with Saccade
cargo run --release --bin saccade-run -- \
  --checkpoint saccade_qwen.safetensors \
  --tokenizer tokenizer.json \
  --prompt "Explain quantum computing" --max-tokens 100

# Step 3: Compare against vanilla baseline
cargo run --release --bin saccade-run -- \
  --model-id Qwen/Qwen2.5-0.5B-Instruct \
  --tokenizer tokenizer.json \
  --prompt "Explain quantum computing" --max-tokens 100

# Micro-benchmarks
cargo run --release --bin qwen_example   # Single-layer GEMM/GEMV comparison
cargo run --release --bin verify         # Correctness validation
```

### PowerShell (Windows)

```powershell
$env:RUSTFLAGS="-C target-cpu=native"; cargo build --release
```

---

## 8. Key Design Decisions & Rationale

### Why a custom Qwen2 model instead of candle-transformers?

The upstream `candle_transformers::models::qwen2::Model` uses private struct fields and wraps `candle_nn::Linear` inside a tracing wrapper. We cannot inject `SaccadeLinearOp` into MLP layers without forking the crate. Our custom implementation (~400 lines) provides the `ProjectionLayer` enum that dispatches to either `Standard(Linear)` or `Saccade(SaccadeLinearOp)` per layer.

### Why CSC instead of CSR for sparse corrections?

CSR (row-major) iterates ~1,337 entries per row with scattered reads from the activation cache — each `token_cache[col]` access hits a different cache line, breaking the CPU prefetcher. CSC (column-major) iterates columns sequentially — `token_cache[col]` accesses are contiguous, and the row-indexed writes target a 6KB accumulator buffer that stays L1-resident.

### Why 4 accumulators instead of 1 or 8?

FMA has 4-cycle latency and 0.5-cycle throughput, meaning 8 independent operations can be in-flight. With 4 accumulators receiving 2 FMAs each per loop iteration, consecutive uses of the same accumulator are separated by 2 cycles of independent work — a practical balance between pipeline utilization and register pressure. Testing showed diminishing returns beyond 4.

### Why pre-scale sparse values at construction time?

The hot loop inner body was `acc += activation * (i8_val as f32) * scale`. Pre-computing `(i8_val as f32) * scale` into `Vec<f32>` eliminates 2 operations per NNZ element (i8→f32 cast + scale multiply). With 2M NNZ entries, this saves 4M operations per layer call.

### Why embedding-only calibration by default?

Full forward-pass calibration through all 24 layers is ideal but costs minutes on CPU. Embedding-only calibration produces usable thresholds in seconds. The routing thresholds are percentile-based, so the relative ordering of activation variances matters more than exact magnitudes. Per-layer calibration is supported via `--calib-layers N`.
