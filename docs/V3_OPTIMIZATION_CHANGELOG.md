# Saccade V3 Engine Optimization & Benchmarking Changelog

This document details the architectural refactoring applied to the Saccade C-TARQ runtime engine, the construction of the comparative benchmarking harness, and the graduation to a production CLI toolkit.

---

## Problem Statement

The V3 engine compiled and executed without panics, but exhibited a **flat-throughput simulation illusion** and critical evaluation gaps:

1. **Rayon thread-pool coordination trap** — fork-joining across matrix rows inside a sequential token loop overwhelmed CPU scheduling.
2. **Scalar nibble unpacking** — nested loops blocked compiler auto-vectorization (SIMD).
3. **Silent delta drop** — `sparse_delta_fp16` hardcoded to `None`; high-volatility tokens silently skipped corrections.
4. **Redundant f16→f32 conversions** — 43M scalar type conversions per batch where 48K sufficed.
5. **Single-threaded kernel** — lost multi-core utilization; vanilla baseline used all cores.
6. **4.03 BPT illusion** — RMS-based delta threshold produced 99.6% sparsity, running as an uncorrected 4-bit baseline.
7. **Single-layer evaluation** — benchmarks targeted only `model.layers.0.mlp.down_proj`.

---

## Optimization History

### Stage 1: Core Kernel Fixes
- Replaced scalar nibble loop with 8-lane unrolled extraction
- Unified Q8/FP16 delta fallback
- Fixed threshold extraction (rank-0 vs rank-1 tensor handling)
- Replaced borrowed CSR guards with owned Vec buffers

### Stage 2: HPC Kernel Optimizations
- Upfront f16→f32 cache conversion (896x fewer type conversions)
- Factored row-scale multiplication (4,864x fewer FP muls)
- Unchecked pointer access for SIMD vectorization

### Stage 3: Parallelism & Evaluation Methodology
- Re-introduced coarse-grained Rayon row parallelism with adaptive threshold
- Added GEMM (batch=10) and GEMV (batch=1) dual benchmarks
- Upgraded to Qwen2-1.5B-Instruct for larger weight matrices

### Stage 4: Production Toolkit (Current)
- **Percentile-based delta threshold** — ensures consistent ~15% fill rate across model sizes
- **Full-model compression** — all MLP layers (gate_proj, up_proj, down_proj × 24 layers)
- **saccade-compile CLI** — end-to-end compilation from HF model + calibration text
- **saccade-run CLI** — streaming inference with vanilla/Saccade dual mode + telemetry
- **Custom Qwen2 model** — full transformer with ProjectionLayer dual-mode for layer interception

---

## Production Toolkit Architecture

### saccade-compile

Compresses a HuggingFace model into Saccade C-TARQ format:

```bash
saccade-compile \
  --model-id Qwen/Qwen2.5-0.5B-Instruct \
  --calib-file calibration.txt \
  --output-path saccade_model.safetensors \
  --target-fill 0.15
```

**Pipeline:**
1. Download model + tokenizer from HF Hub
2. Parse config.json for architecture parameters
3. Tokenize calibration text, run hybrid calibration (first N layers)
4. Extract t4/t8 routing thresholds via ProfileRunner
5. Compress all 72 MLP projections (24 layers × 3 projections) with per-layer percentile thresholds
6. Serialize unified safetensors with compressed + uncompressed tensors

### saccade-run

Streaming inference with telemetry:

```bash
# Saccade mode
saccade-run --checkpoint saccade_model.safetensors --prompt "Explain quantum computing" --max-tokens 100

# Vanilla baseline mode
saccade-run --model-id Qwen/Qwen2.5-0.5B-Instruct --prompt "Explain quantum computing" --max-tokens 100
```

**Features:**
- Real-time token streaming to stdout
- Qwen2 chat template wrapping
- Configurable temperature, top-p, seed
- Performance telemetry dashboard at completion

---

## Key Technical Decisions

### Percentile-Based Delta Threshold

Replaced the RMS-based threshold (`rms_weight * 0.6` → 99.6% sparsity) with a percentile-based approach:

```rust
pub fn compute_percentile_threshold(tensor, target_fill_pct) -> f32 {
    // 1. Quantize all elements to 4-bit, compute reconstruction errors
    // 2. Sort absolute errors
    // 3. Return error at (1.0 - target_fill_pct) percentile
}
```

With `target_fill_pct = 0.15`, this ensures exactly 15% of weight elements receive sparse delta corrections, producing BPT in the 5.11–5.29 range regardless of model size or weight distribution.

### ProjectionLayer Abstraction

The custom Qwen2 model uses an enum to dispatch MLP operations:

```rust
pub enum ProjectionLayer {
    Standard(candle_nn::Linear),   // Vanilla FP16
    Saccade(SaccadeLinearOp),      // 4-bit packed + sparse CSR
}
```

Attention layers always use `Standard` (precision-sensitive). MLP layers use `Saccade` when loaded from a compressed checkpoint.

### Adaptive Parallelism

The `cpu_fwd` kernel selects sequential or parallel execution based on matrix size:

```rust
if out_features * packed_per_row >= 65536 {
    out_slice.par_iter_mut().enumerate().for_each(compute_row);
} else {
    out_slice.iter_mut().enumerate().for_each(compute_row);
}
```

---

## Build & Run

```bash
# Build all binaries with native SIMD
RUSTFLAGS="-C target-cpu=native" cargo build --release

# Compile a model
cargo run --release --bin saccade-compile -- \
  --model-id Qwen/Qwen2.5-0.5B-Instruct \
  --calib-file calibration.txt \
  --output-path saccade_qwen.safetensors

# Run inference (Saccade)
cargo run --release --bin saccade-run -- \
  --checkpoint saccade_qwen.safetensors \
  --prompt "Hello, world" --max-tokens 50

# Run inference (vanilla baseline)
cargo run --release --bin saccade-run -- \
  --model-id Qwen/Qwen2.5-0.5B-Instruct \
  --prompt "Hello, world" --max-tokens 50

# Existing benchmarks and verification
cargo run --release --bin qwen_example
cargo run --release --bin verify
```
