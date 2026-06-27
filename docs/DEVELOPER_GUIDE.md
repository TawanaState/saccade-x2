# Saccade-Candle Developer Guide

Welcome to the developer guide for Saccade V3, a token-adaptive matrix compression engine built on Hugging Face's Candle framework. This guide covers the full pipeline: from compressing a standard HuggingFace model to running streaming inference with real-time telemetry.

---

## The Objective

Saccade executes matrix multiplications mathematically equivalent to $Y = X \cdot W^T$ while reducing memory footprint and improving inference throughput on CPU hardware. Instead of computing in dense FP16, it processes tightly-packed 4-bit base weights with sparse INT8 corrections (CSC format), dynamically adjusting precision per token based on activation complexity.

### Proven Results (Qwen2.5-0.5B-Instruct, 24 layers)

| Metric | Vanilla FP16 | Saccade C-TARQ |
|--------|-------------|----------------|
| Decode speed | 5.8 tok/s | **7.4 tok/s (1.28x faster)** |
| Memory footprint | 1264.81 MB | **718.27 MB (1.76x smaller)** |
| Precision budget | 16.00 BPT | ~5.19 BPT |

---

## Quick Start

### Prerequisites

- Rust toolchain (edition 2021+)
- A calibration text file (any `.txt` with representative text)
- A tokenizer.json for your target model (download from HuggingFace)

### 1. Compile a Model

```bash
cargo run --release --bin saccade-compile -- \
  --model-id Qwen/Qwen2.5-0.5B-Instruct \
  --calib-file calibration.txt \
  --output-path saccade_qwen.safetensors \
  --tokenizer tokenizer.json
```

This downloads the model from HF Hub, compresses all 72 MLP projections (24 layers × gate/up/down), and outputs a unified safetensors archive.

### 2. Run Inference (Saccade)

```bash
cargo run --release --bin saccade-run -- \
  --checkpoint saccade_qwen.safetensors \
  --tokenizer tokenizer.json \
  --prompt "Explain how prime numbers work." \
  --max-tokens 100
```

### 3. Compare Against Vanilla Baseline

```bash
cargo run --release --bin saccade-run -- \
  --model-id Qwen/Qwen2.5-0.5B-Instruct \
  --tokenizer tokenizer.json \
  --prompt "Explain how prime numbers work." \
  --max-tokens 100
```

### PowerShell (Windows) — Enable Native SIMD

```powershell
$env:RUSTFLAGS="-C target-cpu=native"
cargo build --release
```

---

## Architecture Overview

```
┌────────────────────────────────────────────────────────────────┐
│                    SACCADE TOOLKIT PIPELINE                     │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  [HF Model] ──► saccade-compile ──► [Saccade Checkpoint]      │
│                      ▲                       │                 │
│               [calibration.txt]              ▼                 │
│                                        saccade-run             │
│  [User Prompt] ──────────────────────►       │                 │
│                                              ▼                 │
│                                     Terminal Stream            │
│                                     + Telemetry Dashboard      │
└────────────────────────────────────────────────────────────────┘
```

---

## 1. The Compression Pipeline (`saccade-compile`)

### What It Does

For each MLP linear projection (gate_proj, up_proj, down_proj) across all transformer layers:

1. **4-bit base quantization:** Each row's weights are symmetrically quantized to signed 4-bit integers ([-8, +7]) using row-wise max-abs scaling. Eight 4-bit values pack into one `u32`.

2. **Sparse delta extraction:** Reconstruction errors exceeding a percentile-based threshold are stored as INT8 values in Compressed Sparse Column (CSC) format. The threshold is computed per-layer from the actual error distribution to guarantee a target fill rate (default 15%).

3. **Routing threshold calibration:** Token activation variance is profiled using the calibration text. Percentile-based thresholds (t4, t8) are embedded into the checkpoint to drive runtime precision routing.

### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model-id` | required | HuggingFace repository (e.g., `Qwen/Qwen2.5-0.5B-Instruct`) |
| `--calib-file` | required | Plain-text file for calibration profiling |
| `--output-path` | `saccade_model.safetensors` | Output checkpoint path |
| `--tokenizer` | auto-download | Path to tokenizer.json |
| `--target-fill` | `0.15` | Fraction of weights receiving sparse corrections |
| `--pct-t4` | `0.80` | Percentile for medium-volatility routing threshold |
| `--pct-t8` | `0.95` | Percentile for high-volatility routing threshold |

### Output Format

The checkpoint is a standard HuggingFace safetensors file containing:

**Compressed MLP layers** (per layer per projection):
- `model.layers.{i}.mlp.{proj}.saccade_packed_base` — u32 packed 4-bit weights
- `model.layers.{i}.mlp.{proj}.saccade_scale_base` — f16 row-wise scales
- `model.layers.{i}.mlp.{proj}.saccade_delta_*` — CSR sparse corrections

**Uncompressed layers** (kept as-is):
- Attention projections (q/k/v/o), layer norms, embeddings, lm_head

**Routing metadata:**
- `model.layers.{i}.saccade_t4` / `saccade_t8` — per-layer routing thresholds

---

## 2. The Runtime Engine (`saccade-run`)

### Dual-Mode Architecture

**Saccade mode** (`--checkpoint`): Loads compressed safetensors, constructs `Qwen2Model` with `ProjectionLayer::Saccade` for MLP layers. Attention layers remain standard `candle_nn::Linear`.

**Vanilla mode** (`--model-id`): Downloads uncompressed model, constructs the same `Qwen2Model` with `ProjectionLayer::Standard` for all layers. This ensures a fair comparison — same model code, same generation loop, only the MLP kernel differs.

### CLI Options

| Flag | Description |
|------|-------------|
| `--checkpoint <path>` | Saccade mode: path to compiled safetensors |
| `--model-id <repo>` | Vanilla mode: HF model repository |
| `--prompt <text>` | Input prompt |
| `--tokenizer <path>` | Path to tokenizer.json |
| `--max-tokens <N>` | Maximum tokens to generate (default: 100) |
| `--temperature <f>` | Sampling temperature (default: 0.7, 0 = greedy) |
| `--top-p <f>` | Nucleus sampling threshold |
| `--seed <N>` | Random seed (default: 42) |

### Telemetry Output

```
================================================================
           SACCADE PERFORMANCE AUDIT TELEMETRY LOG
================================================================
Execution Mode:          Saccade C-TARQ Adaptive
Total Tokens Decoded:    50
----------------------------------------------------------------
Prefill Latency:         1134.0 ms
Decode Latency:          135.45 ms/token
Generation Speed:        7.4 tokens/second
Weight Memory Footprint: 718.27 MB
================================================================
```

---

## 3. Custom Heuristics

The routing system supports pluggable complexity metrics via function pointers:

```rust
use saccade_core::{SaccadeConfig, variance_heuristic};

// Built-in: statistical variance (recommended)
let config = SaccadeConfig {
    t4: 0.000252,
    t8: 0.000341,
    block_size: 16,
    heuristic: variance_heuristic,
};

// Custom: route by absolute maximum spike
fn max_activation_heuristic(tokens: &[half::f16]) -> f32 {
    tokens.iter().map(|t| t.to_f32().abs()).fold(0.0f32, f32::max)
}
```

---

## 4. The Execution Kernel

The `SaccadeLinearOp` implements Candle's `CustomOp1` trait with a three-phase kernel:

**Phase 1 (Base):** Rayon-parallel 4-bit dot product with 4 pipelined FMA accumulators. Each u32 contains 8 nibble-packed weights that are extracted with constant shifts, multiplied against pre-cached f32 activations, and accumulated into independent accumulators to break FMA serial dependency chains.

**Phase 2 (Sparse):** Sequential CSC column-sweep. For tokens exceeding the routing threshold, sparse INT8 corrections are applied via column-sequential iteration — contiguous activation reads, L1-hot accumulator writes.

**Phase 3 (Convert):** f32 accumulator to f16 output.

All kernel data (packed weights, scales, CSC arrays) is pre-extracted from Tensors at `SaccadeLinearOp::new()` construction time, eliminating per-forward-call overhead.

---

## 5. Micro-Benchmarks

### Single-Layer GEMM/GEMV Comparison

```bash
cargo run --release --bin qwen_example
```

Targets `model.layers.0.mlp.down_proj` on Qwen2-1.5B-Instruct with GEMM (batch=10, prefill) and GEMV (batch=1, autoregressive) benchmarks.

### Correctness Validation

```bash
cargo run --release --bin verify
```

Constructs a mock layer, serializes/loads via safetensors, and verifies that low-variance tokens use base-only computation while high-variance tokens trigger sparse corrections.

---

## 6. Project Structure

| File | Purpose |
|------|---------|
| `saccade-core/src/config.rs` | Core types: `SaccadeLinearOp`, `KernelCache`, `CachedCsc` |
| `saccade-core/src/op.rs` | `CustomOp1` impl: 3-phase pipelined kernel |
| `saccade-core/src/compress.rs` | 4-bit quantization + sparse delta extraction |
| `saccade-core/src/engine.rs` | `SaccadeEngine::compile_model_topology` |
| `saccade-core/src/calibration.rs` | `ProfileRunner::calibrate` — percentile threshold extraction |
| `saccade-core/src/heuristics.rs` | `variance_heuristic`, `l2_norm_heuristic` |
| `saccade-runner/src/model.rs` | Qwen2 transformer with `ProjectionLayer` dual-mode |
| `saccade-runner/src/bin/compile.rs` | `saccade-compile` CLI |
| `saccade-runner/src/bin/run.rs` | `saccade-run` CLI |
| `saccade-runner/src/bin/qwen_example.rs` | GEMM/GEMV micro-benchmark |
| `saccade-runner/src/bin/verify.rs` | Correctness validation |
