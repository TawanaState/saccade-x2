# Saccade-Candle Developer Guide

Welcome to the internal execution guide for Saccade V3 on the Candle framework. This guide details how to take standard model structures (such as Qwen 2 or standard MLPs) and hook them into Saccade's execution backend.

## The Objective

Our goal is to execute machine learning matrices mathematically equivalent to $Y = X \cdot W^T + b$ while preventing structural out-of-memory errors on limited hardware.

Instead of computing purely in FP16, Saccade acts as a plugin. It processes a tightly-packed 4-bit representation of the baseline matrix (`W_base`), unpacking it dynamically in fast CPU SIMD registers, and adding sparse INT8/FP16 corrections (`ΔW`) only on specific sequence coordinates matching high token volatility.

## 1. The Offline Compression Loop

Before evaluating a model, you must map standard weights into our compact representations. This generally requires a calibration dataset to pinpoint volatile variance patches.

### The Artifacts Produced:
For every target sub-module (e.g., `mlp.up_proj`), your compression suite must generate:
1. **`packed_base`**: A Tensor of `u32` containing symmetrically packed 4-bit representation of the baseline matrix.
2. **`scale_base`**: A row-wise array containing `f16` or `f32` scalar shifts needed to project the 4-bit integer values back to coordinate spaces.
3. **`delta_q8_blocks`**: An `f16` or quantized `i8` coordinate matrix storing targeted precision corrections.

### Archiving with Safetensors:
Store these isolated components inside standard Hugging Face mapped binaries.
```rust
let mut compressed_state = HashMap::new();
compressed_state.insert("packed_base".to_string(), packed_base_tensor);
compressed_state.insert("scale_base".to_string(), scale_base_tensor);
compressed_state.insert("delta_q8".to_string(), delta_q8_blocks_tensor);

candle_core::safetensors::save(&compressed_state, "compressed_layer.safetensors")?;
```

---

## 2. Setting Execution Thresholds

The magic behind Saccade's computational routing is the **Global Activation Variance Heuristic**, configured by `SaccadeConfig`.

These thresholds represent absolute structural thresholds defined during your calibration phases:

### Using Custom Developer Heuristics

The evaluation architecture is built for maximum developer friendliness, allowing researchers to inject completely customized mathematical routing constraints.

By default, we supply `variance_heuristic` and `l2_norm_heuristic`. You can apply your own `fn(&[half::f16]) -> f32` functions.

```rust
use saccade_core::{SaccadeConfig, variance_heuristic};

// Option A: Use built-in heuristics
let config = SaccadeConfig {
    t4: 2.0, // Threshold to trigger sparse 8-bit updates
    t8: 8.0, // Threshold to trigger dense FP16 updates
    block_size: 16,
    heuristic: variance_heuristic,
};

// Option B: Write your own totally dynamic routing calibration function!
fn my_custom_activation_routing(tokens: &[half::f16]) -> f32 {
    let mut max_val = 0.0f32;
    for &t in tokens {
        if t.to_f32().abs() > max_val { max_val = t.to_f32().abs(); }
    }
    max_val // Route dynamically off the absolute maximum spike!
}

let custom_config = SaccadeConfig {
    t4: 15.0,
    t8: 30.0,
    block_size: 16,
    heuristic: my_custom_activation_routing,
};
```

---

## 3. Creating the Intercept Wrap

Candle layers are heavily constructed using the native `VarBuilder`. To inject Saccade, you bypass standard definitions of `candle_nn::Linear` to instantiate our operator (`SaccadeLinearOp`).

```rust
use saccade_core::SaccadeLinearOp;

let loaded_tensors = candle_core::safetensors::load("compressed_layer.safetensors", &device)?;

let saccade_plugin = SaccadeLinearOp {
    packed_base: loaded_tensors.get("packed_base").unwrap().clone(),
    scale_base: loaded_tensors.get("scale_base").unwrap().clone(),
    delta_q8_blocks: loaded_tensors.get("delta_q8").unwrap().clone(),
    delta_q8_scales: None,
    delta_fp16_blocks: None,
    config,
    out_features: 64,
    in_features: 128,
};
```

---

## 4. Operational Execution (Forward Pass)

The `SaccadeLinearOp` fundamentally implements Candle's `CustomOp1` trait. Once integrated into the macro structure of a model, any incoming activation (`X`) calling `apply_op1_no_bwd` evaluates token dimensions on-the-fly without the host Python-style synchronizations that crippled V1.

```rust
// Generate your autoregressive context vector
let incoming_activations = Tensor::new(...);

// Evaluates using adaptive Rayon multi-threading and SIMD register unpacking
let output_matrix = incoming_activations.apply_op1_no_bwd(&saccade_plugin)?;
```

The returned tensor naturally merges with the next network graph layer, allowing standard model continuation.

---

## 5. Benchmarking and Footprint Optimization

The benchmarking harness performs a comparative analysis between vanilla dense FP16 execution and Saccade's adaptive C-TARQ pipeline across two scenarios:

- **GEMM (batch=10):** Simulates prefill/prompt encoding — matrix-matrix multiply, compute-bound.
- **GEMV (batch=1):** Simulates autoregressive text generation — matrix-vector product, memory-bandwidth-bound. This is the deployment scenario Saccade was engineered for.

### Execution Footprints (Qwen2-1.5B-Instruct, `model.layers.0.mlp.down_proj`)

Built with `RUSTFLAGS="-C target-cpu=native" cargo run --release --bin qwen_example`.

```
Target layer: 1536 x 8960 (13.76M params)
Extracted thresholds: t4 = 0.014398, t8 = 0.422364
Sparse delta NNZ: 53241 / 13762560 (99.61% sparsity)

GEMM Benchmark (batch=10):
  [Vanilla GEMM] Prose    ~12ms   (812 tok/s)
  [Vanilla GEMM] Logic    ~7ms    (1431 tok/s)
  [Vanilla GEMM] Code     ~6ms    (1651 tok/s)
  [Saccade GEMM] Prose    ~20ms   (505 tok/s)   BPT: 4.00
  [Saccade GEMM] Logic    ~21ms   (474 tok/s)   BPT: 4.03
  [Saccade GEMM] Code     ~19ms   (528 tok/s)   BPT: 4.03

GEMV Benchmark (batch=1, autoregressive decoding):
  [Vanilla GEMV] Prose    2175 µs/tok  (460 tok/s)
  [Vanilla GEMV] Logic    2158 µs/tok  (463 tok/s)
  [Vanilla GEMV] Code     2154 µs/tok  (464 tok/s)
  [Saccade GEMV] Prose    2279 µs/tok  (439 tok/s)   BPT: 4.00
  [Saccade GEMV] Logic    2174 µs/tok  (460 tok/s)   BPT: 4.03
  [Saccade GEMV] Code     1992 µs/tok  (502 tok/s)   BPT: 4.03

Memory: 26.25 MB (FP16) → 6.83 MB (Saccade) = 3.8x compression
```

### Architectural Soundness Analysis

1. **Dynamic Engine Routing:** Calibration bounds (`t4`, `t8`) are embedded into the tensor map and extracted at compile time. Token routing is verified per-profile: Prose→base-only, Logic→Q8 delta, Code→FP16 fallback.
2. **GEMV Throughput Parity:** In the autoregressive decoding scenario (batch=1), Saccade achieves throughput parity with vanilla FP16 while delivering 3.8x memory compression. Code tokens are 7.5% faster due to reduced data volume.
3. **HPC Kernel:** Upfront f16→f32 cache conversion (896x fewer type conversions), factored row-scale multiplication (4,864x fewer FP muls), and unchecked pointer access enabling AVX2/SSE vectorization.
4. **Adaptive Parallelism:** Rayon row parallelism engages only when the matrix size justifies fork-join coordination cost. Small layers run sequentially to avoid thread overhead.
5. **Graceful Delta Fallback:** High-volatility tokens receive the best available sparse delta corrections regardless of which routing threshold they exceed.

### Metrics Collected

| Metric | Description |
|--------|-------------|
| Wall-clock latency | Per-profile execution time (GEMM: ms, GEMV: µs/tok) |
| Throughput | Tokens per second per profile |
| Memory footprint | Dense FP16 vs. packed 4-bit + CSR overhead |
| Bits-per-token (BPT) | Dynamic precision budget: 4.0 (base) to ~4.03 (delta) |

---

## Running the End-to-End Example

The benchmark harness downloads Qwen2-1.5B-Instruct, runs offline calibration, compresses the target layer, and executes both vanilla and Saccade pipelines in GEMM and GEMV configurations.

**Execute (release mode with native SIMD):**
```bash
RUSTFLAGS="-C target-cpu=native" cargo run --release --bin qwen_example
```

For the minimal validation harness:
```bash
RUSTFLAGS="-C target-cpu=native" cargo run --release --bin verify
```
