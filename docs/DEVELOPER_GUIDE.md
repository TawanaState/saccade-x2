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

```rust
use saccade_core::SaccadeConfig;

let config = SaccadeConfig {
    t4: 2.0, // Variance threshold to trigger sparse 8-bit updates
    t8: 8.0, // Variance threshold to trigger dense FP16 updates
    block_size: 16,
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

// Evaluates register-level math using Rayon multi-threading
let output_matrix = incoming_activations.apply_op1_no_bwd(&saccade_plugin)?;
```

The returned tensor naturally merges with the next network graph layer, allowing standard model continuation.

---

## Running the End-to-End Example

We have shipped a self-contained mock simulating this entire environment pipeline. It compresses an isolated linear shape simulating a standard Qwen intermediate MLP mapping, generates varied execution tokens, and executes the sequence dynamically.

**Execute:**
```bash
cargo run --bin qwen_example
```
