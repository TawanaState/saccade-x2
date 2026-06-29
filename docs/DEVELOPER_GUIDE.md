# Saccade V4: Developer & Integration Guide 🚀

Welcome to the engineering guide for **Saccade V4**, a high-performance, token-adaptive matrix compression engine extending Hugging Face's Candle framework. 

Saccade accelerates and compresses matrix operations by running a packed 4-bit integer base matrix on predictable tokens and dynamically applying coordinate-masked sparse INT8 corrections (**C-TARQ** - *Causal Token-Adaptive Residual Quantization*) strictly when token activation variance indicates that higher precision is required.

---

## 1. Architectural History & Crate Version Decisions

### Upgrade to Candle `v0.11.0`
Saccade V4 is built on top of the latest **Candle `v0.11.0`** release. This upgrade enables universal multi-model support, allowing Saccade to seamlessly load, quantize, and execute new model architectures directly from HuggingFace without maintaining custom forked structures. 

During this migration, we verified the following core compiler and system realities:

1. **`CustomOp1` Trait Availability**:
   The `CustomOp1` (unary), `CustomOp2` (binary), and `CustomOp3` (ternary) traits remain fully exported in `candle-core` alongside their new in-place counterparts (`InplaceOp1`, etc.). Saccade V4's execution kernel implements `CustomOp1` with zero modification to its forward execution loop.

2. **GGML Isolation**:
   Candle `v0.11.0` implements static quantization (such as GGUF and GGML formats) inside the isolated `QTensor` container and the `candle_transformers::quantized_nn` submodule. Because Saccade intercepts standard float projections (`candle_nn::Linear`) at the API boundary, it never collides with or depends on internal static GGML paths, preserving Saccade's adaptive token-routing as a pure competitive differentiator.

3. **SIMD Instruction Scheduling**:
   LLVM allocates vector registers (AVX2/AVX-512) and schedules pipeline resources directly from our raw pointer arithmetic within `saccade-core/src/op.rs`. Framework upgrades modify the memory orchestrator wrappers but leave our optimized mathematical kernels executing at full native CPU speed.

---

## 2. Core V4 Feature Specifications

Saccade V4 extends V3 with developer-friendly diagnostics, global runtime control, lock-free telemetry, and automated calibration.

```
┌────────────────────────────────────────────────────────────────────────┐
│                        SACCADE V4 DIAGNOSTIC PIPELINE                  │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  [Model SafeTensors] ──► saccade-verify (Dual-Mode Verification)       │
│                                │                                       │
│                ┌───────────────┴───────────────┐                       │
│                ▼                               ▼                       │
│      [C-TARQ Enabled]                 [C-TARQ Bypassed]                │
│      - 3-Phase SIMD Unpacking         - Reconstructed Dense FP16       │
│      - Heuristic Routing              - Standard GEMM                  │
│                │                               │                       │
│                └───────────────┬───────────────┘                       │
│                                ▼                                       │
│                       Accuracy Audit Report                            │
│                       - Logit Cosine Similarity                        │
│                       - Logit RMSE                                     │
│                       - Real-Time BPT & Speedup                        │
└────────────────────────────────────────────────────────────────────────┘
```

### A. The Bypass Switch (`BYPASS_C_TARQ`)
Developers can disable C-TARQ dynamically at runtime to run unquantized baseline comparisons.
* **Global Control**: Toggle programmatically using `saccade_core::set_bypass_c_tarq(bool)`.
* **Execution Bypass Path**: When active, Saccade bypasses the 3-phase kernel. Instead, it runs standard parallel matrix multiplication against a pre-dequantized dense weight matrix (`dequantized_weight_f32` containing base weights + sparse corrections), providing the exact mathematical baseline.
* **CLI Trigger**: Run `saccade-run` with the `--bypass` flag to evaluate baseline speed and output quality.

### B. Lock-Free Telemetry & Bits-Per-Token (BPT)
To measure performance without degrading runtime throughput, Saccade V4 features a lock-free telemetry register bank using **Atomics** and **Thread-Local Storage**:
* **Base Bits**: Accumulates `in_features * out_features * 4` per token.
* **Sparse Bits**: Accumulates `csc_non_zeros * 8` strictly for active tokens.
* **Parameter Calls**: Accumulates layer weights to calculate average model BPT.
* **Kernel Latency**: Tracks exact duration spent inside Saccade kernels.
* **Mechanism**: Aggregates metrics locally in thread-local storage and flushes them to global registers periodically (every 64 calls) to prevent CPU cache contention.

### C. Seamless calibration
The `saccade-compile` utility supports automatic dataset fetching and custom parameters:
* **Hugging Face Hub Ingestion**: If no local calibration file is provided, Saccade auto-downloads the validation split of the `wikitext` dataset (`wiki.valid.raw` from the hub).
* **Token Budget Control**: Configure exact calibration corpus token limits with `--calib-tokens <N>` (defaults to 512).

---

## 3. Quick Start Command Reference

### Step 1: Automated Model Compilation & Calibration
Compile a HuggingFace model, downloading the wikitext calibration dataset automatically and limiting the run to 256 profiling tokens:
```bash
cargo run --release --bin saccade-compile -- \
  --model-id Qwen/Qwen2.5-0.5B-Instruct \
  --dataset wikitext \
  --calib-tokens 256 \
  --output-path saccade_qwen.safetensors
```

### Step 2: Running Inference with Telemetry
Run inference on the compiled checkpoint using C-TARQ adaptive routing:
```bash
cargo run --release --bin saccade-run -- \
  --checkpoint saccade_qwen.safetensors \
  --prompt "Explain quantum computing in simple terms." \
  --max-tokens 50
```

Run the same prompt in **Bypass Mode** to compare output quality and latency:
```bash
cargo run --release --bin saccade-run -- \
  --checkpoint saccade_qwen.safetensors \
  --prompt "Explain quantum computing in simple terms." \
  --max-tokens 50 \
  --bypass
```

### Step 3: Running the Automated Verification Suite
Run side-by-side correctness auditing on the compiled checkpoint. This runs a prompt twice (with and without C-TARQ), matching generated tokens and calculating logit similarity:
```bash
cargo run --release --bin verify -- \
  --checkpoint saccade_qwen.safetensors \
  --max-tokens 30
```

---

## 4. Verification Output Diagnostics

Below is an example output of the V4 automated audit report:
```
================================================================
            SACCADE SYSTEM VERIFICATION REPORT
================================================================
Checkpoint Evaluated:    "saccade_qwen.safetensors"
Reference Baseline:      Vanilla FP16 Dequantized (Bypass)
Number of Steps Run:     30 steps
----------------------------------------------------------------
NUMERICAL ACCURACY METRICS:
  Avg Logit Cosine Similarity: 0.999992 (Target: >0.998)
  Avg Logit RMSE:              0.002862 (Target: <0.005)
----------------------------------------------------------------
PERFORMANCE AND QUANTIZATION AUDIT:
  C-TARQ End-to-End Latency:   4118.05 ms (7.28 tokens/sec)
  Bypass End-to-End Latency:   13567.92 ms (2.21 tokens/sec)
  Saccade C-TARQ BPT Budget:   5.19 BPT
  Dequantized Bypass BPT:      16.00 BPT
  Kernel Compute Speedup:      4.06x
================================================================
Status: VERIFICATION SUCCESSFUL (Accuracy bounds maintained)
================================================================
```

---

## 5. Directory & Module Reference

| Crate / File | Module | Role / Updates in V4 |
|---|---|---|
| **`saccade-core`** | | Core dynamic custom operations and mathematical helpers. |
| ↳ [`config.rs`](file:///C:/Users/user/Desktop/WORK/RESEARCH/saccade-x2/saccade-core/src/config.rs) | `config` | Holds `KernelCache` and exposes `set_bypass_c_tarq` / `is_c_tarq_bypassed`. |
| ↳ [`op.rs`](file:///C:/Users/user/Desktop/WORK/RESEARCH/saccade-x2/saccade-core/src/op.rs) | `op` | Custom `cpu_fwd` execution loop containing the fast parallel GEMM bypass path. |
| ↳ [`telemetry.rs`](file:///C:/Users/user/Desktop/WORK/RESEARCH/saccade-x2/saccade-core/src/telemetry.rs) | `telemetry` | Lock-free, thread-local register aggregates for BPT and kernel duration. |
| **`saccade-runner`** | | Running executables, CLI layers, and model representations. |
| ↳ [`model.rs`](file:///C:/Users/user/Desktop/WORK/RESEARCH/saccade-x2/saccade-runner/src/model.rs) | `model` | Qwen2 transformer architecture loading both standard and Saccade projections. |
| ↳ [`compile.rs`](file:///C:/Users/user/Desktop/WORK/RESEARCH/saccade-x2/saccade-runner/src/bin/compile.rs) | CLI binary | Quantization tool supporting wikitext auto-download and token limits. |
| ↳ [`run.rs`](file:///C:/Users/user/Desktop/WORK/RESEARCH/saccade-x2/saccade-runner/src/bin/run.rs) | CLI binary | Text generator streaming assistants, supporting `--bypass` and telemetry reporting. |
| ↳ [`verify.rs`](file:///C:/Users/user/Desktop/WORK/RESEARCH/saccade-x2/saccade-runner/src/bin/verify.rs) | CLI binary | Dual-mode accuracy audit comparing logit outputs and calculating speedup. |
| ↳ [`api.rs`](file:///C:/Users/user/Desktop/WORK/RESEARCH/saccade-x2/saccade-runner/src/api.rs) | `api` | Developer API exposing bypass control, telemetry logs, and custom model compilation helpers. |

---

## 6. Developer API Reference

Saccade V4 provides a clean, unified programmatic API inside the `saccade-runner` crate for integration with benchmarking tools, custom runners, or evaluation suites.

### A. API Overview
Access the API by importing:
```rust
use saccade_runner::{SaccadeModelApi, SaccadeMetrics};
```

### B. API Methods

#### `SaccadeModelApi::set_bypass(enabled: bool)`
Toggles the global C-TARQ bypass switch.
* `true`: All linear projections execute standard matrix multiplications using reconstructed weights, bypassing token routing.
* `false`: Enables the adaptive C-TARQ 3-phase kernel.

#### `SaccadeModelApi::reset_telemetry()`
Resets the global thread-safe telemetry registry. Call before starting a benchmark or generation cycle to ensure clean readings.

#### `SaccadeModelApi::get_metrics() -> SaccadeMetrics`
Retrieves aggregated telemetry metrics from the runtime:
```rust
pub struct SaccadeMetrics {
    /// Average Bits Per Token (BPT) of Saccade projections
    pub average_bpt: f64,
    /// Total duration spent inside Saccade kernels (in milliseconds)
    pub kernel_ms: f64,
    /// Total count of layer-tokens evaluated
    pub layer_tokens_processed: u64,
}
```

#### `SaccadeModelApi::compile_tensors(...) -> Result<HashMap<String, Tensor>>`
Compiles an in-memory hashmap of standard dense weights into a Saccade C-TARQ quantized state map.
* **Arguments**:
  * `tensors`: Reference to `HashMap<String, Tensor>` loaded from standard model weights.
  * `target_layers`: Substrings of weights to target (e.g. `&["gate_proj", "up_proj"]`).
  * `calibration_activations`: 2D activation tensor used to calibrate thresholds.
  * `target_fill_rate`: Fraction of elements with sparse deltas (e.g. `0.15`).
  * `pct_t4` and `pct_t8`: Percentiles for routing thresholds (e.g. `0.80`, `0.95`).

