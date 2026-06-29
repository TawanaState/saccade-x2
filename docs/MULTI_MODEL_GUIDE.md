# Saccade V4: Universal Multi-Model Execution & Compilation Guide 🌐

Saccade V4 introduces **Framework-Level Interception**. By modifying the standard `candle_nn::Linear` projection layer and its integration with `VarBuilder` directly in a global crate fork, any neural network built on Hugging Face's Candle framework—such as Large Language Models (LLMs), Vision Transformers (ViTs), and Audio Diffusion architectures—can be compiled and run with C-TARQ acceleration without modifying the model's source code.

---

## 1. How Framework-Level Interception Works

The global dependency routing map is configured as follows:

```
┌────────────────────────────────────────────────────────┐
│                   YOUR RUNNER APPLICATION              │
└───────────────────────────┬────────────────────────────┘
                            │ (Uses official model crate)
                            ▼
┌────────────────────────────────────────────────────────┐
│            candle-transformers (Qwen / Whisper)        │
└───────────────────────────┬────────────────────────────┘
                            │ (Calls candle_nn::linear)
                            ▼
┌────────────────────────────────────────────────────────┐
│            candle-nn (Saccade Patched Crate)           │
├────────────────────────────────────────────────────────┤
│  Checks for: "{prefix}.saccade_packed_base"            │
│    ├─► YES: Instantiates Saccade Linear Backend        │
│    └─► NO:  Instantiates Standard Dense Backend        │
└───────────────────────────┬────────────────────────────┘
                            │ (Forwards matrix math)
                            ▼
┌────────────────────────────────────────────────────────┐
│    saccade-core (Custom INT4/INT8 SIMD Kernels)        │
└────────────────────────────────────────────────────────┘
```

> [!NOTE]
> Because Saccade intercepts weight loading inside `candle_nn::linear`, model loaders automatically instantiate Saccade projection nodes when loading Saccade-compiled Safetensors archives.

---

## 2. Generic API Reference

The developer API is exposed via the `saccade-runner` crate:
```rust
use saccade_runner::{SaccadeModelApi, SaccadeMetrics};
```

### Telemetry Logs & Bypass Control
```rust
// Toggles the dynamic bypass switch programmatically
SaccadeModelApi::set_bypass(false); // false = C-TARQ, true = Bypass GEMM

// Resets telemetry registers before starting a task
SaccadeModelApi::reset_telemetry();

// Retrieve BPT and kernel compute overhead
let metrics = SaccadeModelApi::get_metrics();
println!("Performance: {:.2} BPT, {:.2} ms in kernels", metrics.average_bpt, metrics.kernel_ms);
```

---

## 3. Step-by-Step Multi-Model Support

To run any model on Saccade:

```
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│ 1. Download Model│ ───► │ 2. Profile/Calib│ ───► │  3. Save & Run  │
│  - hf_hub weights│      │  - compile_tensors│      │  - load via VB  │
└─────────────────┘      └─────────────────┘      └─────────────────┘
```

### Step 1: Model Compilation
Load standard model weights into memory and run them through `SaccadeModelApi::compile_tensors`:
```rust
let tensors = candle_core::safetensors::load("original_weights.safetensors", &device)?;
let target_layers = vec!["gate_proj", "up_proj", "down_proj", "fc1", "fc2"];

let compiled = SaccadeModelApi::compile_tensors(
    &tensors,
    &target_layers,
    &calibration_activations, // (num_tokens, hidden_dim)
    0.15, // 15% sparse correction budget
    0.80, // t4 routing percentile
    0.95, // t8 routing percentile
)?;
candle_core::safetensors::save(&compiled, "saccade_checkpoint.safetensors")?;
```

### Step 2: Load and Run
Initialize the model using the standard Candle builder pointing to the compiled Saccade archive:
```rust
let vb = unsafe {
    VarBuilder::from_mmaped_safetensors(&["saccade_checkpoint.safetensors"], DType::F16, &device)?
};
let mut model = Model::new(&config, vb)?;
```

---

## 4. Specific Model Examples

### A. Qwen-2 / Qwen-2.5 (LLM)
For large language models, Saccade accelerates the large MLP down-projections and gate-projections.
* **Target Layers**: `["gate_proj", "up_proj", "down_proj"]`
* **Run Compilation & Benchmarking**:
  ```bash
  cargo run --release --bin qwen_example
  ```

### B. Whisper (Audio Recognition)
For audio-to-text models like OpenAI's Whisper, Saccade optimizes the Feed-Forward Network (FFN) layers in the encoder and decoder.
* **Target Layers**: `["fc1", "fc2"]`
* **Run Compilation & Benchmarking**:
  ```bash
  cargo run --release --bin whisper_example
  ```

### C. Gemma (LLM)
For Google's Gemma models, the weight naming conventions are similar to Llama.
* **Target Layers**: `["gate_proj", "up_proj", "down_proj"]`
* **Compiling & Running Gemma**:
  Use the generic compile CLI:
  ```bash
  cargo run --release --bin saccade-compile -- \
    --model-id google/gemma-2b-it \
    --dataset wikitext \
    --calib-tokens 256 \
    --output-path gemma_saccade.safetensors
  ```
  Run text generation:
  ```bash
  cargo run --release --bin saccade-run -- \
    --checkpoint gemma_saccade.safetensors \
    --prompt "Write a short poem about space exploration."
  ```

---

## 5. Summary of Accuracy vs. Speed Targets

For any architecture, Saccade maintains high-fidelity execution boundaries:

| Model Architecture | Quantization Mode | Target Accuracy Bounds | Expected Kernel Speedup |
|---|---|---|---|
| **Qwen-2** (Text LLM) | C-TARQ (Q4 Base + 15% Delta) | Cosine Sim > `0.999` / RMSE < `0.003` | `4.0x - 6.5x` |
| **Whisper** (Speech-to-Text) | C-TARQ (Q4 Base + 15% Delta) | Cosine Sim > `0.999` / RMSE < `0.003` | `5.0x - 6.8x` |
| **Gemma** (Text LLM) | C-TARQ (Q4 Base + 15% Delta) | Cosine Sim > `0.999` / RMSE < `0.003` | `4.0x - 6.2x` |
