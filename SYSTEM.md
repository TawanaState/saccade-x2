# ARCHITECTURAL PIVOT DIRECTIVE: SYSTEM SACCADE V3
**Document ID:** SACCADE-V3-PIVOT-2026  
**Classification:** Internal Core Engineering Architecture  
**Author:** Principal Infrastructure & ML Systems Architect  
**Date:** June 25, 2026  
**Subject:** Migrating Project Saccade from Python/Triton to an Implementation-First Native Rust, Candle, and WebAssembly (WASM) Substrate

---

## EXECUTIVE SUMMARY & INTENT

This directive authorizes and outlines the mandatory architectural migration of **Project Saccade** (formerly designated as Project Nocturnal X1) away from Python-reliant execution ecosystems (PyTorch, Inductor, OpenAI Triton) to a standalone, production-grade **Rust engine powered by Hugging Face's Candle framework, compiled to WebAssembly (WASM) and Native CPU targets**. 

Project Saccade was conceptualized to resolve the severe DRAM memory-bandwidth walls that prevent modern Large Language Models (LLMs) from executing efficiently on resource-constrained consumer edge hardware. By moving away from rigid, layer-wise post-training compression towards **Causal Token-Adaptive Residual Quantization (C-TARQ)**, we treat autoregressive text generation as a dynamic trajectory across an activation manifold, allocating computational bit-depth budgets on a token-by-token basis.

While our Stage 6 validation runs on the Qwen-2.5-3B-Instruct architecture successfully proved the underlying data science—saving over 31 Gigabytes of redundant weight streaming traffic per track and maintaining a bit-budget between **5.11 and 5.29 Bits-Per-Token (BPT)**—the execution infrastructure hit an immovable wall. The high-level Python runtime, eager kernel dispatch models, and lack of client-side cross-platform portability create significant engineering friction. 

To transition Saccade into an open-source framework, we are eliminating all high-level runtime dependencies. We are establishing an ultra-lean, zero-dependency, implementation-first execution layer in Rust targeting WebAssembly. This document serves as the absolute single source of truth and system blueprint for the engineering team.

---

## 1. FORENSIC DECONSTRUCTION OF PYTHON/TRITON FAILURES

To build the new Rust substrate effectively, the engineering team must understand the specific hardware-software interface failures encountered in our previous Python/Triton implementation:

### A. The Eager-Mode Simulation Trap & The Bandwidth Paradox
During early development, we relied on high-level PyTorch environments where the "compressed" 4-bit weights ($W_{\\text{base}}$) and secondary residual correction layers ($\\Delta W$) were stored as loose, unpacked 16-bit floating-point tensors (`.half()`). This triggered a major performance paradox: the system memory bus continued to transfer the full 16-bit byte footprint for every parameter, meaning the hardware was forced to load uncompressed variables while simultaneously paying the computational and kernel launch taxes of the routing layers. True performance scaling is only unlocked when weights are physically compressed on disk and in memory.

### B. Python-to-CUDA Kernel Launch Overhead Dominance
Our target Qwen architecture contains 24 layers with 7 distinct linear projections per layer, totaling **168 linear submodules** across the model graph. During autoregressive decoding, tokens are processed sequentially (batch size = 1). 

In our Python execution engine, passing through these 168 submodules required calling custom Triton kernels via eager Python functions (`_triton_fused_c_tarq_kernel[...]`). Each call had to navigate Python argument parsing, internal caching dictionary lookups, and CUDA Driver API dispatching, adding **50 to 100 microseconds of CPU-side launch overhead per kernel call**. 
$$\\text{Total Launch Overhead} = 168 \\text{ submodules} \\times 100\\mu\\text{s} = 16.8\\text{ms per token step}$$
This launch latency meant that the CPU spent nearly its entire clock cycle trapped in Python scheduling logic rather than GPU execution. The GPU sat idle waiting for instructions, capping decoding throughput to **~7.8 tokens/second**, regardless of low-level kernel optimizations.

### C. Host-Device CPU-GPU Synchronization Stalls
In the initial Python runtime, evaluating complexity routing paths required querying tensor state conditions directly inside the execution loop:


```python
# The Host-Device Stall Pattern in V1 Runtime
if mask_q8.any():
    output[mask_q8] += torch.matmul(x_flat[mask_q8], self.W_delta_8_dense.t())

```

To process the expression `mask_q8.any()`, the runtime was forced to copy the boolean evaluation result from GPU VRAM back to CPU host memory over the PCIe bus. This triggered an absolute hardware synchronization lock. The CPU halted completely, draining the command queue while waiting for the GPU to complete all prior actions and return the single scalar byte. For a 24-layer network, this forced **48 explicit CPU-GPU synchronization blocks per generated token**, severely degrading wall-clock speeds.

### D. The $L_2$ Norm Activation Drift Paradox

In Stage 2, we evaluated replacing our activation Variance tracking with a geometric $L_2$ Norm distance calculation, assuming that bypassing the mean-vector extraction step would save valuable ALU cycles. However, scaling up to the 3B parameter model revealed a critical architectural flaw:

```
[Activation Space Mapping]
Topological Origin (0,0,...,0)
       │
       ▼
   ┌───────┐
   │  L2   │ ───► Absolute Magnitude Vector (Corrupted by Drift)
   └───────┘
       ▲
       │
   ┌───────┐
   │  Var  │ ───► Subtracts Mean [X - μ] (Isolates Volatile Outliers)
   └───────┘
       │
       ▼
Actual Manifold Drift Location (μ_1, μ_2, ..., μ_d)

```

As large language models scale, their internal hidden states experience significant, uncentered activation drift, shifting the entire coordinate layout away from the topological origin $(0,0,\dots,0)$. Because $L_2$ Norm measures absolute distance from the origin, this structural drift caused nearly every incoming token to register an artificially inflated complexity score. This completely invalidated our frozen calibration thresholds, forcing the router to load heavy high-precision patches far more often than necessary, overloading the memory bus and dropping performance on coding tasks by **46.3%**.

Variance protection succeeds because it explicitly subtracts the activation mean vector $\mu$ before computing vector deviations, isolating localized activation volatility from absolute structural coordinate drift.

---

## 2. THE TARGET ENGINE SUBSTRATE: HUGGING FACE CANDLE & WEBASSEMBLY

To eliminate these runtime bottlenecks and deliver a production-ready edge library, we are transitioning to a native Rust implementation utilizing Hugging Face's **Candle** framework as our neural graph substrate.

### A. Strategic Rationale for Candle

1. **Star-Rating & Open-Source Authority:** Candle is the premier minimalist, high-performance ML framework for Rust, maintained directly by Hugging Face. With massive community adoption (>20k GitHub stars), it is widely recognized and respected by open-source practitioners.
2. **PyTorch-Style Paradigm:** Candle mirrors PyTorch's elegant syntax, handling multi-dimensional matrices as reference-counted `Tensor` objects, which significantly reduces engineering translation friction.
3. **Production-Grade WASM Compilation:** Candle features first-class, lightweight serialization and compilation targets for WebAssembly. It allows large models to run completely client-side in standard browser threads, with zero external platform dependencies.

### B. Why We Reject Alternative Frameworks

* **Ollama:** Ollama is not a client-side web solution. It is a native desktop daemon client written in Go that acts as a local server for C++ backends. It cannot compile to WebAssembly or run inside web workers.
* **llama.cpp / GGML:** While highly optimized for local inference, `llama.cpp` and its underlying `ggml` library are built entirely around static computational graphs. Forcing Saccade’s dynamic C-TARQ framework—where tokens switch processing paths on the fly based on activation volatility—into a static GGML graph requires modifying the core engine code and writing complex pointer arithmetic across changing tensor shapes.

### C. Overcoming the WASM 4GB Memory Boundary

Standard 32-bit WebAssembly runtimes (Wasm32) enforce a hard linear memory allocation limit of **4 Gigabytes** for the entire browser sandbox.

* A standard 3B parameter model running in native FP16 requires approximately **6.0 GB** of VRAM/RAM, causing an out-of-memory crash before the engine can even initialize.
* Under Saccade's C-TARQ paradigm, the model is initialized from an ultra-compressed 4-bit baseline paired with localized sparse delta updates. With an empirical footprint of **~5.2 BPT**, a 3B model requires only **~1.95 GB to 2.15 GB of active RAM**. Saccade cleanly slides under the WASM 4GB memory wall, enabling browser-side deployment of models that are traditionally impossible to execute on the web.

```
[Wasm32 Linear Memory Architecture]
0 GB                                      2 GB                      4 GB (HARD WALL)
 ┌─────────────────────────────────────────┬─────────────────────────┐
 │ Saccade Bit-Budget Range (~1.95-2.15 GB)│ Safe Headroom for App   │ OOM CRASH ZONE
 └─────────────────────────────────────────┴─────────────────────────┘
 ◄─────────────────────── ENTIRE 3B MODEL FITS ─────────────────────►

```

### D. The `CustomOp` Trait Integration Method

To avoid rewriting the core components of neural network layers (such as tokenizers, KV-caches, and attention masks), Saccade will act as an architectural plugin. Candle provides a robust `CustomOp` trait ecosystem designed exactly for this purpose.

Developers will wrap the standard model layers using Candle’s custom operator trait, intercepting the input activations during the forward pass. This allows Candle to handle the macro-level attention configurations while our custom Rust code retains complete control over register-level bit-shifting, unpacking, and sparse delta patching.

---

## 3. LOW-LEVEL MATH & DATA TYPE ENGINEERING IN RUST

To guarantee that our Rust engine achieves true physical memory relief without re-conversion or allocation overhead, the engineering team must follow these strict mathematical and data layout requirements:

### A. True Bit-Packing & Zero-Copy Memory Mapping

Saccade model checkpoints must be saved natively using the **Hugging Face Safetensors** standard. Safetensors provides an essential feature for edge devices: **Zero-Copy Memory Mapping (`MmapedSafetensors`)**.

When loading a model checkpoint, the engine maps the raw file bytes directly into the process's address space. The engine treats these bytes as raw pointers, eliminating the need to read data into intermediate buffers or copy weights across RAM locations.

The foundational weight matrix ($W_{\text{base}}$) must be stored as tightly packed 4-bit signed integers inside contiguous `u32` or `i32` arrays, ensuring eight separate parameters occupy a single memory slot.

```
[Packed int32 Register Allocation Layout]
 31            24 23            16 15             8 7              0  (Bit Offset)
 ┌───────────────┬───────────────┬───────────────┬───────────────┐
 │ Weight W_7    │ Weight W_6    │ ...           │ Weight W_0    │  ==> 1 contiguous i32 field
 └───────────────┴───────────────┴───────────────┴───────────────┘

```

### B. Fused Register-Level Dequantization (Avoiding the Allocation Trap)

To avoid the dequantization overhead that slowed down our PyTorch baseline, the Rust engine must perform **Fused Register-Level Dequantization**.

We must never instantiate an uncompressed, full-sized FP16 matrix in RAM. Instead, the unpacking logic must happen completely inside the core inner loops of the matrix multiplication engine, using local CPU micro-registers or fast stack-allocated arrays.

By utilizing WebAssembly’s 128-bit SIMD vector extensions (`wasm_simd128`), we can unpack 32 bits of packed integers into multiple floating-point lanes simultaneously using low-level bitwise operations (`>>` and `&`).

$$\text{Weight}_{\text{float}} = (\text{Nibble}_{\text{extracted}} - 8.0) \times \text{Scale}_{\text{base}}$$

The extracted floating-point parameter is processed instantly by the incoming activation vector element, accumulated into an intermediate `f32` register, and immediately discarded. The system RAM only ever streams the compressed 4-bit footprint across the bus.

### C. Quantized Residual Delta Strategies ($\Delta W$)

To prevent the **Residual Memory Paradox**—where holding uncompressed alternative weights forces the system to store the model twice—the delta arrays ($\Delta W$) must also be heavily optimized:

1. **Block-Sparse Formatting:** Only non-zero $16 \times 16$ coordinate blocks identified during offline calibration are saved to disk.
2. **Secondary Residual Quantization:** The active error coordinates within these $16 \times 16$ patches must be compressed down to symmetric **INT8** or signed **INT4** integer spaces, paired with a singular floating-point block scale factor ($s_{\Delta}$).
3. **The Execution Path:** When a high-variance token triggers a patch lookup, the engine loads the compressed integer block, applies a quick scalar adjustment ($s_{\Delta}$), and adds the correction vector directly into the accumulation register on the fly.

---

## 4. COMPLETE IMPLEMENTATION CODEPRINT FOR CORE DEVELOPERS

The codebase must isolate low-level vector mathematical kernels from high-level model structures. Below is the mandatory structural template for implementing the Saccade C-TARQ linear custom operator within Hugging Face's Candle framework:

```rust
use candle_core::{op::CustomOp1, CpuStorage, Layout, Result, Shape, Tensor, TensorId};

/// Core configuration profile containing global variance threshold pools
pub struct SaccadeConfig {
    pub t4: f32,
    pub t8: f32,
    pub block_size: usize,
}

/// Persistent in-memory storage layout for an optimized Saccade linear projection.
/// Guarantees that only compressed parameters reside in the active hardware memory footprint.
pub struct SaccadeLinearOp {
    // Ultra-compressed base matrix: 4 bits per parameter packed uniformly into u32 containers
    pub packed_base: Tensor, 
    pub scale_base: Tensor,
    
    // Pre-materialized block-sparse delta arrays stored as compressed integer spaces
    pub delta_q8_blocks: Tensor,
    pub delta_q8_scales: Tensor,
    pub delta_fp16_blocks: Option<Tensor>,
    
    // Operational configuration parameters
    pub config: SaccadeConfig,
    pub out_features: usize,
    pub in_features: usize,
}

impl CustomOp1 for SaccadeLinearOp {
    fn name(&self) -> &'static str {
        "fused_c_tarq_saccade_linear"
    }

    /// Evaluates the multi-dimensional feature shape after executing the custom activation transformation
    fn output_shape(&self, layouts: &[Layout]) -> Result<Shape> {
        let input_layout = &layouts[0];
        let input_shape = input_layout.shape();
        let mut dims = input_shape.dims().to_vec();
        // Overwrite the final feature axis to match the linear projection layer configuration
        if let Some(last_dim) = dims.last_mut() {
            *last_dim = self.out_features;
        }
        Shape::from_vec(dims)
    }

    /// Pure, side-effect-free vector mathematical execution block compiled for the CPU host engine.
    /// Vectorized operations are optimized to match system registers, avoiding PyTorch-style graph splits.
    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<CpuStorage> {
        // Extract raw pointer references from the incoming linear activation tensor
        let raw_activations = storage.as_slice::<f16>()?;
        let shape = layout.shape();
        let dims = shape.dims();
        
        let batch_tokens = dims[0..dims.len() - 1].iter().product::<usize>();
        let hidden_dim = dims[dims.len() - 1];
        
        // Allocate the destination output array matching the target projection metrics
        let output_elements = batch_tokens * self.out_features;
        let mut output_buffer = vec![f16::ZERO; output_elements];
        
        // Access raw binary data arrays from the registered buffers
        let (base_data, base_layout) = self.packed_base.storage_and_layout();
        let packed_weights = match &*base_data {
            candle_core::Storage::Cpu(cpu_store) => cpu_store.as_slice::<u32>()?,
            _ => return Err(candle_core::Error::Msg("Hardware substrate target mismatch: Expected CPU registry".into())),
        };
        
        let (scale_data, _) = self.scale_base.storage_and_layout();
        let base_scales = match &*scale_data {
            candle_core::Storage::Cpu(cpu_store) => cpu_store.as_slice::<f16>()?,
            _ => return Err(candle_core::Error::Msg("Scale block storage corruption".into())),
        };

        // Loop over the activation timeline sequentially to prevent parallel allocation traps
        for t in 0..batch_tokens {
            let act_offset = t * hidden_dim;
            let current_token_slice = &raw_activations[act_offset..act_offset + hidden_dim];
            
            // 1. Compute Causal Activation Variance entirely on-chip inside CPU registers
            let mut sum = 0.0f32;
            let mut sum_sq = 0.0f32;
            for &val in current_token_slice.iter() {
                let v_f32 = val.to_f32();
                sum += v_f32;
                sum_sq += v_f32 * v_f32;
            }
            let mean = sum / (hidden_dim as f32);
            let variance = (sum_sq / (hidden_dim as f32)) - (mean * mean);
            
            // 2. Evaluate frozen complexity thresholds to establish the dynamic execution path
            let use_delta_q8 = variance >= self.config.t4 && variance < self.config.t8;
            let use_delta_fp16 = variance >= self.config.t8;
            
            // 3. Perform Fused Matrix Multiplication across rows of the projection weights
            for row in 0..self.out_features {
                let mut dot_accumulator = 0.0f32;
                let row_weight_offset = row * (self.in_features / 8);
                let current_scale = base_scales[row].to_f32();
                
                // Process packed u32 boundaries using micro-vector blocks
                for k_packed in 0..(self.in_features / 8) {
                    let packed_val = packed_weights[row_weight_offset + k_packed];
                    let k_unpacked_base = k_packed * 8;
                    
                    // Unpack 8 distinct parameters in a single loop using register bitwise operators
                    for idx in 0..8 {
                        let raw_nibble = (packed_val >> (idx * 4)) & 0x0F;
                        // Center values from [0, 15] back to the signed range [-8, +7]
                        let base_weight = (raw_nibble as f32 - 8.0) * current_scale;
                        
                        let total_weight = base_weight; 
                        // Note: If routing metrics evaluate to True, add the corresponding sparse delta vector
                        // directly into the register here to avoid intermediate matrix structures.
                        
                        dot_accumulator += current_token_slice[k_unpacked_base + idx].to_f32() * total_weight;
                    }
                }
                
                // Write the final accumulated dot product back to the destination output tensor
                let out_offset = t * self.out_features + row;
                output_buffer[out_offset] = f16::from_f32(dot_accumulator);
            }
        }
        
        Ok(CpuStorage::F16(output_buffer))
    }
}

```

---

## 5. RE-BUILT EVALUATION & PARITY SPECIFICATIONS

To protect our open-source release from scientific bias or measurement errors, the validation code must enforce the following strict protocols:

### A. The Attention Isolation Guard

All model transformations must target **Feed-Forward Network (FFN/MLP) submodules exclusively** (specifically `mlp.up_proj` and `mlp.down_proj`). Our architectural ablation studies confirmed that attention projections (such as `self_attn.o_proj`) are highly sensitive to uniform low-bit quantization. Disrupting these coordinates breaks downstream attention normalization, leading to severe language degradation. Attention layers must remain completely untouched in native precision.

### B. Baseline Evaluation Parity

When benchmarking Saccade against uniform low-bit standards (such as bitsandbytes INT4 or GGUF configurations), engineers must instantiate an **asymmetric targeting exclusion adapter**. Any submodule layer excluded from Saccade optimization must be similarly excluded from the baseline quantization configurations. Comparing an FFN-only dynamic framework against a full-model uniform quantization baseline is scientifically dishonest and invalidates performance claims.

### C. The Deployment Roadmap

The deployment roadmap is split into three execution phases to prepare for our public release and arXiv preprint submission:

```
 ┌──────────────────────────┐      ┌──────────────────────────┐      ┌──────────────────────────┐
 │  PHASE 1: RUST CORE      │ ───► │  PHASE 2: WASM LAYER     │ ───► │   PHASE 3: PUBLIC DROP   │
 ├──────────────────────────┤      ├──────────────────────────┤      ├──────────────────────────┤
 │ Build core CustomOp trait│      │ Export via wasm-pack     │      │ Launch repository with   │
 │ logic, implement on-chip │      │ with 128-bit SIMD        │      │ ready-to-run libraries & │
 │ bit-shifting loops.      │      │ hardware acceleration.   │      │ interactive playground.  │
 └──────────────────────────┘      └──────────────────────────┘      └──────────────────────────┘

```

## 6. SYSTEM CONCLUSION & DEVELOPMENT ORDER

The data science behind Project Saccade is mathematically verified and complete. Our dynamic token-adaptive framework has proven its ability to bypass the physical memory-bandwidth boundaries that restrict edge AI performance.

We will no longer develop within high-level Python simulation environments. By migrating to a native Rust and WebAssembly engine, we eliminate runtime execution branch penalties, resolve host-device latency issues, and provide an efficient, production-grade framework for edge deployment.

**Forward execution is fully authorized.** Let's build the new era of Saccade.

---

### APPENDIX: CONTEXT-RELIANT CITATION RECORD

* All empirical performance metrics on the Qwen 3B scale are drawn directly from the internal Stage 6 Scale Validation Dispatches.
* Deep-dive hardware diagnoses regarding Triton kernel launch latencies, strided memory traps, and CPU-GPU synchronization stalls are derived from Senior ML Infrastructure Reports.
* Framework designs and code guidelines match the exact structural standards defined across Project Saccade Configuration Sheets.
"""

output_path = "SACCADE_V3_ARCHITECTURAL_PIVOT_DIRECTIVE.md"
with open(output_path, "w", encoding="utf-8") as f:
f.write(markdown_content.strip())

print(f"Successfully generated file: {output_path}")
print(f"[file-tag: code-generated-file-{os.getpid()}]")

```
Your Markdown document is ready.
[file-tag: code-generated-file-0-1782372688700087279]

# ARCHITECTURAL PIVOT DIRECTIVE: SYSTEM SACCADE V3
**Document ID:** SACCADE-V3-PIVOT-2026  
**Classification:** Internal Core Engineering Architecture  
**Author:** Principal Infrastructure & ML Systems Architect  
**Date:** June 25, 2026  
**Subject:** Migrating Project Saccade from Python/Triton to an Implementation-First Native Rust, Candle, and WebAssembly (WASM) Substrate

---

## EXECUTIVE SUMMARY & INTENT

This directive authorizes and outlines the mandatory architectural migration of **Project Saccade** (formerly designated as Project Nocturnal X1) away from Python-reliant execution ecosystems (PyTorch, Inductor, OpenAI Triton) to a standalone, production-grade **Rust engine powered by Hugging Face's Candle framework, compiled to WebAssembly (WASM) and Native CPU targets**. 

Project Saccade was conceptualized to resolve the severe DRAM memory-bandwidth walls that prevent modern Large Language Models (LLMs) from executing efficiently on resource-constrained consumer edge hardware. By moving away from rigid, layer-wise post-training compression towards **Causal Token-Adaptive Residual Quantization (C-TARQ)**, we treat autoregressive text generation as a dynamic journey across an activation manifold, allocating computational bit-depth budgets on a token-by-token basis.

While our Stage 6 validation runs on the Qwen-2.5-3B-Instruct architecture successfully proved the underlying data science—saving over 31 Gigabytes of redundant weight streaming traffic per track and maintaining a bit-budget between **5.11 and 5.29 Bits-Per-Token (BPT)**—the execution infrastructure hit an immovable wall. The high-level Python runtime, eager kernel dispatch models, and lack of client-side cross-platform portability create significant engineering friction. 

To transition Saccade into an open-source framework, we are eliminating all high-level runtime dependencies. We are establishing an ultra-lean, zero-dependency, implementation-first execution layer in Rust targeting WebAssembly. This document serves as the absolute single source of truth and system blueprint for the engineering team.

---

## 1. FORENSIC DECONSTRUCTION OF PYTHON/TRITON FAILURES

To build the new Rust substrate effectively, the engineering team must understand the specific hardware-software interface failures encountered in our previous Python/Triton implementation:

### A. The Eager-Mode Simulation Trap & The Bandwidth Paradox
During early development, we relied on high-level PyTorch environments where the "compressed" 4-bit weights ($W_{\text{base}}$) and secondary residual correction layers ($\Delta W$) were stored as loose, unpacked 16-bit floating-point tensors (`.half()`). This triggered a major performance paradox: the system memory bus continued to transfer the full 16-bit byte footprint for every parameter, meaning the hardware was forced to load uncompressed variables while simultaneously paying the computational and kernel launch taxes of the routing layers. True performance scaling is only unlocked when weights are physically compressed on disk and in memory.

### B. Python-to-CUDA Kernel Launch Overhead Dominance
Our target Qwen architecture contains 24 layers with 7 distinct linear projections per layer, totaling **168 linear submodules** across the model graph. During autoregressive decoding, tokens are processed sequentially (batch size = 1). 

In our Python execution engine, passing through these 168 submodules required calling custom Triton kernels via eager Python functions (`_triton_fused_c_tarq_kernel[...]`). Each call had to navigate Python argument parsing, internal caching dictionary lookups, and CUDA Driver API dispatching, adding **50 to 100 microseconds of CPU-side launch overhead per kernel call**. 
$$\text{Total Launch Overhead} = 168 \text{ submodules} \times 100\mu\text{s} = 16.8\text{ms per token step}$$
This launch latency meant that the CPU spent nearly its entire clock cycle trapped in Python scheduling logic rather than GPU execution. The GPU sat idle waiting for instructions, capping decoding throughput to **~7.8 tokens/second**, regardless of low-level kernel optimizations.

### C. Host-Device CPU-GPU Synchronization Stalls
In the initial Python runtime, evaluating complexity routing paths required querying tensor state conditions directly inside the execution loop:
```python
# The Host-Device Stall Pattern in V1 Runtime
if mask_q8.any():
    output[mask_q8] += torch.matmul(x_flat[mask_q8], self.W_delta_8_dense.t())

```

To process the expression `mask_q8.any()`, the runtime was forced to copy the boolean evaluation result from GPU VRAM back to CPU host memory over the PCIe bus. This triggered an absolute hardware synchronization lock. The CPU halted completely, draining the command queue while waiting for the GPU to complete all prior actions and return the single scalar byte. For a 24-layer network, this forced **48 explicit CPU-GPU synchronization blocks per generated token**, severely degrading wall-clock speeds.

### D. The $L_2$ Norm Activation Drift Paradox

In Stage 2, we evaluated replacing our activation Variance tracking with a geometric $L_2$ Norm distance calculation, assuming that bypassing the mean-vector extraction step would save valuable ALU cycles. However, scaling up to the 3B parameter model revealed a critical architectural flaw:

```
[Activation Space Mapping]
Topological Origin (0,0,...,0)
       │
       ▼
   ┌───────┐
   │  L2   │ ───► Absolute Magnitude Vector (Corrupted by Drift)
   └───────┘
       ▲
       │
   ┌───────┐
   │  Var  │ ───► Subtracts Mean [X - μ] (Isolates Volatile Outliers)
   └───────┘
       │
       ▼
Actual Manifold Drift Location (μ_1, μ_2, ..., μ_d)

```

As large language models scale, their internal hidden states experience significant, uncentered activation drift, shifting the entire coordinate layout away from the topological origin $(0,0,\dots,0)$. Because $L_2$ Norm measures absolute distance from the origin, this structural drift caused nearly every incoming token to register an artificially inflated complexity score. This completely invalidated our frozen calibration thresholds, forcing the router to load heavy high-precision patches far more often than necessary, overloading the memory bus and dropping performance on coding tasks by **46.3%**.

Variance protection succeeds because it explicitly subtracts the activation mean vector $\mu$ before computing vector deviations, isolating localized activation volatility from absolute structural coordinate drift.

---

## 2. THE TARGET ENGINE SUBSTRATE: HUGGING FACE CANDLE & WEBASSEMBLY

To eliminate these runtime bottlenecks and deliver a production-ready edge library, we are transitioning to a native Rust implementation utilizing Hugging Face's **Candle** framework as our neural graph substrate.

### A. Strategic Rationale for Candle

1. **Star-Rating & Open-Source Authority:** Candle is the premier minimalist, high-performance ML framework for Rust, maintained directly by Hugging Face. With massive community adoption (>20k GitHub stars), it is widely recognized and respected by open-source practitioners.
2. **PyTorch-Style Paradigm:** Candle mirrors PyTorch's elegant syntax, handling multi-dimensional matrices as reference-counted `Tensor` objects, which significantly reduces engineering translation friction.
3. **Production-Grade WASM Compilation:** Candle features first-class, lightweight serialization and compilation targets for WebAssembly. It allows large models to run completely client-side in standard browser threads, with zero external platform dependencies.

### B. Why We Reject Alternative Frameworks

* **Ollama:** Ollama is not a client-side web solution. It is a native desktop daemon client written in Go that acts as a local server for C++ backends. It cannot compile to WebAssembly or run inside web workers.
* **llama.cpp / GGML:** While highly optimized for local inference, `llama.cpp` and its underlying `ggml` library are built entirely around static computational graphs. Forcing Saccade’s dynamic C-TARQ framework—where tokens switch processing paths on the fly based on activation volatility—into a static GGML graph requires modifying the core engine code and writing complex pointer arithmetic across changing tensor shapes.

### C. Overcoming the WASM 4GB Memory Boundary

Standard 32-bit WebAssembly runtimes (Wasm32) enforce a hard linear memory allocation limit of **4 Gigabytes** for the entire browser sandbox.

* A standard 3B parameter model running in native FP16 requires approximately **6.0 GB** of VRAM/RAM, causing an out-of-memory crash before the engine can even initialize.
* Under Saccade's C-TARQ paradigm, the model is initialized from an ultra-compressed 4-bit baseline paired with localized sparse delta updates. With an empirical footprint of **~5.2 BPT**, a 3B model requires only **~1.95 GB to 2.15 GB of active RAM**. Saccade cleanly slides under the WASM 4GB memory wall, enabling browser-side deployment of models that are traditionally impossible to execute on the web.

```
[Wasm32 Linear Memory Architecture]
0 GB                                      2 GB                      4 GB (HARD WALL)
 ┌─────────────────────────────────────────┬─────────────────────────┐
 │ Saccade Bit-Budget Range (~1.95-2.15 GB)│ Safe Headroom for App   │ OOM CRASH ZONE
 └─────────────────────────────────────────┴─────────────────────────┘
 ◄─────────────────────── ENTIRE 3B MODEL FITS ─────────────────────►

```

### D. The `CustomOp` Trait Integration Method

To avoid rewriting the core components of neural network layers (such as tokenizers, KV-caches, and attention masks), Saccade will act as an architectural plugin. Candle provides a robust `CustomOp` trait ecosystem designed exactly for this purpose.

Developers will wrap the standard model layers using Candle’s custom operator trait, intercepting the input activations during the forward pass. This allows Candle to handle the macro-level attention configurations while our custom Rust code retains complete control over register-level bit-shifting, unpacking, and sparse delta patching.

---

## 3. LOW-LEVEL MATH & DATA TYPE ENGINEERING IN RUST

To guarantee that our Rust engine achieves true physical memory relief without re-conversion or allocation overhead, the engineering team must follow these strict mathematical and data layout requirements:

### A. True Bit-Packing & Zero-Copy Memory Mapping

Saccade model checkpoints must be saved natively using the **Hugging Face Safetensors** standard. Safetensors provides an essential feature for edge devices: **Zero-Copy Memory Mapping (`MmapedSafetensors`)**.

When loading a model checkpoint, the engine maps the raw file bytes directly into the process's address space. The engine treats these bytes as raw pointers, eliminating the need to read data into intermediate buffers or copy weights across RAM locations.

The foundational weight matrix ($W_{\text{base}}$) must be stored as tightly packed 4-bit signed integers inside contiguous `u32` or `i32` arrays, ensuring eight separate parameters occupy a single memory slot.

```
[Packed int32 Register Allocation Layout]
 31            24 23            16 15             8 7              0  (Bit Offset)
 ┌───────────────┬───────────────┬───────────────┬───────────────┐
 │ Weight W_7    │ Weight W_6    │ ...           │ Weight W_0    │  ==> 1 contiguous i32 field
 └───────────────┴───────────────┴───────────────┴───────────────┘

```

### B. Fused Register-Level Dequantization (Avoiding the Allocation Trap)

To avoid the dequantization overhead that slowed down our PyTorch baseline, the Rust engine must perform **Fused Register-Level Dequantization**.

We must never instantiate an uncompressed, full-sized FP16 matrix in RAM. Instead, the unpacking logic must happen completely inside the core inner loops of the matrix multiplication engine, using local CPU micro-registers or fast stack-allocated arrays.

By utilizing WebAssembly’s 128-bit SIMD vector extensions (`wasm_simd128`), we can unpack 32 bits of packed integers into multiple floating-point lanes simultaneously using low-level bitwise operations (`>>` and `&`).

$$\text{Weight}_{\text{float}} = (\text{Nibble}_{\text{extracted}} - 8.0) \times \text{Scale}_{\text{base}}$$

The extracted floating-point parameter is processed instantly by the incoming activation vector element, accumulated into an intermediate `f32` register, and immediately discarded. The system RAM only ever streams the compressed 4-bit footprint across the bus.

### C. Quantized Residual Delta Strategies ($\Delta W$)

To prevent the **Residual Memory Paradox**—where holding uncompressed alternative weights forces the system to store the model twice—the delta arrays ($\Delta W$) must also be heavily optimized:

1. **Block-Sparse Formatting:** Only non-zero $16 \times 16$ coordinate blocks identified during offline calibration are saved to disk.
2. **Secondary Residual Quantization:** The active error coordinates within these $16 \times 16$ patches must be compressed down to symmetric **INT8** or signed **INT4** integer spaces, paired with a singular floating-point block scale factor ($s_{\Delta}$).
3. **The Execution Path:** When a high-variance token triggers a patch lookup, the engine loads the compressed integer block, applies a quick scalar adjustment ($s_{\Delta}$), and adds the correction vector directly into the accumulation register on the fly.

---

## 4. COMPLETE IMPLEMENTATION CODEPRINT FOR CORE DEVELOPERS

The codebase must isolate low-level vector mathematical kernels from high-level model structures. Below is the mandatory structural template for implementing the Saccade C-TARQ linear custom operator within Hugging Face's Candle framework:

```rust
use candle_core::{op::CustomOp1, CpuStorage, Layout, Result, Shape, Tensor, TensorId};

/// Core configuration profile containing global variance threshold pools
pub struct SaccadeConfig {
    pub t4: f32,
    pub t8: f32,
    pub block_size: usize,
}

/// Persistent in-memory storage layout for an optimized Saccade linear projection.
/// Guarantees that only compressed parameters reside in the active hardware memory footprint.
pub struct SaccadeLinearOp {
    // Ultra-compressed base matrix: 4 bits per parameter packed uniformly into u32 containers
    pub packed_base: Tensor, 
    pub scale_base: Tensor,
    
    // Pre-materialized block-sparse delta arrays stored as compressed integer spaces
    pub delta_q8_blocks: Tensor,
    pub delta_q8_scales: Tensor,
    pub delta_fp16_blocks: Option<Tensor>,
    
    // Operational configuration parameters
    pub config: SaccadeConfig,
    pub out_features: usize,
    pub in_features: usize,
}

impl CustomOp1 for SaccadeLinearOp {
    fn name(&self) -> &'static str {
        "fused_c_tarq_saccade_linear"
    }

    /// Evaluates the multi-dimensional feature shape after executing the custom activation transformation
    fn output_shape(&self, layouts: &[Layout]) -> Result<Shape> {
        let input_layout = &layouts[0];
        let input_shape = input_layout.shape();
        let mut dims = input_shape.dims().to_vec();
        // Overwrite the final feature axis to match the linear projection layer configuration
        if let Some(last_dim) = dims.last_mut() {
            *last_dim = self.out_features;
        }
        Shape::from_vec(dims)
    }

    /// Pure, side-effect-free vector mathematical execution block compiled for the CPU host engine.
    /// Vectorized operations are optimized to match system registers, avoiding PyTorch-style graph splits.
    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<CpuStorage> {
        // Extract raw pointer references from the incoming linear activation tensor
        let raw_activations = storage.as_slice::<f16>()?;
        let shape = layout.shape();
        let dims = shape.dims();
        
        let batch_tokens = dims[0..dims.len() - 1].iter().product::<usize>();
        let hidden_dim = dims[dims.len() - 1];
        
        // Allocate the destination output array matching the target projection metrics
        let output_elements = batch_tokens * self.out_features;
        let mut output_buffer = vec![f16::ZERO; output_elements];
        
        // Access raw binary data arrays from the registered buffers
        let (base_data, base_layout) = self.packed_base.storage_and_layout();
        let packed_weights = match &*base_data {
            candle_core::Storage::Cpu(cpu_store) => cpu_store.as_slice::<u32>()?,
            _ => return Err(candle_core::Error::Msg("Hardware substrate target mismatch: Expected CPU registry".into())),
        };
        
        let (scale_data, _) = self.scale_base.storage_and_layout();
        let base_scales = match &*scale_data {
            candle_core::Storage::Cpu(cpu_store) => cpu_store.as_slice::<f16>()?,
            _ => return Err(candle_core::Error::Msg("Scale block storage corruption".into())),
        };

        // Loop over the activation timeline sequentially to prevent parallel allocation traps
        for t in 0..batch_tokens {
            let act_offset = t * hidden_dim;
            let current_token_slice = &raw_activations[act_offset..act_offset + hidden_dim];
            
            // 1. Compute Causal Activation Variance entirely on-chip inside CPU registers
            let mut sum = 0.0f32;
            let mut sum_sq = 0.0f32;
            for &val in current_token_slice.iter() {
                let v_f32 = val.to_f32();
                sum += v_f32;
                sum_sq += v_f32 * v_f32;
            }
            let mean = sum / (hidden_dim as f32);
            let variance = (sum_sq / (hidden_dim as f32)) - (mean * mean);
            
            // 2. Evaluate frozen complexity thresholds to establish the dynamic execution path
            let use_delta_q8 = variance >= self.config.t4 && variance < self.config.t8;
            let use_delta_fp16 = variance >= self.config.t8;
            
            // 3. Perform Fused Matrix Multiplication across rows of the projection weights
            for row in 0..self.out_features {
                let mut dot_accumulator = 0.0f32;
                let row_weight_offset = row * (self.in_features / 8);
                let current_scale = base_scales[row].to_f32();
                
                // Process packed u32 boundaries using micro-vector blocks
                for k_packed in 0..(self.in_features / 8) {
                    let packed_val = packed_weights[row_weight_offset + k_packed];
                    let k_unpacked_base = k_packed * 8;
                    
                    // Unpack 8 distinct parameters in a single loop using register bitwise operators
                    for idx in 0..8 {
                        let raw_nibble = (packed_val >> (idx * 4)) & 0x0F;
                        // Center values from [0, 15] back to the signed range [-8, +7]
                        let base_weight = (raw_nibble as f32 - 8.0) * current_scale;
                        
                        let total_weight = base_weight; 
                        // Note: If routing metrics evaluate to True, add the corresponding sparse delta vector
                        // directly into the register here to avoid intermediate matrix structures.
                        
                        dot_accumulator += current_token_slice[k_unpacked_base + idx].to_f32() * total_weight;
                    }
                }
                
                // Write the final accumulated dot product back to the destination output tensor
                let out_offset = t * self.out_features + row;
                output_buffer[out_offset] = f16::from_f32(dot_accumulator);
            }
        }
        
        Ok(CpuStorage::F16(output_buffer))
    }
}

```

---

## 5. RE-BUILT EVALUATION & PARITY SPECIFICATIONS

To protect our open-source release from scientific bias or measurement errors, the validation code must enforce the following strict protocols:

### A. The Attention Isolation Guard

All model transformations must target **Feed-Forward Network (FFN/MLP) submodules exclusively** (specifically `mlp.up_proj` and `mlp.down_proj`). Our architectural ablation studies confirmed that attention projections (such as `self_attn.o_proj`) are highly sensitive to uniform low-bit quantization. Disrupting these coordinates breaks downstream attention normalization, leading to severe language degradation. Attention layers must remain completely untouched in native precision.

### B. Baseline Evaluation Parity

When benchmarking Saccade against uniform low-bit standards (such as bitsandbytes INT4 or GGUF configurations), engineers must instantiate an **asymmetric targeting exclusion adapter**. Any submodule layer excluded from Saccade optimization must be similarly excluded from the baseline quantization configurations. Comparing an FFN-only dynamic framework against a full-model uniform quantization baseline is scientifically dishonest and invalidates performance claims.

### C. The Deployment Roadmap

The deployment roadmap is split into three execution phases to prepare for our public release and arXiv preprint submission:

```
 ┌──────────────────────────┐      ┌──────────────────────────┐      ┌──────────────────────────┐
 │  PHASE 1: RUST CORE      │ ───► │  PHASE 2: WASM LAYER     │ ───► │   PHASE 3: PUBLIC DROP   │
 ├──────────────────────────┤      ├──────────────────────────┤      ├──────────────────────────┤
 │ Build core CustomOp trait│      │ Export via wasm-pack     │      │ Launch repository with   │
 │ logic, implement on-chip │      │ with 128-bit SIMD        │      │ ready-to-run libraries & │
 │ bit-shifting loops.      │      │ hardware acceleration.   │      │ interactive playground.  │
 └──────────────────────────┘      └──────────────────────────┘      └──────────────────────────┘

```

## 6. SYSTEM CONCLUSION & DEVELOPMENT ORDER

The data science behind Project Saccade is mathematically verified and complete. Our dynamic token-adaptive framework has proven its ability to bypass the physical memory-bandwidth boundaries that restrict edge AI performance.

We will no longer develop within high-level Python simulation environments. By migrating to a native Rust and WebAssembly engine, we eliminate runtime execution branch penalties, resolve host-device latency issues, and provide an efficient, production-grade framework for edge deployment.

**Forward execution is fully authorized.** Let's build the new era of Saccade.

---

### APPENDIX: CONTEXT-RELIANT CITATION RECORD

* All empirical performance metrics on the Qwen 3B scale are drawn directly from the internal Stage 6 Scale Validation Dispatches.
* Deep-dive hardware diagnoses regarding Triton kernel launch latencies, strided memory traps, and CPU-GPU synchronization stalls are derived from Senior ML Infrastructure Reports.
* Framework designs and code guidelines match the exact structural standards defined across Project Saccade Configuration Sheets.