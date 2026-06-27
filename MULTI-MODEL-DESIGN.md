# ENGINEERING DESIGN RFC: UNIVERSAL MULTI-MODEL SUPPORT VIA CORE RUNTIME HIJACKING

**To:** Core ML Systems Team / Open-Source Release Committee

**From:** Principal ML Systems Architect

**Status:** Approved for Core Subsystem Refactoring

---

## 1. Executive Summary

Project Saccade V3 has demonstrated excellent end-to-end performance on the `Qwen2.5-0.5B-Instruct` architecture, proving the validity of the C-TARQ thesis by delivering a **1.28× wall-clock decoding speedup** and a **1.76× memory footprint reduction** on physical hardware. However, our current method for scaling to new models is an engineering bottleneck: it requires forking high-level model definitions (such as our custom ~400-line `model.rs`) because upstream `candle-transformers` encapsulates internal linear weights as private fields.

To turn Saccade into an easy-to-use, architecture-agnostic framework for the open-source community, we must transition from **Model-Level Forcing** to **Framework-Level Hijacking**. This RFC outlines a design to modify the core `candle-nn::Linear` layer directly inside a global crate fork. This modification allows any neural network in the ecosystem—including Large Language Models (LLMs), Vision Transformers (ViTs), and Diffusion U-Nets—to seamlessly run on Saccade's optimized 3-phase kernel with zero high-level code alterations.

---

## 2. Structural Analysis of the Current Bottleneck

Our current implementation relies on a custom `ProjectionLayer` enum container to route execution paths:

```rust
// Current manual model-level encapsulation pattern
pub enum ProjectionLayer {
    Standard(candle_nn::Linear),
    Saccade(SaccadeLinearOp),
}

```

While functional for our initial validation runs, this structure has several key limitations:

*  nominal Compile-Time Type Layouts: Rust enforces absolute nominal type safety and computes precise struct memory offsets at compile time. Upstream models in `candle-transformers` hardcode fields like `pub down_proj: candle_nn::Linear`. We cannot substitute these fields at runtime with our custom `SaccadeLinearOp` because their underlying memory allocations and types do not match.
* **Encapsulation Barriers:** Upstream model structures keep internal weights and attention projection submodules private. Forcing downstream users to manually fork, modify, and maintain separate code paths for every new architecture (Llama, Gemma, Phi) introduces massive friction and limits developer adoption.
* **Lack of Modality Generalization:** The custom enum wrapper is explicitly tied to our language model text generation loops, blocking its use in other highly parallel modalities that can benefit from memory bandwidth optimization.

---

## 3. The Proposed Universal Interception Architecture

Rather than maintaining a sprawling registry of individual model forks, we will modify the core foundational layout of `candle-nn` itself. By embedding Saccade's conditional routing layer directly into the base framework, we can intercept operations globally.

```
┌────────────────────────────────────────────────────────────────────────┐
│                      SACCADE DECOUPLING PARADIGM                       │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  [candle-transformers] ──► Instantiates candle_nn::Linear              │
│                                           │                            │
│                                           ▼                            │
│                        [Custom candle-nn Crate Fork]                   │
│                        Is Linear state compressed?                     │
│                             ╱                       ╲                  │
│                           YES                        NO                │
│                           ╱                            ╲               │
│                          ▼                              ▼              │
│               [Saccade Backend]                [Vanilla Backend]       │
│               3-Phase SIMD Unpacking           Standard BLAS GEMM/GEMV │
│               + Sparse CSC Deltas              Dense FP16 Matrix Op    │
└────────────────────────────────────────────────────────────────────────┘

```

### A. Core Struct Overhaul (`candle-nn/src/linear.rs`)

We will rewrite the standard `Linear` structure into a unified backend interface. When initialized normally via standard weights, it utilizes native execution layers; when compiled via `saccade-compile`, it automatically triggers our memory-bandwidth-optimized execution loops.

```rust
// Refactored foundational linear layer interface within candle-nn
pub enum LinearBackend {
    Standard {
        weight: candle_core::Tensor,
        bias: Option<candle_core::Tensor>,
    },
    Saccade {
        cache: std::sync::Arc<saccade_core::KernelCache>,
        config: saccade_core::SaccadeConfig,
        in_features: usize,
        out_features: usize,
    },
}

pub struct Linear {
    backend: LinearBackend,
}

impl Linear {
    pub fn forward(&self, x: &candle_core::Tensor) -> candle_core::Result<candle_core::Tensor> {
        match &self.backend {
            LinearBackend::Standard { weight, bias } => {
                // Keep the standard hardware-accelerated BLAS execution path intact
                let y = x.matmul(&weight.t()?)?;
                match bias {
                    Some(b) => y.broadcast_add(b),
                    None => Ok(y),
                }
            },
            LinearBackend::Saccade { cache, config, in_features, out_features } => {
                // Transparently forward execution to Saccade's unrolled 3-phase execution kernel
                x.apply_op1_no_bwd(saccade_core::SaccadeLinearOp::from_cache(
                    cache.clone(),
                    config.clone(),
                    *in_features,
                    *out_features,
                ))
            }
        }
    }
}

```

### B. Global Crate Overriding Configuration

To activate this framework-wide update across all upstream libraries without editing their source code, we use Cargo's `[patch]` feature in the root `Cargo.toml`. This directs the compiler to use our optimized fork for all downstream dependencies:

```toml
[patch.crates-io]
candle-core = { git = "https://github.com/saccade-engine/candle.git", branch = "main" }
candle-nn = { git = "https://github.com/saccade-engine/candle.git", branch = "main" }
candle-transformers = { git = "https://github.com/saccade-engine/candle.git", branch = "main" }

```

---

## 4. Modality Expansion Strategy (Beyond LLMs)

Moving affine transformations into the framework core allows Saccade to optimize any architecture that can be expressed as a directed acyclic graph of tensor multiplications. This structure enables seamless optimization across diverse model modalities:

### A. Vision Transformers (ViTs)

* **The Workflow:** Input images are tokenized into independent visual spatial coordinate arrays.
* **The Saccade Optimization:** The model's query, key, value, and multi-layer perceptron (MLP) projections run through our core `Linear` structure. During inference, background patches with low visual complexity are routed through the high-efficiency 4-bit base layer. The sparse CSC correction paths are dynamically reserved for highly complex features, structural edges, or rapid scene changes, optimizing memory transit across visual processing tasks.

### B. Stable Diffusion & Flow-Matching U-Nets

* **The Workflow:** Cross-attention mechanics project textual embedding matrices onto changing spatial latents over sequential image generation steps.
* **The Saccade Optimization:** During initial high-noise denoising passes, activation states exhibit severe structural volatility, dynamically engaging our high-precision sparse correction paths. As the generation loop stabilizes and settles on clean image layouts, activation variance drops significantly, allowing Saccade to automatically shift down to low-bit execution modes to accelerate processing throughput.

---

## 5. Simplifying User Ingestion Pipelines

To ensure a smooth user experience for the open-source community, the compilation and runtime flows must be completely automated through simple command-line tools:

1. **One-Command Model Compilations (`saccade-compile`):**
Users provide a Hugging Face repository identifier and a local plain-text calibration file. The utility downloads the files, profiles the model's activation manifolds, calculates percentile-based quantization thresholds, and packs the base parameters, sparse CSC matrices, and metadata into a single unified archive.
```bash
saccade-compile --model-id Qwen/Qwen2.5-3B-Instruct --calib-file domain_text.txt --output-path saccade_model.safetensors

```


2. **Universal Streaming Inference Runs (`saccade-run`):**
The runtime utility infers model configurations and parameters directly from the serialized safetensors header metadata. It instantiates the framework layers automatically, streams token outputs to the terminal in real time, and prints a comprehensive system performance dashboard upon sequence completion.
```bash
saccade-run --checkpoint saccade_model.safetensors --prompt "Implement a lock-free ring buffer in Rust." --max-tokens 128

```



---

## 6. Action Plan for the Engineering Team

To implement this global architecture framework, the development team will prioritize the following deployment actions:

* **Phase 1: Establish the Core Framework Fork (Target: 3 Days)**
Clone the main Hugging Face `candle` repository. Refactor `candle-nn::Linear` to support our dual-backend system, ensuring that non-quantized weights flow through native unquantized execution paths with zero performance loss.
* **Phase 2: Migrate Core Operators to KernelCache Layouts (Target: 3 Days)**
Update `SaccadeLinearOp` to ingest pre-compiled data states from our new framework-level `LinearBackend` configuration. Port our unrolled 4-accumulator FMA pipelines and column-major CSC transformation loops directly into the core crate layout.
* **Phase 3: Update Serialization & Verification Suites (Target: 2 Days)**
Modify `saccade-compile` to export layers directly using standard `candle_nn::Linear` names, embedding routing thresholds ($t_4, t_8$) and compression tracking parameters cleanly into the Safetensors JSON header dictionary. Run end-to-end decoding benchmarks across the 0.5B and 1.5B scales to confirm performance stability.

By shifting our implementation from custom model forks to a framework-level injection layer, we eliminate architectural debt, provide universal multi-model compatibility, and position Saccade as a highly practical platform for accelerating diverse models on consumer edge devices.