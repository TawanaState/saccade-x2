# DEVELOPMENT DEPLOYMENT ROADMAP: DIRECTION.md

**Project Substrate:** Saccade Adaptive Matrix Engine (Saccade V3)

**Target Core Framework:** Hugging Face Candle (`candle-core` / `candle-transformers`)

**Primary Execution Targets:** Local CPU (Rayon / Parallel Matrix Vector Intrinsics) $\rightarrow$ WebAssembly (Wasm32 SIMD128)

---

## 1. THE DOMAIN-AGNOSTIC SHIFT: BEYOND LARGE LANGUAGE MODELS

Project Saccade is transitioning from an LLM-specific compression method into a **domain-agnostic, token-adaptive plug-and-play matrix execution engine**.

At the hardware-software interface, nearly all modern deep neural networks—whether they are Autoregressive Transformers, Vision Transformers (ViTs), Diffusion U-Nets, or Multilayer Perceptrons (MLPs)—are fundamentally directed acyclic graphs of affine transformations:

$$Y = X \cdot W^T + b$$

Saccade treats this baseline operation as a dynamic, coordinate-masked tensor calculation. Any machine learning model can be accelerated and compressed under our architecture by replacing standard dense linear layers with our adaptive matrix engine.

### Saccade Affine Transformation Execution Paths

* **Predictable Activation Manifolds (Low Volatility):** Calculated using an ultra-compressed 4-bit packed baseline ($W_{\text{base}}$), reducing memory bus congestion by up to 75%.
* **Volatile Outlier Coordinates (High Volatility):** Dynamically patches the output tensor register using highly localized, block-sparse coordinate corrections ($\Delta W_{q8}$ or $\Delta W_{fp16}$).

This domain-agnostic approach allows the engine to optimize any network layout, from language models to vision pipelines, by focusing purely on activation variance.

---

## 2. CHASSIS TARGET FOR INITIAL LOCAL CPU VALIDATION

To test and validate Saccade's native Rust implementation locally before building toward WebAssembly, developers must utilize a stable, highly modular architecture already present in the source tree:

### Target Architecture Recommendation: Qwen2 / Qwen2.5

* **Repository Path:** `candle-transformers/src/models/qwen2.rs`
* **Strategic Rationale:** The Qwen2 and Qwen2.5 implementations inside the Candle tree feature exceptionally clean and isolated multi-layer perceptron (MLP) blocks. The forward passes are mapped explicitly, making it trivial to locate, intercept, and patch the feed-forward network (FFN) layers without breaking surrounding model logic.

### The Attention Isolation Guard (Mandatory)

When implementing Saccade across model architectures, developers must isolate attention elements to protect them from quantization noise. **Do not modify the linear operators within the Attention submodules** (`q_proj`, `k_proj`, `v_proj`, `o_proj`).

Our previous validation experiments confirmed that attention projection spaces are highly sensitive to low-bit quantization. Introducing compression noise here degrades downstream attention normalization, destabilizing model accuracy. Saccade must target **Feed-Forward Networks exclusively** (e.g., `mlp.up_proj`, `mlp.down_proj`, or equivalent gating submodules in non-LLM networks) to maximize memory bus relief while preserving core intelligence.

---

## 3. CANDLE PRIMITIVES & TENSOR API DIRECTORY

Developers must understand Candle's immutable, reference-counted tensor management system. Below is the API mapping of core operations required to build the Saccade routing framework:

### Core Tensor Primitives Map

* **Matrix Multiplication (`Tensor::matmul`):** Computes foundational linear projections. Located in `candle-core/src/tensor.rs`.
* **Element-Wise Accumulation (`Tensor::add` / `Tensor::broadcast_add`):** Natively fuses bias terms and sparse delta corrections into active accumulation registers without altering the base shape.
* **Geometry Transformations (`Tensor::reshape` / `Tensor::transpose`):** Used to reconstruct inputs and align bit-packed byte strides before executing kernel operations.
* **Precision Transformations (`Tensor::to_dtype`):** Explicitly casts scalar buffers into matching register formats (`DType::F16`, `DType::F32`) inside custom macro loops.

### Intercepting Forward Graphs via `CustomOp1`

To create our token-adaptive routing engine, developers must implement Candle’s native custom operator trait:

* **Source File Reference:** `candle-core/src/custom_op.rs`
* **Implementation Standard:** Create a struct that implements `CustomOp1`. This allows you to write low-level C-style vector arithmetic loops in the `cpu_fwd` execution hook, bypassing Candle's standard high-level operators to handle bit-shifting directly within local registers.
* **Reference Example:** To see how to structure and register user-defined operators, developers should study the internal test definitions located in `candle-core/tests/custom_op_tests.rs`.

---

## 4. PLUG-AND-PLAY PLUG-IN ARCHITECTURE

Saccade integrates seamlessly into Candle models as an architectural extension. Instead of modifying the core engine or writing complex pointer layers, developers will use a clean wrap-and-replace approach:

```
[Candle Standard Layer Loading Sequence]
 Safetensors Archive ──► VarBuilder ──► candle_nn::Linear::new() ──► Dense FP16 Graph

[Saccade V3 Plugin Deployment Flow]
 Safetensors Archive ──► VarBuilder ──► saccade::LinearOp::wrap() ──► Token-Adaptive Graph
                                                  │
                                                  ▼
                                      [ CustomOp1 Intercept Hook ]
                                        - Bitwise Register Shifts
                                        - Dynamic Variance Routing

```

### The Implementation Contract

1. **Intercept the Initialization Engine:** Intercept Candle's model topology builder (`candle_nn::VarBuilder`) when loading weight files from disk.
2. **Layer Replacement:** Replace standard linear initializers (`candle_nn::linear`) inside target submodules with our custom wrapper struct (`SaccadeLinearOp`).
3. **Encapsulate Weights Natively:** The custom wrapper struct loads the packed 4-bit baseline matrix and the sparse coordinate array keys directly from the native `safetensors` archive.
4. **Graph Execution:** During the forward computation graph loop, the wrapper intercepts incoming features, executes on-chip activation variance routing, and dynamically combines precision layers into a single, optimized output allocation buffer.

---

## 5. OFFLINE CALIBRATION & RE-CONVERSION SAFETY PROTOCOL

To prevent threshold alignment errors between the training and runtime environments, the engine must execute under a strict evaluation contract:

### A. Scale-Invariant Variance Extraction

The global activation variance heuristic must be calculated on-chip using a rolling single-pass calculation during calibration steps:

$$\sigma^2 = \frac{1}{d} \sum_{i=1}^{d} (x_i - \mu)^2$$

Bypassing the mean subtraction step to use raw $L_2$ Euclidean magnitudes is strictly prohibited. Hidden layer activations across deep networks encounter significant uncentered drift over time. Subtracting the local mean vector $\mu$ isolates true token complexity from background architectural coordinate drift, preventing threshold distortion.

### B. Dual-Phase Invariance Lock

The exact same `Strategy Heuristic Object` initialized during the offline data calibration phase must be anchored to the online generation runtime. Hardcoded thresholds are banned; variance boundary scales ($t_4$, $t_8$) must be read directly from the metadata blocks inside the `safetensors` model container.

---

## 6. LOCAL CPU OPTIMIZATION & MULTITHREADING SCHEDULING

Autoregressive token generation or single-sample inference at the edge runs at a batch size of 1, making execution inherently memory-bandwidth bound. To maximize performance on local CPU architectures, developers must follow these strict multithreading and compilation rules:

### A. Mitigating Thread Divergence on Host CPUs

On parallel GPU systems, dynamic routing logic can cause severe "thread divergence" penalties that halt processing units. Host CPUs, however, utilize advanced out-of-order execution pipelines and hardware branch predictors. Evaluating token complexity metrics and branching into different execution paths introduces virtually zero instruction penalty on a CPU.

### B. CPU Thread Caching and Core Allocation

* **Vectorized Processing Loops:** Low-level bitwise shifts and integer unpacking tasks must be structured to fit entirely within the processor’s fast L1/L2 caches, completely avoiding system RAM allocation steps during matrix transformations.
* **Leveraging Rayon and Fused Backends:** Candle uses the **Rayon** library to automatically handle CPU thread pooling. Saccade loops should remain single-threaded across the token timeline dimension to prevent thread allocation overhead. Parallel threads should be applied across the weight matrix row dimension (`out_features`) during matrix multiplication, allowing individual CPU cores to stream and unpack compressed parameters independently.
* **SIMD Native Vectorization:** To ensure high performance on local host architectures, developers must enable target extension compilation flags (`RUSTFLAGS="-C target-cpu=native"`) to force the compiler to generate optimized AVX2 or AVX-512 assembly instructions.

---

## 7. DOCUMENTATION & REPOSITORY EXPLORATION GUIDE

Developers onboarding onto the Project Saccade V3 codebase must familiarize themselves with the following key locations in the Candle workspace to master the framework's architecture:

### 1. Mathematical Storage Substrate (`candle-core/src/`)

* **`tensor.rs`:** Master multi-dimensional array management methods. Understand tensor layout strides, reference allocations, and memory pointer transformations.
* **`cpu_backend/`:** Contains core execution loops for host chip math. Inspect how raw slices are modified within standard memory spaces.
* **`safetensors.rs`:** Native parsing logic for memory-mapped weight archives. Focus on how files are chunked into zero-copy references.

### 2. Quantization Framework Models (`candle-core/src/quantized/`)

* Developers must study `gguf_file.rs` and `k_quants.rs`. This directory contains `llama.cpp`-style integer block dequantization mappings. It provides an excellent reference for writing optimized register bit-packing loops in Rust.

### 3. Production Model Architecture Examples (`candle-transformers/src/models/`)

* Study the structural layouts of `qwen2.rs`, `gemma.rs`, and `stable_diffusion/`. Use these reference files to understand how layers are instantiated via `VarBuilder` and how forward execution passes are constructed.

### 4. Client-Side WASM Deployments (`candle-wasm-examples/`)

* Inspect the `llama2-c/` and `quant-qwen3/` subfolders. These examples demonstrate how to handle linear web memory buffers, configure async web worker messaging pipelines, and utilize compilation wrappers to target the web browser canvas.

---

## CONCRETE DEV ACTION ITEMS

1. **Set Up Local Sandbox:** Initialize a local Rust project, import `candle-core` and `candle-transformers`, and configure a clean execution runner for a standard Qwen2 model architecture.
2. **Implement custom `LinearOp` struct:** Code the baseline wrapper framework using Candle's `CustomOp1` trait, ensuring it routes inputs cleanly.
3. **Verify Register Math:** Write the register-level 4-bit unpacking macro using bitwise operators (`>>` and `&`), ensuring parameter scales apply correctly without allocating full intermediate tables in system memory.
4. **Compile & Benchmark:** Run local CPU validation benchmarks with `target-cpu=native` enabled to verify bit-unpacking speeds before initializing target compilation setups for WebAssembly.