# 02. Structural Module Map

This document serves as the developer orientation for the `saccade/` internal source tree. The package is divided cleanly into AOT compilation routines, real-time native execution hooks, hardware-compliant packing, and evaluation harnessing.

## Directory Tree Outline

```text
saccade/
├── __init__.py           # High-level convenience API interface
├── compiler.py           # AOT parameter calibration and layout replacement
├── runtime.py            # Inductor-optimized forward pass hooks
├── serialization.py      # Safetensors zero-copy disk mapping and metadata parsing
├── benchmarking.py       # Eager-mode analytical tracking routines
├── core/
│   └── packing.py        # Physical tensor packing logic (INT32 Bitshift & Block Sparsity)
├── adapters/
│   └── routing.py        # Abstract token entropy/complexity heuristic functions
└── cli/                  # Command-line suite mapping the Python API to developer workflows
    ├── main.py
    ├── benchmark_harness.py
    ├── data_ingestion.py
    └── plots.py
```

## Module Execution Responsibilities

### 1. `compiler.py` (The Transformation Engine)
**Role:** Ingests unquantized huggingface transformers.
*   **Targeting:** Parses `target_layers` (e.g., FFN projections, or "all" for attention parity).
*   **Calibration Loop:** Mounts `register_forward_hook` onto linear layers, runs a dataset payload, and derives $t_4$ and $t_8$ global percentile thresholds natively.
*   **Structural Mutability:** Unpacks HuggingFace FP16 parameters, extracts integer quantiles, determines precision residuals, delegates to `core/packing.py`, and destructively swaps standard `nn.Linear` layers with custom `SaccadeLinearPlugin` modules.

### 2. `runtime.py` (The Hot-Path Plugin)
**Role:** Intercepts incoming `forward()` calls dynamically during autoregressive generation.
*   `execute_fused_saccade_block`: A pure functional tracing function isolated for `torch.compile` (`mode="reduce-overhead"`, `fullgraph=True`). Contains zero Python control flow and zero dynamic tensor shapes, enabling a single monolithic CUDA Graph with no graph breaks.
*   **Zero-Branch Mask-Broadcast:** Instead of dynamically slicing token rows via `torch.where` (which produced value-dependent shapes and graph breaks), the block runs three unconditional parallel matmuls (base, delta Q8, delta FP16) for all tokens, then zeroes out irrelevant contributions via binary mask broadcasting. This trades marginal extra FLOPs for full static-graph compilability and CUDA Graph fusion.
*   **On-the-Fly INT32 Dequantization:** `_unpack_and_dequantize()` runs inside the compiled graph, performing vectorized INT32 → FP16 conversion each forward pass. Only the packed INT32 tensor (4x smaller than FP16) persists in VRAM — the unpacked FP16 exists transiently in GPU registers/L2 cache during kernel execution.
*   **Pre-Materialized Dense Deltas:** Block-sparse delta matrices are converted to dense format once at plugin init and stored as persistent registered buffers. This eliminates the per-forward `.to_dense()` allocation that caused memory fragmentation and was untraceable by TorchInductor.

### 3. `core/packing.py` (Hardware Memory Controller)
**Role:** Executes the physical data mutations to drop tensor footprints.
*   `SaccadeBitPacker`: Runs bitwise `|` and `<<` operators to cram eight distinct `INT4` weight parameters directly inside standard PyTorch `INT32` tensor allocations.
*   `BlockSparseMatrix`: Re-aligns random spatial high-variance residual values into rigid $16 \times 16$ bounds. Random sparse arrays force cache-miss overhead. Block structures match L1/L2 physical cache line ingestion speeds.

### 4. `serialization.py` (Zero-Copy Mounter)
**Role:** Provides SafeTensors save/load capabilities without intermediary RAM bloat.
*   **Metadata Embedding:** Logs precision parameters (`layer_name_t4`), heuristic configurations (`variance`), tensor padded shapes, and routing strategy natively into the JSON Safetensor header.
*   **Zero-Copy Loading:** Streams disk weights via `f.get_tensor()` direct to GPU allocations `f(..., device="cuda")` circumventing standard PyTorch memory duplications.

### 5. `cli/benchmark_harness.py` (Evaluation Parity Controller)
**Role:** Standardizes the empirical 4-stage evaluation baseline.
*   Automates memory destruction between FP16, INT4 (`bitsandbytes`), INT8, and C-TARQ instances to prevent OOM.
*   **Dynamic Skipping:** Uses `llm_int8_skip_modules` calculated against unquantized FP16 module structures mapped by `--target_layers`. This guarantees that Saccade is mathematically benchmarked against identical configurations (e.g., comparing MLP-only Saccade to an MLP-only INT4 baseline, preventing unfair accuracy bias.)

### 6. `adapters/routing.py`
**Role:** Exposes `BaseRoutingAdapter` for user extensions. By default, standard deviation/variance mapping scales efficiently against intermediate hidden features, acting as an excellent predictor of computational importance. Developers can subclass this to supply explicit custom routing logics (e.g., attention head entropy tracking).