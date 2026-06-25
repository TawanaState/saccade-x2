# 01. Architecture & Design Rationale

Project Saccade implements **Causal Token-Adaptive Residual Quantization (C-TARQ)**. This document outlines the fundamental technical decisions, physics constraints, and historical iteration that define the architecture.

## 1. The Bottleneck: Memory Bandwidth vs. FLOPs

When executing autoregressive LLMs on consumer edge devices (e.g., T4/L4 GPUs, NPUs, unified memory silicon), generation speed is bottlenecked by DRAM memory bandwidth, not raw compute (FLOPs). Every single token generation requires loading the entire model parameter space from main memory into chip registers.

*   **Uniform Low Precision (INT4):** Maximizes bandwidth efficiency but irreversibly destroys the geometric feature layout necessary for logical reasoning, coding, and mathematical syntax.
*   **Uniform High Precision (BF16/FP32):** Preserves feature fidelity but heavily saturates memory buses, resulting in commercially unviable throughput speeds.

**The Saccade Hypothesis:** Complexity in language is non-uniform. Syntactic glue ("the", "and") requires minimal representational precision, whereas reasoning anchors (code variables, logic operators) require high-fidelity states. Precision must scale dynamically across the temporal sequence, token-by-token.

## 2. Core Architectural Pillars

### A. The Compressed Baseline Matrix ($W_{\text{base}}$)
The entire model is globally quantized to a statically uniform INT4 baseline layout.
*   **Implementation:** `SaccadeBitPacker`. 8 distinct 4-bit weights are physically bitwise-shifted into a single, contiguous `int32` tensor.
*   **Result:** The absolute physical memory bus footprint drops by 75% compared to FP16, guaranteeing residency bounds on tight hardware limits.

### B. The Residual Error Arrays ($\Delta W$)
Instead of abandoning precision lost during the INT4 transformation, C-TARQ precomputes the structural error: $\Delta W = W_{\text{full}} - W_{\text{base}}$.
*   **Sparsity Constraints:** $\Delta W$ arrays are enforced into strict $16 \times 16$ block-sparse topologies (`BlockSparseMatrix`). Unstructured sparsity destroys continuous memory read patterns; block-sparsity ensures contiguous bus streaming, maximizing cache hit rates during selective reconstruction.
*   **Stratification:** We maintain distinct sparse structures for INT8 and FP16 tier delta corrections.

### C. The Causal Entropy Router
Activations are evaluated against threshold invariants *per-layer* before matrix multiplication occurs.
*   **Operation:** If token variance (or L2 norm) exceeds a specific threshold, the layer triggers a sparse lookup, applying the high-precision delta specifically for that token slice.
*   **Mathematical Layout:** $Y = X \cdot W_{\text{base}}^T + X \cdot \Delta W^T$ (Applied sparsely).
*   **V2 Execution Model:** To eliminate memory bandwidth overhead, the runtime runs a compiled static base pass (unpacking + matmul) via `torch.compile(fullgraph=True)`, but keeps delta matrix multiplications conditional in eager mode. This prevents useless FLOPs and memory reads for inactive tokens while keeping compilation clean and safe from recompilation spikes.

## 3. Historical Pivots & Failure Analysis

To understand why the codebase operates exactly as it does, it's critical to review past design failures.

### The "Stateful Routing" Failure (MoE Parallels)
*   **Approach:** Attempted routing tokens between 4 parallel expert configurations (INT4, INT8, BF16, FP32).
*   **Failure:** Required all 4 model copies to maintain hot residency in RAM. It mathematically improved speed on paper, but violated physical VRAM boundaries on edge hardware.
*   **Pivot:** Shifted to the Base + Residual Delta matrix, resolving the residency conflict.

### The "Oracle Bias" Failure
*   **Approach:** Previous iterations dynamically computed threshold quantiles over the *entire incoming sequence batch* during forward passes.
*   **Failure:** This gave the router "look-ahead" knowledge. Standard autoregressive models process tokens one-by-one. Calculating sequence variance dynamically during inference is mathematically invalid in live deployment.
*   **Pivot:** C-TARQ introduces an explicit *Offline Causal Calibration Phase*. The compiler runs a calibration payload dataset, records static scale-invariant threshold triggers per layer, and permanently freezes them into the execution module. No runtime look-ahead occurs.

## 4. Hardware Independence

The architecture does not depend on custom CUDA C++ kernels, which typically fragment deployment across changing NVCC versions or hardware boundaries (AMD ROCm, Apple Metal). By leveraging standard PyTorch primitives coupled closely with PyTorch 2.x `torch.compile` (Inductor AOT mapping), the fusion logic is universally port-agnostic and relies entirely on native graph tracing.