# Saccade C-TARQ Engine 🚀

A domain-agnostic, token-adaptive matrix execution engine extending Hugging Face's Candle framework.

## Project Objective
Saccade accelerates and compresses standard neural operations (`Y = X * W^T + b`) by executing a packed 4-bit integer base matrix on predictable tokens. It dynamically loads coordinate-masked sparse precision updates (`ΔW`) strictly when token activation variance dictates higher precision is necessary.

Saccade resolves severe DRAM memory-bandwidth constraints natively in Rust, achieving true edge execution locally on CPUs and browsers via WebAssembly.

## Core Features
1. **Domain-Agnostic Adaptation**: Works generically for any linear projection, allowing acceleration outside of language generation (e.g. LLMs, Vision models, generic MLPs).
2. **Allocation Trap Bypass**: Prevents dequantization overhead by unpacking tightly-bound 4-bit weights strictly within execution loops, using standard native SIMD hardware registers instead of broad memory allocations.
3. **Candle Substrate Extension**: Employs Candle's PyTorch-style API and `CustomOp1` traits for transparent injection into standard deep learning evaluation loops without complex core framework modifications.

## Compilation & Usage

This project contains a highly modular engine (`saccade-core`) and an evaluation/test runner (`saccade-runner`).

### Local Verification
To run the Saccade evaluation runtime to prove execution branches correctly path into sparse operations on highly active token trajectories:

```bash
cargo run --bin verify
```

### High-Performance Production Host Compilations
Native host CPU instruction paths (such as AVX-512 and AVX2 bit shifts) should be requested by exporting the target hardware flags:

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

## Future Roadmap (WASM Target)
Saccade's ultra-low operational memory footprint ensures LLMs traditionally constrained by the 4GB Wasm32 boundary can fit actively in browser tabs. In future milestones, target the WASM stack to enable zero-latency, full-client privacy implementations using:

```bash
wasm-pack build --target web -- --features wasm_simd128
```
