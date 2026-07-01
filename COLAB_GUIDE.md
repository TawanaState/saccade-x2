# Google Colab Setup and Testing Guide for Saccade C-TARQ Engine 🚀

This guide provides step-by-step instructions to set up a common testing environment on **Google Colab** and run the Saccade V4 suite, including model compilation, verification, Qwen text generation, and Whisper audio tests.

---

## 1. Environment Setup

To run Saccade on Google Colab, you need to configure the Linux environment, install Rust/Cargo, clone the repository with its submodules, and compile the code.

Follow these steps in a new Google Colab notebook.

### A. Install Rust and System Dependencies

Colab runtimes do not have Rust/Cargo in their default PATH. Run this Python block in Colab to install Rustup, configure your toolchain, and set up your system PATH.

```python
# Cell 1: Install Rustup and configure system PATH
!curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

import os
os.environ["PATH"] += ":/root/.cargo/bin"

# Verify installation
!cargo --version
```

### B. Clone the Repository (with Submodules)

Saccade relies on a patched version of Hugging Face Candle located in the `candle/` submodule. It is **critical** to clone recursively to fetch the submodule contents and audio assets.

```bash
# Cell 2: Clone repository recursively
!git clone --recursive https://github.com/TawanaState/saccade-x2.git
%cd saccade-x2
```

> [!NOTE]
> If you have already cloned the repository without the submodules, run the following command in the repository root to initialize them:
> ```bash
> !git submodule update --init --recursive
> ```

---

## 2. Running the Test Suite

Once the environment is configured, you can run the primary Saccade binaries. All tests can be run on CPU or GPU runtimes (though Saccade is optimized for high-performance CPU execution).

### A. Model Compilation (`saccade-compile`)

The [compile.rs](file:///C:/Users/user/Desktop/WORK/RESEARCH/saccade-x2/saccade-runner/src/bin/compile.rs) script fetches a standard Hugging Face transformer model, calibrates it using a dataset, and compresses it into the token-adaptive Saccade format.

Run the following command to download `Qwen/Qwen2.5-0.5B-Instruct` and compile it into a Saccade checkpoint:

```bash
# Cell 3: Compile Qwen model to Saccade format
!cargo run --release --bin saccade-compile -- \
  --model-id "Qwen/Qwen2.5-0.5B-Instruct" \
  --output-path "saccade_qwen.safetensors"
```

#### Key Arguments:
- `--model-id`: Hugging Face repository name.
- `--output-path`: Target output file name (defaults to `saccade_qwen.safetensors` for verification).
- `--target-fill`: Fill rate for coordinate-masked sparse precision updates (default: `0.15` / 15%).
- `--pct-t4` / `--pct-t8`: Percentiles for routing thresholds (defaults: `0.80` / `0.95`).

---

### B. Accuracy and Performance Verification (`verify`)

The [verify.rs](file:///C:/Users/user/Desktop/WORK/RESEARCH/saccade-x2/saccade-runner/src/bin/verify.rs) binary verifies that the token-adaptive routing engine functions correctly. It runs two evaluation passes for comparative metrics:
1. **Phase A (C-TARQ Routing)**: Dynamic token routing using compressed 4-bit weights and sparse updates.
2. **Phase B (Bypass Mode)**: Fully dequantized standard dense matrix multiplication.

It then performs a logit-level audit to ensure numerical correctness.

```bash
# Cell 4: Run verification suite
!cargo run --release --bin verify -- \
  --checkpoint "saccade_qwen.safetensors" \
  --max-tokens 30
```

#### Expected Report Output:
- **Numerical Accuracy**: Average logit Cosine Similarity (Target: `> 0.998`) and RMSE (Target: `< 0.005`).
- **Speedup & Budget**: Kernel speedup multiplier and average bits-per-token (BPT) budget.
- **Verification Status**: `VERIFICATION SUCCESSFUL` if accuracy bounds are maintained.

---

### C. Qwen Text Generation Example (`qwen_example`)

The [qwen_example.rs](file:///C:/Users/user/Desktop/WORK/RESEARCH/saccade-x2/saccade-runner/src/bin/qwen_example.rs) script is a self-contained demonstration. It loads the `Qwen2.5-0.5B-Instruct` model, generates synthetic profile calibrations, runs on-the-fly Saccade compilation, saves the checkpoint, and compares text generation results.

```bash
# Cell 5: Run Qwen example
!cargo run --release --bin qwen_example
```

> [!TIP]
> This command is perfect for checking if the entire framework (from downloading to intercepting linear layers to generating text) compiles and runs end-to-end without needing manual threshold setup.

---

### D. Whisper Audio Transcription Test (`whisper_example`)

The [whisper_example.rs](file:///C:/Users/user/Desktop/WORK/RESEARCH/saccade-x2/saccade-runner/src/bin/whisper_example.rs) script showcases Saccade on audio/speech models. It downloads `openai/whisper-tiny`, decodes a mono WAV file, calculates log-mel spectrogram features, performs calibration and compilation on real activations, and benchmarks encoder performance.

```bash
# Cell 6: Run Whisper audio example
!cargo run --release --bin whisper_example
```

> [!IMPORTANT]
> The whisper example reads audio mel filter assets from `candle/candle-examples/examples/whisper/melfilters.bytes`. This file will only be present if the submodules were cloned recursively.

---

## 3. High-Performance Host Optimizations

To force the compiler to utilize vectorized CPU instruction paths (AVX2, FMA, etc.) on Colab's Intel Xeon or AMD EPYC CPU nodes, prepend `RUSTFLAGS="-C target-cpu=native"` to your compilation commands:

```bash
# Cell 7: Compile with native CPU target optimizations
!RUSTFLAGS="-C target-cpu=native" cargo build --release
```

Using `target-cpu=native` triggers native SIMD instruction generation, which dramatically speeds up the low-level bit-unpacking and sparse matrix patching loops in Saccade's [CustomOp1](file:///C:/Users/user/Desktop/WORK/RESEARCH/saccade-x2/GUIDE.md#L58-L65) intercepts.
