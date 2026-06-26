use candle_core::{DType, Device, Tensor};
use saccade_core::{SaccadeConfig, SaccadeEngine, calibration::ProfileRunner, variance_heuristic};
use hf_hub::api::sync::Api;

/// Tracks per-token routing decisions for dynamic BPT calculation.
/// Each token is classified into one of three precision tiers based on
/// its activation variance relative to the calibrated thresholds.
struct RouteStats {
    base_only: usize,
    delta_q8: usize,
    delta_fp16: usize,
}

impl RouteStats {
    fn classify(
        tokens: &[half::f16],
        hidden_dim: usize,
        batch_tokens: usize,
        t4: f32,
        t8: f32,
    ) -> Self {
        let mut base_only = 0usize;
        let mut delta_q8 = 0usize;
        let mut delta_fp16 = 0usize;

        for t in 0..batch_tokens {
            let offset = t * hidden_dim;
            let slice = &tokens[offset..offset + hidden_dim];
            let score = variance_heuristic(slice);

            if score >= t8 {
                delta_fp16 += 1;
            } else if score >= t4 {
                delta_q8 += 1;
            } else {
                base_only += 1;
            }
        }
        Self { base_only, delta_q8, delta_fp16 }
    }

    fn total(&self) -> usize {
        self.base_only + self.delta_q8 + self.delta_fp16
    }

    /// Computes the average bits-per-token across the routing distribution.
    /// Base-only tokens cost 4 bits/weight; delta tokens add the sparse CSR overhead
    /// proportional to the number of non-zero correction entries.
    fn avg_bpt(&self, nnz: usize, total_params: usize) -> f32 {
        let total = self.total() as f32;
        if total == 0.0 {
            return 4.0;
        }
        let base_bits = 4.0f32;
        let delta_overhead = if total_params > 0 {
            (nnz as f32 * 8.0) / (total_params as f32)
        } else {
            0.0
        };
        let bpt_base = base_bits;
        let bpt_delta = base_bits + delta_overhead;

        ((self.base_only as f32 * bpt_base)
            + (self.delta_q8 as f32 * bpt_delta)
            + (self.delta_fp16 as f32 * bpt_delta))
            / total
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;

    println!("================================================================");
    println!("  Saccade V3 C-TARQ — Comparative Benchmarking Harness");
    println!("================================================================\n");

    // ----------------------------------------------------------------
    // Phase 1: Model Acquisition
    // ----------------------------------------------------------------
    println!("=== Phase 1: Downloading Qwen2-1.5B-Instruct Model ===");
    let api = Api::new()?;
    let repo = api.model("Qwen/Qwen2-1.5B-Instruct".to_string());
    let model_file = repo.get("model.safetensors")?;
    println!("Downloaded to: {:?}", model_file);

    let mut tensors = candle_core::safetensors::load(model_file, &device)?;

    // ----------------------------------------------------------------
    // Phase 2: Offline Calibration
    // ----------------------------------------------------------------
    println!("\n=== Phase 2: Offline Calibration ===");

    let target_name = "model.layers.0.mlp.down_proj";

    // Dynamically read the weight dimensions from the actual model tensor
    // rather than hardcoding them — this adapts to any Qwen variant.
    let target_weight = tensors
        .get(&format!("{}.weight", target_name))
        .expect("Target weight must exist in model");
    let target_dims = target_weight.shape().dims();
    let hidden_size = target_dims[1]; // in_features of down_proj = intermediate_size
    println!("Target layer: {} ({}x{})", target_name, target_dims[0], target_dims[1]);

    let calib_tokens = 300;

    // Calibration data with variance spread across the full hidden dimension.
    // Three bands simulate the expected activation distribution in production:
    //   - 15% high-variance (code-like tokens)
    //   - 65% medium-variance (reasoning tokens)
    //   - 20% low-variance (predictable prose)
    let mut calib_data = vec![half::f16::from_f32(0.01); calib_tokens * hidden_size];

    for t in 0..45 {
        for h in 0..hidden_size {
            let sign = if h % 2 == 0 { 1.0 } else { -1.0 };
            calib_data[t * hidden_size + h] = half::f16::from_f32(sign * 0.65);
        }
    }
    for t in 45..240 {
        for h in 0..hidden_size {
            let sign = if h % 2 == 0 { 1.0 } else { -1.0 };
            calib_data[t * hidden_size + h] = half::f16::from_f32(sign * 0.12);
        }
    }

    let calib_tensor = Tensor::from_vec(calib_data, (calib_tokens, hidden_size), &device)?;
    let (t4, t8) = ProfileRunner::calibrate(&calib_tensor, 0.20, 0.85)?;
    println!("Extracted thresholds: t4 = {:.6}, t8 = {:.6}", t4, t8);

    tensors.insert(
        format!("{}.saccade_t4", target_name),
        Tensor::from_vec(vec![t4], (1,), &device)?,
    );
    tensors.insert(
        format!("{}.saccade_t8", target_name),
        Tensor::from_vec(vec![t8], (1,), &device)?,
    );

    // ----------------------------------------------------------------
    // Phase 3: Engine Compilation
    // ----------------------------------------------------------------
    println!("\n=== Phase 3: Model Compression via SaccadeEngine ===");

    let config = SaccadeConfig {
        t4: 999.0,
        t8: 999.0,
        block_size: 16,
        heuristic: variance_heuristic,
    };

    let target_substrings = vec![target_name];
    let compiled_layers = SaccadeEngine::compile_model_topology(&tensors, &target_substrings, config)?;
    println!("Compiled {} layer(s).", compiled_layers.len());

    let saccade_linear = compiled_layers
        .get(target_name)
        .expect("Target layer must be compiled");

    let nnz = saccade_linear
        .sparse_delta_q8
        .as_ref()
        .map(|sp| {
            let (v_store, _) = sp.values.storage_and_layout();
            match &*v_store {
                candle_core::Storage::Cpu(cpu) => cpu.as_slice::<u8>().map(|s| s.len()).unwrap_or(0),
                _ => 0,
            }
        })
        .unwrap_or(0);
    let total_params = saccade_linear.out_features * saccade_linear.in_features;
    let active_t4 = saccade_linear.config.t4;
    let active_t8 = saccade_linear.config.t8;

    println!("Active thresholds -> t4: {:.6}, t8: {:.6}", active_t4, active_t8);
    println!(
        "Sparse delta NNZ: {} / {} total params ({:.2}% sparsity)",
        nnz, total_params,
        if total_params > 0 { (1.0 - (nnz as f64 / total_params as f64)) * 100.0 } else { 0.0 }
    );

    // Retain the original dense weight for vanilla baseline comparisons
    let original_weight = tensors
        .get(&format!("{}.weight", target_name))
        .expect("Original weight tensor must exist")
        .clone();

    // ----------------------------------------------------------------
    // Phase 4: Build Test Inputs
    // ----------------------------------------------------------------
    println!("\n=== Phase 4: Constructing Test Inputs ===");

    let hidden_dim = saccade_linear.in_features;

    // Test tokens designed with dimensionally-spread variance so the heuristic
    // can accurately classify them. Single-index variance injection (prior bug)
    // produced near-zero variance across 4,864 dimensions.

    // Prose: uniform values => variance ≈ 0 => routes to base-only
    let prose_data_1 = vec![half::f16::from_f32(0.02); hidden_dim];

    // Logic: alternating ±0.12 => medium variance => routes to Q8 delta
    let mut logic_data_1 = vec![half::f16::from_f32(0.0); hidden_dim];
    for h in 0..hidden_dim {
        let sign = if h % 2 == 0 { 1.0 } else { -1.0 };
        logic_data_1[h] = half::f16::from_f32(sign * 0.12);
    }

    // Code: alternating ±0.65 => high variance => routes to FP16/fallback delta
    let mut code_data_1 = vec![half::f16::from_f32(0.0); hidden_dim];
    for h in 0..hidden_dim {
        let sign = if h % 2 == 0 { 1.0 } else { -1.0 };
        code_data_1[h] = half::f16::from_f32(sign * 0.65);
    }

    // Verify routing classification on single tokens
    let prose_routes_1 = RouteStats::classify(&prose_data_1, hidden_dim, 1, active_t4, active_t8);
    let logic_routes_1 = RouteStats::classify(&logic_data_1, hidden_dim, 1, active_t4, active_t8);
    let code_routes_1 = RouteStats::classify(&code_data_1, hidden_dim, 1, active_t4, active_t8);

    println!("  Prose routing:  base={}, q8={}, fp16={}", prose_routes_1.base_only, prose_routes_1.delta_q8, prose_routes_1.delta_fp16);
    println!("  Logic routing:  base={}, q8={}, fp16={}", logic_routes_1.base_only, logic_routes_1.delta_q8, logic_routes_1.delta_fp16);
    println!("  Code  routing:  base={}, q8={}, fp16={}", code_routes_1.base_only, code_routes_1.delta_q8, code_routes_1.delta_fp16);

    // Build batched (GEMM) and single-token (GEMV) tensors
    let gemm_batch = 10;
    let gemv_iters = 50; // Simulate 50-step autoregressive decode

    // GEMM tensors: 10 tokens batched
    let prose_gemm = Tensor::from_vec(prose_data_1.repeat(gemm_batch), (gemm_batch, hidden_dim), &device)?;
    let logic_gemm = Tensor::from_vec(logic_data_1.repeat(gemm_batch), (gemm_batch, hidden_dim), &device)?;
    let code_gemm = Tensor::from_vec(code_data_1.repeat(gemm_batch), (gemm_batch, hidden_dim), &device)?;

    // GEMV tensors: single token
    let prose_gemv = Tensor::from_vec(prose_data_1.clone(), (1, hidden_dim), &device)?;
    let logic_gemv = Tensor::from_vec(logic_data_1.clone(), (1, hidden_dim), &device)?;
    let code_gemv = Tensor::from_vec(code_data_1.clone(), (1, hidden_dim), &device)?;

    let prose_routes_b = RouteStats::classify(&prose_data_1.repeat(gemm_batch), hidden_dim, gemm_batch, active_t4, active_t8);
    let logic_routes_b = RouteStats::classify(&logic_data_1.repeat(gemm_batch), hidden_dim, gemm_batch, active_t4, active_t8);
    let code_routes_b = RouteStats::classify(&code_data_1.repeat(gemm_batch), hidden_dim, gemm_batch, active_t4, active_t8);

    // Prepare vanilla baseline
    let weight_f16 = original_weight.to_dtype(DType::F16)?;
    let weight_t = weight_f16.t()?;

    let vanilla_mem_bytes = {
        let w_dims = weight_f16.dims();
        w_dims[0] * w_dims[1] * 2
    };
    let saccade_base_bytes = total_params / 2;
    let saccade_scale_bytes = saccade_linear.out_features * 2;
    let saccade_delta_bytes = nnz * (4 + 1)
        + (saccade_linear.out_features + 1) * 4
        + 2;
    let saccade_total_bytes = saccade_base_bytes + saccade_scale_bytes + saccade_delta_bytes;

    // ================================================================
    // Phase 5: GEMM Benchmark (batch=10, matrix-matrix multiply)
    //
    // This scenario represents prefill / prompt encoding where multiple
    // tokens are processed simultaneously. Dense FP16 matmul is highly
    // optimized for this layout (parallel, compute-heavy). Saccade's
    // advantage here is memory footprint, not raw throughput.
    // ================================================================
    println!("\n=== Phase 5: GEMM Benchmark (batch={}) ===", gemm_batch);
    println!("Matrix-matrix multiply — favors dense FP16 parallelism.\n");

    let gemm_tests = vec![
        ("Prose (Low Volatility)", &prose_gemm, &prose_routes_b),
        ("Logic (Medium Volatility)", &logic_gemm, &logic_routes_b),
        ("Code  (High Volatility)", &code_gemm, &code_routes_b),
    ];

    // Vanilla GEMM
    for &(desc, ref input, _) in &gemm_tests {
        let input_f16 = input.to_dtype(DType::F16)?;
        let _ = input_f16.matmul(&weight_t)?; // warm-up
        let start = std::time::Instant::now();
        let _out = input_f16.matmul(&weight_t)?;
        let elapsed = start.elapsed();
        println!(
            "  [Vanilla GEMM] {:<28} {:>9.3}ms  ({:.0} tok/s)",
            desc, elapsed.as_secs_f64() * 1000.0,
            gemm_batch as f64 / elapsed.as_secs_f64(),
        );
    }

    // Saccade GEMM
    for &(desc, ref input, ref routes) in &gemm_tests {
        let _ = input.apply_op1_no_bwd(saccade_linear)?; // warm-up
        let start = std::time::Instant::now();
        let _out = input.apply_op1_no_bwd(saccade_linear)?;
        let elapsed = start.elapsed();
        let bpt = routes.avg_bpt(nnz, total_params);
        println!(
            "  [Saccade GEMM] {:<28} {:>9.3}ms  ({:.0} tok/s)  BPT: {:.2}",
            desc, elapsed.as_secs_f64() * 1000.0,
            gemm_batch as f64 / elapsed.as_secs_f64(), bpt,
        );
    }

    // ================================================================
    // Phase 6: GEMV Benchmark (batch=1, autoregressive decoding)
    //
    // This is the scenario Saccade was engineered for. Autoregressive
    // text generation processes one token at a time, turning the workload
    // into a matrix-vector product (GEMV). GEMV has low arithmetic
    // intensity and is dominated by memory-bandwidth — exactly where
    // Saccade's 4-bit packed weights reduce DRAM traffic.
    //
    // We loop `gemv_iters` single-token passes to get stable timing,
    // simulating a real decode sequence.
    // ================================================================
    println!("\n=== Phase 6: GEMV Benchmark (batch=1, {} iterations) ===", gemv_iters);
    println!("Matrix-vector product — simulates autoregressive decoding.\n");

    let gemv_tests = vec![
        ("Prose (Low Volatility)", &prose_gemv, &prose_routes_1),
        ("Logic (Medium Volatility)", &logic_gemv, &logic_routes_1),
        ("Code  (High Volatility)", &code_gemv, &code_routes_1),
    ];

    // Vanilla GEMV
    for &(desc, ref input, _) in &gemv_tests {
        let input_f16 = input.to_dtype(DType::F16)?;
        // Warm up the pipeline
        for _ in 0..5 { let _ = input_f16.matmul(&weight_t)?; }

        let start = std::time::Instant::now();
        for _ in 0..gemv_iters {
            let _out = input_f16.matmul(&weight_t)?;
        }
        let elapsed = start.elapsed();
        let per_token_us = elapsed.as_micros() as f64 / gemv_iters as f64;
        println!(
            "  [Vanilla GEMV] {:<28} {:>9.1}us/tok  ({:.0} tok/s)",
            desc, per_token_us,
            1_000_000.0 / per_token_us,
        );
    }

    // Saccade GEMV
    for &(desc, ref input, ref routes) in &gemv_tests {
        // Warm up
        for _ in 0..5 { let _ = input.apply_op1_no_bwd(saccade_linear)?; }

        let start = std::time::Instant::now();
        for _ in 0..gemv_iters {
            let _out = input.apply_op1_no_bwd(saccade_linear)?;
        }
        let elapsed = start.elapsed();
        let per_token_us = elapsed.as_micros() as f64 / gemv_iters as f64;
        let bpt = routes.avg_bpt(nnz, total_params);
        println!(
            "  [Saccade GEMV] {:<28} {:>9.1}us/tok  ({:.0} tok/s)  BPT: {:.2}",
            desc, per_token_us,
            1_000_000.0 / per_token_us, bpt,
        );
    }

    // ----------------------------------------------------------------
    // Phase 7: Comparative Summary
    // ----------------------------------------------------------------
    println!("\n=== Phase 7: Comparative Summary ===");
    let compression_ratio = vanilla_mem_bytes as f64 / saccade_total_bytes as f64;
    println!(
        "  Memory: {:.2} MB (FP16) -> {:.2} MB (Saccade) = {:.1}x compression",
        vanilla_mem_bytes as f64 / (1024.0 * 1024.0),
        saccade_total_bytes as f64 / (1024.0 * 1024.0),
        compression_ratio,
    );
    println!("  Vanilla BPT: 16.00 (dense FP16, all profiles)");
    println!("  Saccade BPT [Prose]: {:.2} (base-only)", prose_routes_1.avg_bpt(nnz, total_params));
    println!("  Saccade BPT [Logic]: {:.2} (Q8 delta)", logic_routes_1.avg_bpt(nnz, total_params));
    println!("  Saccade BPT [Code]:  {:.2} (FP16 fallback)", code_routes_1.avg_bpt(nnz, total_params));

    println!(
        "\n  NOTE: This benchmark targets a single layer ({:.2} MB dense). On CPUs",
        vanilla_mem_bytes as f64 / (1024.0 * 1024.0)
    );
    println!("  with large L3 caches, the weights may remain cache-resident, hiding");
    println!("  DRAM bandwidth costs. Saccade's compression advantage grows when");
    println!("  full-model inference forces weight streaming from main memory.");

    println!("\nDone.");
    Ok(())
}
