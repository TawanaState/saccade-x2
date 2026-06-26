use candle_core::{DType, Device, Tensor};
use saccade_core::{SaccadeConfig, SaccadeEngine, calibration::ProfileRunner, variance_heuristic};
use hf_hub::api::sync::Api;

// Tracks per-token routing decisions for BPT calculation.
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
    println!("=== Phase 1: Downloading Qwen2-0.5B-Instruct Model ===");
    let api = Api::new()?;
    let repo = api.model("Qwen/Qwen2-0.5B-Instruct".to_string());
    let model_file = repo.get("model.safetensors")?;
    println!("Downloaded to: {:?}", model_file);

    let mut tensors = candle_core::safetensors::load(model_file, &device)?;

    // ----------------------------------------------------------------
    // Phase 2: Offline Calibration
    // ----------------------------------------------------------------
    println!("\n=== Phase 2: Offline Calibration ===");

    let calib_tokens = 300;
    let hidden_size = 4864;

    // Build calibration data with variance spread across the full hidden dimension
    // rather than concentrating signal in a single index.
    let mut calib_data = vec![half::f16::from_f32(0.01); calib_tokens * hidden_size];

    // High-variance band: ~15% of tokens — alternating large amplitudes across all dims
    for t in 0..45 {
        for h in 0..hidden_size {
            let sign = if h % 2 == 0 { 1.0 } else { -1.0 };
            calib_data[t * hidden_size + h] = half::f16::from_f32(sign * 0.65);
        }
    }
    // Medium-variance band: ~65% of tokens
    for t in 45..240 {
        for h in 0..hidden_size {
            let sign = if h % 2 == 0 { 1.0 } else { -1.0 };
            calib_data[t * hidden_size + h] = half::f16::from_f32(sign * 0.12);
        }
    }
    // Remaining 60 tokens stay flat at 0.01 (low variance)

    let calib_tensor = Tensor::from_vec(calib_data, (calib_tokens, hidden_size), &device)?;

    let (t4, t8) = ProfileRunner::calibrate(&calib_tensor, 0.20, 0.85)?;
    println!("Extracted thresholds: t4 = {:.6}, t8 = {:.6}", t4, t8);

    let target_name = "model.layers.0.mlp.down_proj";
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

    // Extract sparse NNZ count for BPT calculation
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

    println!(
        "Active thresholds -> t4: {:.6}, t8: {:.6}",
        active_t4, active_t8
    );
    println!(
        "Sparse delta NNZ: {} / {} total params ({:.2}% sparsity)",
        nnz,
        total_params,
        if total_params > 0 {
            (1.0 - (nnz as f64 / total_params as f64)) * 100.0
        } else {
            0.0
        }
    );

    // Keep the original weight tensor for the vanilla baseline
    let original_weight = tensors
        .get(&format!("{}.weight", target_name))
        .expect("Original weight tensor must exist")
        .clone();

    // ----------------------------------------------------------------
    // Phase 4: Build Test Inputs
    // ----------------------------------------------------------------
    println!("\n=== Phase 4: Constructing Test Inputs ===");

    let test_size = 10;
    let hidden_dim = saccade_linear.in_features;

    // Prose: flat distribution — all values identical => variance ~ 0
    let prose_data = vec![half::f16::from_f32(0.02); test_size * hidden_dim];
    let prose_tensor = Tensor::from_vec(prose_data.clone(), (test_size, hidden_dim), &device)?;

    // Logic: alternating moderate amplitudes => medium variance
    let mut logic_data = vec![half::f16::from_f32(0.0); test_size * hidden_dim];
    for t in 0..test_size {
        for h in 0..hidden_dim {
            let sign = if h % 2 == 0 { 1.0 } else { -1.0 };
            logic_data[t * hidden_dim + h] = half::f16::from_f32(sign * 0.12);
        }
    }
    let logic_tensor = Tensor::from_vec(logic_data.clone(), (test_size, hidden_dim), &device)?;

    // Code: high-amplitude oscillations => high variance
    let mut code_data = vec![half::f16::from_f32(0.0); test_size * hidden_dim];
    for t in 0..test_size {
        for h in 0..hidden_dim {
            let sign = if h % 2 == 0 { 1.0 } else { -1.0 };
            code_data[t * hidden_dim + h] = half::f16::from_f32(sign * 0.65);
        }
    }
    let code_tensor = Tensor::from_vec(code_data.clone(), (test_size, hidden_dim), &device)?;

    // Verify routing classification
    let prose_routes = RouteStats::classify(&prose_data, hidden_dim, test_size, active_t4, active_t8);
    let logic_routes = RouteStats::classify(&logic_data, hidden_dim, test_size, active_t4, active_t8);
    let code_routes = RouteStats::classify(&code_data, hidden_dim, test_size, active_t4, active_t8);

    println!(
        "  Prose routing:  base={}, q8={}, fp16={}",
        prose_routes.base_only, prose_routes.delta_q8, prose_routes.delta_fp16
    );
    println!(
        "  Logic routing:  base={}, q8={}, fp16={}",
        logic_routes.base_only, logic_routes.delta_q8, logic_routes.delta_fp16
    );
    println!(
        "  Code  routing:  base={}, q8={}, fp16={}",
        code_routes.base_only, code_routes.delta_q8, code_routes.delta_fp16
    );

    let test_inputs = vec![
        ("Prose (Low Volatility)", &prose_tensor, &prose_data, &prose_routes),
        ("Logic (Medium Volatility)", &logic_tensor, &logic_data, &logic_routes),
        ("Code  (High Volatility)", &code_tensor, &code_data, &code_routes),
    ];

    // ----------------------------------------------------------------
    // Phase 5: Run 1 — Vanilla Dense FP16 Baseline
    // ----------------------------------------------------------------
    println!("\n=== Phase 5: Run 1 — Vanilla Dense FP16 Baseline ===");
    println!("Executing standard Y = X * W^T using native Candle matmul.\n");

    let weight_f16 = original_weight.to_dtype(DType::F16)?;
    let weight_t = weight_f16.t()?;

    let vanilla_mem_bytes = {
        let w_dims = weight_f16.dims();
        w_dims[0] * w_dims[1] * 2 // f16 = 2 bytes each
    };

    for &(desc, ref input, _, _) in &test_inputs {
        let input_f16 = input.to_dtype(DType::F16)?;

        // Warm-up pass
        let _ = input_f16.matmul(&weight_t)?;

        let start = std::time::Instant::now();
        let _out = input_f16.matmul(&weight_t)?;
        let elapsed = start.elapsed();

        let tokens_per_sec = test_size as f64 / elapsed.as_secs_f64();
        println!(
            "  [Vanilla] {:<30} {:>9.3}ms  ({:.0} tok/s)  BPT: 16.00",
            desc,
            elapsed.as_secs_f64() * 1000.0,
            tokens_per_sec,
        );
    }
    println!(
        "  [Vanilla] Weight memory footprint: {:.2} MB (dense FP16)",
        vanilla_mem_bytes as f64 / (1024.0 * 1024.0)
    );

    // ----------------------------------------------------------------
    // Phase 6: Run 2 — Saccade Adaptive C-TARQ Execution
    // ----------------------------------------------------------------
    println!("\n=== Phase 6: Run 2 — Saccade C-TARQ Adaptive Execution ===");
    println!("Executing with fused 4-bit register unpacking + sparse CSR deltas.\n");

    // Saccade memory footprint: packed_base (4 bits/param) + scales + sparse delta overhead
    let saccade_base_bytes = total_params / 2; // 4 bits per weight
    let saccade_scale_bytes = saccade_linear.out_features * 2; // f16 per row
    let saccade_delta_bytes = nnz * (4 + 1) // col_index(u32) + value(u8)
        + (saccade_linear.out_features + 1) * 4 // row_ptrs(u32)
        + 2; // scale(f16)
    let saccade_total_bytes = saccade_base_bytes + saccade_scale_bytes + saccade_delta_bytes;

    for &(desc, ref input, _, ref routes) in &test_inputs {
        // Warm-up pass
        let _ = input.apply_op1_no_bwd(saccade_linear)?;

        let start = std::time::Instant::now();
        let _out = input.apply_op1_no_bwd(saccade_linear)?;
        let elapsed = start.elapsed();

        let tokens_per_sec = test_size as f64 / elapsed.as_secs_f64();
        let bpt = routes.avg_bpt(nnz, total_params);
        println!(
            "  [Saccade] {:<30} {:>9.3}ms  ({:.0} tok/s)  BPT: {:.2}",
            desc,
            elapsed.as_secs_f64() * 1000.0,
            tokens_per_sec,
            bpt,
        );
    }
    println!(
        "  [Saccade] Weight memory footprint: {:.2} MB (4-bit packed + sparse CSR)",
        saccade_total_bytes as f64 / (1024.0 * 1024.0)
    );

    // ----------------------------------------------------------------
    // Phase 7: Comparative Summary
    // ----------------------------------------------------------------
    println!("\n=== Phase 7: Comparative Summary ===");
    let compression_ratio = vanilla_mem_bytes as f64 / saccade_total_bytes as f64;
    println!(
        "  Memory reduction: {:.2} MB -> {:.2} MB ({:.1}x compression)",
        vanilla_mem_bytes as f64 / (1024.0 * 1024.0),
        saccade_total_bytes as f64 / (1024.0 * 1024.0),
        compression_ratio,
    );
    println!(
        "  Vanilla BPT: 16.00 (dense FP16) across all profiles"
    );

    for &(desc, _, _, ref routes) in &test_inputs {
        let bpt = routes.avg_bpt(nnz, total_params);
        println!("  Saccade BPT [{}]: {:.2}", desc, bpt);
    }

    println!("\nDone.");
    Ok(())
}
