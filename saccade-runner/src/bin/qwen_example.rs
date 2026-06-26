use candle_core::{DType, Device, Tensor};
use saccade_core::{SaccadeConfig, SaccadeEngine, calibration::ProfileRunner, variance_heuristic};
use hf_hub::api::sync::Api;
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;

    println!("=== Phase 1: Downloading Qwen2-0.5B-Instruct Model ===");
    let api = Api::new()?;
    let repo = api.model("Qwen/Qwen2-0.5B-Instruct".to_string());

    // We only need the safetensors weights
    let model_file = repo.get("model.safetensors")?;
    println!("Downloaded to: {:?}", model_file);

    // Load the model weights raw directly into a map for compiler
    let mut tensors = candle_core::safetensors::load(model_file, &device)?;

    println!("\n=== Phase 2: Offline Calibration ===");
    // Generate mock calibration sequences covering different distribution structures
    let calib_tokens = 300;
    let hidden_size = 4864;
    let mut calib_data = vec![half::f16::from_f32(0.01); calib_tokens * hidden_size];
    
    // Inject high variance for ~15% of tokens simulating highly complex paths
    for t in 0..45 {
        calib_data[t * hidden_size + 0] = half::f16::from_f32(12.0);
    }
    // Inject medium variance for ~80% of tokens
    for t in 45..240 {
        calib_data[t * hidden_size + 0] = half::f16::from_f32(2.5);
    }
    // Rest remain flat low variance (predictable paths)

    let calib_tensor = Tensor::from_vec(calib_data, (calib_tokens, hidden_size), &device)?;
    
    // Target ~80% of tokens bypass dense and utilize T4, ~15% utilize T8
    let (t4, t8) = ProfileRunner::calibrate(&calib_tensor, 0.20, 0.85)?;
    println!("Offline Data-Driven Thresholds Extracted: t4 = {:.4}, t8 = {:.4}", t4, t8);

    // Embed data-driven thresholds straight into the model container registry
    let target_name = "model.layers.0.mlp.down_proj";
    tensors.insert(format!("{}.saccade_t4", target_name), Tensor::from_vec(vec![t4], (1,), &device)?);
    tensors.insert(format!("{}.saccade_t8", target_name), Tensor::from_vec(vec![t8], (1,), &device)?);

    println!("\n=== Phase 3: Real Model Compression via SaccadeEngine ===");
    // Configure the Saccade runtime engine default heuristics (to be overridden by map metadata)
    let config = SaccadeConfig {
        t4: 999.0, // High default to ensure override proves it works
        t8: 999.0, // High default
        block_size: 16,
        heuristic: variance_heuristic,
    };

    let target_substrings = vec![target_name];

    let compiled_layers = SaccadeEngine::compile_model_topology(
        &tensors,
        &target_substrings,
        config,
    )?;

    println!("Successfully compiled {} layers.", compiled_layers.len());
    let saccade_linear = compiled_layers.get(target_name).expect("Layer should be compiled");

    println!("\n=== Phase 4: Online Inference Benchmarks ===");
    
    // Test sets
    let test_size = 10;
    
    // 1. Prose Token (Predictable, Low Variance)
    let prose_data = vec![half::f16::from_f32(0.02); test_size * hidden_size];
    let prose_tensor = Tensor::from_vec(prose_data, (test_size, hidden_size), &device)?;
    
    // 2. Logic Token (Medium Variance)
    let mut logic_data = vec![half::f16::from_f32(0.01); test_size * hidden_size];
    for t in 0..test_size { logic_data[t * hidden_size + 0] = half::f16::from_f32(2.5); }
    let logic_tensor = Tensor::from_vec(logic_data, (test_size, hidden_size), &device)?;

    // 3. Code Token (High Volatility)
    let mut code_data = vec![half::f16::from_f32(0.01); test_size * hidden_size];
    for t in 0..test_size { code_data[t * hidden_size + 0] = half::f16::from_f32(15.0); }
    let code_tensor = Tensor::from_vec(code_data, (test_size, hidden_size), &device)?;

    let tests = vec![
        ("Prose (Low Volatility)", prose_tensor),
        ("Logic (Medium Volatility)", logic_tensor),
        ("Code (High Volatility)", code_tensor),
    ];

    for (desc, input) in tests {
        let start = std::time::Instant::now();
        let _out = input.apply_op1_no_bwd(saccade_linear)?;
        let duration = start.elapsed();
        println!("Benchmark [{}]: Execution Time = {:?}", desc, duration);
    }

    Ok(())
}
