use candle_core::{DType, Device, Tensor};
use saccade_core::{SaccadeConfig, SaccadeEngine};
use hf_hub::api::sync::Api;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;

    println!("=== Phase 1: Downloading Qwen2-0.5B-Instruct Model ===");
    let api = Api::new()?;
    let repo = api.model("Qwen/Qwen2-0.5B-Instruct".to_string());

    // We only need the safetensors weights
    let model_file = repo.get("model.safetensors")?;
    println!("Downloaded to: {:?}", model_file);

    // Load the model weights raw directly into a map for compiler
    let tensors = candle_core::safetensors::load(model_file, &device)?;

    println!("\n=== Phase 2: Real Model Compression via SaccadeEngine ===");
    // Configure the Saccade runtime engine heuristics
    let config = SaccadeConfig {
        t4: 0.1, // A very low threshold to trigger deltas for our test token
        t8: 999.0, // FP16 threshold not used
        block_size: 16,
        heuristic: saccade_core::variance_heuristic,
    };

    let target_substrings = vec!["model.layers.0.mlp.down_proj"];

    let compiled_layers = SaccadeEngine::compile_model_topology(
        &tensors,
        &target_substrings,
        config,
    )?;

    println!("Successfully compiled {} layers.", compiled_layers.len());

    println!("\n=== Phase 3: Online Inference Execution & Comparison ===");
    let target_name = "model.layers.0.mlp.down_proj";
    let saccade_linear = compiled_layers.get(target_name).expect("Layer should be compiled");

    // Generate test activations
    // We'll use a batch of 2 tokens.
    let batch_tokens = 2;
    let hidden_size = 4864;
    let mut activation_data = vec![half::f16::from_f32(0.01); batch_tokens * hidden_size];

    // Inject variance into token 1
    activation_data[1 * hidden_size + 0] = half::f16::from_f32(5.0);

    let incoming_activations = Tensor::from_vec(activation_data, (batch_tokens, hidden_size), &device)?;

    println!("Input Activation Shape: {:?}", incoming_activations.shape());

    // Execute Saccade forward pass
    let start_saccade = std::time::Instant::now();
    let output_saccade = incoming_activations.apply_op1_no_bwd(saccade_linear)?;
    let saccade_duration = start_saccade.elapsed();

    // Execute Dense forward pass
    // y = x @ W.t()
    let dense_weight = tensors.get("model.layers.0.mlp.down_proj.weight").unwrap();
    let start_dense = std::time::Instant::now();
    // Qwen model weights might be loaded as BF16, let's cast to F16 for parity
    let dense_weight_f16 = dense_weight.to_dtype(DType::F16)?;
    let dense_weight_t = dense_weight_f16.transpose(0, 1)?;
    let output_dense = incoming_activations.matmul(&dense_weight_t)?;
    let dense_duration = start_dense.elapsed();

    println!("Output Projection Shape: {:?}", output_saccade.shape());
    println!("Saccade Engine Execution Time: {:?}", saccade_duration);
    println!("Dense Engine Execution Time: {:?}", dense_duration);

    // Calculate error
    let saccade_f32 = output_saccade.to_dtype(DType::F32)?;
    let dense_f32 = output_dense.to_dtype(DType::F32)?;
    let diff = saccade_f32.sub(&dense_f32)?;
    let sq_diff = diff.mul(&diff)?;
    let sum_sq = sq_diff.sum_all()?.to_scalar::<f32>()?;
    let mse = sum_sq / (batch_tokens * 896) as f32;

    println!("Mean Squared Error vs Dense: {:.6}", mse);

    // Memory footprint comparison
    // Original Dense FP16: 896 * 4864 * 2 bytes
    let dense_bytes = 896 * 4864 * 2;
    // Compressed Saccade:
    // packed_base: 896 * (4864 / 8) * 4 bytes
    let packed_bytes = 896 * (4864 / 8) * 4;
    // scale_base: 896 * 2 bytes
    let scale_bytes = 896 * 2;

    // In a true sparse format, it would be nnz * (1 byte value + 4 byte coord).
    let mut true_sparse_delta_bytes = 0;
    if let Some(sp) = &saccade_linear.sparse_delta_q8 {
        let nnz = sp.values.elem_count();
        // values (1 byte) + col_indices (4 bytes) + row_ptrs ((896 + 1) * 4 bytes)
        true_sparse_delta_bytes = nnz * 1 + nnz * 4 + (896 + 1) * 4;
    }

    let total_saccade_bytes = packed_bytes + scale_bytes + true_sparse_delta_bytes;

    println!("\n=== Memory Footprint Comparison ===");
    println!("Original Dense FP16 Footprint:  {} bytes", dense_bytes);
    println!("Saccade True Sparse Footprint:  {} bytes ({} packed + {} scale + {} sparse delta)",
             total_saccade_bytes, packed_bytes, scale_bytes, true_sparse_delta_bytes);
    println!("Compression Ratio: {:.2}x", dense_bytes as f32 / total_saccade_bytes as f32);

    Ok(())
}
