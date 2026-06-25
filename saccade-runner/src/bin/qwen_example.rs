use candle_core::{DType, Device, Tensor};
use saccade_core::{SaccadeConfig, SaccadeLinearOp, compress_tensor_to_saccade};
use hf_hub::api::sync::Api;
use candle_nn::VarBuilder;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;

    println!("=== Phase 1: Downloading Qwen2-0.5B-Instruct Model ===");
    let api = Api::new()?;
    let repo = api.model("Qwen/Qwen2-0.5B-Instruct".to_string());

    // We only need the safetensors weights
    let model_file = repo.get("model.safetensors")?;
    println!("Downloaded to: {:?}", model_file);

    // Load the model weights
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_file], DType::F16, &device)? };

    // Extract a specific target layer, e.g., mlp.down_proj of layer 0
    println!("Extracting model.layers.0.mlp.down_proj.weight...");
    let dense_weight = vb.get((896, 4864), "model.layers.0.mlp.down_proj.weight")?;

    println!("Extracted Shape: {:?}", dense_weight.shape());

    println!("\n=== Phase 2: Real Model Compression ===");
    // Compress the tensor using our custom pipeline.
    // We use an error threshold of 0.05. If quantization error > 0.05, save the delta.
    let delta_threshold = 0.05;
    let compressed_tensors = compress_tensor_to_saccade(&dense_weight, delta_threshold)?;

    let model_path = "compressed_qwen_layer.safetensors";
    candle_core::safetensors::save(&compressed_tensors, model_path)?;
    println!("Successfully saved compressed model artifacts to `{}`", model_path);

    println!("\n=== Phase 3: Online Inference Execution & Comparison ===");
    // Load the compressed model back
    let loaded_tensors = candle_core::safetensors::load(model_path, &device)?;
    let loaded_packed_base = loaded_tensors.get("packed_base").unwrap().clone();
    let loaded_scale_base = loaded_tensors.get("scale_base").unwrap().clone();
    let loaded_delta_q8 = loaded_tensors.get("delta_q8").unwrap().clone();

    // Configure the Saccade runtime engine heuristics
    let config = SaccadeConfig {
        t4: 0.1, // A very low threshold to trigger deltas for our test token
        t8: 999.0, // FP16 threshold not used
        block_size: 16,
        heuristic: saccade_core::variance_heuristic,
    };

    let saccade_linear = SaccadeLinearOp {
        packed_base: loaded_packed_base,
        scale_base: loaded_scale_base,
        delta_q8_blocks: loaded_delta_q8,
        delta_q8_scales: None,
        delta_fp16_blocks: None,
        config,
        out_features: 896,
        in_features: 4864,
    };

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
    let output_saccade = incoming_activations.apply_op1_no_bwd(&saccade_linear)?;
    let saccade_duration = start_saccade.elapsed();

    // Execute Dense forward pass
    // y = x @ W.t()
    let start_dense = std::time::Instant::now();
    let dense_weight_t = dense_weight.transpose(0, 1)?;
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
    // delta_q8 (dense storage for now): 896 * 4864 * 2 bytes.
    // In a true sparse format, it would be nnz * (2 byte value + 2 byte coord).
    // Let's count the actual non-zeros from our generation pipeline to represent true sparse footprint
    let d8_f32 = loaded_tensors.get("delta_q8").unwrap().to_dtype(DType::F32)?.to_vec2::<f32>()?;
    let mut nnz = 0;
    for r in 0..896 {
        for c in 0..4864 {
            if d8_f32[r][c].abs() > 0.0 {
                nnz += 1;
            }
        }
    }
    let true_sparse_delta_bytes = nnz * 4; // 2 bytes value + 2 bytes coord

    let total_saccade_bytes = packed_bytes + scale_bytes + true_sparse_delta_bytes;

    println!("\n=== Memory Footprint Comparison ===");
    println!("Original Dense FP16 Footprint:  {} bytes", dense_bytes);
    println!("Saccade True Sparse Footprint:  {} bytes ({} packed + {} scale + {} sparse delta)",
             total_saccade_bytes, packed_bytes, scale_bytes, true_sparse_delta_bytes);
    println!("Compression Ratio: {:.2}x", dense_bytes as f32 / total_saccade_bytes as f32);

    // Clean up temporary execution artifact
    let _ = std::fs::remove_file(model_path);
    Ok(())
}
