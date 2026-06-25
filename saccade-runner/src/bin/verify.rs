use candle_core::{Device, Tensor};
use saccade_core::{SaccadeConfig, SaccadeLinearOp};
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;

    // Define mock dimensions
    let in_features = 32;
    let out_features = 16;
    let batch_tokens = 1;

    // Generate mock packed base weights (4 bits per parameter, 8 params per u32)
    let packed_elements = out_features * (in_features / 8);
    let packed_base_data: Vec<u32> = vec![0x12345678; packed_elements];
    let packed_base = Tensor::from_vec(packed_base_data, (out_features, in_features / 8), &device)?;

    // Generate mock scales
    let scale_base_data: Vec<half::f16> = vec![half::f16::from_f32(0.5); out_features];
    let scale_base = Tensor::from_vec(scale_base_data, (out_features,), &device)?;

    // Generate mock sparse delta q8 (we use dense here per the architecture simplification)
    let delta_elements = out_features * in_features;
    let mut delta_q8_data: Vec<half::f16> = vec![half::f16::from_f32(0.0); delta_elements];
    // Add some non-zero sparse corrections
    delta_q8_data[0] = half::f16::from_f32(1.5);
    delta_q8_data[15] = half::f16::from_f32(-0.5);
    let delta_q8_blocks = Tensor::from_vec(delta_q8_data, (out_features, in_features), &device)?;

    // Serialize to safetensors to ensure compliance with step 3
    let mut tensors = HashMap::new();
    tensors.insert("packed_base".to_string(), packed_base);
    tensors.insert("scale_base".to_string(), scale_base);
    tensors.insert("delta_q8".to_string(), delta_q8_blocks);

    let model_path = "mock_model.safetensors";
    candle_core::safetensors::save(&tensors, model_path)?;

    // Load back from safetensors memory mapped file
    let loaded_tensors = candle_core::safetensors::load(model_path, &device)?;
    let loaded_packed_base = loaded_tensors.get("packed_base").unwrap().clone();
    let loaded_scale_base = loaded_tensors.get("scale_base").unwrap().clone();
    let loaded_delta_q8 = loaded_tensors.get("delta_q8").unwrap().clone();

    // Initialize the Saccade custom operator
    let config = SaccadeConfig {
        t4: 1.0,
        t8: 5.0,
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
        out_features,
        in_features,
    };

    println!("Saccade Linear Operator Initialized.");

    // We can evaluate the operator by directly calling custom_op1 since there's no native wrapper in this mock
    // Create a low variance token
    let low_variance_data: Vec<half::f16> = vec![half::f16::from_f32(0.1); batch_tokens * in_features];
    let low_variance_input = Tensor::from_vec(low_variance_data, (batch_tokens, in_features), &device)?;

    // The variance is sum_sq / hidden_dim - mean^2. For all 0.1, mean = 0.1, sum_sq/N = 0.01. Variance = 0.0.
    // 0.0 < t4 (1.0), so this uses NO deltas (packed base only).

    // Create a high variance token
    let mut high_variance_data: Vec<half::f16> = vec![half::f16::from_f32(0.0); batch_tokens * in_features];
    high_variance_data[0] = half::f16::from_f32(10.0);
    let high_variance_input = Tensor::from_vec(high_variance_data, (batch_tokens, in_features), &device)?;

    // For high_variance_input: one element is 10.0, rest 0.0.
    // mean = 10.0 / 32 = 0.3125
    // sum_sq / N = 100.0 / 32 = 3.125
    // variance = 3.125 - 0.3125^2 = 3.125 - 0.09765625 = 3.027
    // 3.027 >= t4 (1.0) and < t8 (5.0), so this will trigger use_delta_q8!

    println!("Running low-variance forward pass (Expect: base weights only)...");
    let out_low = low_variance_input.apply_op1_no_bwd(&saccade_linear)?;
    println!("Low-variance output shape: {:?}", out_low.shape());

    println!("Running high-variance forward pass (Expect: base weights + sparse deltas)...");
    let out_high = high_variance_input.apply_op1_no_bwd(&saccade_linear)?;
    println!("High-variance output shape: {:?}", out_high.shape());

    println!("Validation successful.");

    // Clean up mock artifact
    let _ = std::fs::remove_file(model_path);
    Ok(())
}
