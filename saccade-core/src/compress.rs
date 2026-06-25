use candle_core::{DType, Device, Tensor};
use std::collections::HashMap;

/// Compresses a dense FP16 or FP32 tensor into a packed 4-bit representation with row-wise scaling
/// and calculates the sparse delta based on reconstruction error magnitude.
pub fn compress_tensor_to_saccade(
    base_tensor: &Tensor,
    delta_threshold: f32, // The error threshold above which a sparse delta is allocated
) -> candle_core::Result<HashMap<String, Tensor>> {
    let device = base_tensor.device();
    let shape = base_tensor.shape().dims();
    if shape.len() != 2 {
        return Err(candle_core::Error::Msg("Only 2D tensors are supported".into()));
    }
    let out_features = shape[0];
    let in_features = shape[1];

    if in_features % 8 != 0 {
        return Err(candle_core::Error::Msg("in_features must be a multiple of 8".into()));
    }

    // Convert to F32 for CPU processing
    let base_f32 = base_tensor.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;
    let base_data = base_f32.to_vec2::<f32>()?;

    let mut packed_base_data = vec![0u32; out_features * (in_features / 8)];
    let mut scale_base_data = vec![0.0f32; out_features];
    let mut delta_q8_data = vec![0.0f32; out_features * in_features];

    for row in 0..out_features {
        // Find max absolute value in the row to determine symmetric scale
        let mut max_abs = 0.0f32;
        for col in 0..in_features {
            let val = base_data[row][col].abs();
            if val > max_abs {
                max_abs = val;
            }
        }

        // The values are mapped to [-8, 7] (16 values).
        // We use 7 as the max magnitude for scaling to avoid overflowing the 4-bit signed representation.
        let scale = if max_abs > 0.0 { max_abs / 7.0 } else { 1.0 };
        scale_base_data[row] = scale;

        for k_packed in 0..(in_features / 8) {
            let mut packed_u32: u32 = 0;
            for idx in 0..8 {
                let col = k_packed * 8 + idx;
                let val = base_data[row][col];

                // Quantize to [-8, 7]
                let mut q_val = (val / scale).round() as i32;
                q_val = q_val.max(-8).min(7);

                // Center to [0, 15] for storing in unsigned bits
                let u_val = (q_val + 8) as u32;

                // Pack into the u32
                packed_u32 |= u_val << (idx * 4);

                // Calculate reconstruction error
                let dequantized = (q_val as f32) * scale;
                let error = val - dequantized;

                // If error magnitude exceeds threshold, store it in the delta matrix
                if error.abs() > delta_threshold {
                    delta_q8_data[row * in_features + col] = error;
                }
            }
            packed_base_data[row * (in_features / 8) + k_packed] = packed_u32;
        }
    }

    let packed_base = Tensor::from_vec(packed_base_data, (out_features, in_features / 8), device)?;

    // Scale must be F16 for SaccadeLinearOp
    let scale_f16 = scale_base_data.iter().map(|&v| half::f16::from_f32(v)).collect::<Vec<_>>();
    let scale_base = Tensor::from_vec(scale_f16, (out_features,), device)?;

    // Delta must be F16 for SaccadeLinearOp
    let delta_f16 = delta_q8_data.iter().map(|&v| half::f16::from_f32(v)).collect::<Vec<_>>();
    let delta_q8_blocks = Tensor::from_vec(delta_f16, (out_features, in_features), device)?;

    let mut compressed_state = HashMap::new();
    compressed_state.insert("packed_base".to_string(), packed_base);
    compressed_state.insert("scale_base".to_string(), scale_base);
    compressed_state.insert("delta_q8".to_string(), delta_q8_blocks);

    Ok(compressed_state)
}

/// Iterates through an entire tensor map (e.g., from a safetensors file) and compresses
/// specific target linear projections to Saccade format. Unmatched layers are kept untouched.
pub fn compress_model_layers(
    tensors: &HashMap<String, Tensor>,
    target_layers: &[&str],
    delta_threshold: f32,
) -> candle_core::Result<HashMap<String, Tensor>> {
    let mut new_state = HashMap::new();

    for (name, tensor) in tensors.iter() {
        // Check if the current tensor name matches any of the target substrings
        let should_compress = target_layers.iter().any(|&target| name.contains(target));

        if should_compress {
            // Usually, biases are left untouched and only weight matrices are compressed
            if name.ends_with(".weight") && tensor.dims().len() == 2 {
                println!("Compressing target layer: {}", name);
                let compressed_blocks = compress_tensor_to_saccade(tensor, delta_threshold)?;

                // Mount the compressed artifacts with the layer prefix
                let prefix = name.trim_end_matches(".weight");
                for (suffix, comp_tensor) in compressed_blocks {
                    new_state.insert(format!("{}.saccade_{}", prefix, suffix), comp_tensor);
                }
            } else {
                new_state.insert(name.clone(), tensor.clone());
            }
        } else {
            // Keep unchanged
            new_state.insert(name.clone(), tensor.clone());
        }
    }

    Ok(new_state)
}
