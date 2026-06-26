use std::collections::HashMap;
use candle_core::Tensor;
use crate::config::{SaccadeConfig, SaccadeLinearOp, SparseDeltaMatrix};
use crate::compress::compress_tensor_to_saccade;

pub struct SaccadeEngine;

impl SaccadeEngine {
    fn extract_scalar_f32(t: &Tensor) -> candle_core::Result<f32> {
        // Handle both rank-0 scalars and rank-1 (1,) tensors from safetensors metadata.
        let flat = t.flatten_all()?.to_dtype(candle_core::DType::F32)?;
        let vals = flat.to_vec1::<f32>()?;
        vals.first().copied().ok_or_else(|| candle_core::Error::Msg("Empty threshold tensor".into()))
    }

    pub fn compile_model_topology<'a>(
        tensors: &HashMap<String, Tensor>,
        target_substrings: &[&str],
        config: SaccadeConfig,
    ) -> candle_core::Result<HashMap<String, SaccadeLinearOp>> {
        let mut layers = HashMap::new();

        for (name, tensor) in tensors.iter() {
            let is_target = target_substrings.iter().any(|&sub| name.contains(sub));

            if is_target && name.ends_with(".weight") && tensor.dims().len() == 2 {
                println!("Saccade: Intercepting and compiling {}", name);
                
                let dims = tensor.shape().dims();
                let out_features = dims[0];
                let in_features = dims[1];

                // Delta threshold controls the sparse correction fill rate.
                // Adaptive: scale with the weight matrix's Frobenius norm to handle
                // different model sizes. Larger models have larger weight magnitudes,
                // producing more reconstruction errors above a fixed threshold.
                let weight_f32 = tensor.to_dtype(candle_core::DType::F32)?;
                let sq = weight_f32.sqr()?;
                let frobenius_sq = sq.sum_all()?.to_scalar::<f32>()?;
                let rms_weight = (frobenius_sq / (out_features * in_features) as f32).sqrt();

                // Target ~10-15% fill: threshold at ~60% of the per-element RMS weight.
                // This captures the tail of the reconstruction error distribution while
                // keeping the sparse delta compact enough for meaningful compression.
                let delta_threshold = rms_weight * 0.6;
                let blocks = compress_tensor_to_saccade(tensor, delta_threshold)?;

                let packed_base = blocks.get("packed_base").unwrap().clone();
                let scale_base = blocks.get("scale_base").unwrap().clone();

                let mut sparse_delta_q8 = None;
                if blocks.contains_key("delta_row_ptrs") {
                    sparse_delta_q8 = Some(SparseDeltaMatrix {
                        row_ptrs: blocks.get("delta_row_ptrs").unwrap().clone(),
                        col_indices: blocks.get("delta_col_indices").unwrap().clone(),
                        values: blocks.get("delta_values").unwrap().clone(),
                        scale: blocks.get("delta_scale").unwrap().clone(),
                    });
                }

                let base_name = name.trim_end_matches(".weight");
                
                // If thresholds are embedded natively in the tensor map metadata, override config script defaults.
                // This ensures engine is purely data-driven.
                let mut layer_config = config.clone();
                let t4_key = format!("{}.saccade_t4", base_name);
                let t8_key = format!("{}.saccade_t8", base_name);

                if let Some(t) = tensors.get(&t4_key) {
                    if let Ok(v) = Self::extract_scalar_f32(t) {
                        layer_config.t4 = v;
                    }
                }
                if let Some(t) = tensors.get(&t8_key) {
                    if let Ok(v) = Self::extract_scalar_f32(t) {
                        layer_config.t8 = v;
                    }
                }

                let saccade_op = SaccadeLinearOp {
                    packed_base,
                    scale_base,
                    sparse_delta_q8,
                    sparse_delta_fp16: None, // Simplified to symmetric Q8 blocks as per v3 specs
                    config: layer_config,
                    out_features,
                    in_features,
                };
                
                layers.insert(base_name.to_string(), saccade_op);
            }
        }

        Ok(layers)
    }
}
