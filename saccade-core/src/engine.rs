use std::collections::HashMap;
use candle_core::Tensor;
use crate::config::{SaccadeConfig, SaccadeLinearOp, SparseDeltaMatrix};
use crate::compress::compress_tensor_to_saccade;

pub struct SaccadeEngine;

impl SaccadeEngine {
    /// Compiles a model's linear projections dynamically, intercepting requested targets,
    /// compressing them on the fly, and substituting the Saccade custom operator.
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

                // Currently we run single delta_threshold. For automated intercept
                // we assume a fixed threshold matching V3 specs. We will default to 0.05.
                let blocks = compress_tensor_to_saccade(tensor, 0.05)?;

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

                let saccade_op = SaccadeLinearOp {
                    packed_base,
                    scale_base,
                    sparse_delta_q8,
                    sparse_delta_fp16: None, // Simplified to symmetric Q8 blocks as per v3 specs
                    config: config.clone(), // Clone config to inject to multiple instances
                    out_features,
                    in_features,
                };

                let base_name = name.trim_end_matches(".weight");
                layers.insert(base_name.to_string(), saccade_op);
            }
        }

        Ok(layers)
    }
}
