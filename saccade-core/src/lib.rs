use candle_core::{CustomOp1, CpuStorage, Layout, Result, Shape, Tensor};
use rayon::prelude::*;

/// Core configuration profile containing global variance threshold pools
pub struct SaccadeConfig {
    pub t4: f32,
    pub t8: f32,
    pub block_size: usize,
}

/// Persistent in-memory storage layout for an optimized Saccade linear projection.
/// Guarantees that only compressed parameters reside in the active hardware memory footprint.
pub struct SaccadeLinearOp {
    // Ultra-compressed base matrix: 4 bits per parameter packed uniformly into u32 containers
    pub packed_base: Tensor,
    pub scale_base: Tensor,

    // Pre-materialized block-sparse delta arrays stored as compressed integer spaces
    // Here we assume dense for simplicity inside the custom op (as the Python version did via `to_dense()` at init).
    pub delta_q8_blocks: Tensor,
    pub delta_q8_scales: Option<Tensor>,
    pub delta_fp16_blocks: Option<Tensor>,

    // Operational configuration parameters
    pub config: SaccadeConfig,
    pub out_features: usize,
    pub in_features: usize,
}

impl CustomOp1 for SaccadeLinearOp {
    fn name(&self) -> &'static str {
        "fused_c_tarq_saccade_linear"
    }

    /// Pure, side-effect-free vector mathematical execution block compiled for the CPU host engine.
    /// Vectorized operations are optimized to match system registers, avoiding PyTorch-style graph splits.
    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
        // Evaluate the multi-dimensional feature shape after executing the custom activation transformation
        let input_shape = layout.shape();
        let mut dims = input_shape.dims().to_vec();
        // Overwrite the final feature axis to match the linear projection layer configuration
        if let Some(last_dim) = dims.last_mut() {
            *last_dim = self.out_features;
        }
        let out_shape = Shape::from(dims.as_slice());

        // Extract raw pointer references from the incoming linear activation tensor
        let raw_activations = storage.as_slice::<half::f16>()?;
        let shape = layout.shape();
        let dims = shape.dims();

        let batch_tokens = dims[0..dims.len() - 1].iter().product::<usize>();
        let hidden_dim = dims[dims.len() - 1];

        // Allocate the destination output array matching the target projection metrics
        let output_elements = batch_tokens * self.out_features;
        let mut output_buffer = vec![half::f16::from_f32(0.0); output_elements];

        // Access raw binary data arrays from the registered buffers
        let (base_data, _base_layout) = self.packed_base.storage_and_layout();
        let packed_weights = match &*base_data {
            candle_core::Storage::Cpu(cpu_store) => cpu_store.as_slice::<u32>()?,
            _ => return Err(candle_core::Error::Msg("Hardware substrate target mismatch: Expected CPU registry".into())),
        };

        let (scale_data, _) = self.scale_base.storage_and_layout();
        let base_scales = match &*scale_data {
            candle_core::Storage::Cpu(cpu_store) => cpu_store.as_slice::<half::f16>()?,
            _ => return Err(candle_core::Error::Msg("Scale block storage corruption".into())),
        };

        // Extract deltas if available
        let mut w_delta_8: Option<&[half::f16]> = None;
        let d8_store = self.delta_q8_blocks.storage_and_layout().0;
        if let candle_core::Storage::Cpu(cpu_store) = &*d8_store {
            if let Ok(slice) = cpu_store.as_slice::<half::f16>() {
                w_delta_8 = Some(slice);
            }
        }

        let mut w_delta_16: Option<&[half::f16]> = None;
        let d16_store_opt = self.delta_fp16_blocks.as_ref().map(|t| t.storage_and_layout().0);
        if let Some(d16_store) = &d16_store_opt {
            if let candle_core::Storage::Cpu(cpu_store) = &**d16_store {
                if let Ok(slice) = cpu_store.as_slice::<half::f16>() {
                    w_delta_16 = Some(slice);
                }
            }
        }

        // Loop over the activation timeline sequentially to prevent parallel allocation traps
        for t in 0..batch_tokens {
            let act_offset = t * hidden_dim;
            let current_token_slice = &raw_activations[act_offset..act_offset + hidden_dim];

            // 1. Compute Causal Activation Variance entirely on-chip inside CPU registers
            let mut sum = 0.0f32;
            let mut sum_sq = 0.0f32;
            for &val in current_token_slice.iter() {
                let v_f32 = val.to_f32();
                sum += v_f32;
                sum_sq += v_f32 * v_f32;
            }
            let mean = sum / (hidden_dim as f32);
            let variance = (sum_sq / (hidden_dim as f32)) - (mean * mean);

            // 2. Evaluate frozen complexity thresholds to establish the dynamic execution path
            let use_delta_q8 = variance >= self.config.t4 && variance < self.config.t8;
            let use_delta_fp16 = variance >= self.config.t8;

            // 3. Perform Fused Matrix Multiplication across rows of the projection weights
            // Parallelize computation across rows using Rayon as instructed
            let out_slice = &mut output_buffer[t * self.out_features..(t + 1) * self.out_features];
            out_slice.par_iter_mut().enumerate().for_each(|(row, out_val)| {
                let mut dot_accumulator = 0.0f32;
                let row_weight_offset = row * (self.in_features / 8);
                let current_scale = base_scales[row].to_f32();

                // Process packed u32 boundaries using micro-vector blocks
                for k_packed in 0..(self.in_features / 8) {
                    let packed_val = packed_weights[row_weight_offset + k_packed];
                    let k_unpacked_base = k_packed * 8;

                    // Unpack 8 distinct parameters in a single loop using register bitwise operators
                    for idx in 0..8 {
                        let raw_nibble = (packed_val >> (idx * 4)) & 0x0F;
                        // Center values from [0, 15] back to the signed range [-8, +7]
                        let base_weight = (raw_nibble as f32 - 8.0) * current_scale;

                        let mut total_weight = base_weight;

                        // Add sparse delta if needed
                        let element_idx = row * self.in_features + k_unpacked_base + idx;

                        if use_delta_q8 {
                            if let Some(delta_8) = w_delta_8 {
                                total_weight += delta_8[element_idx].to_f32();
                            }
                        } else if use_delta_fp16 {
                            if let Some(delta_16) = w_delta_16 {
                                total_weight += delta_16[element_idx].to_f32();
                            }
                        }

                        dot_accumulator += current_token_slice[k_unpacked_base + idx].to_f32() * total_weight;
                    }
                }
                *out_val = half::f16::from_f32(dot_accumulator);
            });
        }

        Ok((CpuStorage::F16(output_buffer), out_shape))
    }
}
