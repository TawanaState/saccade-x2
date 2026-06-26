use candle_core::{CustomOp1, CpuStorage, Layout, Result, Shape, Tensor};
use rayon::prelude::*;
use crate::config::SaccadeLinearOp;

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

        // We must hold the `Storage` locks outside the loop so the slices live long enough.
        struct CsrSlices<'a> {
            r: &'a [u32],
            c: &'a [u32],
            v: &'a [u8],
            s: f32,
        }

        let mut d8_r_store_opt = None;
        let mut d8_c_store_opt = None;
        let mut d8_v_store_opt = None;
        let mut d8_s_store_opt = None;

        let mut csr_q8 = None;
        if let Some(sp) = &self.sparse_delta_q8 {
            d8_r_store_opt = Some(sp.row_ptrs.storage_and_layout().0);
            d8_c_store_opt = Some(sp.col_indices.storage_and_layout().0);
            d8_v_store_opt = Some(sp.values.storage_and_layout().0);
            d8_s_store_opt = Some(sp.scale.storage_and_layout().0);

            if let (
                Some(r_store),
                Some(c_store),
                Some(v_store),
                Some(s_store)
            ) = (&d8_r_store_opt, &d8_c_store_opt, &d8_v_store_opt, &d8_s_store_opt) {
                if let (
                    candle_core::Storage::Cpu(r_cpu),
                    candle_core::Storage::Cpu(c_cpu),
                    candle_core::Storage::Cpu(v_cpu),
                    candle_core::Storage::Cpu(s_cpu)
                ) = (&**r_store, &**c_store, &**v_store, &**s_store) {
                    if let (Ok(r), Ok(c), Ok(v), Ok(s)) = (
                        r_cpu.as_slice::<u32>(),
                        c_cpu.as_slice::<u32>(),
                        v_cpu.as_slice::<u8>(),
                        s_cpu.as_slice::<half::f16>()
                    ) {
                        csr_q8 = Some(CsrSlices { r, c, v, s: s[0].to_f32() });
                    }
                }
            }
        }

        let mut d16_r_store_opt = None;
        let mut d16_c_store_opt = None;
        let mut d16_v_store_opt = None;
        let mut d16_s_store_opt = None;

        let mut csr_fp16 = None;
        if let Some(sp) = &self.sparse_delta_fp16 {
            d16_r_store_opt = Some(sp.row_ptrs.storage_and_layout().0);
            d16_c_store_opt = Some(sp.col_indices.storage_and_layout().0);
            d16_v_store_opt = Some(sp.values.storage_and_layout().0);
            d16_s_store_opt = Some(sp.scale.storage_and_layout().0);

            if let (
                Some(r_store),
                Some(c_store),
                Some(v_store),
                Some(s_store)
            ) = (&d16_r_store_opt, &d16_c_store_opt, &d16_v_store_opt, &d16_s_store_opt) {
                if let (
                    candle_core::Storage::Cpu(r_cpu),
                    candle_core::Storage::Cpu(c_cpu),
                    candle_core::Storage::Cpu(v_cpu),
                    candle_core::Storage::Cpu(s_cpu)
                ) = (&**r_store, &**c_store, &**v_store, &**s_store) {
                    if let (Ok(r), Ok(c), Ok(v), Ok(s)) = (
                        r_cpu.as_slice::<u32>(),
                        c_cpu.as_slice::<u32>(),
                        v_cpu.as_slice::<u8>(),
                        s_cpu.as_slice::<half::f16>()
                    ) {
                        csr_fp16 = Some(CsrSlices { r, c, v, s: s[0].to_f32() });
                    }
                }
            }
        }

        // Loop over the activation timeline sequentially to prevent parallel allocation traps
        for t in 0..batch_tokens {
            let act_offset = t * hidden_dim;
            let current_token_slice = &raw_activations[act_offset..act_offset + hidden_dim];

            // 1. Compute Complexity Heuristic entirely on-chip inside CPU registers
            let complexity_score = (self.config.heuristic)(current_token_slice);

            // 2. Evaluate frozen complexity thresholds to establish the dynamic execution path
            let use_delta_q8 = complexity_score >= self.config.t4 && complexity_score < self.config.t8;
            let use_delta_fp16 = complexity_score >= self.config.t8;

            // 3. Perform Fused Matrix Multiplication across rows of the projection weights
            // Parallelize computation across rows using Rayon as instructed
            let out_slice = &mut output_buffer[t * self.out_features..(t + 1) * self.out_features];
            out_slice.par_iter_mut().enumerate().for_each(|(row, out_val)| {
                let mut dot_accumulator = 0.0f32;
                let row_weight_offset = row * (self.in_features / 8);
                let current_scale = base_scales[row].to_f32();

                // Process packed u32 boundaries using micro-vector blocks for base
                for k_packed in 0..(self.in_features / 8) {
                    let packed_val = packed_weights[row_weight_offset + k_packed];
                    let k_unpacked_base = k_packed * 8;

                    // Unpack 8 distinct parameters in a single loop using register bitwise operators
                    for idx in 0..8 {
                        let raw_nibble = (packed_val >> (idx * 4)) & 0x0F;
                        // Center values from [0, 15] back to the signed range [-8, +7]
                        let base_weight = (raw_nibble as f32 - 8.0) * current_scale;

                        dot_accumulator += current_token_slice[k_unpacked_base + idx].to_f32() * base_weight;
                    }
                }

                // Add sparse delta updates if thresholds match using CSR traversal
                if use_delta_q8 {
                    if let Some(ref csr) = csr_q8 {
                        let row_start = csr.r[row] as usize;
                        let row_end = csr.r[row + 1] as usize;
                        for i in row_start..row_end {
                            let col = csr.c[i] as usize;
                            let val_i8 = csr.v[i] as i8; // Reinterpret back to signed 8-bit
                            let weight_update = (val_i8 as f32) * csr.s;
                            dot_accumulator += current_token_slice[col].to_f32() * weight_update;
                        }
                    }
                } else if use_delta_fp16 {
                    if let Some(ref csr) = csr_fp16 {
                        let row_start = csr.r[row] as usize;
                        let row_end = csr.r[row + 1] as usize;
                        for i in row_start..row_end {
                            let col = csr.c[i] as usize;
                            let val_i8 = csr.v[i] as i8; // Assumes fp16 uses same quant block logic for now
                            let weight_update = (val_i8 as f32) * csr.s;
                            dot_accumulator += current_token_slice[col].to_f32() * weight_update;
                        }
                    }
                }

                *out_val = half::f16::from_f32(dot_accumulator);
            });
        }

        Ok((CpuStorage::F16(output_buffer), out_shape))
    }
}
