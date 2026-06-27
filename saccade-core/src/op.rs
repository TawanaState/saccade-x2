use candle_core::{CustomOp1, CpuStorage, Layout, Result, Shape};
use rayon::prelude::*;
use crate::config::SaccadeLinearOp;

impl CustomOp1 for SaccadeLinearOp {
    fn name(&self) -> &'static str {
        "fused_c_tarq_saccade_linear"
    }

    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
        let input_shape = layout.shape();
        let mut dims = input_shape.dims().to_vec();
        if let Some(last_dim) = dims.last_mut() {
            *last_dim = self.out_features;
        }
        let out_shape = Shape::from(dims.as_slice());

        let raw_activations = storage.as_slice::<half::f16>()?;
        let shape = layout.shape();
        let dims = shape.dims();

        let batch_tokens = dims[0..dims.len() - 1].iter().product::<usize>();
        let hidden_dim = dims[dims.len() - 1];

        let output_elements = batch_tokens * self.out_features;
        let mut output_buffer = vec![half::f16::from_f32(0.0); output_elements];

        let packed_weights = &self.cache.packed_weights;
        let base_scales = &self.cache.scales_f32;
        let csc_data = &self.cache.csc;

        let mut token_routes: Vec<(bool, bool)> = Vec::with_capacity(batch_tokens);
        for t in 0..batch_tokens {
            let act_offset = t * hidden_dim;
            let slice = &raw_activations[act_offset..act_offset + hidden_dim];
            let score = (self.config.heuristic)(slice);
            token_routes.push((
                score >= self.config.t4 && score < self.config.t8,
                score >= self.config.t8,
            ));
        }

        let out_features = self.out_features;
        let packed_per_row = self.in_features / 8;
        let in_features = self.in_features;

        let mut token_f32 = vec![0.0f32; hidden_dim];
        let mut acc_buffer = vec![0.0f32; out_features];

        for t in 0..batch_tokens {
            let act_offset = t * hidden_dim;
            let current_token_slice = &raw_activations[act_offset..act_offset + hidden_dim];
            let (use_delta_q8, use_delta_fp16) = token_routes[t];

            for i in 0..hidden_dim {
                unsafe {
                    *token_f32.get_unchecked_mut(i) = current_token_slice.get_unchecked(i).to_f32();
                }
            }
            let token_cache: &[f32] = &token_f32;

            // ── PHASE 1: Base 4-bit dot product with pipelined accumulators ────
            //
            // FMA has 4-cycle latency but 0.5-cycle throughput on modern x86.
            // A single accumulator (`acc += a*b`) creates a serial dependency chain
            // that can only issue 1 FMA per 4 cycles — wasting 87.5% of the FPU.
            //
            // Four independent accumulators (a0-a3) break this chain. Each gets
            // 2 FMAs per loop iteration, so consecutive uses of the same accumulator
            // are separated by 4 independent instructions — enough to fill the
            // pipeline and approach the 0.5-cycle throughput limit.
            //
            // Rust does NOT enable -ffast-math, so the compiler cannot reorder
            // `acc += x` into independent chains. This must be done manually.
            acc_buffer.par_iter_mut().enumerate().for_each(|(row, acc_val)| {
                let mut a0 = 0.0f32;
                let mut a1 = 0.0f32;
                let mut a2 = 0.0f32;
                let mut a3 = 0.0f32;
                let row_weight_offset = row * packed_per_row;

                unsafe {
                    for k_packed in 0..packed_per_row {
                        let p = *packed_weights.get_unchecked(row_weight_offset + k_packed);
                        let base = k_packed * 8;

                        let n0 = (p & 0x0F) as f32 - 8.0;
                        let n1 = ((p >> 4) & 0x0F) as f32 - 8.0;
                        let n2 = ((p >> 8) & 0x0F) as f32 - 8.0;
                        let n3 = ((p >> 12) & 0x0F) as f32 - 8.0;
                        let n4 = ((p >> 16) & 0x0F) as f32 - 8.0;
                        let n5 = ((p >> 20) & 0x0F) as f32 - 8.0;
                        let n6 = ((p >> 24) & 0x0F) as f32 - 8.0;
                        let n7 = (p >> 28) as f32 - 8.0;

                        // Distribute across 4 accumulators to break serial FMA chains.
                        // Each accumulator receives 2 FMAs per iteration, with 2
                        // intervening independent FMAs between consecutive uses —
                        // enough pipeline separation to hide the 4-cycle FMA latency.
                        a0 += *token_cache.get_unchecked(base) * n0;
                        a1 += *token_cache.get_unchecked(base + 1) * n1;
                        a2 += *token_cache.get_unchecked(base + 2) * n2;
                        a3 += *token_cache.get_unchecked(base + 3) * n3;
                        a0 += *token_cache.get_unchecked(base + 4) * n4;
                        a1 += *token_cache.get_unchecked(base + 5) * n5;
                        a2 += *token_cache.get_unchecked(base + 6) * n6;
                        a3 += *token_cache.get_unchecked(base + 7) * n7;
                    }

                    // Reduce accumulators with balanced tree to minimize rounding error
                    *acc_val = (a0 + a1 + a2 + a3) * *base_scales.get_unchecked(row);
                }
            });

            // ── PHASE 2: Sparse CSC correction ─────────────────────────────────
            if use_delta_q8 || use_delta_fp16 {
                if let Some(ref csc) = csc_data {
                    unsafe {
                        for col in 0..in_features {
                            let col_start = *csc.col_ptrs.get_unchecked(col) as usize;
                            let col_end = *csc.col_ptrs.get_unchecked(col + 1) as usize;
                            if col_start == col_end { continue; }

                            let activation = *token_cache.get_unchecked(col);

                            for idx in col_start..col_end {
                                let row = *csc.row_indices.get_unchecked(idx) as usize;
                                *acc_buffer.get_unchecked_mut(row) += activation * *csc.values_f32.get_unchecked(idx);
                            }
                        }
                    }
                }
            }

            // ── PHASE 3: Convert f32 → f16 ─────────────────────────────────────
            let out_slice = &mut output_buffer[t * out_features..(t + 1) * out_features];
            for row in 0..out_features {
                unsafe {
                    *out_slice.get_unchecked_mut(row) = half::f16::from_f32(*acc_buffer.get_unchecked(row));
                }
            }
        }

        Ok((CpuStorage::F16(output_buffer), out_shape))
    }
}
