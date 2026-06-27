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

        // All kernel data is pre-computed at construction time — no Tensor guard
        // acquisition, no per-call Vec memcpy. This eliminates ~2ms per layer call
        // that accumulated to ~144ms overhead across 72 MLP layers per token.
        let packed_weights = &self.cache.packed_weights;
        let base_scales = &self.cache.scales_f32;
        let csc_data = &self.cache.csc;

        // Pre-classify token complexity outside the hot path.
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

        // Per-token f32 activation cache and f32 accumulator buffer.
        // Reused across tokens to avoid heap allocation per step.
        let mut token_f32 = vec![0.0f32; hidden_dim];
        let mut acc_buffer = vec![0.0f32; out_features];

        for t in 0..batch_tokens {
            let act_offset = t * hidden_dim;
            let current_token_slice = &raw_activations[act_offset..act_offset + hidden_dim];
            let (use_delta_q8, use_delta_fp16) = token_routes[t];

            // Single f16→f32 pass per token.
            for i in 0..hidden_dim {
                unsafe {
                    *token_f32.get_unchecked_mut(i) = current_token_slice.get_unchecked(i).to_f32();
                }
            }
            let token_cache: &[f32] = &token_f32;

            // ── PHASE 1: Base 4-bit matrix multiplication ──────────────────────
            // Rayon-parallel across rows. Each worker processes independent rows
            // with no shared mutable state. The unrolled 8-lane nibble extraction
            // and factored row-scale enable SIMD auto-vectorization.
            acc_buffer.par_iter_mut().enumerate().for_each(|(row, acc_val)| {
                let mut acc = 0.0f32;
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

                        acc += *token_cache.get_unchecked(base) * n0;
                        acc += *token_cache.get_unchecked(base + 1) * n1;
                        acc += *token_cache.get_unchecked(base + 2) * n2;
                        acc += *token_cache.get_unchecked(base + 3) * n3;
                        acc += *token_cache.get_unchecked(base + 4) * n4;
                        acc += *token_cache.get_unchecked(base + 5) * n5;
                        acc += *token_cache.get_unchecked(base + 6) * n6;
                        acc += *token_cache.get_unchecked(base + 7) * n7;
                    }

                    // Row scale applied once after accumulation.
                    *acc_val = acc * *base_scales.get_unchecked(row);
                }
            });

            // ── PHASE 2: Sparse CSC correction ─────────────────────────────────
            // Column-sequential iteration reads token_cache contiguously (no
            // pointer chasing). Row-indexed writes target acc_buffer (~6KB),
            // which stays fully L1-resident. This replaces the CSR loop that
            // performed ~1,337 scattered reads per row via csr.c[i].
            if use_delta_q8 || use_delta_fp16 {
                if let Some(ref csc) = csc_data {
                    unsafe {
                        for col in 0..in_features {
                            let col_start = *csc.col_ptrs.get_unchecked(col) as usize;
                            let col_end = *csc.col_ptrs.get_unchecked(col + 1) as usize;
                            if col_start == col_end { continue; }

                            // Sequential read from activation cache (contiguous, prefetch-friendly)
                            let activation = *token_cache.get_unchecked(col);

                            // Values are pre-scaled f32 — single FMA per non-zero element
                            for idx in col_start..col_end {
                                let row = *csc.row_indices.get_unchecked(idx) as usize;
                                *acc_buffer.get_unchecked_mut(row) += activation * *csc.values_f32.get_unchecked(idx);
                            }
                        }
                    }
                }
            }

            // ── PHASE 3: Convert f32 accumulator to f16 output ─────────────────
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
