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

        let (base_data, _) = self.packed_base.storage_and_layout();
        let packed_weights = match &*base_data {
            candle_core::Storage::Cpu(cpu_store) => cpu_store.as_slice::<u32>()?,
            _ => return Err(candle_core::Error::Msg("Expected CPU storage for packed_base".into())),
        };

        let (scale_data, _) = self.scale_base.storage_and_layout();
        let base_scales = match &*scale_data {
            candle_core::Storage::Cpu(cpu_store) => cpu_store.as_slice::<half::f16>()?,
            _ => return Err(candle_core::Error::Msg("Expected CPU storage for scale_base".into())),
        };

        // Owned CSR copies — avoids holding Storage guards across Rayon thread boundaries.
        // The one-time memcpy cost is negligible vs. the matrix computation.
        struct OwnedCsr {
            r: Vec<u32>,
            c: Vec<u32>,
            v: Vec<u8>,
            s: f32,
        }

        // SAFETY: OwnedCsr contains only Vec<T> where T: Send+Sync, so it is
        // safe to share immutable references across Rayon worker threads.
        unsafe impl Sync for OwnedCsr {}

        let csr_q8: Option<OwnedCsr> = if let Some(sp) = &self.sparse_delta_q8 {
            let r_guard = sp.row_ptrs.storage_and_layout().0;
            let c_guard = sp.col_indices.storage_and_layout().0;
            let v_guard = sp.values.storage_and_layout().0;
            let s_guard = sp.scale.storage_and_layout().0;

            match (&*r_guard, &*c_guard, &*v_guard, &*s_guard) {
                (
                    candle_core::Storage::Cpu(r_cpu),
                    candle_core::Storage::Cpu(c_cpu),
                    candle_core::Storage::Cpu(v_cpu),
                    candle_core::Storage::Cpu(s_cpu),
                ) => {
                    let r = r_cpu.as_slice::<u32>()?.to_vec();
                    let c = c_cpu.as_slice::<u32>()?.to_vec();
                    let v = v_cpu.as_slice::<u8>()?.to_vec();
                    let s = s_cpu.as_slice::<half::f16>()?[0].to_f32();
                    Some(OwnedCsr { r, c, v, s })
                }
                _ => None,
            }
        } else {
            None
        };

        // Pre-classify token complexity outside the hot path so routing decisions
        // don't pollute the parallel row computation.
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

        let in_features = self.in_features;
        let out_features = self.out_features;
        let packed_per_row = in_features / 8;

        // Per-token f32 activation cache — eliminates redundant f16→f32 conversions.
        // Without this, the same element would be converted once per output row (896x).
        let mut token_f32 = vec![0.0f32; hidden_dim];

        for t in 0..batch_tokens {
            let act_offset = t * hidden_dim;
            let current_token_slice = &raw_activations[act_offset..act_offset + hidden_dim];
            let (use_delta_q8, use_delta_fp16) = token_routes[t];

            // Single f16→f32 pass per token: hidden_dim ops instead of hidden_dim * out_features.
            for i in 0..hidden_dim {
                unsafe {
                    *token_f32.get_unchecked_mut(i) = current_token_slice.get_unchecked(i).to_f32();
                }
            }

            // Reborrow as immutable slice — the mutable write phase is complete,
            // and Rayon workers only need read access to the cached activations.
            let token_cache: &[f32] = &token_f32;
            let out_slice = &mut output_buffer[t * out_features..(t + 1) * out_features];

            // The row kernel closure. Extracted as a local function so both the
            // sequential and parallel paths share the exact same hot loop body
            // without code duplication.
            let compute_row = |(row, out_val): (usize, &mut half::f16)| {
                let mut acc = 0.0f32;
                let row_weight_offset = row * packed_per_row;

                // Inner dot-product: unchecked access eliminates bounds-check branches,
                // allowing LLVM to emit clean AVX2/SSE vector instructions.
                unsafe {
                    for k_packed in 0..packed_per_row {
                        let p = *packed_weights.get_unchecked(row_weight_offset + k_packed);
                        let base = k_packed * 8;

                        // Manual 8-lane unroll: constant shift amounts produce branch-free
                        // nibble extraction that maps directly to SIMD integer lanes.
                        let n0 = (p & 0x0F) as f32 - 8.0;
                        let n1 = ((p >> 4) & 0x0F) as f32 - 8.0;
                        let n2 = ((p >> 8) & 0x0F) as f32 - 8.0;
                        let n3 = ((p >> 12) & 0x0F) as f32 - 8.0;
                        let n4 = ((p >> 16) & 0x0F) as f32 - 8.0;
                        let n5 = ((p >> 20) & 0x0F) as f32 - 8.0;
                        let n6 = ((p >> 24) & 0x0F) as f32 - 8.0;
                        let n7 = (p >> 28) as f32 - 8.0;

                        // Accumulate without scale — factored out to a single multiply below.
                        acc += *token_cache.get_unchecked(base) * n0;
                        acc += *token_cache.get_unchecked(base + 1) * n1;
                        acc += *token_cache.get_unchecked(base + 2) * n2;
                        acc += *token_cache.get_unchecked(base + 3) * n3;
                        acc += *token_cache.get_unchecked(base + 4) * n4;
                        acc += *token_cache.get_unchecked(base + 5) * n5;
                        acc += *token_cache.get_unchecked(base + 6) * n6;
                        acc += *token_cache.get_unchecked(base + 7) * n7;
                    }

                    // Factored row-scale: one multiply per row instead of in_features per row.
                    // The row scale is symmetric (same factor for all columns in the row),
                    // so it distributes over the sum: sum(x_i * w_i * s) = s * sum(x_i * w_i).
                    let scale = base_scales.get_unchecked(row).to_f32();
                    acc *= scale;

                    // Sparse delta correction: both Q8 and FP16 routing thresholds fall back
                    // to the available CSR data. This prevents high-volatility tokens from
                    // silently receiving base-only precision when sparse_delta_fp16 is None.
                    if use_delta_q8 || use_delta_fp16 {
                        if let Some(ref csr) = csr_q8 {
                            let row_start = *csr.r.get_unchecked(row) as usize;
                            let row_end = *csr.r.get_unchecked(row + 1) as usize;
                            for i in row_start..row_end {
                                let col = *csr.c.get_unchecked(i) as usize;
                                let val_i8 = *csr.v.get_unchecked(i) as i8;
                                let weight_update = (val_i8 as f32) * csr.s;
                                acc += *token_cache.get_unchecked(col) * weight_update;
                            }
                        }
                    }

                    *out_val = half::f16::from_f32(acc);
                }
            };

            // Adaptive parallelism: use Rayon when the total work per token is large
            // enough to amortize fork-join coordination cost (~2-5ms). For layers with
            // few output rows or small hidden dims, sequential avoids thread overhead.
            // For production-scale layers (e.g., 1536x8960), multi-core L3 bandwidth
            // is critical — sequential access patterns stall on L3 latency while
            // parallel threads aggregate cache bandwidth across cores.
            const PARALLEL_THRESHOLD: usize = 256 * 256; // ~65K FLOPs minimum
            if out_features * packed_per_row >= PARALLEL_THRESHOLD {
                out_slice.par_iter_mut().enumerate().for_each(compute_row);
            } else {
                out_slice.iter_mut().enumerate().for_each(compute_row);
            }
        }

        Ok((CpuStorage::F16(output_buffer), out_shape))
    }
}
