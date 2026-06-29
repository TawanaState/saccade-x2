use candle_core::Tensor;
use std::sync::atomic::{AtomicBool, Ordering};

pub static BYPASS_C_TARQ: AtomicBool = AtomicBool::new(false);

pub fn set_bypass_c_tarq(val: bool) {
    BYPASS_C_TARQ.store(val, Ordering::SeqCst);
}

pub fn is_c_tarq_bypassed() -> bool {
    BYPASS_C_TARQ.load(Ordering::SeqCst)
}

/// A function pointer defining the dynamic strategy used to calculate a complexity score for a single token slice.
pub type HeuristicFn = fn(&[half::f16]) -> f32;

/// Core configuration profile containing global variance threshold pools and dynamic routing strategies
#[derive(Clone)]
pub struct SaccadeConfig {
    pub t4: f32,
    pub t8: f32,
    pub block_size: usize,
    pub heuristic: HeuristicFn,
}

/// A Coordinate List (COO) or Compressed Sparse Row (CSR) representation of sparse delta patches.
#[derive(Clone, Debug)]
pub struct SparseDeltaMatrix {
    pub row_ptrs: Tensor,
    pub col_indices: Tensor,
    pub values: Tensor,
    pub scale: Tensor,
}

/// Pre-transposed Compressed Sparse Column (CSC) format for cache-friendly sparse correction.
/// Column-sequential iteration reads the activation vector contiguously, and the row-indexed
/// writes target a small accumulator buffer (~6KB) that stays L1-resident.
/// Values are pre-scaled to f32 at construction time, eliminating i8→f32 conversion and
/// scale multiplication from the hot path — leaving just one FMA per non-zero element.
pub struct CachedCsc {
    pub col_ptrs: Vec<u32>,
    pub row_indices: Vec<u32>,
    pub values_f32: Vec<f32>,
}

/// Pre-extracted kernel execution data. Populated once at construction time to eliminate
/// per-forward-call Tensor guard acquisition and Vec memcpy overhead. For full-model
/// inference (72 layer calls per token), this saves ~144ms of per-call extraction overhead.
pub struct KernelCache {
    pub packed_weights: Vec<u32>,
    pub scales_f32: Vec<f32>,
    pub csc: Option<CachedCsc>,
    pub dequantized_weight_f32: Vec<f32>,
}

/// Persistent in-memory storage layout for an optimized Saccade linear projection.
pub struct SaccadeLinearOp {
    pub packed_base: Tensor,
    pub scale_base: Tensor,
    pub sparse_delta_q8: Option<SparseDeltaMatrix>,
    pub sparse_delta_fp16: Option<SparseDeltaMatrix>,
    pub config: SaccadeConfig,
    pub out_features: usize,
    pub in_features: usize,
    pub cache: KernelCache,
}

impl SaccadeLinearOp {
    /// Construct a SaccadeLinearOp and pre-compute all kernel execution data.
    /// Extracts packed weights, scales, and builds the CSC sparse format once
    /// so that cpu_fwd never touches Tensor storage guards.
    pub fn new(
        packed_base: Tensor,
        scale_base: Tensor,
        sparse_delta_q8: Option<SparseDeltaMatrix>,
        config: SaccadeConfig,
        out_features: usize,
        in_features: usize,
    ) -> candle_core::Result<Self> {
        // Extract packed weights from Tensor into a contiguous Vec
        let packed_weights = {
            let (store, _) = packed_base.storage_and_layout();
            match &*store {
                candle_core::Storage::Cpu(cpu) => cpu.as_slice::<u32>()?.to_vec(),
                _ => return Err(candle_core::Error::Msg("Expected CPU storage for packed_base".into())),
            }
        };

        // Extract and pre-convert scales from f16 to f32
        let scales_f32: Vec<f32> = {
            let (store, _) = scale_base.storage_and_layout();
            match &*store {
                candle_core::Storage::Cpu(cpu) => {
                    cpu.as_slice::<half::f16>()?.iter().map(|v| v.to_f32()).collect()
                }
                _ => return Err(candle_core::Error::Msg("Expected CPU storage for scale_base".into())),
            }
        };

        // Build CSC from CSR for cache-friendly sparse correction
        let csc = if let Some(ref sp) = sparse_delta_q8 {
            Some(Self::build_csc(sp, out_features, in_features)?)
        } else {
            None
        };

        // Pre-compute dequantized weights (base + sparse deltas) for standard GEMM bypass
        let mut dequantized_weight_f32 = vec![0.0f32; out_features * in_features];
        let packed_per_row = in_features / 8;
        for row in 0..out_features {
            let scale = scales_f32[row];
            let row_offset = row * packed_per_row;
            let dest_row_offset = row * in_features;
            for k in 0..packed_per_row {
                let p = packed_weights[row_offset + k];
                let base = k * 8;
                for idx in 0..8 {
                    let u_val = (p >> (idx * 4)) & 0x0F;
                    let q_val = (u_val as i32) - 8;
                    dequantized_weight_f32[dest_row_offset + base + idx] = (q_val as f32) * scale;
                }
            }
        }

        // Add sparse corrections to the dequantized weights
        if let Some(ref csc_data) = csc {
            for col in 0..in_features {
                let col_start = csc_data.col_ptrs[col] as usize;
                let col_end = csc_data.col_ptrs[col + 1] as usize;
                for idx in col_start..col_end {
                    let row = csc_data.row_indices[idx] as usize;
                    let val = csc_data.values_f32[idx];
                    dequantized_weight_f32[row * in_features + col] += val;
                }
            }
        }

        let cache = KernelCache { packed_weights, scales_f32, csc, dequantized_weight_f32 };


        Ok(Self {
            packed_base, scale_base, sparse_delta_q8, sparse_delta_fp16: None,
            config, out_features, in_features, cache,
        })
    }

    /// Convert CSR sparse data to CSC (Compressed Sparse Column) format.
    /// CSC allows column-sequential iteration during the sparse correction pass,
    /// enabling contiguous reads from the activation cache instead of scattered
    /// pointer-chasing that breaks the CPU prefetcher.
    fn build_csc(sp: &SparseDeltaMatrix, out_features: usize, in_features: usize) -> candle_core::Result<CachedCsc> {
        let r_guard = sp.row_ptrs.storage_and_layout().0;
        let c_guard = sp.col_indices.storage_and_layout().0;
        let v_guard = sp.values.storage_and_layout().0;
        let s_guard = sp.scale.storage_and_layout().0;

        let (r, c, v, scale) = match (&*r_guard, &*c_guard, &*v_guard, &*s_guard) {
            (
                candle_core::Storage::Cpu(r_cpu),
                candle_core::Storage::Cpu(c_cpu),
                candle_core::Storage::Cpu(v_cpu),
                candle_core::Storage::Cpu(s_cpu),
            ) => {
                let r = r_cpu.as_slice::<u32>()?;
                let c = c_cpu.as_slice::<u32>()?;
                let v = v_cpu.as_slice::<u8>()?;
                let s = s_cpu.as_slice::<half::f16>()?[0].to_f32();
                (r, c, v, s)
            }
            _ => return Err(candle_core::Error::Msg("Expected CPU storage for sparse delta".into())),
        };

        let nnz = v.len();

        // Count entries per column
        let mut col_counts = vec![0u32; in_features + 1];
        for &col_idx in c.iter() {
            col_counts[col_idx as usize + 1] += 1;
        }
        // Prefix sum → column pointers
        for j in 1..=in_features {
            col_counts[j] += col_counts[j - 1];
        }

        // Transpose CSR → CSC
        let mut csc_rows = vec![0u32; nnz];
        let mut csc_vals_f32 = vec![0.0f32; nnz];
        let mut write_pos: Vec<u32> = col_counts[..in_features].to_vec();

        for row in 0..out_features {
            let row_start = r[row] as usize;
            let row_end = r[row + 1] as usize;
            for idx in row_start..row_end {
                let col = c[idx] as usize;
                let pos = write_pos[col] as usize;
                csc_rows[pos] = row as u32;
                // Pre-scale: convert i8 and apply scale factor once during construction
                // so the hot loop only needs a single FMA: acc += activation * val_f32
                csc_vals_f32[pos] = (v[idx] as i8 as f32) * scale;
                write_pos[col] += 1;
            }
        }

        Ok(CachedCsc {
            col_ptrs: col_counts,
            row_indices: csc_rows,
            values_f32: csc_vals_f32,
        })
    }
}
