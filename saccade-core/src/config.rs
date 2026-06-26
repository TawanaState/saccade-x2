use candle_core::Tensor;

/// A function pointer defining the dynamic strategy used to calculate a complexity score for a single token slice.
pub type HeuristicFn = fn(&[half::f16]) -> f32;

/// Core configuration profile containing global variance threshold pools and dynamic routing strategies
#[derive(Clone)]
pub struct SaccadeConfig {
    pub t4: f32,
    pub t8: f32,
    pub block_size: usize,
    /// The dynamic complexity metric calculation function to use per-token (e.g., Variance or L2 Norm).
    pub heuristic: HeuristicFn,
}

/// A Coordinate List (COO) or Compressed Sparse Row (CSR) representation of sparse delta patches.
#[derive(Clone, Debug)]
pub struct SparseDeltaMatrix {
    /// Pointers to the start of each row in col_indices and values. (length = out_features + 1)
    pub row_ptrs: Tensor,
    /// Column indices for the non-zero elements.
    pub col_indices: Tensor,
    /// The quantized integer values.
    pub values: Tensor,
    /// The singular scale factor for this delta block.
    pub scale: Tensor,
}

/// Persistent in-memory storage layout for an optimized Saccade linear projection.
/// Guarantees that only compressed parameters reside in the active hardware memory footprint.
pub struct SaccadeLinearOp {
    // Ultra-compressed base matrix: 4 bits per parameter packed uniformly into u32 containers
    pub packed_base: Tensor,
    pub scale_base: Tensor,

    // Pre-materialized block-sparse delta arrays stored as compressed integer spaces
    pub sparse_delta_q8: Option<SparseDeltaMatrix>,
    pub sparse_delta_fp16: Option<SparseDeltaMatrix>,

    // Operational configuration parameters
    pub config: SaccadeConfig,
    pub out_features: usize,
    pub in_features: usize,
}
