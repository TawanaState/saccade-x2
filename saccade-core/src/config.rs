use candle_core::Tensor;

/// A function pointer defining the dynamic strategy used to calculate a complexity score for a single token slice.
pub type HeuristicFn = fn(&[half::f16]) -> f32;

/// Core configuration profile containing global variance threshold pools and dynamic routing strategies
pub struct SaccadeConfig {
    pub t4: f32,
    pub t8: f32,
    pub block_size: usize,
    /// The dynamic complexity metric calculation function to use per-token (e.g., Variance or L2 Norm).
    pub heuristic: HeuristicFn,
}

/// Persistent in-memory storage layout for an optimized Saccade linear projection.
/// Guarantees that only compressed parameters reside in the active hardware memory footprint.
pub struct SaccadeLinearOp {
    // Ultra-compressed base matrix: 4 bits per parameter packed uniformly into u32 containers
    pub packed_base: Tensor,
    pub scale_base: Tensor,

    // Pre-materialized block-sparse delta arrays stored as compressed integer spaces
    // Here we assume dense for simplicity inside the custom op.
    pub delta_q8_blocks: Tensor,
    pub delta_q8_scales: Option<Tensor>,
    pub delta_fp16_blocks: Option<Tensor>,

    // Operational configuration parameters
    pub config: SaccadeConfig,
    pub out_features: usize,
    pub in_features: usize,
}
