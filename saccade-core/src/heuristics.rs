/// Computes the statistical variance of the token activations.
/// Isolates local volatility by explicitly tracking the mean. Recommended for Saccade.
pub fn variance_heuristic(token_slice: &[half::f16]) -> f32 {
    let hidden_dim = token_slice.len() as f32;
    let mut sum = 0.0f32;
    let mut sum_sq = 0.0f32;
    for &val in token_slice.iter() {
        let v_f32 = val.to_f32();
        sum += v_f32;
        sum_sq += v_f32 * v_f32;
    }
    let mean = sum / hidden_dim;
    (sum_sq / hidden_dim) - (mean * mean)
}

/// Computes the L2 Norm (Euclidean magnitude) of the token activations.
/// This measures absolute magnitude rather than volatility from the mean.
pub fn l2_norm_heuristic(token_slice: &[half::f16]) -> f32 {
    let mut sum_sq = 0.0f32;
    for &val in token_slice.iter() {
        let v_f32 = val.to_f32();
        sum_sq += v_f32 * v_f32;
    }
    sum_sq.sqrt()
}
