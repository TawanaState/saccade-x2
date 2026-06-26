use candle_core::Tensor;
/// Profiling framework to extract scale-invariant runtime routing thresholds
/// from offline sample sequences, avoiding manual heuristic overrides.
pub struct ProfileRunner;

impl ProfileRunner {
    /// Accepts a standard FP16 activation tensor (batch_size, hidden_dim).
    /// Computes token-level variance, and maps the standard deviation distribution
    /// to locate precise routing boundaries for `t4` and `t8`.
    pub fn calibrate(activations: &Tensor, pct_t4: f32, pct_t8: f32) -> candle_core::Result<(f32, f32)> {
        let shape = activations.shape().dims();
        if shape.len() != 2 {
            return Err(candle_core::Error::Msg("Calibration expects 2D activation tensors (batch, hidden_dim)".into()));
        }

        let num_tokens = shape[0];
        let hidden_dim = shape[1];
        
        let acts_f32 = activations.to_dtype(candle_core::DType::F32)?;
        let acts_vec = acts_f32.to_vec2::<f32>()?;
        
        let mut variances = Vec::with_capacity(num_tokens);

        for t in 0..num_tokens {
            let mut sum = 0.0f32;
            let mut sum_sq = 0.0f32;
            for &val in &acts_vec[t] {
                sum += val;
                sum_sq += val * val;
            }
            let mean = sum / (hidden_dim as f32);
            let variance = (sum_sq / (hidden_dim as f32)) - (mean * mean);
            variances.push(variance);
        }

        // Sort variances to find quantile thresholds directly corresponding to 
        // the desired density percentiles (e.g., target 80% tokens via T4, 15% via T8, 5% full)
        variances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let idx_t4 = ((num_tokens as f32 * pct_t4).round() as usize).min(num_tokens.saturating_sub(1));
        let idx_t8 = ((num_tokens as f32 * pct_t8).round() as usize).min(num_tokens.saturating_sub(1));

        let t4 = variances[idx_t4];
        let t8 = variances[idx_t8];

        Ok((t4, t8))
    }
}
