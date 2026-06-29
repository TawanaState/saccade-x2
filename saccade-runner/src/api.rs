use candle_core::{DType, Device, Result, Tensor};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy)]
pub struct SaccadeMetrics {
    pub average_bpt: f64,
    pub kernel_ms: f64,
    pub layer_tokens_processed: u64,
}

pub struct SaccadeModelApi;

impl SaccadeModelApi {
    /// Enable or disable C-TARQ dynamic token-adaptive routing globally.
    pub fn set_bypass(enabled: bool) {
        saccade_core::set_bypass_c_tarq(enabled);
    }

    /// Reset all global telemetry registers.
    pub fn reset_telemetry() {
        saccade_core::telemetry::TELEMETRY.reset();
    }

    /// Retrieve the current performance metrics from the global telemetry registry.
    pub fn get_metrics() -> SaccadeMetrics {
        saccade_core::telemetry::flush_telemetry();
        
        let base_bits = saccade_core::telemetry::TELEMETRY.total_base_bits.load(std::sync::atomic::Ordering::Relaxed);
        let sparse_bits = saccade_core::telemetry::TELEMETRY.total_sparse_bits.load(std::sync::atomic::Ordering::Relaxed);
        let total_param_calls = saccade_core::telemetry::TELEMETRY.total_param_calls.load(std::sync::atomic::Ordering::Relaxed);
        let kernel_ns = saccade_core::telemetry::TELEMETRY.total_elapsed_ns.load(std::sync::atomic::Ordering::Relaxed);
        let tokens = saccade_core::telemetry::TELEMETRY.total_tokens_processed.load(std::sync::atomic::Ordering::Relaxed);

        let average_bpt = if total_param_calls > 0 {
            (base_bits + sparse_bits) as f64 / total_param_calls as f64
        } else {
            16.0
        };

        SaccadeMetrics {
            average_bpt,
            kernel_ms: kernel_ns as f64 / 1_000_000.0,
            layer_tokens_processed: tokens,
        }
    }

    /// Compile a dense model map into a Saccade-quantized state map using a custom calibration dataset.
    /// Works generically on any weight matrix matching the target layer substrings.
    pub fn compile_tensors(
        tensors: &HashMap<String, Tensor>,
        target_layers: &[&str],
        calibration_activations: &Tensor, // shape: (num_tokens, hidden_dim)
        target_fill_rate: f32,
        pct_t4: f32,
        pct_t8: f32,
    ) -> Result<HashMap<String, Tensor>> {
        // Run profile calibration to find t4 & t8 thresholds
        let calib_f16 = calibration_activations.to_dtype(DType::F16)?;
        let (t4, t8) = saccade_core::calibration::ProfileRunner::calibrate(&calib_f16, pct_t4, pct_t8)?;
        
        let mut output_tensors = HashMap::new();

        // Compress layers matching the target layout
        for (name, tensor) in tensors.iter() {
            let is_target_weight = target_layers.iter().any(|&target| name.contains(target)) && name.ends_with(".weight");
            
            if is_target_weight {
                let dims = tensor.shape().dims();
                if dims.len() == 2 && dims[1] % 8 == 0 {
                    let delta_threshold = saccade_core::compress::compute_percentile_threshold(tensor, target_fill_rate)?;
                    let blocks = saccade_core::compress::compress_tensor_to_saccade(tensor, delta_threshold)?;
                    
                    let prefix = name.trim_end_matches(".weight");
                    for (suffix, comp_tensor) in blocks {
                        output_tensors.insert(format!("{}.saccade_{}", prefix, suffix), comp_tensor);
                    }
                    continue;
                }
            }
            output_tensors.insert(name.clone(), tensor.clone());
        }

        // Auto-detect and inject thresholds into each layer's path in the safetensors file
        let mut layer_indices = std::collections::HashSet::new();
        for name in tensors.keys() {
            if let Some(idx) = extract_layer_index(name) {
                layer_indices.insert(idx);
            }
        }

        for idx in layer_indices {
            output_tensors.insert(
                format!("model.layers.{}.saccade_t4", idx),
                Tensor::from_vec(vec![t4], (1,), &Device::Cpu)?,
            );
            output_tensors.insert(
                format!("model.layers.{}.saccade_t8", idx),
                Tensor::from_vec(vec![t8], (1,), &Device::Cpu)?,
            );
        }

        Ok(output_tensors)
    }
}

fn extract_layer_index(name: &str) -> Option<usize> {
    let parts: Vec<&str> = name.split('.').collect();
    for i in 0..parts.len().saturating_sub(1) {
        if parts[i] == "layers" {
            if let Ok(idx) = parts[i + 1].parse::<usize>() {
                return Some(idx);
            }
        }
    }
    None
}
