pub mod config;
pub mod heuristics;
pub mod op;
pub mod compress;
pub mod engine;
pub mod calibration;
pub mod telemetry;

pub use config::{SaccadeConfig, SaccadeLinearOp, HeuristicFn, KernelCache, CachedCsc, set_bypass_c_tarq, is_c_tarq_bypassed};
pub use heuristics::{variance_heuristic, l2_norm_heuristic};
pub use compress::{compress_tensor_to_saccade, compress_model_layers, compute_percentile_threshold};
pub use engine::SaccadeEngine;

