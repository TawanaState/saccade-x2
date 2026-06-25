pub mod config;
pub mod heuristics;
pub mod op;
pub mod compress;

pub use config::{SaccadeConfig, SaccadeLinearOp, HeuristicFn};
pub use heuristics::{variance_heuristic, l2_norm_heuristic};
pub use compress::{compress_tensor_to_saccade, compress_model_layers};
