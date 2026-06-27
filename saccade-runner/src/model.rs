use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{Embedding, VarBuilder};
use saccade_core::config::{SaccadeConfig, SaccadeLinearOp, SparseDeltaMatrix};
use saccade_core::variance_heuristic;
use std::collections::HashMap;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

fn default_rope_theta() -> f64 { 1_000_000.0 }
fn default_rms_eps() -> f64 { 1e-6 }
fn default_sliding_window() -> usize { 32768 }

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Qwen2Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default = "default_rms_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_sliding_window")]
    pub sliding_window: usize,
    #[serde(default)]
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub tie_word_embeddings: bool,
}

impl Qwen2Config {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

// ---------------------------------------------------------------------------
// ProjectionLayer — dual-mode Linear for MLP interception
// ---------------------------------------------------------------------------

pub enum ProjectionLayer {
    Standard(candle_nn::Linear),
    Saccade(SaccadeLinearOp),
}

impl ProjectionLayer {
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Standard(l) => l.forward(xs),
            Self::Saccade(op) => {
                // The Saccade kernel operates on F16 activations internally.
                // Convert F32 model activations → F16 → custom op → F16 output → F32.
                let orig_dtype = xs.dtype();
                let xs_f16 = if orig_dtype != DType::F16 { xs.to_dtype(DType::F16)? } else { xs.clone() };
                let out = xs_f16.apply_op1_no_bwd(op)?;
                if orig_dtype != DType::F16 { out.to_dtype(orig_dtype) } else { Ok(out) }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// RmsNorm
// ---------------------------------------------------------------------------

struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn new(weight: Tensor, eps: f64) -> Self {
        Self { weight, eps }
    }

    fn load(size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(size, "weight")?;
        Ok(Self::new(weight, eps))
    }

    fn from_tensor(weight: Tensor, eps: f64) -> Self {
        Self::new(weight, eps)
    }
}

impl Module for RmsNorm {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let dtype = xs.dtype();
        let xs_f32 = xs.to_dtype(DType::F32)?;
        let variance = xs_f32.sqr()?.mean_keepdim(D::Minus1)?;
        let xs_normed = xs_f32.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        xs_normed.to_dtype(dtype)?.broadcast_mul(&self.weight)
    }
}

// ---------------------------------------------------------------------------
// Rotary Position Embeddings
// ---------------------------------------------------------------------------

struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(dtype: DType, cfg: &Qwen2Config, dev: &Device) -> Result<Self> {
        let head_dim = cfg.head_dim();
        let max_seq = if cfg.max_position_embeddings > 0 {
            cfg.max_position_embeddings
        } else {
            cfg.sliding_window
        };
        let inv_freq: Vec<f32> = (0..head_dim)
            .step_by(2)
            .map(|i| 1.0 / (cfg.rope_theta as f32).powf(i as f32 / head_dim as f32))
            .collect();
        let inv_freq = Tensor::new(inv_freq.as_slice(), dev)?;
        let t = Tensor::arange(0u32, max_seq as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_seq, 1))?;
        let freqs = t.matmul(&inv_freq.reshape((1, head_dim / 2))?)?;
        // Duplicate to full head_dim: each frequency applies to a pair of dimensions
        let freqs_full = Tensor::cat(&[&freqs, &freqs], D::Minus1)?;
        let cos = freqs_full.cos()?.to_dtype(dtype)?;
        let sin = freqs_full.sin()?.to_dtype(dtype)?;
        Ok(Self { sin, cos })
    }

    fn apply(&self, q: &Tensor, k: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        let (_b, _h, seq_len, _d) = q.dims4()?;
        let cos = self.cos.narrow(0, offset, seq_len)?;
        let sin = self.sin.narrow(0, offset, seq_len)?;
        let q_embed = Self::apply_rope(q, &cos, &sin)?;
        let k_embed = Self::apply_rope(k, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }

    // RoPE: rotate each pair of dimensions by the position-dependent angle.
    // x_rot = x * cos + rotate_half(x) * sin
    fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let (_b, _h, _s, d) = x.dims4()?;
        let half = d / 2;
        let x1 = x.narrow(D::Minus1, 0, half)?;
        let x2 = x.narrow(D::Minus1, half, half)?;
        let rotated = Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)?;
        let cos = cos.unsqueeze(0)?.unsqueeze(0)?; // (1, 1, seq, half) → broadcast
        let sin = sin.unsqueeze(0)?.unsqueeze(0)?;
        x.broadcast_mul(&cos)? + rotated.broadcast_mul(&sin)?
    }
}

// ---------------------------------------------------------------------------
// Attention
// ---------------------------------------------------------------------------

struct Attention {
    q_proj: candle_nn::Linear,
    k_proj: candle_nn::Linear,
    v_proj: candle_nn::Linear,
    o_proj: candle_nn::Linear,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    hidden_size: usize,
    rotary_emb: Arc<RotaryEmbedding>,
    kv_cache: Option<(Tensor, Tensor)>,
}

impl Attention {
    fn new(rope: Arc<RotaryEmbedding>, cfg: &Qwen2Config, vb: VarBuilder) -> Result<Self> {
        let hd = cfg.head_dim();
        let q_proj = candle_nn::linear(cfg.hidden_size, cfg.num_attention_heads * hd, vb.pp("q_proj"))?;
        let k_proj = candle_nn::linear(cfg.hidden_size, cfg.num_key_value_heads * hd, vb.pp("k_proj"))?;
        let v_proj = candle_nn::linear(cfg.hidden_size, cfg.num_key_value_heads * hd, vb.pp("v_proj"))?;
        let o_proj = candle_nn::linear_no_bias(cfg.num_attention_heads * hd, cfg.hidden_size, vb.pp("o_proj"))?;
        Ok(Self {
            q_proj, k_proj, v_proj, o_proj,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: hd,
            hidden_size: cfg.hidden_size,
            rotary_emb: rope,
            kv_cache: None,
        })
    }

    fn from_tensors(rope: Arc<RotaryEmbedding>, cfg: &Qwen2Config, tensors: &HashMap<String, Tensor>, prefix: &str) -> Result<Self> {
        // Auto-convert to F32 — Candle CPU doesn't support BF16 matmul
        let get = |name: &str| -> Result<Tensor> {
            let t = tensors.get(&format!("{}.{}", prefix, name))
                .ok_or_else(|| candle_core::Error::Msg(format!("Missing tensor: {}.{}", prefix, name)))?;
            if t.dtype() != DType::F32 { t.to_dtype(DType::F32) } else { Ok(t.clone()) }
        };
        let hd = cfg.head_dim();
        let q_proj = candle_nn::Linear::new(get("q_proj.weight")?, Some(get("q_proj.bias")?));
        let k_proj = candle_nn::Linear::new(get("k_proj.weight")?, Some(get("k_proj.bias")?));
        let v_proj = candle_nn::Linear::new(get("v_proj.weight")?, Some(get("v_proj.bias")?));
        let o_proj = candle_nn::Linear::new(get("o_proj.weight")?, None);
        Ok(Self {
            q_proj, k_proj, v_proj, o_proj,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: hd,
            hidden_size: cfg.hidden_size,
            rotary_emb: rope,
            kv_cache: None,
        })
    }

    fn forward(&mut self, xs: &Tensor, mask: Option<&Tensor>, offset: usize) -> Result<Tensor> {
        let (b, seq_len, _) = xs.dims3()?;
        let q = self.q_proj.forward(xs)?;
        let k = self.k_proj.forward(xs)?;
        let v = self.v_proj.forward(xs)?;

        let q = q.reshape((b, seq_len, self.num_heads, self.head_dim))?.transpose(1, 2)?;
        let k = k.reshape((b, seq_len, self.num_kv_heads, self.head_dim))?.transpose(1, 2)?;
        let v = v.reshape((b, seq_len, self.num_kv_heads, self.head_dim))?.transpose(1, 2)?;

        let (q, k) = self.rotary_emb.apply(&q, &k, offset)?;

        // KV cache
        let (k, v) = match &self.kv_cache {
            Some((pk, pv)) => {
                let k = Tensor::cat(&[pk, &k], 2)?;
                let v = Tensor::cat(&[pv, &v], 2)?;
                (k, v)
            }
            None => (k, v),
        };
        self.kv_cache = Some((k.clone(), v.clone()));

        // GQA: repeat KV heads to match query heads
        let num_groups = self.num_heads / self.num_kv_heads;
        let k = if num_groups > 1 {
            let k = k.unsqueeze(2)?.expand((b, self.num_kv_heads, num_groups, k.dim(2)?, self.head_dim))?;
            k.reshape((b, self.num_heads, (), self.head_dim))?
        } else { k };
        let v = if num_groups > 1 {
            let v = v.unsqueeze(2)?.expand((b, self.num_kv_heads, num_groups, v.dim(2)?, self.head_dim))?;
            v.reshape((b, self.num_heads, (), self.head_dim))?
        } else { v };

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let attn = (q.matmul(&k.transpose(D::Minus2, D::Minus1)?)? * scale)?;
        let attn = match mask {
            Some(m) => attn.broadcast_add(m)?,
            None => attn,
        };
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        let out = attn.matmul(&v)?.transpose(1, 2)?.reshape((b, seq_len, self.hidden_size))?;
        self.o_proj.forward(&out)
    }

    fn clear_kv_cache(&mut self) {
        self.kv_cache = None;
    }
}

// ---------------------------------------------------------------------------
// MLP
// ---------------------------------------------------------------------------

pub struct Mlp {
    gate_proj: ProjectionLayer,
    up_proj: ProjectionLayer,
    down_proj: ProjectionLayer,
}

impl Mlp {
    fn from_vb(cfg: &Qwen2Config, vb: VarBuilder) -> Result<Self> {
        let gate = candle_nn::linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("gate_proj"))?;
        let up = candle_nn::linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("up_proj"))?;
        let down = candle_nn::linear_no_bias(cfg.intermediate_size, cfg.hidden_size, vb.pp("down_proj"))?;
        Ok(Self {
            gate_proj: ProjectionLayer::Standard(gate),
            up_proj: ProjectionLayer::Standard(up),
            down_proj: ProjectionLayer::Standard(down),
        })
    }

    fn from_saccade(gate: SaccadeLinearOp, up: SaccadeLinearOp, down: SaccadeLinearOp) -> Self {
        Self {
            gate_proj: ProjectionLayer::Saccade(gate),
            up_proj: ProjectionLayer::Saccade(up),
            down_proj: ProjectionLayer::Saccade(down),
        }
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(xs)?.apply(&candle_nn::Activation::Silu)?;
        let up = self.up_proj.forward(xs)?;
        self.down_proj.forward(&(gate * up)?)
    }
}

// ---------------------------------------------------------------------------
// DecoderLayer
// ---------------------------------------------------------------------------

struct DecoderLayer {
    self_attn: Attention,
    mlp: Mlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    fn forward(&mut self, xs: &Tensor, mask: Option<&Tensor>, offset: usize) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(&xs, mask, offset)?;
        let xs = (residual + xs)?;
        let residual = &xs;
        let xs = self.post_attention_layernorm.forward(&xs)?;
        let xs = self.mlp.forward(&xs)?;
        residual + xs
    }

    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache();
    }
}

// ---------------------------------------------------------------------------
// Full Model
// ---------------------------------------------------------------------------

pub struct Qwen2Model {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: candle_nn::Linear,
    #[allow(dead_code)]
    sliding_window: usize,
    pub device: Device,
    pub dtype: DType,
    pub cfg: Qwen2Config,
}

impl Qwen2Model {
    /// Load a vanilla (uncompressed) model from a VarBuilder.
    pub fn from_standard(cfg: &Qwen2Config, vb: VarBuilder) -> Result<Self> {
        let embed_tokens = candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("model.embed_tokens"))?;
        let rope = Arc::new(RotaryEmbedding::new(vb.dtype(), cfg, vb.device())?);

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let vb_l = vb.pp(format!("model.layers.{}", i));
            let attn = Attention::new(rope.clone(), cfg, vb_l.pp("self_attn"))?;
            let mlp = Mlp::from_vb(cfg, vb_l.pp("mlp"))?;
            let input_ln = RmsNorm::load(cfg.hidden_size, cfg.rms_norm_eps, vb_l.pp("input_layernorm"))?;
            let post_ln = RmsNorm::load(cfg.hidden_size, cfg.rms_norm_eps, vb_l.pp("post_attention_layernorm"))?;
            layers.push(DecoderLayer { self_attn: attn, mlp, input_layernorm: input_ln, post_attention_layernorm: post_ln });
        }

        let norm = RmsNorm::load(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("model.norm"))?;
        let lm_head = if cfg.tie_word_embeddings || !vb.contains_tensor("lm_head.weight") {
            candle_nn::Linear::new(embed_tokens.embeddings().clone(), None)
        } else {
            candle_nn::linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
        };

        Ok(Self {
            embed_tokens, layers, norm, lm_head,
            sliding_window: cfg.sliding_window,
            device: vb.device().clone(),
            dtype: vb.dtype(),
            cfg: cfg.clone(),
        })
    }

    /// Load a Saccade-compressed model from a raw tensor map.
    /// Detects compressed layers by checking for `saccade_packed_base` keys.
    pub fn from_saccade_checkpoint(cfg: &Qwen2Config, tensors: &HashMap<String, Tensor>, device: &Device) -> Result<Self> {
        let dtype = DType::F32;

        // Helper: convert any tensor to F32 for CPU matmul compatibility.
        // Candle's CPU backend doesn't support BF16 matmul, and many HF models
        // (including Qwen2.5) store weights in BF16.
        let to_f32 = |t: &Tensor| -> Result<Tensor> {
            if t.dtype() != DType::F32 {
                t.to_dtype(DType::F32)
            } else {
                Ok(t.clone())
            }
        };

        let embed_w = tensors.get("model.embed_tokens.weight")
            .ok_or_else(|| candle_core::Error::Msg("Missing model.embed_tokens.weight".into()))?;
        let embed_w_f32 = to_f32(embed_w)?;
        let embed_tokens = Embedding::new(embed_w_f32, cfg.hidden_size);

        let rope = Arc::new(RotaryEmbedding::new(dtype, cfg, device)?);

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let attn_prefix = format!("model.layers.{}.self_attn", i);
            let attn = Attention::from_tensors(rope.clone(), cfg, tensors, &attn_prefix)?;

            let mlp_prefix = format!("model.layers.{}.mlp", i);
            let mlp = Self::load_mlp(cfg, tensors, &mlp_prefix, i)?;

            let ln1_w = tensors.get(&format!("model.layers.{}.input_layernorm.weight", i))
                .ok_or_else(|| candle_core::Error::Msg(format!("Missing input_layernorm.weight for layer {}", i)))?;
            let ln2_w = tensors.get(&format!("model.layers.{}.post_attention_layernorm.weight", i))
                .ok_or_else(|| candle_core::Error::Msg(format!("Missing post_attention_layernorm.weight for layer {}", i)))?;

            let input_ln = RmsNorm::from_tensor(to_f32(ln1_w)?, cfg.rms_norm_eps);
            let post_ln = RmsNorm::from_tensor(to_f32(ln2_w)?, cfg.rms_norm_eps);
            layers.push(DecoderLayer { self_attn: attn, mlp, input_layernorm: input_ln, post_attention_layernorm: post_ln });
        }

        let norm_w = tensors.get("model.norm.weight")
            .ok_or_else(|| candle_core::Error::Msg("Missing model.norm.weight".into()))?;
        let norm = RmsNorm::from_tensor(to_f32(norm_w)?, cfg.rms_norm_eps);

        // Qwen2.5 ties lm_head weights with embeddings — fall back to embed_tokens
        let lm_head_w = tensors.get("lm_head.weight").unwrap_or(embed_w);
        let lm_head = candle_nn::Linear::new(to_f32(lm_head_w)?, None);

        Ok(Self {
            embed_tokens, layers, norm, lm_head,
            sliding_window: cfg.sliding_window,
            device: device.clone(),
            dtype,
            cfg: cfg.clone(),
        })
    }

    /// Detect whether an MLP layer is Saccade-compressed and construct accordingly.
    fn load_mlp(_cfg: &Qwen2Config, tensors: &HashMap<String, Tensor>, prefix: &str, layer_idx: usize) -> Result<Mlp> {
        let packed_key = format!("{}.down_proj.saccade_packed_base", prefix);
        if tensors.contains_key(&packed_key) {
            // Compressed: build SaccadeLinearOp for each projection
            let t4_key = format!("model.layers.{}.saccade_t4", layer_idx);
            let t8_key = format!("model.layers.{}.saccade_t8", layer_idx);
            let t4 = saccade_core::SaccadeEngine::extract_scalar_f32_pub(tensors.get(&t4_key))?;
            let t8 = saccade_core::SaccadeEngine::extract_scalar_f32_pub(tensors.get(&t8_key))?;

            let saccade_cfg = SaccadeConfig { t4, t8, block_size: 16, heuristic: variance_heuristic };

            let gate = Self::build_saccade_op(tensors, &format!("{}.gate_proj", prefix), &saccade_cfg)?;
            let up = Self::build_saccade_op(tensors, &format!("{}.up_proj", prefix), &saccade_cfg)?;
            let down = Self::build_saccade_op(tensors, &format!("{}.down_proj", prefix), &saccade_cfg)?;
            Ok(Mlp::from_saccade(gate, up, down))
        } else {
            // Standard: load raw weights, converting to F32 for CPU matmul
            let get = |name: &str| -> Result<Tensor> {
                let t = tensors.get(&format!("{}.{}", prefix, name))
                    .ok_or_else(|| candle_core::Error::Msg(format!("Missing {}.{}", prefix, name)))?;
                if t.dtype() != DType::F32 { t.to_dtype(DType::F32) } else { Ok(t.clone()) }
            };
            let gate = candle_nn::Linear::new(get("gate_proj.weight")?, None);
            let up = candle_nn::Linear::new(get("up_proj.weight")?, None);
            let down = candle_nn::Linear::new(get("down_proj.weight")?, None);
            Ok(Mlp {
                gate_proj: ProjectionLayer::Standard(gate),
                up_proj: ProjectionLayer::Standard(up),
                down_proj: ProjectionLayer::Standard(down),
            })
        }
    }

    fn build_saccade_op(tensors: &HashMap<String, Tensor>, prefix: &str, cfg: &SaccadeConfig) -> Result<SaccadeLinearOp> {
        let get = |suffix: &str| -> Result<Tensor> {
            tensors.get(&format!("{}.saccade_{}", prefix, suffix))
                .cloned()
                .ok_or_else(|| candle_core::Error::Msg(format!("Missing {}.saccade_{}", prefix, suffix)))
        };

        let packed_base = get("packed_base")?;
        let scale_base = get("scale_base")?;

        let out_features = scale_base.dim(0)?;
        let in_features = packed_base.dim(1)? * 8;

        let sparse_delta_q8 = if let Ok(rp) = get("delta_row_ptrs") {
            Some(SparseDeltaMatrix {
                row_ptrs: rp,
                col_indices: get("delta_col_indices")?,
                values: get("delta_values")?,
                scale: get("delta_scale")?,
            })
        } else {
            None
        };

        SaccadeLinearOp::new(packed_base, scale_base, sparse_delta_q8, cfg.clone(), out_features, in_features)
    }

    fn causal_mask(&self, seq_len: usize, offset: usize, device: &Device) -> Result<Tensor> {
        let mask: Vec<f32> = (0..seq_len)
            .flat_map(|i| {
                (0..seq_len + offset).map(move |j| {
                    if j > i + offset { f32::NEG_INFINITY } else { 0.0 }
                })
            })
            .collect();
        Tensor::from_vec(mask, (1, 1, seq_len, seq_len + offset), device)
    }

    pub fn forward(&mut self, input_ids: &Tensor, offset: usize) -> Result<Tensor> {
        let (_, seq_len) = input_ids.dims2()?;
        let mask = if seq_len > 1 {
            Some(self.causal_mask(seq_len, offset, &self.device)?)
        } else {
            None
        };

        let mut xs = self.embed_tokens.forward(input_ids)?;
        for layer in self.layers.iter_mut() {
            xs = layer.forward(&xs, mask.as_ref(), offset)?;
        }
        let xs = xs.narrow(1, seq_len - 1, 1)?;
        let xs = self.norm.forward(&xs)?;
        self.lm_head.forward(&xs)
    }

    /// Run through the first N layers and return hidden states for calibration.
    pub fn forward_calibrate(&mut self, input_ids: &Tensor, num_layers: usize) -> Result<Tensor> {
        let xs = self.embed_tokens.forward(input_ids)?;
        let mut hidden = xs;
        for (i, layer) in self.layers.iter_mut().enumerate() {
            if i >= num_layers { break; }
            hidden = layer.forward(&hidden, None, 0)?;
        }
        Ok(hidden)
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_kv_cache();
        }
    }
}
