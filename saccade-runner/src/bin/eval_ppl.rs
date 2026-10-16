//! WikiText-2 perplexity evaluator (audit tool).
//!
//! Modes:
//!   fp        original weights, dense path (reference)
//!   adaptive  Saccade checkpoint with frozen thresholds from the checkpoint
//!   bypass    Saccade checkpoint, dequantized base+delta dense path
//!   base4     Saccade checkpoint, routing forced OFF (4-bit base only, no delta)
//!   always    Saccade checkpoint, routing forced ON  (4-bit base + delta for every token)
use candle_core::{DType, Device, Tensor, D};
use clap::Parser;
use saccade_runner::model::{Qwen2Config, Qwen2Model};
use std::collections::HashMap;
use std::path::PathBuf;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    mode: String,
    /// Original model.safetensors (mode=fp) or Saccade checkpoint (other modes)
    #[arg(long)]
    weights: PathBuf,
    #[arg(long)]
    text: PathBuf,
    #[arg(long)]
    tokenizer: PathBuf,
    #[arg(long, default_value_t = 512)]
    ctx: usize,
    #[arg(long, default_value_t = 40)]
    max_windows: usize,
    /// Optional path: dump per-token NLL (little-endian f32) for oracle-routing analysis
    #[arg(long)]
    dump: Option<PathBuf>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let device = Device::Cpu;
    let mut tensors: HashMap<String, Tensor> = candle_core::safetensors::load(&args.weights, &device)?;

    let embed_w = tensors.get("model.embed_tokens.weight").ok_or("no embed")?;
    let (vocab, hidden) = (embed_w.dims()[0], embed_w.dims()[1]);
    let layers = (0..).take_while(|i| tensors.contains_key(&format!("model.layers.{}.input_layernorm.weight", i))).count();
    let inter = tensors
        .get("model.layers.0.mlp.gate_proj.saccade_scale_base")
        .or_else(|| tensors.get("model.layers.0.mlp.gate_proj.weight"))
        .ok_or("no gate")?
        .dim(0)?;
    let q_out = tensors["model.layers.0.self_attn.q_proj.weight"].dim(0)?;
    let k_out = tensors["model.layers.0.self_attn.k_proj.weight"].dim(0)?;
    let head_dim = [2usize, 4, 8, 1].iter().map(|&n| k_out / n)
        .find(|&hd| hd > 0 && q_out % hd == 0 && k_out % hd == 0 && q_out / hd == hidden / hd).unwrap_or(64);
    let cfg = Qwen2Config {
        vocab_size: vocab, hidden_size: hidden, intermediate_size: inter, num_hidden_layers: layers,
        num_attention_heads: q_out / head_dim, num_key_value_heads: k_out / head_dim,
        rope_theta: 1_000_000.0, rms_norm_eps: 1e-6, sliding_window: 32768,
        max_position_embeddings: 32768, tie_word_embeddings: !tensors.contains_key("lm_head.weight"),
    };

    // Reference baselines: simulated group-wise symmetric RTN on the same FFN projections.
    // Mode syntax: rtn<bits>g<group>, e.g. rtn4g128, rtn5g64.
    if let Some(spec) = args.mode.strip_prefix("rtn") {
        let mut it = spec.split('g');
        let bits: i32 = it.next().unwrap().parse()?;
        let group: usize = it.next().ok_or("expected rtn<bits>g<group>")?.parse()?;
        let qmax = ((1i32 << (bits - 1)) - 1) as f32;
        for (name, t) in tensors.iter_mut() {
            let is_ffn = ["gate_proj", "up_proj", "down_proj"].iter().any(|p| name.contains(p));
            if is_ffn && name.ends_with(".weight") {
                let (r, c) = (t.dims()[0], t.dims()[1]);
                let mut v: Vec<f32> = t.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
                for g in v.chunks_mut(group) {
                    let m = g.iter().fold(0f32, |a, &x| a.max(x.abs()));
                    let s = if m > 0.0 { m / qmax } else { 1.0 };
                    for x in g.iter_mut() { *x = (*x / s).round().clamp(-qmax - 1.0, qmax) * s; }
                }
                *t = Tensor::from_vec(v, (r, c), &device)?;
            }
        }
    }

    let force = match args.mode.as_str() {
        "base4" => Some(f32::MAX),
        "always" => Some(-1.0e30f32),
        _ => None,
    };
    if let Some(v) = force {
        for i in 0..layers {
            tensors.insert(format!("model.layers.{}.saccade_t4", i), Tensor::from_vec(vec![v], (1,), &device)?);
            tensors.insert(format!("model.layers.{}.saccade_t8", i), Tensor::from_vec(vec![v], (1,), &device)?);
        }
    }
    saccade_core::set_bypass_c_tarq(args.mode == "bypass");

    let mut model = Qwen2Model::from_saccade_checkpoint(&cfg, &tensors, &device)?;
    drop(tensors);

    let tok = tokenizers::Tokenizer::from_file(&args.tokenizer).map_err(|e| format!("{e}"))?;
    let text = std::fs::read_to_string(&args.text)?;
    let ids = tok.encode(text.as_str(), false).map_err(|e| format!("{e}"))?.get_ids().to_vec();
    let n_windows = (ids.len() / args.ctx).min(args.max_windows);
    eprintln!("mode={} tokens={} windows={}x{}", args.mode, ids.len(), n_windows, args.ctx);

    saccade_core::telemetry::TELEMETRY.reset();
    let (mut nll_sum, mut count) = (0.0f64, 0usize);
    let mut dump_buf: Vec<f32> = Vec::new();
    let t0 = std::time::Instant::now();
    for w in 0..n_windows {
        let win = &ids[w * args.ctx..(w + 1) * args.ctx];
        model.clear_kv_cache();
        let input = Tensor::new(win, &device)?.unsqueeze(0)?;
        let logits = model.forward_full(&input, 0)?.to_dtype(DType::F32)?; // (1, ctx, V)
        let pred = logits.narrow(1, 0, args.ctx - 1)?.squeeze(0)?;
        let logp = candle_nn::ops::log_softmax(&pred, D::Minus1)?;
        let tgt = Tensor::new(&win[1..], &device)?.unsqueeze(1)?;
        let per_tok: Vec<f32> = logp.gather(&tgt, 1)?.neg()?.squeeze(1)?.to_vec1()?;
        let nll: f64 = per_tok.iter().map(|&v| v as f64).sum();
        dump_buf.extend_from_slice(&per_tok);
        nll_sum += nll;
        count += args.ctx - 1;
        if (w + 1) % 10 == 0 || w + 1 == n_windows {
            eprintln!("  win {}/{}  running PPL {:.4}  ({:.0}s)", w + 1, n_windows, (nll_sum / count as f64).exp(), t0.elapsed().as_secs_f64());
        }
    }
    if let Some(p) = &args.dump {
        std::fs::write(p, dump_buf.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
    }
    saccade_core::telemetry::flush_telemetry();
    let t = &saccade_core::telemetry::TELEMETRY;
    let ord = std::sync::atomic::Ordering::Relaxed;
    let (b, s, p) = (t.total_base_bits.load(ord), t.total_sparse_bits.load(ord), t.total_param_calls.load(ord));
    let bpt = if p > 0 { (b + s) as f64 / p as f64 } else { f64::NAN };
    println!("RESULT mode={} ppl={:.4} tokens_scored={} telemetry_bpt={:.4} wall_s={:.1}",
        args.mode, (nll_sum / count as f64).exp(), count, bpt, t0.elapsed().as_secs_f64());
    Ok(())
}
