//! Strict b=1 end-to-end decode benchmark (audit tool).
//! The prompt is fed one token at a time so *every* step is a single-token GEMV step (no GEMM prefill is timed).
//!   modes: fp32 | fp16 | adaptive | bypass
use candle_core::{DType, Device, Tensor};
use clap::Parser;
use hf_hub::api::sync::Api;
use saccade_runner::model::{Qwen2Config, Qwen2Model};
use std::path::PathBuf;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    mode: String,
    #[arg(long)]
    model_id: String,
    #[arg(long)]
    checkpoint: Option<PathBuf>,
    #[arg(long)]
    text: PathBuf,
    #[arg(long)]
    tokenizer: PathBuf,
    #[arg(long, default_value_t = 32)]
    prompt_len: usize,
    #[arg(long, default_value_t = 96)]
    new_tokens: usize,
}

fn rss_mb(key: &str) -> f64 {
    std::fs::read_to_string("/proc/self/status").ok()
        .and_then(|s| s.lines().find(|l| l.starts_with(key)).map(|l| l.split_whitespace().nth(1).unwrap().parse::<f64>().unwrap() / 1024.0))
        .unwrap_or(f64::NAN)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = Args::parse();
    let dev = Device::Cpu;
    let api = Api::new()?;
    let repo = api.model(a.model_id.clone());
    let cfg: Qwen2Config = serde_json::from_str(&std::fs::read_to_string(repo.get("config.json")?)?)?;
    let weights = repo.get("model.safetensors")?;

    saccade_core::set_bypass_c_tarq(a.mode == "bypass");
    let mut model = match a.mode.as_str() {
        "fp32" | "fp16" => {
            let dt = if a.mode == "fp32" { DType::F32 } else { DType::F16 };
            let vb = unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(&[&weights], dt, &dev)? };
            Qwen2Model::from_standard(&cfg, vb)?
        }
        _ => {
            let t = candle_core::safetensors::load(a.checkpoint.as_ref().ok_or("--checkpoint required")?, &dev)?;
            let mut c = cfg.clone();
            c.tie_word_embeddings = !t.contains_key("lm_head.weight");
            Qwen2Model::from_saccade_checkpoint(&c, &t, &dev)?
        }
    };
    // from_saccade_checkpoint's `tensors` map is dropped here (end of match arm); measure steady state afterwards.
    let tok = tokenizers::Tokenizer::from_file(&a.tokenizer).map_err(|e| format!("{e}"))?;
    let text = std::fs::read_to_string(&a.text)?;
    let ids: Vec<u32> = tok.encode(&text[..4000], false).map_err(|e| format!("{e}"))?.get_ids()[..a.prompt_len].to_vec();

    saccade_core::telemetry::TELEMETRY.reset();
    let mut next = ids[0];
    let mut step_ms = Vec::new();
    let mut produced = Vec::new();
    for pos in 0..(a.prompt_len + a.new_tokens - 1) {
        let inp = if pos < a.prompt_len { ids[pos] } else { next };
        let x = Tensor::new(&[inp], &dev)?.unsqueeze(0)?;
        let t0 = std::time::Instant::now();
        let logits = model.forward(&x, pos)?.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
        let v: Vec<f32> = logits.to_vec1()?;
        next = v.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as u32;
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        if pos >= a.prompt_len { step_ms.push(ms); produced.push(next); }
    }
    step_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = step_ms[step_ms.len() / 2];
    let mean = step_ms.iter().sum::<f64>() / step_ms.len() as f64;
    saccade_core::telemetry::flush_telemetry();
    let t = &saccade_core::telemetry::TELEMETRY;
    let o = std::sync::atomic::Ordering::Relaxed;
    let (b, s, p) = (t.total_base_bits.load(o), t.total_sparse_bits.load(o), t.total_param_calls.load(o));
    println!("RESULT mode={:<8} decode_median={:7.2} ms/tok ({:5.1} tok/s)  mean={:7.2}  steady_RSS={:7.0} MB  peak_RSS={:7.0} MB  telemetry_BPT={:.2}  first_toks={:?}",
        a.mode, med, 1e3 / med, mean, rss_mb("VmRSS:"), rss_mb("VmHWM:"), if p > 0 { (b + s) as f64 / p as f64 } else { f64::NAN }, &produced[..6]);
    Ok(())
}
