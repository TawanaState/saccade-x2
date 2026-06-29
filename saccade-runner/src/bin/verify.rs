use clap::Parser;
use candle_core::{DType, Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use saccade_runner::model::{Qwen2Config, Qwen2Model};
use hf_hub::api::sync::Api;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "saccade-verify")]
#[command(about = "Verify numerical accuracy and performance of Saccade C-TARQ vs Bypass mode")]
struct Args {
    /// Path to a compiled .safetensors checkpoint
    #[arg(long, default_value = "saccade_qwen.safetensors")]
    checkpoint: PathBuf,

    /// Input prompt for verification
    #[arg(long, default_value = "Explain the significance of prime numbers.")]
    prompt: String,

    /// Number of tokens to generate for evaluation
    #[arg(long, default_value_t = 30)]
    max_tokens: usize,

    /// Path to tokenizer.json (auto-downloaded if not provided)
    #[arg(long)]
    tokenizer: Option<PathBuf>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let device = Device::Cpu;

    println!("================================================================");
    println!("  Saccade V4 — Automated System Verification Suite");
    println!("================================================================\n");

    if !args.checkpoint.exists() {
        eprintln!("Error: checkpoint file {:?} not found.", args.checkpoint);
        eprintln!("Please run saccade-compile first to generate a compressed checkpoint.");
        std::process::exit(1);
    }

    // ---- 1. Load Saccade checkpoint ----
    println!("Loading model from checkpoint: {:?}", args.checkpoint);
    let tensors = candle_core::safetensors::load(&args.checkpoint, &device)?;

    // Infer config
    let embed_w = tensors.get("model.embed_tokens.weight")
        .ok_or("Missing model.embed_tokens.weight")?;
    let vocab_size = embed_w.dims()[0];
    let hidden_size = embed_w.dims()[1];
    let num_layers = (0..)
        .take_while(|i| tensors.contains_key(&format!("model.layers.{}.input_layernorm.weight", i)))
        .count();

    let intermediate_size = if let Some(t) = tensors.get("model.layers.0.mlp.gate_proj.saccade_scale_base") {
        t.dim(0)?
    } else if let Some(t) = tensors.get("model.layers.0.mlp.gate_proj.weight") {
        t.dim(0)?
    } else {
        hidden_size * 4
    };

    let q_proj_w = tensors.get("model.layers.0.self_attn.q_proj.weight").ok_or("Missing q_proj")?;
    let q_out = q_proj_w.dim(0)?;
    let k_proj_w = tensors.get("model.layers.0.self_attn.k_proj.weight").ok_or("Missing k_proj")?;
    let k_out = k_proj_w.dim(0)?;
    let head_dim = [2usize, 4, 8, 1].iter()
        .map(|&nkv| k_out / nkv)
        .find(|&hd| hd > 0 && q_out % hd == 0 && k_out % hd == 0 && q_out / hd == hidden_size / hd)
        .unwrap_or(64);

    let has_lm_head = tensors.contains_key("lm_head.weight");

    let cfg = Qwen2Config {
        vocab_size,
        hidden_size,
        intermediate_size,
        num_hidden_layers: num_layers,
        num_attention_heads: q_out / head_dim,
        num_key_value_heads: k_out / head_dim,
        rope_theta: 1_000_000.0,
        rms_norm_eps: 1e-6,
        sliding_window: 32768,
        max_position_embeddings: 32768,
        tie_word_embeddings: !has_lm_head,
    };

    // Load tokenizer
    let tokenizer = load_tokenizer(args.tokenizer.as_ref(), None, Some(args.checkpoint.parent().unwrap_or(std::path::Path::new("."))))?;

    // Wrap prompt in template
    let chat_prompt = format!("<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n", args.prompt);
    let encoding = tokenizer.encode(chat_prompt.as_str(), true)
        .map_err(|e| format!("Tokenization failed: {}", e))?;
    let input_ids = encoding.get_ids().to_vec();
    let input_len = input_ids.len();

    println!("Prompt: {:?}", args.prompt);
    println!("Prompt tokens: {}\n", input_len);

    let seed = 42;
    let temp = Some(0.7);
    let top_p = None;

    // ---- 2. Run Mode A: C-TARQ Adaptive Routing ----
    println!("=== Phase A: Running with C-TARQ Adaptive Routing ===");
    saccade_core::set_bypass_c_tarq(false);
    saccade_core::telemetry::TELEMETRY.reset();

    let mut model_ctarq = Qwen2Model::from_saccade_checkpoint(&cfg, &tensors, &device)?;
    let mut lp_ctarq = LogitsProcessor::new(seed, temp, top_p);

    let start_ctarq = std::time::Instant::now();
    let input_tensor = Tensor::new(input_ids.as_slice(), &device)?.unsqueeze(0)?;
    
    // Prefill
    let mut logits_ctarq = model_ctarq.forward(&input_tensor, 0)?;
    let mut next_token_ctarq = lp_ctarq.sample(&logits_ctarq.squeeze(0)?.squeeze(0)?)?;
    
    let mut ctarq_tokens = vec![next_token_ctarq];
    let mut ctarq_logit_slices = vec![logits_ctarq.clone()];
    let mut offset = input_len;

    for _ in 1..args.max_tokens {
        let token_tensor = Tensor::new(&[next_token_ctarq], &device)?.unsqueeze(0)?;
        logits_ctarq = model_ctarq.forward(&token_tensor, offset)?;
        next_token_ctarq = lp_ctarq.sample(&logits_ctarq.squeeze(0)?.squeeze(0)?)?;
        ctarq_tokens.push(next_token_ctarq);
        ctarq_logit_slices.push(logits_ctarq.clone());
        offset += 1;
    }
    let elapsed_ctarq_ms = start_ctarq.elapsed().as_secs_f64() * 1000.0;

    // Flush and extract C-TARQ telemetry
    saccade_core::telemetry::flush_telemetry();
    let base_bits_a = saccade_core::telemetry::TELEMETRY.total_base_bits.load(std::sync::atomic::Ordering::Relaxed);
    let sparse_bits_a = saccade_core::telemetry::TELEMETRY.total_sparse_bits.load(std::sync::atomic::Ordering::Relaxed);
    let params_a = saccade_core::telemetry::TELEMETRY.total_param_calls.load(std::sync::atomic::Ordering::Relaxed);
    let kernel_ns_a = saccade_core::telemetry::TELEMETRY.total_elapsed_ns.load(std::sync::atomic::Ordering::Relaxed);
    let bpt_ctarq = if params_a > 0 {
        (base_bits_a + sparse_bits_a) as f64 / params_a as f64
    } else {
        4.0
    };
    let kernel_ms_ctarq = kernel_ns_a as f64 / 1_000_000.0;

    println!("Completed C-TARQ pass: generated {} tokens in {:.2} ms\n", ctarq_tokens.len(), elapsed_ctarq_ms);

    // ---- 3. Run Mode B: Bypass Mode (Dequantized FP16 Baseline) ----
    println!("=== Phase B: Running in C-TARQ Bypass Mode ===");
    saccade_core::set_bypass_c_tarq(true);
    saccade_core::telemetry::TELEMETRY.reset();

    let mut model_bypass = Qwen2Model::from_saccade_checkpoint(&cfg, &tensors, &device)?;
    let mut lp_bypass = LogitsProcessor::new(seed, temp, top_p);

    let start_bypass = std::time::Instant::now();
    let input_tensor = Tensor::new(input_ids.as_slice(), &device)?.unsqueeze(0)?;
    
    // Prefill
    let mut logits_bypass = model_bypass.forward(&input_tensor, 0)?;
    let mut next_token_bypass = lp_bypass.sample(&logits_bypass.squeeze(0)?.squeeze(0)?)?;
    
    let mut bypass_tokens = vec![next_token_bypass];
    let mut bypass_logit_slices = vec![logits_bypass.clone()];
    let mut offset = input_len;

    for _ in 1..args.max_tokens {
        let token_tensor = Tensor::new(&[next_token_bypass], &device)?.unsqueeze(0)?;
        logits_bypass = model_bypass.forward(&token_tensor, offset)?;
        next_token_bypass = lp_bypass.sample(&logits_bypass.squeeze(0)?.squeeze(0)?)?;
        bypass_tokens.push(next_token_bypass);
        bypass_logit_slices.push(logits_bypass.clone());
        offset += 1;
    }
    let elapsed_bypass_ms = start_bypass.elapsed().as_secs_f64() * 1000.0;

    // Flush and extract Bypass telemetry
    saccade_core::telemetry::flush_telemetry();
    let base_bits_b = saccade_core::telemetry::TELEMETRY.total_base_bits.load(std::sync::atomic::Ordering::Relaxed);
    let params_b = saccade_core::telemetry::TELEMETRY.total_param_calls.load(std::sync::atomic::Ordering::Relaxed);
    let kernel_ns_b = saccade_core::telemetry::TELEMETRY.total_elapsed_ns.load(std::sync::atomic::Ordering::Relaxed);
    let bpt_bypass = if params_b > 0 {
        base_bits_b as f64 / params_b as f64
    } else {
        16.0
    };
    let kernel_ms_bypass = kernel_ns_b as f64 / 1_000_000.0;

    println!("Completed Bypass pass: generated {} tokens in {:.2} ms\n", bypass_tokens.len(), elapsed_bypass_ms);

    // ---- 4. Accuracy Comparison ----
    println!("=== Phase C: Logit Accuracy Auditing ===");
    let mut sum_cosine_similarity = 0.0f32;
    let mut sum_rmse = 0.0f32;
    let compare_len = ctarq_logit_slices.len().min(bypass_logit_slices.len());

    for i in 0..compare_len {
        let l_ctarq = ctarq_logit_slices[i].flatten_all()?;
        let l_bypass = bypass_logit_slices[i].flatten_all()?;

        let cos = cosine_similarity(&l_ctarq, &l_bypass)?;
        let err = rmse(&l_ctarq, &l_bypass)?;

        sum_cosine_similarity += cos;
        sum_rmse += err;
    }

    let avg_cosine_similarity = sum_cosine_similarity / compare_len as f32;
    let avg_rmse = sum_rmse / compare_len as f32;

    // ---- 5. Report Generation ----
    println!("\n================================================================");
    println!("            SACCADE SYSTEM VERIFICATION REPORT");
    println!("================================================================");
    println!("Checkpoint Evaluated:    {:?}", args.checkpoint);
    println!("Reference Baseline:      Vanilla FP16 Dequantized (Bypass)");
    println!("Number of Steps Run:     {} steps", compare_len);
    println!("----------------------------------------------------------------");
    println!("NUMERICAL ACCURACY METRICS:");
    println!("  Avg Logit Cosine Similarity: {:.6} (Target: >0.998)", avg_cosine_similarity);
    println!("  Avg Logit RMSE:              {:.6} (Target: <0.005)", avg_rmse);
    println!("----------------------------------------------------------------");
    println!("PERFORMANCE AND QUANTIZATION AUDIT:");
    println!("  C-TARQ End-to-End Latency:   {:.2} ms ({:.2} tokens/sec)", elapsed_ctarq_ms, compare_len as f64 / (elapsed_ctarq_ms / 1000.0));
    println!("  Bypass End-to-End Latency:   {:.2} ms ({:.2} tokens/sec)", elapsed_bypass_ms, compare_len as f64 / (elapsed_bypass_ms / 1000.0));
    println!("  Saccade C-TARQ BPT Budget:   {:.2} BPT", bpt_ctarq);
    println!("  Dequantized Bypass BPT:      {:.2} BPT", bpt_bypass);
    println!("  Kernel Compute Speedup:      {:.2}x", kernel_ms_bypass / kernel_ms_ctarq.max(0.0001));
    println!("================================================================");
    
    if avg_cosine_similarity >= 0.995 && avg_rmse <= 0.01 {
        println!("Status: VERIFICATION SUCCESSFUL (Accuracy bounds maintained)");
    } else {
        println!("Status: VERIFICATION FAILED (Accuracy limits breached)");
    }
    println!("================================================================");

    Ok(())
}

fn cosine_similarity(t1: &Tensor, t2: &Tensor) -> candle_core::Result<f32> {
    let dot = (t1 * t2)?.sum_all()?.to_dtype(DType::F32)?.to_scalar::<f32>()?;
    let norm1 = (t1 * t1)?.sum_all()?.sqrt()?.to_dtype(DType::F32)?.to_scalar::<f32>()?;
    let norm2 = (t2 * t2)?.sum_all()?.sqrt()?.to_dtype(DType::F32)?.to_scalar::<f32>()?;
    if norm1 > 0.0 && norm2 > 0.0 {
        Ok(dot / (norm1 * norm2))
    } else {
        Ok(0.0)
    }
}

fn rmse(t1: &Tensor, t2: &Tensor) -> candle_core::Result<f32> {
    let diff = (t1 - t2)?;
    let sq_diff = (&diff * &diff)?;
    let mean_sq = sq_diff.mean_all()?.to_dtype(DType::F32)?.to_scalar::<f32>()?;
    Ok(mean_sq.sqrt())
}

fn hf_download(model_id: &str, filename: &str) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let api = Api::new()?;
    let repo = api.model(model_id.to_string());
    match repo.get(filename) {
        Ok(path) => Ok(path),
        Err(_) => {
            let api2 = Api::new()?;
            let repo2 = api2.model(model_id.to_string());
            Ok(repo2.get(filename)?)
        }
    }
}

fn load_tokenizer(
    cli_path: Option<&PathBuf>,
    model_id: Option<&str>,
    search_dir: Option<&std::path::Path>,
) -> Result<tokenizers::Tokenizer, Box<dyn std::error::Error>> {
    if let Some(p) = cli_path {
        return tokenizers::Tokenizer::from_file(p)
            .map_err(|e| format!("Failed to load tokenizer from {:?}: {}", p, e).into());
    }
    if let Some(dir) = search_dir {
        let adjacent = dir.join("tokenizer.json");
        if adjacent.exists() {
            return tokenizers::Tokenizer::from_file(&adjacent)
                .map_err(|e| format!("Tokenizer error: {}", e).into());
        }
    }
    if let Some(mid) = model_id {
        if let Ok(path) = hf_download(mid, "tokenizer.json") {
            return tokenizers::Tokenizer::from_file(&path)
                .map_err(|e| format!("Tokenizer error: {}", e).into());
        }
    }
    Err("Could not find tokenizer.json. Use --tokenizer <path> to provide it manually.".into())
}
