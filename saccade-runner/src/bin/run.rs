use clap::Parser;
use candle_core::{DType, Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use saccade_runner::model::{Qwen2Config, Qwen2Model};
use hf_hub::api::sync::Api;
use std::io::Write;
use std::path::PathBuf;

const EOS_TOKEN: u32 = 151645; // <|im_end|>
const ENDOFTEXT_TOKEN: u32 = 151643; // <|endoftext|>

#[derive(Parser)]
#[command(name = "saccade-run")]
#[command(about = "Stream text generation with vanilla or Saccade-compressed models")]
struct Args {
    /// Saccade mode: path to a compiled .safetensors checkpoint
    #[arg(long)]
    checkpoint: Option<PathBuf>,

    /// Vanilla mode: HuggingFace model repository name
    #[arg(long)]
    model_id: Option<String>,

    /// Input prompt for generation
    #[arg(long)]
    prompt: String,

    /// Maximum tokens to generate
    #[arg(long, default_value_t = 100)]
    max_tokens: usize,

    /// Sampling temperature (0 = greedy)
    #[arg(long, default_value_t = 0.7)]
    temperature: f64,

    /// Top-p nucleus sampling threshold
    #[arg(long)]
    top_p: Option<f64>,

    /// Random seed for reproducibility
    #[arg(long, default_value_t = 42)]
    seed: u64,
}

struct GenerationTelemetry {
    total_tokens: usize,
    prefill_ms: f64,
    decode_ms: f64,
    mode: String,
    weight_bytes: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let device = Device::Cpu;

    let (mut model, tokenizer, _cfg, telemetry_mode, weight_bytes) = match (&args.checkpoint, &args.model_id) {
        (Some(checkpoint), None) => load_saccade(&checkpoint, &device)?,
        (None, Some(model_id)) => load_vanilla(model_id, &device)?,
        _ => {
            eprintln!("Error: provide either --checkpoint (Saccade) or --model-id (vanilla), not both.");
            std::process::exit(1);
        }
    };

    // Wrap prompt in Qwen chat template
    let chat_prompt = format!(
        "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
        args.prompt
    );

    let encoding = tokenizer.encode(chat_prompt.as_str(), true)
        .map_err(|e| format!("Tokenization failed: {}", e))?;
    let input_ids: Vec<u32> = encoding.get_ids().to_vec();
    let input_len = input_ids.len();

    println!("================================================================");
    println!("  Saccade V3 — Streaming Inference Engine");
    println!("  Mode: {}", telemetry_mode);
    println!("  Prompt tokens: {}", input_len);
    println!("================================================================\n");

    let temperature = if args.temperature <= 0.0 { None } else { Some(args.temperature) };
    let mut logits_processor = LogitsProcessor::new(args.seed, temperature, args.top_p);

    // Prefill
    let prefill_start = std::time::Instant::now();
    let input_tensor = Tensor::new(input_ids.as_slice(), &device)?.unsqueeze(0)?;
    let logits = model.forward(&input_tensor, 0)?;
    let prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;

    let logits = logits.squeeze(0)?.squeeze(0)?; // (vocab_size,)
    let mut next_token = logits_processor.sample(&logits)?;
    let mut generated_tokens = vec![next_token];

    // Print first token
    if let Some(text) = decode_token(&tokenizer, next_token) {
        print!("{}", text);
        std::io::stdout().flush()?;
    }

    // Autoregressive decode loop
    let decode_start = std::time::Instant::now();
    let mut offset = input_len;

    for _ in 1..args.max_tokens {
        if next_token == EOS_TOKEN || next_token == ENDOFTEXT_TOKEN {
            break;
        }

        let token_tensor = Tensor::new(&[next_token], &device)?.unsqueeze(0)?;
        let logits = model.forward(&token_tensor, offset)?;
        let logits = logits.squeeze(0)?.squeeze(0)?;
        next_token = logits_processor.sample(&logits)?;
        generated_tokens.push(next_token);
        offset += 1;

        if let Some(text) = decode_token(&tokenizer, next_token) {
            if next_token != EOS_TOKEN && next_token != ENDOFTEXT_TOKEN {
                print!("{}", text);
                std::io::stdout().flush()?;
            }
        }
    }

    let decode_ms = decode_start.elapsed().as_secs_f64() * 1000.0;
    let decode_tokens = generated_tokens.len().saturating_sub(1).max(1);

    println!("\n");

    // Telemetry
    let telemetry = GenerationTelemetry {
        total_tokens: generated_tokens.len(),
        prefill_ms,
        decode_ms,
        mode: telemetry_mode.clone(),
        weight_bytes,
    };
    print_telemetry(&telemetry, decode_tokens);

    Ok(())
}

fn load_vanilla(model_id: &str, device: &Device) -> Result<(Qwen2Model, tokenizers::Tokenizer, Qwen2Config, String, usize), Box<dyn std::error::Error>> {
    println!("Downloading vanilla model: {}", model_id);
    let api = Api::new()?;
    let repo = api.model(model_id.to_string());
    let model_file = repo.get("model.safetensors")?;
    let config_file = repo.get("config.json")?;
    let tokenizer_file = repo.get("tokenizer.json")?;

    let cfg: Qwen2Config = serde_json::from_str(&std::fs::read_to_string(&config_file)?)?;
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_file)
        .map_err(|e| format!("Tokenizer error: {}", e))?;

    println!("Building vanilla model ({} layers, hidden={})...", cfg.num_hidden_layers, cfg.hidden_size);
    let vb = unsafe {
        candle_nn::VarBuilder::from_mmaped_safetensors(&[&model_file], DType::F32, device)?
    };
    let model = Qwen2Model::from_standard(&cfg, vb)?;

    let weight_bytes = cfg.vocab_size * cfg.hidden_size * 2 // embeddings
        + cfg.num_hidden_layers * (
            cfg.hidden_size * cfg.hidden_size * 2 * 4 // attention projections
            + cfg.hidden_size * cfg.intermediate_size * 2 * 3 // MLP projections
        )
        + cfg.vocab_size * cfg.hidden_size * 2; // lm_head

    Ok((model, tokenizer, cfg, "Vanilla FP16 Baseline".into(), weight_bytes))
}

fn load_saccade(checkpoint: &PathBuf, device: &Device) -> Result<(Qwen2Model, tokenizers::Tokenizer, Qwen2Config, String, usize), Box<dyn std::error::Error>> {
    println!("Loading Saccade checkpoint: {:?}", checkpoint);
    let tensors = candle_core::safetensors::load(checkpoint, device)?;

    // Infer config from tensor shapes
    let embed_w = tensors.get("model.embed_tokens.weight")
        .ok_or("Missing model.embed_tokens.weight in checkpoint")?;
    let embed_dims = embed_w.dims();
    let vocab_size = embed_dims[0];
    let hidden_size = embed_dims[1];

    // Count layers
    let num_layers = (0..)
        .take_while(|i| tensors.contains_key(&format!("model.layers.{}.input_layernorm.weight", i)))
        .count();

    // Detect intermediate_size from the first MLP gate_proj
    let intermediate_size = if let Some(t) = tensors.get("model.layers.0.mlp.gate_proj.saccade_scale_base") {
        t.dim(0)?
    } else if let Some(t) = tensors.get("model.layers.0.mlp.gate_proj.weight") {
        t.dim(0)?
    } else {
        hidden_size * 4 // fallback
    };

    // Detect num_attention_heads from q_proj shape
    let q_proj_w = tensors.get("model.layers.0.self_attn.q_proj.weight")
        .ok_or("Missing q_proj.weight")?;
    let q_out = q_proj_w.dim(0)?;

    // Detect num_kv_heads from k_proj shape
    let k_proj_w = tensors.get("model.layers.0.self_attn.k_proj.weight")
        .ok_or("Missing k_proj.weight")?;
    let k_out = k_proj_w.dim(0)?;
    let _head_dim = hidden_size / (q_out / (q_out.max(k_out) / k_out.max(1)));

    // Simpler: assume head_dim is gcd-based
    let head_dim_guess = if k_out > 0 && q_out % k_out == 0 {
        // q_out = num_heads * head_dim, k_out = num_kv_heads * head_dim
        // gcd tells us head_dim
        gcd(q_out, k_out)
    } else {
        64 // fallback
    };

    let cfg = Qwen2Config {
        vocab_size,
        hidden_size,
        intermediate_size,
        num_hidden_layers: num_layers,
        num_attention_heads: q_out / head_dim_guess,
        num_key_value_heads: k_out / head_dim_guess,
        rope_theta: 1_000_000.0,
        rms_norm_eps: 1e-6,
        sliding_window: 32768,
        max_position_embeddings: 32768,
        tie_word_embeddings: false,
    };

    println!("Inferred config: {}x{}, {} layers, {} heads, {} kv_heads",
        cfg.hidden_size, cfg.intermediate_size, cfg.num_hidden_layers,
        cfg.num_attention_heads, cfg.num_key_value_heads);

    // Load tokenizer — try same directory as checkpoint, fall back to downloading
    let checkpoint_dir = checkpoint.parent().unwrap_or(std::path::Path::new("."));
    let tokenizer_path = checkpoint_dir.join("tokenizer.json");
    let tokenizer = if tokenizer_path.exists() {
        tokenizers::Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| format!("Tokenizer error: {}", e))?
    } else {
        eprintln!("Warning: tokenizer.json not found alongside checkpoint, downloading from HF...");
        // Try to infer source model from vocab_size
        let model_id = if vocab_size == 151936 { "Qwen/Qwen2.5-0.5B-Instruct" } else { "Qwen/Qwen2.5-0.5B-Instruct" };
        let api = Api::new()?;
        let repo = api.model(model_id.to_string());
        let tok_file = repo.get("tokenizer.json")?;
        tokenizers::Tokenizer::from_file(&tok_file)
            .map_err(|e| format!("Tokenizer error: {}", e))?
    };

    let model = Qwen2Model::from_saccade_checkpoint(&cfg, &tensors, device)?;

    // Compute weight memory footprint
    let weight_bytes: usize = tensors.values().map(|t| {
        let elems: usize = t.dims().iter().product();
        let bpe = match t.dtype() {
            candle_core::DType::F16 => 2, candle_core::DType::F32 => 4,
            candle_core::DType::U32 => 4, candle_core::DType::U8 => 1, _ => 2,
        };
        elems * bpe
    }).sum();

    Ok((model, tokenizer, cfg, "Saccade C-TARQ Adaptive".into(), weight_bytes))
}

fn decode_token(tokenizer: &tokenizers::Tokenizer, token_id: u32) -> Option<String> {
    tokenizer.decode(&[token_id], false).ok()
}

fn print_telemetry(t: &GenerationTelemetry, decode_tokens: usize) {
    let decode_tok_per_sec = if t.decode_ms > 0.0 {
        decode_tokens as f64 / (t.decode_ms / 1000.0)
    } else { 0.0 };
    let ms_per_tok = if decode_tokens > 0 { t.decode_ms / decode_tokens as f64 } else { 0.0 };

    println!("================================================================");
    println!("           SACCADE PERFORMANCE AUDIT TELEMETRY LOG");
    println!("================================================================");
    println!("Execution Mode:          {}", t.mode);
    println!("Total Tokens Decoded:    {}", t.total_tokens);
    println!("----------------------------------------------------------------");
    println!("Prefill Latency:         {:.1} ms", t.prefill_ms);
    println!("Decode Latency:          {:.2} ms/token", ms_per_tok);
    println!("Generation Speed:        {:.1} tokens/second", decode_tok_per_sec);
    println!("Weight Memory Footprint: {:.2} MB", t.weight_bytes as f64 / (1024.0 * 1024.0));
    println!("================================================================");
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}
