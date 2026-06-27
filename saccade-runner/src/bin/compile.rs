use clap::Parser;
use candle_core::{DType, Device, Module, Tensor};
use saccade_core::{
    calibration::ProfileRunner,
    compress_tensor_to_saccade, compute_percentile_threshold,
};
use saccade_runner::model::Qwen2Config;
use hf_hub::api::sync::Api;
use std::collections::HashMap;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "saccade-compile")]
#[command(about = "Compress a HuggingFace model into Saccade C-TARQ format")]
struct Args {
    /// HuggingFace model repository (e.g., "Qwen/Qwen2.5-0.5B-Instruct")
    #[arg(long)]
    model_id: String,

    /// Path to a plain-text calibration file for threshold extraction
    #[arg(long)]
    calib_file: PathBuf,

    /// Output path for the compressed safetensors archive
    #[arg(long, default_value = "saccade_model.safetensors")]
    output_path: PathBuf,

    /// Target delta fill rate (0.15 = 15% of weights get sparse corrections)
    #[arg(long, default_value_t = 0.15)]
    target_fill: f32,

    /// Percentile for t4 routing threshold (fraction of tokens below t4)
    #[arg(long, default_value_t = 0.80)]
    pct_t4: f32,

    /// Percentile for t8 routing threshold (fraction of tokens below t8)
    #[arg(long, default_value_t = 0.95)]
    pct_t8: f32,

    /// Number of layers to run for hybrid calibration (0 = embedding-only)
    #[arg(long, default_value_t = 0)]
    calib_layers: usize,

    /// Path to tokenizer.json (downloaded automatically if not provided)
    #[arg(long)]
    tokenizer: Option<PathBuf>,
}

/// Download a single file from HF Hub, retrying with a fresh API handle on failure.
fn hf_download(model_id: &str, filename: &str) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let api = Api::new()?;
    let repo = api.model(model_id.to_string());
    match repo.get(filename) {
        Ok(path) => Ok(path),
        Err(_) => {
            // Retry with a fresh API instance (works around hf-hub URL caching bugs)
            let api2 = Api::new()?;
            let repo2 = api2.model(model_id.to_string());
            Ok(repo2.get(filename)?)
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let device = Device::Cpu;

    println!("================================================================");
    println!("  Saccade V3 C-TARQ — Model Compilation Engine");
    println!("================================================================\n");

    // ---- 1. Download model files (each in a separate API call to avoid URL bugs) ----
    println!("=== Phase 1: Downloading {} ===", args.model_id);
    let model_file = hf_download(&args.model_id, "model.safetensors")?;
    println!("Weights: {:?}", model_file);

    // ---- 2. Load tensors and infer config from tensor shapes ----
    println!("\n=== Phase 2: Loading model tensors ===");
    let tensors = candle_core::safetensors::load(&model_file, &device)?;
    println!("Loaded {} tensors.", tensors.len());

    let cfg = infer_config(&tensors)?;
    println!("Inferred config: hidden={}, intermediate={}, layers={}, vocab={}",
        cfg.hidden_size, cfg.intermediate_size, cfg.num_hidden_layers, cfg.vocab_size);

    // ---- 3. Calibration ----
    println!("\n=== Phase 3: Calibration ===");
    let calib_text = std::fs::read_to_string(&args.calib_file)?;
    println!("Calibration text: {} chars", calib_text.len());

    // Load tokenizer — from CLI flag, or download from HF Hub
    let tokenizer_path = if let Some(ref p) = args.tokenizer {
        p.clone()
    } else {
        println!("Downloading tokenizer...");
        hf_download(&args.model_id, "tokenizer.json")?
    };

    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| format!("Failed to load tokenizer: {}", e))?;
    let encoding = tokenizer.encode(calib_text.as_str(), true)
        .map_err(|e| format!("Tokenization failed: {}", e))?;
    let token_ids: Vec<u32> = encoding.get_ids().to_vec();
    let num_tokens = token_ids.len().min(512);
    let token_ids = &token_ids[..num_tokens];
    println!("Tokenized {} calibration tokens.", num_tokens);

    let token_tensor = Tensor::new(token_ids, &device)?.unsqueeze(0)?;

    let activations = if args.calib_layers > 0 {
        println!("Running hybrid calibration through {} layers...", args.calib_layers);
        let config_file = hf_download(&args.model_id, "config.json")?;
        let config_str = std::fs::read_to_string(&config_file)?;
        let parsed_cfg: Qwen2Config = serde_json::from_str(&config_str)?;
        let vb = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(&[&model_file], DType::F32, &device)?
        };
        let mut model = saccade_runner::model::Qwen2Model::from_standard(&parsed_cfg, vb)?;
        let hidden = model.forward_calibrate(&token_tensor, args.calib_layers)?;
        hidden.squeeze(0)?
    } else {
        // Embedding-only calibration (fast, no full model needed)
        println!("Running embedding-only calibration...");
        let embed_w = tensors.get("model.embed_tokens.weight")
            .ok_or("Missing embedding weights")?;
        let embedding = candle_nn::Embedding::new(embed_w.clone(), cfg.hidden_size);
        embedding.forward(&token_tensor.squeeze(0)?)?
    };

    let calib_f16 = activations.to_dtype(DType::F16)?;
    let (t4, t8) = ProfileRunner::calibrate(&calib_f16, args.pct_t4, args.pct_t8)?;
    println!("Extracted thresholds: t4 = {:.6}, t8 = {:.6}", t4, t8);

    // ---- 4. Compress all MLP layers ----
    println!("\n=== Phase 4: Compressing MLP layers ===");
    let target_projs = ["gate_proj", "up_proj", "down_proj"];
    let mut output_tensors: HashMap<String, Tensor> = HashMap::new();

    // Copy non-MLP tensors unchanged
    for (name, tensor) in tensors.iter() {
        let is_mlp_weight = target_projs.iter().any(|p| name.contains(p)) && name.ends_with(".weight");
        if !is_mlp_weight {
            output_tensors.insert(name.clone(), tensor.clone());
        }
    }

    let mut total_compressed = 0usize;
    for layer_idx in 0..cfg.num_hidden_layers {
        output_tensors.insert(
            format!("model.layers.{}.saccade_t4", layer_idx),
            Tensor::from_vec(vec![t4], (1,), &device)?,
        );
        output_tensors.insert(
            format!("model.layers.{}.saccade_t8", layer_idx),
            Tensor::from_vec(vec![t8], (1,), &device)?,
        );

        for proj in &target_projs {
            let weight_key = format!("model.layers.{}.mlp.{}.weight", layer_idx, proj);
            let tensor = match tensors.get(&weight_key) {
                Some(t) => t,
                None => {
                    eprintln!("Warning: missing {}, skipping", weight_key);
                    continue;
                }
            };

            let dims = tensor.shape().dims();
            if dims.len() != 2 || dims[1] % 8 != 0 {
                eprintln!("Warning: {} has incompatible shape {:?}, keeping as-is", weight_key, dims);
                output_tensors.insert(weight_key, tensor.clone());
                continue;
            }

            let delta_threshold = compute_percentile_threshold(tensor, args.target_fill)?;
            let blocks = compress_tensor_to_saccade(tensor, delta_threshold)?;

            let prefix = format!("model.layers.{}.mlp.{}", layer_idx, proj);
            for (suffix, comp_tensor) in blocks {
                output_tensors.insert(format!("{}.saccade_{}", prefix, suffix), comp_tensor);
            }
            total_compressed += 1;
        }

        if (layer_idx + 1) % 8 == 0 || layer_idx == cfg.num_hidden_layers - 1 {
            println!("  Compressed layers 0-{} ({} projections done)", layer_idx, total_compressed);
        }
    }
    println!("Total compressed projections: {}", total_compressed);

    // ---- 5. Save ----
    println!("\n=== Phase 5: Saving compressed checkpoint ===");
    println!("Output: {:?}", args.output_path);
    candle_core::safetensors::save(&output_tensors, &args.output_path)?;

    let orig_size: usize = tensors.values().map(|t| {
        let elems: usize = t.dims().iter().product();
        let bpe = match t.dtype() { DType::F16 => 2, DType::F32 => 4, DType::U32 => 4, DType::U8 => 1, _ => 2 };
        elems * bpe
    }).sum();

    let comp_size: usize = output_tensors.values().map(|t| {
        let elems: usize = t.dims().iter().product();
        let bpe = match t.dtype() { DType::F16 => 2, DType::F32 => 4, DType::U32 => 4, DType::U8 => 1, _ => 2 };
        elems * bpe
    }).sum();

    println!("\n=== Compilation Summary ===");
    println!("  Source model: {}", args.model_id);
    println!("  Layers compressed: {} ({} projections)", cfg.num_hidden_layers, total_compressed);
    println!("  Original size: {:.2} MB", orig_size as f64 / (1024.0 * 1024.0));
    println!("  Compressed size: {:.2} MB", comp_size as f64 / (1024.0 * 1024.0));
    println!("  Compression ratio: {:.2}x", orig_size as f64 / comp_size as f64);
    println!("  Target fill rate: {:.0}%", args.target_fill * 100.0);
    println!("  Routing thresholds: t4={:.6}, t8={:.6}", t4, t8);
    println!("\nDone.");
    Ok(())
}

/// Infer Qwen2Config from tensor shapes — avoids needing to download config.json.
fn infer_config(tensors: &HashMap<String, Tensor>) -> Result<Qwen2Config, Box<dyn std::error::Error>> {
    let embed_w = tensors.get("model.embed_tokens.weight")
        .ok_or("Missing model.embed_tokens.weight")?;
    let embed_dims = embed_w.dims();
    let vocab_size = embed_dims[0];
    let hidden_size = embed_dims[1];

    let num_hidden_layers = (0..)
        .take_while(|i| tensors.contains_key(&format!("model.layers.{}.input_layernorm.weight", i)))
        .count();

    let intermediate_size = tensors
        .get("model.layers.0.mlp.gate_proj.weight")
        .map(|t| t.dims()[0])
        .unwrap_or(hidden_size * 4);

    let q_proj_w = tensors.get("model.layers.0.self_attn.q_proj.weight")
        .ok_or("Missing q_proj.weight")?;
    let k_proj_w = tensors.get("model.layers.0.self_attn.k_proj.weight")
        .ok_or("Missing k_proj.weight")?;
    let q_out = q_proj_w.dims()[0];
    let k_out = k_proj_w.dims()[0];

    // head_dim = hidden_size / num_attention_heads. For Qwen2, num_attention_heads
    // is encoded as q_proj_out / head_dim. We derive head_dim from the k_proj: since
    // k_out = num_kv_heads * head_dim and num_kv_heads is small (typically 2),
    // head_dim is k_out / num_kv_heads. We test common kv_head counts (2, 4, 8).
    let head_dim = [2usize, 4, 8, 1].iter()
        .map(|&nkv| k_out / nkv)
        .find(|&hd| hd > 0 && q_out % hd == 0 && k_out % hd == 0 && q_out / hd == hidden_size / hd)
        .unwrap_or(64);

    Ok(Qwen2Config {
        vocab_size,
        hidden_size,
        intermediate_size,
        num_hidden_layers,
        num_attention_heads: q_out / head_dim,
        num_key_value_heads: k_out / head_dim,
        rope_theta: 1_000_000.0,
        rms_norm_eps: 1e-6,
        sliding_window: 32768,
        max_position_embeddings: 32768,
        tie_word_embeddings: false,
    })
}
