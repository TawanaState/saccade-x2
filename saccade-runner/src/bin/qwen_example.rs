use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::qwen2::{Config, ModelForCausalLM};
use hf_hub::api::sync::Api;
use tokenizers::Tokenizer;
use saccade_runner::{SaccadeModelApi, SaccadeMetrics};

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let device = Device::Cpu;

    println!("================================================================");
    println!("  Saccade V4 Qwen Example — Universal Interception Bench");
    println!("================================================================\n");

    // ----------------------------------------------------------------
    // Phase 1: Load tokenizer and configuration
    // ----------------------------------------------------------------
    println!("=== Phase 1: Acquiring Qwen2.5-0.5B-Instruct Model ===");
    let api = Api::new()?;
    let repo = api.model("Qwen/Qwen2.5-0.5B-Instruct".to_string());
    
    let config_file = repo.get("config.json")?;
    let tokenizer_file = repo.get("tokenizer.json")?;
    let weights_file = repo.get("model.safetensors")?;
    
    let config: Config = serde_json::from_reader(std::fs::File::open(config_file)?)?;
    let tokenizer = Tokenizer::from_file(tokenizer_file)?;
    
    println!("Loading weights into memory...");
    let tensors = candle_core::safetensors::load(&weights_file, &device)?;
    println!("Successfully loaded {} weight tensors.", tensors.len());

    // ----------------------------------------------------------------
    // Phase 2: Calibration & Compilation via Developer API
    // ----------------------------------------------------------------
    println!("\n=== Phase 2: Running Calibration & Compilation ===");
    
    // Simulate typical calibration activations (300 tokens of intermediate MLP dimension)
    // For intermediate MLP layers in Qwen2.5-0.5B, intermediate size = 4864
    let calib_tokens = 300;
    let intermediate_size = config.intermediate_size;
    
    let mut calib_data = vec![half::f16::from_f32(0.01); calib_tokens * intermediate_size];
    // Inject some structured high-variance features to simulate profile runs
    for t in 0..45 {
        for h in 0..intermediate_size {
            let sign = if h % 2 == 0 { 1.0 } else { -1.0 };
            calib_data[t * intermediate_size + h] = half::f16::from_f32(sign * 0.65);
        }
    }
    let calibration_activations = Tensor::from_vec(calib_data, (calib_tokens, intermediate_size), &device)?;
    
    // Target MLP projections for Saccade quantization
    let target_layers = vec!["gate_proj", "up_proj", "down_proj"];
    
    println!("Compiling tensors via SaccadeModelApi...");
    let compiled_tensors = SaccadeModelApi::compile_tensors(
        &tensors,
        &target_layers,
        &calibration_activations,
        0.15, // target sparse fill rate (15%)
        0.80, // pct_t4
        0.95, // pct_t8
    )?;

    // Save Saccade archive
    let output_path = "qwen2_saccade_example.safetensors";
    println!("Saving Saccade compiled archive to: {}", output_path);
    candle_core::safetensors::save(&compiled_tensors, output_path)?;

    // ----------------------------------------------------------------
    // Phase 3: Loading the Intercepted Model
    // ----------------------------------------------------------------
    println!("\n=== Phase 3: Instantiating Official Qwen2 model ===");
    
    // Create the VarBuilder pointing directly to our Saccade checkpoint.
    // The global candle_nn::Linear interception will catch any gate_proj, up_proj,
    // or down_proj and instantiate Saccade Linear layers automatically!
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[output_path], DType::F16, &device)?
    };
    
    let mut model = ModelForCausalLM::new(&config, vb)?;
    println!("Model loaded successfully! Saccade backends mounted automatically.");

    // ----------------------------------------------------------------
    // Phase 4: Token Generation & Benchmarking
    // ----------------------------------------------------------------
    println!("\n=== Phase 4: Evaluating Inference & Telemetry ===");
    
    let prompt = "Explain quantum physics in a single sentence.";
    let tokens = tokenizer.encode(prompt, true)?;
    let prompt_tokens = tokens.get_ids();
    
    let mut run_eval = |mode_name: &str, bypass: bool| -> Result<(String, SaccadeMetrics), Box<dyn std::error::Error + Send + Sync>> {
        println!("\n--- Running in {} Mode ---", mode_name);
        SaccadeModelApi::set_bypass(bypass);
        SaccadeModelApi::reset_telemetry();
        
        let mut tokens = prompt_tokens.to_vec();
        let mut generated_text = String::new();
        let start_time = std::time::Instant::now();
        
        for _step in 0..15 {
            let context_len = tokens.len();
            let input_tensor = Tensor::new(&tokens[context_len - 1..], &device)?.unsqueeze(0)?;
            let logits = model.forward(&input_tensor, context_len - 1)?;
            let logits = logits.squeeze(0)?.squeeze(0)?;
            
            // Greedy decode
            let next_token = logits.argmax(0)?.to_scalar::<u32>()?;
            tokens.push(next_token);
            
            let word = tokenizer.decode(&[next_token], true)?;
            generated_text.push_str(&word);
            generated_text.push(' ');
        }
        
        let elapsed = start_time.elapsed();
        let metrics = SaccadeModelApi::get_metrics();
        
        println!("Output: {}", generated_text.trim());
        println!("Latency: {:.2} ms", elapsed.as_secs_f64() * 1000.0);
        println!("Average Bits-Per-Token: {:.2} BPT", metrics.average_bpt);
        println!("Kernel Time: {:.2} ms", metrics.kernel_ms);
        
        Ok((generated_text, metrics))
    };

    let (ctarq_text, ctarq_metrics) = run_eval("C-TARQ Routing", false)?;
    let (bypass_text, bypass_metrics) = run_eval("Bypass Base-Only", true)?;

    println!("\n================================================================");
    println!("               SACCADE COMPARATIVE BENCHMARK REPORT");
    println!("================================================================");
    println!("C-TARQ Routing BPT:     {:.2} BPT", ctarq_metrics.average_bpt);
    println!("Bypass Baseline BPT:    {:.2} BPT", bypass_metrics.average_bpt);
    println!("Kernel Speedup:         {:.2}x", bypass_metrics.kernel_ms / ctarq_metrics.kernel_ms);
    println!("Generated text matches: {}", ctarq_text == bypass_text);
    println!("================================================================");

    Ok(())
}
