use candle_core::{DType, Device, Tensor};
use candle_nn::{Module, VarBuilder};
use candle_transformers::models::whisper::{Config, model::Whisper};
use hf_hub::api::sync::Api;
use saccade_runner::{SaccadeModelApi, SaccadeMetrics};
use clap::Parser;

#[derive(Parser, Debug)]
#[command(author, version, about = "Saccade Whisper Audio Calibration & Benchmarking Example")]
struct Args {
    /// Path to a custom WAV file to use for calibration. If omitted, downloads 'samples_jfk.wav'.
    #[arg(long)]
    wav_calib: Option<std::path::PathBuf>,

    /// Path to a custom WAV file to decode and transcribe. If omitted, downloads 'samples_jfk.wav'.
    #[arg(long)]
    wav_input: Option<std::path::PathBuf>,
}

fn pcm_decode<P: AsRef<std::path::Path>>(path: P) -> Result<(Vec<f32>, u32), Box<dyn std::error::Error + Send + Sync>> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    if spec.channels != 1 {
        return Err("WAV file must be mono".into());
    }
    let pcm_data = match spec.sample_format {
        hound::SampleFormat::Int => {
            let max_val = 2.0f32.powi(spec.bits_per_sample as i32 - 1);
            reader.samples::<i32>()
                .map(|s| s.map(|v| v as f32 / max_val))
                .collect::<Result<Vec<_>, _>>()?
        }
        hound::SampleFormat::Float => {
            reader.samples::<f32>()
                .collect::<Result<Vec<_>, _>>()?
        }
    };
    Ok((pcm_data, spec.sample_rate))
}

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let args = Args::parse();
    let device = Device::Cpu;

    println!("================================================================");
    println!("  Saccade V4 Whisper Example — Real Audio Calibration & Acceleration");
    println!("================================================================\n");

    // ----------------------------------------------------------------
    // Phase 1: Load config and weights from HF Hub
    // ----------------------------------------------------------------
    println!("=== Phase 1: Acquiring Whisper-Tiny Model & Default Samples ===");
    let api = Api::new()?;
    let repo = api.model("openai/whisper-tiny".to_string());
    
    println!("Downloading config.json...");
    let config_file = repo.get("config.json")?;
    println!("Downloading model.safetensors...");
    let weights_file = repo.get("model.safetensors")?;
    
    let config: Config = serde_json::from_reader(std::fs::File::open(config_file)?)?;
    println!("Model config loaded: d_model = {}, layers = {}", config.d_model, config.encoder_layers);
    
    println!("Loading weights into memory...");
    let tensors = candle_core::safetensors::load(&weights_file, &device)?;
    
    // Handle WAV files (User-supplied or default samples_jfk.wav)
    println!("Downloading samples_jfk.wav from Narsil/candle_demo...");
    let default_wav = {
        let dataset = api.dataset("Narsil/candle_demo".to_string());
        dataset.get("samples_jfk.wav")?
    };
    
    let calib_wav = args.wav_calib.unwrap_or(default_wav.clone());
    let input_wav = args.wav_input.unwrap_or(default_wav);
    
    println!("Calibration WAV: {:?}", calib_wav);
    println!("Execution WAV: {:?}", input_wav);

    // ----------------------------------------------------------------
    // Phase 2: Mel-Filter setup & Audio Decoding
    // ----------------------------------------------------------------
    println!("\n=== Phase 2: Decoding Audio & Computing Mel Spectrograms ===");
    
    // Load static mel filters from candle examples directory
    let mel_bytes = match config.num_mel_bins {
        80 => std::fs::read("candle/candle-examples/examples/whisper/melfilters.bytes")?,
        128 => std::fs::read("candle/candle-examples/examples/whisper/melfilters128.bytes")?,
        n => return Err(format!("unexpected num_mel_bins {}", n).into()),
    };
    let mut mel_filters = vec![0f32; mel_bytes.len() / 4];
    <byteorder::LittleEndian as byteorder::ByteOrder>::read_f32_into(&mel_bytes, &mut mel_filters);

    // Helper to decode PCM and compute log-mel spectrogram
    let wav_to_mel_tensor = |path: &std::path::PathBuf| -> Result<Tensor, Box<dyn std::error::Error + Send + Sync>> {
        let (pcm_data, sample_rate) = pcm_decode(path)?;
        if sample_rate != 16000 {
            return Err(format!("WAV file must be 16000Hz, found {}Hz", sample_rate).into());
        }
        let mel = candle_transformers::models::whisper::audio::pcm_to_mel(&config, &pcm_data, &mel_filters);
        let mel_len = mel.len();
        let t = Tensor::from_vec(
            mel,
            (1, config.num_mel_bins, mel_len / config.num_mel_bins),
            &device,
        )?;
        Ok(t)
    };

    println!("Processing calibration audio...");
    let calib_mel = wav_to_mel_tensor(&calib_wav)?;
    println!("Processing execution audio...");
    let input_mel = wav_to_mel_tensor(&input_wav)?;

    // ----------------------------------------------------------------
    // Phase 3: Extract Real Activations & Calibrate
    // ----------------------------------------------------------------
    println!("\n=== Phase 3: Extracting Real Activation Profiles & Compiling ===");
    
    // Extract convolution layers weights to run initial encoder passes on the CPU
    let conv1_w = tensors.get("model.encoder.conv1.weight").ok_or("missing conv1 weight")?;
    let conv1_b = tensors.get("model.encoder.conv1.bias").ok_or("missing conv1 bias")?;
    let conv2_w = tensors.get("model.encoder.conv2.weight").ok_or("missing conv2 weight")?;
    let conv2_b = tensors.get("model.encoder.conv2.bias").ok_or("missing conv2 bias")?;
    
    let conv1 = candle_nn::Conv1d::new(
        conv1_w.clone(),
        Some(conv1_b.clone()),
        candle_nn::Conv1dConfig { padding: 1, stride: 1, groups: 1, dilation: 1, cudnn_fwd_algo: None }
    );
    let conv2 = candle_nn::Conv1d::new(
        conv2_w.clone(),
        Some(conv2_b.clone()),
        candle_nn::Conv1dConfig { padding: 1, stride: 2, groups: 1, dilation: 1, cudnn_fwd_algo: None }
    );
    
    // Forward through first stages to get post-conv activations
    let x = conv1.forward(&calib_mel)?.gelu()?;
    let x = conv2.forward(&x)?.gelu()?;
    let x = x.transpose(1, 2)?;
    let (_, seq_len, d_model) = x.dims3()?;
    let calibration_activations = x.reshape((seq_len, d_model))?;
    
    println!("Profile size extracted: {:?}", calibration_activations.shape());

    // Target the encoder and decoder FFN projections: "fc1" and "fc2"
    let target_layers = vec!["fc1", "fc2"];
    
    println!("Compiling tensors via SaccadeModelApi...");
    let compiled_tensors = SaccadeModelApi::compile_tensors(
        &tensors,
        &target_layers,
        &calibration_activations,
        0.15, // target sparse fill rate (15%)
        0.80, // pct_t4
        0.95, // pct_t8
    )?;

    let output_path = "whisper_saccade_example.safetensors";
    println!("Saving Saccade compiled archive to: {}", output_path);
    candle_core::safetensors::save(&compiled_tensors, output_path)?;

    // ----------------------------------------------------------------
    // Phase 4: Loading the Intercepted Whisper Model
    // ----------------------------------------------------------------
    println!("\n=== Phase 4: Instantiating Official Whisper model ===");
    
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[output_path], DType::F32, &device)?
    };
    
    let mut model = Whisper::load(&vb, config)?;
    println!("Whisper model loaded successfully! C-TARQ layers mounted.");

    // ----------------------------------------------------------------
    // Phase 5: Execution & Telemetry Benchmarking
    // ----------------------------------------------------------------
    println!("\n=== Phase 5: Running Encoder Inference ===");
    
    let mut run_encoder = |mode_name: &str, bypass: bool| -> Result<(Tensor, SaccadeMetrics), Box<dyn std::error::Error + Send + Sync>> {
        println!("\n--- Running Encoder in {} Mode ---", mode_name);
        SaccadeModelApi::set_bypass(bypass);
        SaccadeModelApi::reset_telemetry();
        
        let start_time = std::time::Instant::now();
        let encoder_out = model.encoder.forward(&input_mel, true)?;
        let elapsed = start_time.elapsed();
        
        let metrics = SaccadeModelApi::get_metrics();
        
        println!("Output features shape: {:?}", encoder_out.shape());
        println!("Latency: {:.2} ms", elapsed.as_secs_f64() * 1000.0);
        println!("Average Bits-Per-Token: {:.2} BPT", metrics.average_bpt);
        println!("Kernel Time: {:.2} ms", metrics.kernel_ms);
        
        Ok((encoder_out, metrics))
    };

    let (ctarq_out, ctarq_metrics) = run_encoder("C-TARQ Routing", false)?;
    let (bypass_out, bypass_metrics) = run_encoder("Bypass Base-Only", true)?;

    // Calculate accuracy similarity between C-TARQ and Bypass dequantized outputs
    let diff = (&ctarq_out - &bypass_out)?;
    let rmse = diff.sqr()?.mean_all()?.to_scalar::<f32>()?.sqrt();
    
    let dot = (&ctarq_out * &bypass_out)?.sum_all()?.to_scalar::<f32>()?;
    let norm_ctarq = ctarq_out.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
    let norm_bypass = bypass_out.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
    let cosine_sim = dot / (norm_ctarq * norm_bypass);

    println!("\n================================================================");
    println!("               SACCADE WHISPER COMPARATIVE BENCHMARK");
    println!("================================================================");
    println!("C-TARQ Routing BPT:     {:.2} BPT", ctarq_metrics.average_bpt);
    println!("Bypass Baseline BPT:    {:.2} BPT", bypass_metrics.average_bpt);
    println!("Avg Logit Similarity:   {:.6}", cosine_sim);
    println!("Avg Logit RMSE:         {:.6}", rmse);
    println!("Kernel Speedup:         {:.2}x", bypass_metrics.kernel_ms / ctarq_metrics.kernel_ms);
    println!("================================================================");

    Ok(())
}
