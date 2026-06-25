# Saccade (Project Nocturnal X1)

**Causal Token-Adaptive Residual Quantization (C-TARQ)**

Saccade is an implementation-first, hardware-compliant sparse residual weight streaming framework. It solves the rigid memory bandwidth limitations that choke large transformer architectures on consumer edge hardware by executing token-by-token dynamic mixed-precision routing.

## Why Saccade?

Traditional compression methods force an entire model to become small and fast at the cost of its intelligence, or they keep the model smart but too heavy to run on consumer hardware.

Saccade introduces a third way: it allows an AI to dynamically adjust its internal precision on a token-by-token basis. Predictable sequence steps are run strictly inside a lightweight 4-bit baseline math matrix, while complex tokens instantly trigger a sparse memory fetch of structural delta block arrays to patch precision errors on the fly.

## Features

*   **True 4-Bit Integer Matrix Packing:** Compresses 8 distinct 4-bit weights uniformly into a single, contiguous `int32` memory field, physically reducing RAM footprint by 4x vs FP16.
*   **On-the-Fly INT32 Dequantization:** Base weights stay packed in VRAM; the compiled kernel unpacks INT32→FP16 on-chip each forward pass, eliminating persistent cache duplication.
*   **Structured 16x16 Block-Sparsity:** Bounding residual secondary arrays into strict sparse blocks for contiguous streaming across memory buses.
*   **Hybrid Compilation & Conditional Routing (V2):** Fuses static-shaped base weight unpacking and GEMM passes into an optimized `torch.compile(fullgraph=True)` graph, while executing sparse delta additions conditionally to avoid useless FLOPs and memory bandwidth for inactive tokens.
*   **Developer Customizable Heuristics:** Open adapter ecosystem to define how your project classifies token complexities.

## Installation

```bash
pip install -e .
```

## CLI Usage

The `saccade` package includes a powerful, developer-friendly unified CLI to handle compilation and benchmarking.

```bash
# Standard Compilation
saccade compile \
    --model_id "Qwen/Qwen2.5-0.5B-Instruct" \
    --target_bpt 5.20 \
    --routing_type "variance" \
    --sparsity_density 0.25 \
    --target_layers "all" \
    --calibration_data "./dataset/eval_prompts.json" \
    --output_path "./weights/saccade_0.5b.safetensors"

# Automated A/B Evaluation Benchmark
saccade compile \
    --model_id "Qwen/Qwen2.5-0.5B-Instruct" \
    --calibration_data "./dataset/eval_prompts.json" \
    --output_path "./weights/saccade_0.5b.safetensors" \
    --target_layers "all" \
    --benchmark \
    --generate_plots
```

The `--target_layers` parameter accepts `"all"`, a comma-separated string like `"q_proj,v_proj"`, or defaults to FFN layers (`"mlp.down_proj, mlp.up_proj"`) if omitted.

## Quick Start (Python API)

You can compile and profile any HuggingFace model dynamically:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import saccade

# 1. Load model and tokenizer
model_id = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).cuda()

# 2. Compile model with C-TARQ
compiled_model, tracker = saccade.compile(
    model=model,
    tokenizer=tokenizer,
    target_bpt=5.20,
    routing_adapter=saccade.VarianceRoutingAdapter(),
    sparsity_density=0.25,
    target_layers="all"  # or specific layers like "q_proj, v_proj"
)

# 3. Profile execution speed and bits-per-token overhead
metrics = saccade.PerformanceProfiler.profile_generation_speed(
    compiled_model,
    tokenizer,
    text_payload="The classical framework of deep architectural topologies demands a rigorous balance of memory operations.",
    tracker=tracker
)

print(metrics)

# 4. Save optimized and compiled state
saccade.save_checkpoint(compiled_model, "saccade_optimized.safetensors", target_layers="all")
```

### Loading Checkpoints

To easily load a compiled and optimized checkpoint later:

```python
import torch
from transformers import AutoModelForCausalLM
import saccade

# 1. Load the original raw architecture
model_id = "Qwen/Qwen2.5-0.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)

# 2. Inject Saccade weights and kernels dynamically
compiled_model, tracker = saccade.load_checkpoint(
    model=model,
    file_path="saccade_optimized.safetensors",
    device="cuda"
)

# Now `compiled_model` is ready for high-performance generation!
```

## Custom Routing Heuristics

Developers can create custom heuristic routers by subclassing `BaseRoutingAdapter`:

```python
from saccade import BaseRoutingAdapter
import torch

class MyCustomHeuristic(BaseRoutingAdapter):
    def compute_token_complexity(self, x: torch.Tensor) -> torch.Tensor:
        # Example logic: Absolute maximum magnitude
        return torch.abs(torch.max(x.float(), dim=-1).values)

# Pass it during compilation:
compiled_model, tracker = saccade.compile(
    model=model,
    tokenizer=tokenizer,
    routing_adapter=MyCustomHeuristic()
)
```

## Advanced Usage
See `example.py` for a full runnable suite benchmarking several different downstream capability domains.