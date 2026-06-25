import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np

import saccade

# Establish execution residency constraints
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f">> Hardware Accelerator Validated: [{torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU Vector Engine'}]")

def run_example():
    target_hf_model = "Qwen/Qwen2.5-0.5B-Instruct"  # Using 0.5B for faster local testing, can be 3B
    print(f"[System Init] Loading raw target configurations: {target_hf_model}")

    tokenizer = AutoTokenizer.from_pretrained(target_hf_model)
    vanilla_model = AutoModelForCausalLM.from_pretrained(target_hf_model, torch_dtype=torch.float16).to(device)

    tasks = {
        "WikiText-2 (Prose)": "The classical framework of deep architectural topologies demands a rigorous balance of memory operations.",
        "GSM8K / MATH (Reasoning)": "Question: A vector space has a dimension of 512. If a transformation matrix reduces its basis by 128 coefficients, find rank. 512 - 128 = 384.",
        "HumanEval (Coding Syntax)": "def optimized_fused_gemm(X, W, Block_Size=16):\n    shared_memory = alloc_blocks(Block_Size)\n    return advanced_accumulation_path(X, W)",
        "MMLU (General Knowledge)": "The primary structure of chemical proteins relies heavily on stable configurations and molecular alignments across atomic bonds."
    }

    # Example of a custom routing heuristic
    class MyCustomHeuristic(saccade.BaseRoutingAdapter):
        def compute_token_complexity(self, x: torch.Tensor) -> torch.Tensor:
            # A simple heuristic: max activation per token
            return torch.abs(torch.max(x.float(), dim=-1).values)

    # Compile model using Saccade directly
    compiled_model, runtime_tracker = saccade.compile(
        model=vanilla_model,
        tokenizer=tokenizer,
        target_bpt=5.20,
        routing_adapter=saccade.VarianceRoutingAdapter(), # Or use MyCustomHeuristic()
        sparsity_density=0.25,
        target_layers="all"  # Enable targeting across all transformer projections for accurate benchmarking
    )

    # Run benchmarks across tasks
    empirical_records = {}
    for task_name, text in tasks.items():
        print(f" ↳ Monitoring hardware bus transitions on track: [{task_name}]")
        metrics = saccade.PerformanceProfiler.profile_generation_speed(compiled_model, tokenizer, text, runtime_tracker)
        empirical_records[task_name] = metrics

    # Serialize verified graph layers directly to safetensors files
    saccade.save_checkpoint(compiled_model, "saccade_example_model.safetensors", routing_heuristic_name="variance", target_layers="all")

    print("\n" + "="*115)
    print(f"{'EVALUATED DOWNSTREAM CAPABILITY TARGETS':<38} | {'BPT FOOTPRINT':<14} | {'THROUGHPUT (t/s)':<18} | {'BUS RELIEF SAVINGS'}")
    print("-"*115)
    for track, m in empirical_records.items():
        print(f"🚀 {track:<35} | {m['bpt']:<14.2f} | {m['speed']:<18.1f} | {m['saved_mb']:.2f} MB")
    print("="*115 + "\n")

if __name__ == "__main__":
    run_example()
