import torch
import torch.nn as nn
import numpy as np

from .core.packing import SaccadeBitPacker, BlockSparseMatrix
from .runtime import SaccadeLinearPlugin
from .adapters.routing import BaseRoutingAdapter, VarianceRoutingAdapter

class SaccadeCompiler:
    """Handles parameter compression, automated error extraction, and block-sparsity styling."""

    @staticmethod
    def compile_model(
        model,
        tokenizer,
        calibration_text: str = None,
        target_bpt: float = 5.25,
        routing_adapter: BaseRoutingAdapter = None,
        sparsity_density: float = 0.30,
        target_layers: list = None
    ):
        """
        Extracts precision weights, maps dequantized layouts,
        and computes the high-precision error delta matrix.

        Args:
            model: The base HuggingFace model to compile.
            tokenizer: The tokenizer associated with the model.
            calibration_text: Optional custom text to calibrate against.
            target_bpt: The target bits-per-token limit for memory bandwidth profiling.
            routing_adapter: An instantiated adapter inherited from BaseRoutingAdapter.
            sparsity_density: Density of the 16x16 sparse blocks.
            target_layers: List of layer substring identifiers to quantize.
        """
        if isinstance(target_layers, str):
            if target_layers.lower() == "all":
                target_substrings = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            else:
                target_substrings = [s.strip() for s in target_layers.split(",")]
        else:
            target_substrings = target_layers or ["mlp.down_proj", "mlp.up_proj"]

        if routing_adapter is None:
            routing_adapter = VarianceRoutingAdapter()

        if not isinstance(routing_adapter, BaseRoutingAdapter):
            raise ValueError("routing_adapter must be an instance of a subclass of BaseRoutingAdapter.")

        print("\n" + "="*80)
        print("          LAUNCHING SACCADE SYSTEM GRAPH COMPILE INTERFACE              ")
        print("="*80)
        print(f"[Compiler] Initializing Global Optimization (Budget target: {target_bpt} BPT)...")

        if calibration_text is None:
            from datasets import load_dataset
            print("[Compiler] No calibration dataset provided. Defaulting to wikitext-2-raw-v1.")
            calib_dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
            calibration_text = "\n".join([text for text in calib_dataset["text"][:15] if len(text.strip()) > 0])

        layer_metrics = {}
        hooks = []
        def make_hook(name):
            def hook(module, input, output):
                x = input[0].detach()
                scores = routing_adapter.compute_token_complexity(x.view(-1, x.size(-1)))
                layer_metrics[name].extend(scores.cpu().numpy())
            return hook

        target_tuple = tuple(target_substrings)
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                for target in target_tuple:
                    if target in name:
                        layer_metrics[name] = []
                        hooks.append(module.register_forward_hook(make_hook(name)))
                        break

        tokens = tokenizer(calibration_text, return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            model(**tokens.to(model.device))

        for hook in hooks:
            hook.remove()

        # Synthesize global activation targets across unified information canvas
        all_scores = np.array([score for scores in layer_metrics.values() for score in scores])
        pct_4 = max(0.01, min(0.94, (8.4 - target_bpt) / 4.0))
        global_t4 = float(np.quantile(all_scores, pct_4))
        global_t8 = float(np.quantile(all_scores, pct_4 + (1.0 - pct_4 - 0.05)))
        thresholds = (global_t4, global_t8)

        print(f"[Compiler] Unified scale-invariant limits frozen: t4={global_t4:.4f}, t8={global_t8:.4f}")

        tracker = {"tokens_q4": 0, "tokens_q8": 0, "tokens_fp16": 0, "total_tokens": 0, "bytes_saved": 0}
        modules_to_replace = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                for target in target_tuple:
                    if target in name:
                        modules_to_replace.append((name, module))
                        break

        print(f"[Compiler] Unpacking execution arrays and performing manual structural int32 packing allocation...")
        for name, orig_linear in modules_to_replace:
            W_full = orig_linear.weight.data.clone()

            # Formulate asymmetric quantization profile limits
            q_levels = 15
            scale_base = W_full.abs().max(dim=-1, keepdim=True).values / (q_levels / 2)
            scale_base[scale_base == 0] = 1.0
            W_base_raw = torch.round(W_full / scale_base) + 8.0

            # Execute manual 4-bit packaging routine (Saves file space / footprint on device RAM)
            packed_base = SaccadeBitPacker.pack_weights_to_int32(W_base_raw)
            W_dequant = (W_base_raw - 8.0) * scale_base
            W_delta_raw = W_full - W_dequant

            # Apply structured block spatial constraints
            sparse_delta_16 = BlockSparseMatrix(W_delta_raw, block_size=16, sparsity_density=sparsity_density)
            scale_8 = W_delta_raw.abs().max(dim=-1, keepdim=True).values / (255 / 2)
            scale_8[scale_8 == 0] = 1.0
            W_delta_8_raw = torch.round(W_delta_raw / scale_8) * scale_8
            sparse_delta_8 = BlockSparseMatrix(W_delta_8_raw, block_size=16, sparsity_density=sparsity_density)

            plugin_layer = SaccadeLinearPlugin(
                orig_linear, packed_base, scale_base.half(),
                sparse_delta_8, sparse_delta_16, thresholds, name, routing_adapter, tracker
            )

            parts = name.split('.')
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], plugin_layer)

        print("[Saccade] Compilation sequence terminated successfully.")
        return model, tracker
