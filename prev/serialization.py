import os
import torch
from safetensors.torch import save_file
from safetensors import safe_open

from .runtime import SaccadeLinearPlugin
from .adapters.routing import L2NormRoutingAdapter, VarianceRoutingAdapter, BaseRoutingAdapter

class SaccadeModelSaver:
    """Manages metadata wrapping and true zero-copy weight serialization and deserialization."""

    @staticmethod
    def save_checkpoint(model, file_path: str, routing_heuristic_name: str = "variance", target_layers=None):
        """
        Serializes Saccade-compiled layers alongside their operational execution profiles.

        V2 Storage Layout:
        ──────────────────
        Per SaccadeLinearPlugin layer, the checkpoint stores:
          - packed_base (INT32): Bit-packed 4-bit base weights
          - scale_base (FP16): Per-row dequantization scale factors
          - W_delta_8_dense (FP16): Pre-materialized 8-bit delta correction matrix
          - W_delta_16_dense (FP16): Pre-materialized 16-bit delta correction matrix
          - bias (FP16, optional): Original linear bias

        Metadata includes per-layer threshold scalars (t4, t8) and global config.
        """
        state_dict = {}
        meta = {
            "routing_heuristic": routing_heuristic_name,
            "quantization_backend": "custom_int32_packed_ctarq",
            "runtime_version": "2",  # Marks the V2 dense-delta format
        }

        if target_layers is not None:
            if isinstance(target_layers, list):
                meta["target_layers"] = ",".join(target_layers)
            else:
                meta["target_layers"] = str(target_layers)

        for name, module in model.named_modules():
            if isinstance(module, SaccadeLinearPlugin):
                state_dict[f"{name}.packed_base"] = module.packed_base.cpu()
                state_dict[f"{name}.scale_base"] = module.scale_base.cpu()

                # V2: Store pre-materialized dense delta matrices directly.
                # These are already dense FP16 buffers — no sparse reconstruction needed at load.
                state_dict[f"{name}.W_delta_8_dense"] = module.W_delta_8_dense.cpu()
                state_dict[f"{name}.W_delta_16_dense"] = module.W_delta_16_dense.cpu()

                if module.bias is not None:
                    state_dict[f"{name}.bias"] = module.bias.cpu()

                meta[f"{name}_t4"] = str(module.t4)
                meta[f"{name}_t8"] = str(module.t8)

        dir_path = os.path.dirname(file_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        save_file(state_dict, file_path, metadata=meta)
        print(f"[Serialization] Safe archive written successfully to storage disk: {file_path}")

    @staticmethod
    def load_checkpoint(model, file_path: str, device="cpu", routing_heuristic: BaseRoutingAdapter = None):
        """
        Deserializes a safetensors archive and dynamically transforms the appropriate
        linear layers of a raw HuggingFace model into SaccadeLinearPlugins.

        V2 Loading Strategy:
        ────────────────────
        1. Reads pre-materialized dense delta matrices directly (no sparse reconstruction)
        2. Creates lightweight BlockSparseMatrix wrappers for API compatibility with
           SaccadeLinearPlugin.__init__ (which calls .to_dense() once at init anyway)
        3. Streams weights via zero-copy safe_open directly to target device

        Also supports V1 checkpoints by detecting sparse state keys and reconstructing
        dense deltas on the fly during loading.
        """
        print(f"[Serialization] Loading Saccade optimized checkpoint from: {file_path}")

        with safe_open(file_path, framework="pt", device=str(device)) as f:
            metadata = f.metadata() or {}

            if routing_heuristic is None:
                routing_heuristic_name = metadata.get("routing_heuristic", "variance")
                if routing_heuristic_name == "variance":
                    routing_adapter = VarianceRoutingAdapter()
                elif routing_heuristic_name == "l2_norm":
                    routing_adapter = L2NormRoutingAdapter()
                else:
                    print(f"[Serialization] Unknown routing heuristic '{routing_heuristic_name}'. Defaulting to variance.")
                    routing_adapter = VarianceRoutingAdapter()
            else:
                routing_adapter = routing_heuristic

            # Store file keys as a set for O(1) membership validation
            file_keys = f.keys()
            file_keys_set = set(file_keys)

            runtime_version = metadata.get("runtime_version", "1")

            # Dynamically derive unique plugin names from the file structure
            plugin_names = set([k.rsplit(".", 1)[0] for k in file_keys])
            tracker = {"tokens_q4": 0, "tokens_q8": 0, "tokens_fp16": 0, "total_tokens": 0, "bytes_saved": 0}

            print(f"[Serialization] Graph Mutation Engine processing {len(plugin_names)} target submodules (format v{runtime_version})...")

            for name in plugin_names:
                # O(1) targeted module lookup - skips walking the entire model tree graph
                try:
                    orig_linear = model.get_submodule(name)
                except AttributeError:
                    print(f"[Serialization] Warning: Submodule {name} not found in model hierarchy. Skipping.")
                    continue

                t4 = float(metadata.get(f"{name}_t4", 0.0))
                t8 = float(metadata.get(f"{name}_t8", 0.0))

                # Zero-Copy Streaming: Pull weights from disk directly onto target device
                packed_base = f.get_tensor(f"{name}.packed_base")
                scale_base = f.get_tensor(f"{name}.scale_base")

                bias_key = f"{name}.bias"
                bias = f.get_tensor(bias_key) if bias_key in file_keys_set else None

                if bias is not None:
                    orig_linear.bias = torch.nn.Parameter(bias)

                # Detect checkpoint format and load accordingly
                delta_8_dense_key = f"{name}.W_delta_8_dense"
                delta_16_dense_key = f"{name}.W_delta_16_dense"

                if delta_8_dense_key in file_keys_set and delta_16_dense_key in file_keys_set:
                    # V2 format: dense deltas stored directly
                    W_delta_8_dense = f.get_tensor(delta_8_dense_key)
                    W_delta_16_dense = f.get_tensor(delta_16_dense_key)
                else:
                    # V1 backward compatibility: reconstruct dense from sparse state
                    from .core.packing import BlockSparseMatrix

                    def parse_tuple(s):
                        return tuple(map(int, s.strip('()').split(', ')))

                    shape_8_padded = parse_tuple(metadata.get(f"{name}_8_padded"))
                    shape_8_orig = parse_tuple(metadata.get(f"{name}_8_orig"))
                    shape_16_padded = parse_tuple(metadata.get(f"{name}_16_padded"))
                    shape_16_orig = parse_tuple(metadata.get(f"{name}_16_orig"))

                    W_delta_8_active = f.get_tensor(f"{name}.W_delta_8_active_blocks")
                    W_delta_8_mask = f.get_tensor(f"{name}.W_delta_8_block_mask")
                    W_delta_16_active = f.get_tensor(f"{name}.W_delta_16_active_blocks")
                    W_delta_16_mask = f.get_tensor(f"{name}.W_delta_16_block_mask")

                    sparse_8 = BlockSparseMatrix(
                        active_blocks_data=W_delta_8_active,
                        block_mask=W_delta_8_mask,
                        padded_shape=shape_8_padded,
                        orig_shape=shape_8_orig
                    )
                    sparse_16 = BlockSparseMatrix(
                        active_blocks_data=W_delta_16_active,
                        block_mask=W_delta_16_mask,
                        padded_shape=shape_16_padded,
                        orig_shape=shape_16_orig
                    )

                    W_delta_8_dense = sparse_8.to_dense(torch.device(device)).to(torch.float16)
                    W_delta_16_dense = sparse_16.to_dense(torch.device(device)).to(torch.float16)

                    print(f"  [V1→V2 migration] Converted sparse deltas to dense for: {name}")

                # Construct the plugin layer directly with pre-materialized dense deltas.
                # The _DenseAsSparseBridge wraps dense tensors with a .to_dense() method
                # for API compatibility with SaccadeLinearPlugin.__init__.
                bridge_8 = _DenseAsSparseBridge(W_delta_8_dense)
                bridge_16 = _DenseAsSparseBridge(W_delta_16_dense)

                plugin_layer = SaccadeLinearPlugin(
                    orig_linear, packed_base, scale_base,
                    bridge_8, bridge_16, (t4, t8), name, routing_adapter, tracker
                )

                # Fast parent mapping using standard PyTorch rsplit patterns
                if '.' in name:
                    parent_name, child_name = name.rsplit('.', 1)
                    parent = model.get_submodule(parent_name)
                else:
                    parent = model
                    child_name = name

                setattr(parent, child_name, plugin_layer)

        print(f"[Serialization] Saccade checkpoint loaded successfully and injected into execution graph.")
        return model, tracker


class _DenseAsSparseBridge:
    """
    Lightweight shim that wraps a pre-materialized dense tensor to satisfy
    the .to_dense(device) interface expected by SaccadeLinearPlugin.__init__.

    Used during checkpoint loading when delta matrices are already in dense
    format (V2 checkpoints or V1→V2 migrations). Avoids constructing full
    BlockSparseMatrix objects for data that's already dense.
    """
    def __init__(self, dense_tensor):
        self._dense = dense_tensor

    def to_dense(self, target_device):
        return self._dense.to(target_device)

