import torch

class SaccadeBitPacker:
    """
    Executes true physical bit-packing. Compresses 8 distinct 4-bit weights
    into a singular, contiguous int32 memory container to yield a clean 75% footprint drop.
    """
    @staticmethod
    def pack_weights_to_int32(weight_tensor_int8: torch.Tensor) -> torch.Tensor:
        N, K = weight_tensor_int8.shape
        assert K % 8 == 0, f"Matrix dimension K={K} must be divisible by 8 for uniform packing."

        device_target = weight_tensor_int8.device
        clamped_w = torch.clamp(weight_tensor_int8, 0, 15).to(torch.int32)
        packed = torch.zeros((N, K // 8), dtype=torch.int32, device=device_target)

        for i in range(8):
            packed |= (clamped_w[:, i::8] & 0xF) << (i * 4)

        return packed

class BlockSparseMatrix:
    """
    Enforces rigid block-sparsity constraints on residual arrays to protect
    contiguous hardware memory bus streaming lines.
    """
    def __init__(self, dense_matrix: torch.Tensor=None, block_size: int = 16, sparsity_density: float = 0.25,
                 active_blocks_data: torch.Tensor = None, block_mask: torch.Tensor = None,
                 padded_shape: tuple = None, orig_shape: tuple = None):
        self.block_size = block_size

        # Fast initialization from pre-computed sparse states (Deserialization)
        if active_blocks_data is not None and block_mask is not None:
            self.active_blocks_data = active_blocks_data
            self.block_mask = block_mask
            self.padded_N, self.padded_K = padded_shape
            self.N, self.K = orig_shape
            return

        # Standard Dense -> Sparse Compilation
        if dense_matrix is None:
            raise ValueError("Must provide either dense_matrix or precomputed sparse structures.")

        self.N, self.K = dense_matrix.shape

        pad_n = (block_size - (self.N % block_size)) % block_size
        pad_k = (block_size - (self.K % block_size)) % block_size
        padded = torch.nn.functional.pad(dense_matrix, (0, pad_k, 0, pad_n)) if (pad_n > 0 or pad_k > 0) else dense_matrix.clone()

        self.padded_N, self.padded_K = padded.shape
        n_blocks = self.padded_N // block_size
        k_blocks = self.padded_K // block_size

        blocks = padded.view(n_blocks, block_size, k_blocks, block_size).permute(0, 2, 1, 3)
        block_norms = torch.norm(blocks.float(), p=2, dim=(2, 3))

        k_elements = max(1, int(block_norms.numel() * sparsity_density))
        threshold = torch.topk(block_norms.view(-1), k_elements).values[-1]
        self.block_mask = block_norms >= threshold

        self.active_blocks_data = blocks[self.block_mask].half()

    def get_sparse_state(self):
        """Returns the compressed structures for native buffer residency and serialization."""
        return {
            "active_blocks_data": self.active_blocks_data,
            "block_mask": self.block_mask,
            "padded_shape": (self.padded_N, self.padded_K),
            "orig_shape": (self.N, self.K)
        }

    def to_dense(self, target_device: torch.device) -> torch.Tensor:
        """
        Reconstructs the full zero-padded dense tensor.
        Only used internally for specific AOT compilation traces if dynamic shapes aren't viable,
        or for fallback debugging.
        """
        n_blocks = self.padded_N // self.block_size
        k_blocks = self.padded_K // self.block_size
        reconstructed = torch.zeros(n_blocks, k_blocks, self.block_size, self.block_size, device=target_device, dtype=torch.float16)
        reconstructed[self.block_mask.to(target_device)] = self.active_blocks_data.to(target_device)
        dense = reconstructed.permute(0, 2, 1, 3).reshape(self.padded_N, self.padded_K)
        return dense[:self.N, :self.K]
