import torch

class BaseRoutingAdapter:
    """
    Polymorphic base adapter class for routing logic.
    Developers can extend this class to implement custom complexity metrics.
    """
    def compute_token_complexity(self, x: torch.Tensor) -> torch.Tensor:
        """
        Accepts flat activation states and returns a 1D tensor of scores.
        """
        raise NotImplementedError("Developers must implement compute_token_complexity.")

class L2NormRoutingAdapter(BaseRoutingAdapter):
    """
    Computes geometric vector distance in a single hardware reduction pass.
    Fast on physical chips, but may suffer from uncentered activation drift.
    """
    def compute_token_complexity(self, x: torch.Tensor) -> torch.Tensor:
        return torch.norm(x.float(), p=2, dim=-1)

class VarianceRoutingAdapter(BaseRoutingAdapter):
    """
    Highly effective at identifying giant outlier activation peaks by
    subtracting the vector mean first. This protects against uncentered activation drift.
    """
    def compute_token_complexity(self, x: torch.Tensor) -> torch.Tensor:
        return x.float().var(dim=-1)
