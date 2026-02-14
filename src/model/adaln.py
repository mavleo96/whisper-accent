import torch
import torch.nn as nn


# Note: This is not a usual implementation of Adaptive Layer Normalization;
# This allows to wrap an existing LayerNorm and add a modulation layer to it;
# Also the gamma_cond is going to impact the beta_old in this implementation;
# We use GeLu here to match activation in whisper architecture;
class AdaptiveLayerNorm(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        condition_dim: int,
        eps: float = 1e-5,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.hidden_dim = hidden_dim
        self.condition_dim = condition_dim
        output_dim = hidden_dim * 2 if bias else hidden_dim

        self.norm = nn.LayerNorm(
            hidden_dim, eps=eps, elementwise_affine=True, device=device, dtype=dtype
        )
        self.modulation = nn.Sequential(
            nn.GELU(),
            nn.Linear(condition_dim, output_dim, bias=False, device=device, dtype=dtype),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.ones_(self.norm.weight)
        nn.init.zeros_(self.norm.bias)

    def forward(self, input: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.modulation(condition).chunk(2, dim=-1)
        return (1 + gamma) * self.norm(input) + beta


__all__ = ["AdaptiveLayerNorm"]
