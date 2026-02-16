import torch
import torch.nn as nn


# Note: This is not a usual implementation of Adaptive Layer Normalization;
# This allows to initialize old gamma and beta as bias in the modulation layer;
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

        self.norm = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)

        # Modulation layer to learn the weights for the adaptive normalization
        self.modulation = nn.Sequential(
            nn.GELU(),
            nn.Linear(condition_dim, output_dim, bias=True, device=device, dtype=dtype),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.modulation[-1].weight)

        # init modulation bias first half to 1 and second half to 0
        nn.init.ones_(self.modulation[-1].bias[: self.hidden_dim])
        nn.init.zeros_(self.modulation[-1].bias[self.hidden_dim :])

    def forward(self, input: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.modulation(condition).chunk(2, dim=-1)
        return gamma * self.norm(input) + beta


__all__ = ["AdaptiveLayerNorm"]
