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
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.condition_dim = condition_dim
        self.has_bias = bias
        self.eps = eps

        # Base normalization without learnable affine parameters
        self.norm = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)

        # Modulation network: condition -> (gamma, beta)
        output_dim = hidden_dim * 2 if bias else hidden_dim
        self.modulation = nn.Sequential(
            nn.GELU(),
            nn.Linear(
                condition_dim,
                output_dim,
                bias=True,
                device=device,
                dtype=dtype,
            ),
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.modulation[-1].weight)

        # Initialize modulation bias
        # gamma bias to 1.0, beta bias to 0.0
        nn.init.ones_(self.modulation[-1].bias[: self.hidden_dim])
        if self.has_bias:
            nn.init.zeros_(self.modulation[-1].bias[self.hidden_dim :])

    def forward(
        self,
        input: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        # Compute modulation parameters
        mod_params = self.modulation(condition.unsqueeze(1))
        if self.has_bias:
            gamma, beta = mod_params.chunk(2, dim=-1)
        else:
            gamma = mod_params
            beta = None

        # Apply adaptive norm
        normalized = self.norm(input)
        return gamma * normalized + (beta if beta is not None else 0)


__all__ = ["AdaptiveLayerNorm"]
