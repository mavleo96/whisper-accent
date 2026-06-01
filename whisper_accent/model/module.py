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


class MultiHeadAttentionPooling(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_seeds: int = 1,
        dropout: float = 0.0,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()

        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
            )

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_seeds = num_seeds
        self.dropout = dropout

        # Learnable pooling queries
        self.seeds = nn.Parameter(torch.empty(num_seeds, hidden_dim, device=device, dtype=dtype))

        # Multi-head attention layer
        self.mha = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
            batch_first=True,
            device=device,
            dtype=dtype,
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.seeds, mean=0.0, std=0.02)
        self.mha._reset_parameters()

    def forward(
        self,
        input: torch.Tensor,
        key_padding_mask: torch.BoolTensor | None = None,
    ) -> torch.Tensor:
        B, _, _ = input.shape

        # Learnable queries broadcast across batch
        query = self.seeds.unsqueeze(0).expand(B, -1, -1)  # [B, S, D]

        # Pooling input feature over queries
        out, _ = self.mha(
            query=query,
            key=input,
            value=input,
            key_padding_mask=key_padding_mask,
            need_weights=False,  # enables optimized SDPA backend
            is_causal=False,
        )  # -> [B, num_seeds, D]

        if self.num_seeds == 1:
            out = out.squeeze(1)

        return out


__all__ = ["AdaptiveLayerNorm", "MultiHeadAttentionPooling"]
